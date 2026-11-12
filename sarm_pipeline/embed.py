"""Precompute frozen SigLIP-2 embeddings for every frame of the selected episodes.

SigLIP is frozen in SARM2, so a frame always maps to the same vector. Computing them
once means training never decodes video or runs the image encoder.

Each mp4 is decoded once, front to back, by a pool of worker processes. Frames are
resized straight to 224x224 by the decoder (area interpolation), then normalized the
way the SigLIP image processor does: x / 255, then (x - 0.5) / 0.5. Training and
labeling read the same cache, so they always see identical preprocessing.

Cache layout (cache_dir):
  index.parquet   episode_index, length, row_start   (rows are episode frames, in order)
  state.npy       (N, state_dim) float32, raw joint state
  emb_<cam>.npy   (N, 768) float16, one per camera
  segments.json   stage boundaries per episode
  text.pt         SigLIP text embeddings of the stage names, "dummy" and the task
  meta.json       encoder settings and which video files are done (for resuming)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoModel, AutoProcessor

from .config import Config
from .v3 import V3Dataset

DUMMY = "dummy"


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SiglipEncoder:
    def __init__(self, ckpt: str, device: torch.device):
        self.device = device
        self.dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
        self.model = AutoModel.from_pretrained(ckpt, dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

    @torch.no_grad()
    def images(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, H, W, 3) uint8 at the encoder resolution -> (B, D) float32."""
        x = frames.to(self.device, non_blocking=True).permute(0, 3, 1, 2).to(self.dtype)
        x = (x / 255.0 - 0.5) / 0.5
        return self.model.get_image_features(pixel_values=x).float()

    @torch.no_grad()
    def texts(self, texts: list[str]) -> torch.Tensor:
        # Same tokenization as opensarm's FrozenSiglipEncoder.encode_text.
        inputs = self.processor(text=texts, return_tensors="pt", padding="max_length",
                                max_length=64, truncation=True).to(self.device)
        return self.model.get_text_features(**inputs).float()


class _DecodeJobs(IterableDataset):
    """Each job is one mp4; yields chunks of (rows, frames) for the frames the cache needs."""

    def __init__(self, jobs: list[dict], fps: int, size: int, threads: int, chunk: int):
        self.jobs, self.fps, self.size, self.threads, self.chunk = jobs, fps, size, threads, chunk

    def __iter__(self):
        info = get_worker_info()
        jobs = self.jobs if info is None else self.jobs[info.id::info.num_workers]
        for job in jobs:
            yield from self._decode(job)

    def _decode(self, job: dict):
        wanted = dict(zip(job["video_frames"], job["rows"]))
        first, last = min(wanted), max(wanted)
        found, rows, frames = 0, [], []
        with av.open(job["path"]) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = self.threads
            if first > 0:
                container.seek(int(first / self.fps / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                vf = round(float(frame.pts * stream.time_base) * self.fps)
                if vf > last:
                    break
                row = wanted.get(vf)
                if row is None:
                    continue
                img = frame.reformat(width=self.size, height=self.size, format="rgb24", interpolation="AREA")
                rows.append(row)
                frames.append(img.to_ndarray())
                found += 1
                if len(rows) == self.chunk:
                    yield {"job": job["id"], "camera": job["camera"], "rows": np.array(rows),
                           "frames": np.stack(frames), "last": False}
                    rows, frames = [], []
        if found != len(wanted):
            raise RuntimeError(f"{job['path']}: decoded {found} of {len(wanted)} wanted frames")
        yield {"job": job["id"], "camera": job["camera"], "rows": np.array(rows, dtype=np.int64),
               "frames": np.stack(frames) if frames else np.zeros((0, self.size, self.size, 3), np.uint8),
               "last": True}


def build_cache(cfg: Config, ds: V3Dataset, episodes: list[int]) -> Path:
    out = Path(cfg.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cameras = dict(cfg.dataset.cameras)
    lengths = [int(ds.episodes.loc[e, "length"]) for e in episodes]
    index = pd.DataFrame({"episode_index": episodes, "length": lengths,
                          "row_start": np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)})
    n_rows = int(sum(lengths))
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fresh = meta.get("episodes") != episodes or meta.get("encoder") != cfg.encoder.ckpt
    if fresh:
        meta = {"episodes": episodes, "encoder": cfg.encoder.ckpt, "image_size": cfg.encoder.image_size,
                "interpolation": "area", "cameras": cameras, "done_jobs": []}
    index.to_parquet(out / "index.parquet")

    # Joint state and stage boundaries (cheap; rewritten every time).
    states = ds.read_column(episodes, cfg.dataset.state_key)
    np.save(out / "state.npy", np.concatenate([states[e] for e in episodes]))
    segments = {str(e): ds.stage_segments(e, cfg.dataset.stage_prefix) for e in episodes}
    (out / "segments.json").write_text(json.dumps(segments))

    device = pick_device(cfg.train.device)
    encoder = SiglipEncoder(cfg.encoder.ckpt, device)
    stage_names = list(cfg.stages.names) + [DUMMY]
    torch.save({"stage_names": stage_names, "stage_emb": encoder.texts(stage_names).cpu(),
                "task": cfg.dataset.task, "task_emb": encoder.texts([cfg.dataset.task])[0].cpu()},
               out / "text.pt")

    # One decode job per (camera, mp4).
    pm = ds.pointer_map(episodes, cameras).merge(index[["episode_index", "row_start"]], on="episode_index")
    jobs = []
    for (cam, key, chunk, file), group in pm.groupby(["camera", "video_key", "chunk_index", "file_index"]):
        vfs, rows = [], []
        for r in group.itertuples():
            vfs.extend(range(r.video_start_frame, r.video_start_frame + r.length))
            rows.extend(range(r.row_start, r.row_start + r.length))
        jobs.append({"id": f"{cam}/{chunk}/{file}", "camera": cam, "path": str(ds.video_path(key, chunk, file)),
                     "video_frames": vfs, "rows": rows})
    done = set(meta["done_jobs"])
    pending = [j for j in jobs if j["id"] not in done]

    size = int(cfg.encoder.image_size)
    dim = encoder.images(torch.zeros(1, size, size, 3, dtype=torch.uint8)).shape[-1]
    embs = {}
    for cam in cameras:
        path = out / f"emb_{cam}.npy"
        if fresh or not path.exists():
            embs[cam] = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n_rows, dim))
        else:
            embs[cam] = np.load(path, mmap_mode="r+")
    meta_path.write_text(json.dumps(meta))
    print(f"[embed] {n_rows} frames x {len(cameras)} cameras, {len(pending)}/{len(jobs)} video files to do, "
          f"device={device}")
    if not pending:
        return out

    workers = int(cfg.encoder.decode_workers)
    loader = DataLoader(_DecodeJobs(pending, ds.fps, cfg.encoder.image_size, cfg.encoder.decode_threads,
                                    cfg.encoder.chunk_frames),
                        batch_size=None, num_workers=workers, prefetch_factor=4 if workers else None)
    t0, n_done = time.time(), 0
    bs = int(cfg.encoder.batch_size)
    for item in loader:
        frames, rows = torch.as_tensor(item["frames"]), np.asarray(item["rows"])
        for i in range(0, len(rows), bs):
            embs[item["camera"]][rows[i:i + bs]] = encoder.images(frames[i:i + bs]).cpu().numpy().astype(np.float16)
        n_done += len(rows)
        if item["last"]:
            embs[item["camera"]].flush()
            meta["done_jobs"].append(item["job"])
            meta_path.write_text(json.dumps(meta))
            rate = n_done / (time.time() - t0)
            left = n_rows * len(cameras) - n_done - sum(len(j["rows"]) for j in jobs if j["id"] in done)
            print(f"[embed] {len(meta['done_jobs'])}/{len(jobs)} files, {rate:.0f} frames/s, "
                  f"~{max(left, 0) / max(rate, 1) / 60:.1f} min left", flush=True)
    return out
