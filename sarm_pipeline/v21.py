"""Read local LeRobot v2.1 datasets (one parquet + one mp4 per camera per episode), as the pipeline's dataset.

Exposes the subset of V3Dataset's interface that `embed` and the labeler use. Several v2.1 datasets can be read
as one: each source gets an episode-index offset (source 0 keeps its indices, source 1 starts at 1000, ...).
Everything is resampled to one frame rate (`fps`, default 30) so that windows (`frame_gap`) mean the same
time in every source: pipeline frame i of an episode is native frame round(i * native_fps / fps).

    sources: [{root: data/client/fold_clothes, offset: 0}, {root: data/client/nurvai2, offset: 1000}]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class V21Dataset:
    def __init__(self, sources: list[dict], fps: int = 30):
        self.fps = int(fps)
        rows, self.sources = [], []
        for src in sources:
            root = Path(src["root"])
            info = json.loads((root / "meta" / "info.json").read_text())
            if info.get("codebase_version") != "v2.1":
                raise ValueError(f"{root}: expected v2.1, got {info.get('codebase_version')}")
            self.sources.append({"root": root, "info": info})
            for line in (root / "meta" / "episodes.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                e = json.loads(line)
                native = int(e["length"])
                nfps = int(info["fps"])
                rows.append({"episode_index": int(src.get("offset", 0)) + int(e["episode_index"]),
                             "source": len(self.sources) - 1, "native_index": int(e["episode_index"]),
                             "native_fps": nfps, "native_length": native, "tasks": list(e.get("tasks", [])),
                             "length": int(np.floor((native - 1) * self.fps / nfps)) + 1})
        self.episodes = pd.DataFrame(rows).set_index("episode_index").sort_index()
        self.video_keys = sorted({k for s in self.sources for k, f in s["info"]["features"].items()
                                  if f["dtype"] == "video"})

    # --- helpers ---------------------------------------------------------------------------
    def native_frames(self, ep: int) -> np.ndarray:
        """Native frame index of every pipeline frame of `ep`."""
        m = self.episodes.loc[ep]
        i = np.arange(int(m["length"]))
        return np.minimum(np.round(i * m["native_fps"] / self.fps).astype(np.int64), int(m["native_length"]) - 1)

    def _path(self, ep: int, kind: str, video_key: str | None = None) -> Path:
        m = self.episodes.loc[ep]
        src = self.sources[int(m["source"])]
        chunk = int(m["native_index"]) // int(src["info"].get("chunks_size", 1000))
        fmt = src["info"]["data_path" if kind == "data" else "video_path"]
        return src["root"] / fmt.format(episode_chunk=chunk, episode_index=int(m["native_index"]), video_key=video_key)

    # --- interface used by embed ------------------------------------------------------------
    def annotated_episodes(self, stage_prefix: str) -> list[int]:
        return []

    def pointer_map(self, episodes: list[int], cameras: dict[str, str]) -> pd.DataFrame:
        """One row per (episode, camera). `video_frames` lists the native frames to decode (resampled)."""
        rows = []
        for ep in episodes:
            frames = self.native_frames(ep)
            for cam, key in cameras.items():
                rows.append({"episode_index": ep, "camera": cam, "video_key": key, "chunk_index": 0,
                             "file_index": ep, "video_start_frame": 0, "length": len(frames),
                             "video_frames": frames.tolist(), "native_fps": int(self.episodes.loc[ep, "native_fps"])})
        return pd.DataFrame(rows)

    def video_path(self, video_key: str, chunk: int, file: int) -> Path:
        return self._path(int(file), "video", video_key)        # file_index is the episode (see pointer_map)

    def read_column(self, episodes: list[int], column: str) -> dict[int, np.ndarray]:
        out = {}
        for ep in episodes:
            df = pq.read_table(self._path(ep, "data"), columns=["frame_index", column]).to_pandas()
            df = df.sort_values("frame_index")
            arr = np.stack(df[column].to_numpy()).astype(np.float32)
            if len(arr) != int(self.episodes.loc[ep, "native_length"]):
                raise ValueError(f"episode {ep}: {len(arr)} rows of {column}, metadata says "
                                 f"{self.episodes.loc[ep, 'native_length']}")
            out[ep] = arr[self.native_frames(ep)]
        return out

    def stage_segments(self, episode: int, stage_prefix: str) -> tuple[list[str], list[int], list[int]]:
        return [], [], []                                        # client data: stages come from hand labels
