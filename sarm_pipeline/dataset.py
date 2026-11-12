"""Training windows built from the embedding cache.

Everything lives in memory as flat per-frame arrays (one row per episode frame):
SigLIP vectors per camera, normalized joint state, stage and progress labels. A sample
is an anchor frame plus a short history window, gathered by row index.

Window layout, matching opensarm's FrameGapLeRobotDataset.get_frame_indices with
n_obs_steps = 6 and frame_gap = 30:
  reward model  [episode start] + 5 frames 1 s apart before the anchor + [anchor]  = 7
  stage model                     5 frames 1 s apart before the anchor + [anchor]  = 6
With less than n_obs_steps * frame_gap frames of history, the history frames are spaced
evenly between the episode start and the anchor instead.

Reward-model training samples may also get 1-3 "rewind" frames appended: recent frames
in reverse order with their (lower) progress, so the model learns that undoing work
lowers progress. Validation and labeling never rewind.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import episode_labels, stage_shares


@dataclass
class Cache:
    episodes: list[int]
    length: dict[int, int]
    row_start: dict[int, int]
    emb: dict[str, np.ndarray]        # camera -> (N, D) float16
    state: np.ndarray                 # (N, state_dim) float32, normalized
    stage: np.ndarray                 # (N,) int64, dummy = number of stages
    progress: np.ndarray              # (N,) float32
    stage_names: list[str]            # includes "dummy" last
    stage_emb: torch.Tensor           # (S + 1, D)
    task_emb: torch.Tensor            # (D,)
    state_mean: np.ndarray
    state_std: np.ndarray
    alpha: np.ndarray                 # stage shares used for stage-aware progress

    @property
    def cameras(self) -> list[str]:
        return list(self.emb)


def load_cache(cache_dir: str | Path, cameras: list[str], stage_names: list[str], progress_mode: str,
               train_episodes: list[int], in_memory: bool = True, segments_override: str | None = None) -> Cache:
    """Load the cache and build labels. Stats (state normalization, stage shares) come from train episodes only."""
    cache_dir = Path(cache_dir)
    index = pd.read_parquet(cache_dir / "index.parquet")
    episodes = [int(e) for e in index["episode_index"]]
    length = dict(zip(episodes, index["length"].astype(int)))
    row_start = dict(zip(episodes, index["row_start"].astype(int)))
    segments = {int(k): v for k, v in json.loads((cache_dir / "segments.json").read_text()).items()}
    if segments_override:          # relabeled episodes (e.g. labels/hand_train.json) replace the dataset's
        segments.update({int(k): v for k, v in json.loads(Path(segments_override).read_text()).items()
                         if not k.startswith("_")})
    text = torch.load(cache_dir / "text.pt")
    if text["stage_names"][:-1] != list(stage_names):
        raise ValueError("cache was built with different stage names; rebuild it with `embed`")

    train_rows = np.concatenate([np.arange(row_start[e], row_start[e] + length[e]) for e in train_episodes])
    raw_state = np.load(cache_dir / "state.npy")
    mean = raw_state[train_rows].mean(0)
    std = np.maximum(raw_state[train_rows].std(0), 1e-6)

    alpha = stage_shares([segments[e] for e in train_episodes], list(stage_names))
    stage = np.empty(len(raw_state), dtype=np.int64)
    progress = np.empty(len(raw_state), dtype=np.float32)
    for e in episodes:
        s, p = episode_labels(length[e], segments[e], list(stage_names), alpha, progress_mode)
        stage[row_start[e]:row_start[e] + length[e]] = s
        progress[row_start[e]:row_start[e] + length[e]] = p

    emb = {}
    for cam in cameras:
        arr = np.load(cache_dir / f"emb_{cam}.npy", mmap_mode="r")
        emb[cam] = np.ascontiguousarray(arr) if in_memory else arr
    return Cache(episodes, length, row_start, emb, ((raw_state - mean) / std).astype(np.float32), stage, progress,
                 text["stage_names"], text["stage_emb"], text["task_emb"], mean, std, alpha)


def frame_window(idx: int, ep_start: int, n_obs_steps: int, frame_gap: int, include_start: bool) -> list[int]:
    """opensarm's get_frame_indices: [ep_start] + (n_obs_steps - 1) history frames + [idx]."""
    if n_obs_steps <= 1:
        return ([ep_start] if include_start else []) + [idx]
    distance = idx - ep_start
    if distance >= frame_gap * n_obs_steps:
        mid = [idx - frame_gap * j for j in range(n_obs_steps - 1, 0, -1)]
    else:
        mid = [ep_start + round(distance * k / n_obs_steps) for k in range(1, n_obs_steps)]
    frames = ([ep_start] if include_start else []) + mid + [idx]
    for i in range(1, len(frames)):
        frames[i] = max(frames[i], frames[i - 1])
    return frames


def segment_bounds(stage: np.ndarray, row: int, lo: int, hi: int) -> int:
    """First row of the contiguous same-stage run containing `row` (within [lo, hi))."""
    start = row
    while start > lo and stage[start - 1] == stage[row]:
        start -= 1
    return start


class WindowDataset(Dataset):
    """mode = "stage" (stage estimator) or "reward" (reward model)."""

    def __init__(self, cache: Cache, episodes: list[int], mode: str, n_obs_steps: int, frame_gap: int,
                 max_rewind_steps: int = 0, rewind_prob: float = 0.0, stage_window: str = "episode",
                 anchors: np.ndarray | None = None):
        if mode not in ("stage", "reward"):
            raise ValueError(mode)
        self.c, self.mode = cache, mode
        self.n_obs, self.gap = n_obs_steps, frame_gap
        self.max_rewind = max_rewind_steps if mode == "reward" else 0
        self.rewind_prob = rewind_prob if mode == "reward" else 0.0
        self.stage_window = stage_window
        rows = [np.arange(cache.row_start[e], cache.row_start[e] + cache.length[e]) for e in episodes]
        # row -> (episode start row, episode end row exclusive)
        self.ep_bounds = np.zeros((len(cache.stage), 2), dtype=np.int64)
        for e in cache.episodes:
            s, n = cache.row_start[e], cache.length[e]
            self.ep_bounds[s:s + n] = (s, s + n)
        self.anchors = np.concatenate(rows) if anchors is None else np.asarray(anchors)
        self.slots = (1 + n_obs_steps if mode == "reward" else n_obs_steps) + self.max_rewind

    def __len__(self) -> int:
        return len(self.anchors)

    def window(self, row: int, rewind: int = 0) -> list[int]:
        start, end = self.ep_bounds[row]
        if self.mode == "reward":
            frames = frame_window(row, start, self.n_obs, self.gap, include_start=True)
            # Rewind: the frames just before the anchor, newest first (already in the window).
            frames += [row - self.gap * j for j in range(1, rewind + 1)]
            return frames
        if self.stage_window == "segment":
            start = segment_bounds(self.c.stage, row, start, end)
        return frame_window(row, start, self.n_obs, self.gap, include_start=False)

    def sample_rewind(self, row: int) -> int:
        # torch RNG: DataLoader reseeds it per worker (numpy's would repeat across workers).
        start = self.ep_bounds[row, 0]
        if self.max_rewind == 0 or row <= start + self.n_obs * self.gap or torch.rand(1).item() >= self.rewind_prob:
            return 0
        max_rewind = min(self.max_rewind, (row - start) // self.gap)
        return int(torch.randint(1, max_rewind + 1, (1,)).item())

    def __getitem__(self, i: int) -> dict:
        row = int(self.anchors[i])
        frames = self.window(row, self.sample_rewind(row))
        n = len(frames)
        idx = np.asarray(frames)
        c = self.c
        img = np.zeros((len(c.cameras), self.slots, next(iter(c.emb.values())).shape[1]), dtype=np.float32)
        for k, cam in enumerate(c.cameras):
            img[k, :n] = c.emb[cam][idx]
        state = np.zeros((self.slots, c.state.shape[1]), dtype=np.float32)
        state[:n] = c.state[idx]
        progress = np.zeros(self.slots, dtype=np.float32)
        progress[:n] = c.progress[idx]
        return {
            "img": torch.from_numpy(img),                       # (N_cam, slots, D)
            "state": torch.from_numpy(state),                   # (slots, state_dim)
            "progress": torch.from_numpy(progress),             # (slots,), padding = 0
            "length": torch.tensor(n, dtype=torch.int32),
            "stage": torch.tensor(c.stage[row], dtype=torch.long),   # anchor stage
            "row": torch.tensor(row, dtype=torch.long),
        }


def fixed_anchors(cache: Cache, episodes: list[int], n: int, seed: int) -> np.ndarray:
    """A seeded random subset of frames, reused at every validation check."""
    rows = np.concatenate([np.arange(cache.row_start[e], cache.row_start[e] + cache.length[e]) for e in episodes])
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=min(n, len(rows)), replace=False))
