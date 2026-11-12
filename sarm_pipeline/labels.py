"""Per-frame stage and progress labels from per-episode stage boundaries.

Stages come as (name, start_frame, end_frame) with exclusive ends. Frames outside
every stage get the extra "dummy" class.

Progress modes:
  linear       progress = frame / (length - 1)
  stage_aware  progress = P[k] + alpha[k] * tau      (SARM's formula 2)
               alpha[k] = average share of annotated time spent in stage k (train split)
               P[k]     = sum of alpha before stage k
               tau      = position within the stage, 0 -> 1
               Before the first stage progress is 0, after the last it is 1, and in gaps it
               holds the end value of the previous stage.
"""
from __future__ import annotations

import numpy as np

Segments = tuple[list[str], list[int], list[int]]


def stage_indices(names: list[str], stage_names: list[str]) -> list[int]:
    unknown = sorted(set(names) - set(stage_names))
    if unknown:
        raise ValueError(f"unknown stage names {unknown}; config stages.names must list them")
    return [stage_names.index(n) for n in names]


def stage_shares(segments: list[Segments], stage_names: list[str]) -> np.ndarray:
    """Average fraction of annotated time spent in each stage, across episodes."""
    shares = []
    for names, starts, ends in segments:
        dur = np.zeros(len(stage_names))
        for k, s, e in zip(stage_indices(names, stage_names), starts, ends):
            dur[k] += e - s
        if dur.sum() > 0:
            shares.append(dur / dur.sum())
    if not shares:
        raise ValueError("no annotated episodes to compute stage shares from")
    alpha = np.mean(shares, axis=0)
    return alpha / alpha.sum()


def episode_labels(length: int, segments: Segments, stage_names: list[str], alpha: np.ndarray,
                   mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (stage int16 [length], progress float32 [length]); dummy = len(stage_names)."""
    dummy = len(stage_names)
    stage = np.full(length, dummy, dtype=np.int16)
    names, starts, ends = segments
    ks = stage_indices(names, stage_names)
    for k, s, e in zip(ks, starts, ends):
        stage[max(s, 0):min(e, length)] = k

    if mode == "linear":
        return stage, (np.arange(length) / max(length - 1, 1)).astype(np.float32)
    if mode != "stage_aware":
        raise ValueError(f"unknown progress mode {mode!r}")

    if not ks:
        raise ValueError("stage_aware progress needs stage segments")
    prior = np.concatenate([[0.0], np.cumsum(alpha)[:-1]])
    progress = np.zeros(length, dtype=np.float64)
    held, cursor = 0.0, 0
    for k, s, e in sorted(zip(ks, starts, ends), key=lambda x: x[1]):
        s, e = max(s, cursor), min(e, length)
        progress[cursor:s] = held                       # before the first stage, or a gap
        if e > s:
            progress[s:e] = prior[k] + alpha[k] * (np.arange(s, e) - s) / (e - s)
        held, cursor = prior[k] + alpha[k], max(cursor, e)
    progress[cursor:] = 1.0                             # after the last stage
    return stage, progress.astype(np.float32)
