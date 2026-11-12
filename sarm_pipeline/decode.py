"""Ordered decoding of per-frame stage probabilities over a whole episode.

A demo moves through the stages in order, so instead of taking each frame's argmax we find
the most likely stage sequence that only moves forward (Viterbi):

  [dummy before start] -> stage 0 -> stage 1 -> ... -> stage S-1 -> [dummy after end]

Moving to the next state costs `switch_penalty` (in log-probability units), skipping k states
costs an extra `skip_penalty * k`. Backward moves are not allowed; failure episodes that undo
work need a different decoder (or the raw per-frame argmax).
"""
from __future__ import annotations

import numpy as np


def ordered_viterbi(probs: np.ndarray, switch_penalty: float = 5.0, skip_penalty: float = 5.0) -> np.ndarray:
    """probs: (T, S + 1) per-frame probabilities, last column = dummy. Returns (T,) stage indices."""
    n_stages = probs.shape[1] - 1
    dummy = n_stages
    emit_class = np.array([dummy] + list(range(n_stages)) + [dummy])     # chain state -> class
    logp = np.log(np.clip(probs, 1e-6, 1.0))[:, emit_class]               # (T, n_states)
    n = len(emit_class)

    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    trans = np.full((n, n), -np.inf)                                      # trans[from, to]
    trans[i == j] = 0.0
    fwd = j > i
    trans[fwd] = -(switch_penalty + skip_penalty * (j[fwd] - i[fwd] - 1))

    T = len(logp)
    score = logp[0] + np.where(np.arange(n) <= 1, 0.0, -np.inf)           # start in pre-dummy or stage 0
    back = np.zeros((T, n), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + trans                                     # (from, to)
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return emit_class[path]
