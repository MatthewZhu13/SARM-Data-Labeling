"""Read a LeRobot v3.0 dataset and map every episode frame to where it is stored.

v3.0 packs many episodes into each data parquet and each camera mp4. For each
(episode, camera) the episode metadata gives the mp4 and the timestamp where the
episode starts inside it, so

    video frame = round(from_timestamp * fps) + episode frame

and the frame's state/action row in the data parquet is

    data row = dataset_from_index + episode frame
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class V3Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError(f"{self.root}: expected a v3.0 dataset, got {self.info.get('codebase_version')}")
        self.fps = int(self.info["fps"])
        self.episodes = _read_episode_meta(self.root)
        self.video_keys = [k for k, f in self.info["features"].items() if f["dtype"] == "video"]

    # --- selection -------------------------------------------------------------------------
    def annotated_episodes(self, stage_prefix: str) -> list[int]:
        col = f"{stage_prefix}_names"
        if col not in self.episodes:
            return []
        return [int(e) for e in self.episodes.index[self.episodes[col].notna()]]

    # --- pointer map -----------------------------------------------------------------------
    def pointer_map(self, episodes: list[int], cameras: dict[str, str]) -> pd.DataFrame:
        """One row per (episode, camera): video file, first video frame, length, data row start."""
        rows = []
        for ep in episodes:
            meta = self.episodes.loc[ep]
            length = int(meta["length"])
            for cam, key in cameras.items():
                start_ts = float(meta[f"videos/{key}/from_timestamp"])
                end_ts = float(meta[f"videos/{key}/to_timestamp"])
                start_frame = round(start_ts * self.fps)
                span = round((end_ts - start_ts) * self.fps)
                if span != length:
                    raise ValueError(f"episode {ep} camera {cam}: video span {span} frames != length {length}")
                rows.append({
                    "episode_index": ep,
                    "camera": cam,
                    "video_key": key,
                    "chunk_index": int(meta[f"videos/{key}/chunk_index"]),
                    "file_index": int(meta[f"videos/{key}/file_index"]),
                    "video_start_frame": start_frame,
                    "length": length,
                    "data_row_start": int(meta["dataset_from_index"]),
                })
        return pd.DataFrame(rows)

    # --- paths -----------------------------------------------------------------------------
    def video_path(self, video_key: str, chunk: int, file: int) -> Path:
        return self.root / self.info["video_path"].format(video_key=video_key, chunk_index=chunk, file_index=file)

    def data_path(self, chunk: int, file: int) -> Path:
        return self.root / self.info["data_path"].format(chunk_index=chunk, file_index=file)

    def data_files(self, episodes: list[int]) -> list[tuple[int, int]]:
        meta = self.episodes.loc[episodes]
        return sorted({(int(c), int(f)) for c, f in zip(meta["data/chunk_index"], meta["data/file_index"])})

    def video_files(self, episodes: list[int], cameras: dict[str, str]) -> list[tuple[str, int, int]]:
        meta = self.episodes.loc[episodes]
        out = set()
        for key in cameras.values():
            for c, f in zip(meta[f"videos/{key}/chunk_index"], meta[f"videos/{key}/file_index"]):
                out.add((key, int(c), int(f)))
        return sorted(out)

    # --- frame data ------------------------------------------------------------------------
    def read_column(self, episodes: list[int], column: str) -> dict[int, np.ndarray]:
        """Per-episode arrays of one data column (e.g. observation.state), ordered by frame."""
        wanted = set(episodes)
        out: dict[int, np.ndarray] = {}
        for chunk, file in self.data_files(episodes):
            table = pq.read_table(self.data_path(chunk, file), columns=["episode_index", "frame_index", column])
            df = table.to_pandas()
            df = df[df["episode_index"].isin(wanted)]
            for ep, group in df.groupby("episode_index"):
                group = group.sort_values("frame_index")
                out[int(ep)] = np.stack(group[column].to_numpy()).astype(np.float32)
        for ep in episodes:
            length = int(self.episodes.loc[ep, "length"])
            if ep not in out or len(out[ep]) != length:
                got = 0 if ep not in out else len(out[ep])
                raise ValueError(f"episode {ep}: {got} rows of {column}, expected {length}")
        return out

    def stage_segments(self, episode: int, stage_prefix: str) -> tuple[list[str], list[int], list[int]]:
        meta = self.episodes.loc[episode]
        names = meta.get(f"{stage_prefix}_names")        # absent in unlabeled datasets (e.g. full_folding)
        if names is None or (isinstance(names, float) and np.isnan(names)):
            return [], [], []
        return (list(names),
                [int(x) for x in meta[f"{stage_prefix}_start_frames"]],
                [int(x) for x in meta[f"{stage_prefix}_end_frames"]])


def _read_episode_meta(root: Path) -> pd.DataFrame:
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root / 'meta' / 'episodes'}")
    frames = []
    for path in files:
        columns = [c for c in pq.read_schema(path).names if not c.startswith("stats/")]
        frames.append(pq.read_table(path, columns=columns).to_pandas())
    return pd.concat(frames).set_index("episode_index").sort_index()


def parse_episode_spec(spec, available: list[int]) -> list[int]:
    """'annotated' handled by the caller; accepts a list, '537-987', or '1,2,5-9'."""
    if isinstance(spec, list):
        return [int(e) for e in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    missing = sorted(set(out) - set(available))
    if missing:
        raise ValueError(f"episodes not in dataset: {missing[:10]}")
    return out
