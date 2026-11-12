"""Download only the files a run needs, choose its episodes, and split them by episode."""
from __future__ import annotations

import json
import random
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import Config
from .v3 import V3Dataset, parse_episode_spec


def open_local(cfg: Config):
    """Local v2.1 sources (client data: `dataset.format: v2.1`, `dataset.sources`); never downloaded."""
    from .v21 import V21Dataset
    return V21Dataset(list(cfg.dataset.sources), int(cfg.dataset.get("fps", 30)))


def fetch_metadata(cfg: Config) -> V3Dataset:
    if cfg.dataset.get("format") == "v2.1":
        return open_local(cfg)
    snapshot_download(cfg.dataset.repo_id, repo_type="dataset", revision=cfg.dataset.revision,
                      local_dir=cfg.dataset.root, allow_patterns=["meta/*"])
    return V3Dataset(cfg.dataset.root)


def select_episodes(cfg: Config, ds: V3Dataset) -> list[int]:
    spec = cfg.dataset.episodes
    if spec == "all":
        episodes = [int(e) for e in ds.episodes.index]
    elif spec == "annotated":
        episodes = ds.annotated_episodes(cfg.dataset.stage_prefix)
        if not episodes:
            raise ValueError(f"no episodes have {cfg.dataset.stage_prefix}_* stage labels")
    else:
        episodes = parse_episode_spec(spec, list(ds.episodes.index))
    if cfg.dataset.max_episodes:
        episodes = episodes[: cfg.dataset.max_episodes]
    return sorted(episodes)


def fetch(cfg: Config) -> tuple[V3Dataset, list[int]]:
    """Download metadata, then the data parquets and camera videos holding the selected episodes."""
    ds = fetch_metadata(cfg)
    episodes = select_episodes(cfg, ds)
    if cfg.dataset.get("format") == "v2.1":
        print(f"[fetch] local v2.1 dataset, {len(episodes)} episodes, nothing to download")
        return ds, episodes
    cameras = dict(cfg.dataset.cameras)
    patterns = [str(ds.data_path(c, f).relative_to(ds.root)) for c, f in ds.data_files(episodes)]
    patterns += [str(ds.video_path(k, c, f).relative_to(ds.root)) for k, c, f in ds.video_files(episodes, cameras)]
    print(f"[fetch] {len(episodes)} episodes -> {len(patterns)} files")
    snapshot_download(cfg.dataset.repo_id, repo_type="dataset", revision=cfg.dataset.revision,
                      local_dir=cfg.dataset.root, allow_patterns=patterns, max_workers=8)
    return V3Dataset(cfg.dataset.root), episodes


def make_split(cfg: Config, episodes: list[int]) -> dict[str, list[int]]:
    """Split by whole episodes with a fixed seed. Reuses the saved split if it matches."""
    path = Path(cfg.split.file)
    if path.exists():
        saved = json.loads(path.read_text())
        if sorted(saved["episodes"]) == sorted(episodes):
            return {k: saved[k] for k in ("train", "val", "test")}
        raise ValueError(f"{path} was made for a different episode set; delete it or set split.file")

    fr = cfg.split.fractions
    shuffled = list(episodes)
    random.Random(cfg.split.seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * fr["val"]))
    n_test = max(1, round(len(shuffled) * fr["test"]))
    split = {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test:n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val:]),
    }
    if not split["train"]:
        raise ValueError("split left no training episodes")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": cfg.split.seed, "fractions": dict(fr), "episodes": sorted(episodes),
                                **split}, indent=1))
    return split
