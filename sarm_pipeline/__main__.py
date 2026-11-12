"""python -m sarm_pipeline <step> --config configs/hqf_pilot.yaml [--set key=value ...]

Steps, in order (`all` runs them in sequence):
  fetch         download metadata, data parquets and videos for the selected episodes
  prep          split episodes into train/val/test (saved to split.file)
  embed         decode videos once and cache SigLIP-2 embeddings, state and stage boundaries
  train-stage   train the stage estimator
  train-reward  train the reward model (loads the stage estimator)
  eval          frame-by-frame evaluation on the test split
  episode-stage train the offline whole-episode stage labeler; reports val and test
  episode-stage-oof  k-fold out-of-fold predictions on train+val, to find episodes with bad labels
"""
from __future__ import annotations

import argparse
import json

from .config import load_config

STEPS = ["fetch", "prep", "embed", "train-stage", "train-reward", "eval", "episode-stage"]
EXTRA_STEPS = ["episode-stage-oof"]


def run(step: str, cfg, resume: bool, split_name: str) -> None:
    if step == "fetch":
        from .data import fetch
        fetch(cfg)
    elif step == "prep":
        from .data import fetch_metadata, make_split, select_episodes
        ds = fetch_metadata(cfg)
        split = make_split(cfg, select_episodes(cfg, ds))
        print(f"[prep] split {cfg.split.file}: " + ", ".join(f"{k} {len(v)}" for k, v in split.items()))
    elif step == "embed":
        from .data import open_local, select_episodes
        from .embed import build_cache
        from .v3 import V3Dataset
        ds = open_local(cfg) if cfg.dataset.get("format") == "v2.1" else V3Dataset(cfg.dataset.root)
        build_cache(cfg, ds, select_episodes(cfg, ds))
    elif step == "train-stage":
        from .train import train_stage
        print(json.dumps(train_stage(cfg, resume=resume), indent=1))
    elif step == "train-reward":
        from .train import train_reward
        print(json.dumps(train_reward(cfg, resume=resume), indent=1))
    elif step == "eval":
        from .evaluate import predict
        _, metrics, out = predict(cfg, split_name)
        print(json.dumps({k: metrics[k] for k in ("progress", "monotonicity", "stage_boundary_error_s")}, indent=1))
        acc = {k: round(metrics[k]["accuracy"], 3) for k in metrics if k.startswith("stage_") and "accuracy" in metrics[k]}
        print(f"[eval] stage accuracy {acc}; outputs in {out}")
    elif step == "episode-stage":
        from .episode_stage import train_episode_stage
        print(json.dumps(train_episode_stage(cfg), indent=1))
    elif step == "episode-stage-oof":
        from .episode_stage import out_of_fold
        out_of_fold(cfg, int(cfg.episode_stage.get("folds", 5)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="sarm_pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS + EXTRA_STEPS + ["all"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--resume", action="store_true", help="continue training from latest.pt")
    parser.add_argument("--split", default="test", help="split to evaluate (eval step)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    for step in STEPS if args.step == "all" else [args.step]:
        print(f"===== {step} =====", flush=True)
        run(step, cfg, args.resume, args.split)


if __name__ == "__main__":
    main()
