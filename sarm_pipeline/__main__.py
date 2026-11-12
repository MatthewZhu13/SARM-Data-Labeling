"""python -m sarm_pipeline <step> --config configs/hqf_pilot.yaml [--set key=value ...]

Steps, in order (`all` runs them in sequence):
  fetch         download metadata, data parquets and videos for the selected episodes
  prep          split episodes into train/val/test (saved to split.file)
  embed         decode videos once and cache SigLIP-2 embeddings, state and stage boundaries
  train-stage   train the stage estimator
  train-reward  train the reward model (loads the stage estimator)
  eval          frame-by-frame evaluation on the test split
"""
from __future__ import annotations

import argparse
import json

from .config import load_config

STEPS = ["fetch", "prep", "embed", "train-stage", "train-reward", "eval"]


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
        from .data import select_episodes
        from .embed import build_cache
        from .v3 import V3Dataset
        ds = V3Dataset(cfg.dataset.root)
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
        print(f"[eval] stage accuracy alone {metrics['stage_estimator_alone']['accuracy']:.3f}, "
              f"finetuned {metrics['stage_estimator_finetuned']['accuracy']:.3f}; outputs in {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="sarm_pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS + ["all"])
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
