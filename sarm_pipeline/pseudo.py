"""Build stage-label override files from the hand-label ensemble (docs/GOLD.md).

    python -m sarm_pipeline.pseudo --config configs/hqf_pilot.yaml --runs es_hand_s0 es_hand_s1 es_hand_s2

Writes two files in the cache's segments.json format, for `--set labels.segments_override=<file>`:
  labels/pseudo_hand_ens.json   train + val: the ensemble's decoded stages; hand-labeled episodes
                                (labels/hand_train.json) keep their hand labels.
  labels/hand_ens_all.json      the same plus the blind test labels (labels/test_gold_v2.json), so that
                                SARM2 evaluation scores progress and stage against the hand labels.
                                Training only ever reads train episodes, so test labels never reach a model.
Test episodes marked `unclear` keep the dataset's labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .episode_stage import ensemble_predict, predictions_to_segments

FPS = 30


def gold_segments(gold: dict, lengths: dict[int, int], names: list[str]) -> dict:
    """test_gold_v2 `stages_s` ([start, end) seconds; last end may be 9999) -> segments format in frames."""
    out = {}
    for key, c in gold.items():
        if key.startswith("_") or c.get("unclear") or "stages_s" not in c:
            continue
        ep = int(key)
        starts = [int(s * FPS) for s, _ in c["stages_s"]]
        ends = starts[1:] + [min(int(c["stages_s"][-1][1] * FPS), lengths[ep])]
        out[ep] = [names[: len(starts)], starts, ends]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="episode-stage runs to ensemble")
    ap.add_argument("--hand", default="labels/hand_train.json")
    ap.add_argument("--gold", default="labels/test_gold_v2.json")
    ap.add_argument("--out", default="labels/pseudo_hand_ens.json")
    ap.add_argument("--out-all", default="labels/hand_ens_all.json")
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    names = list(cfg.stages.names)

    preds = pd.concat([ensemble_predict(cfg, args.runs, s) for s in ("train", "val")], ignore_index=True)
    segs = {str(k): v for k, v in predictions_to_segments(preds, names).items()}
    hand = {k: v for k, v in json.loads(Path(args.hand).read_text()).items() if not k.startswith("_")}
    segs.update(hand)
    meta = {"_notes": f"train+val stages from the ensemble of {args.runs}; {len(hand)} hand-labeled episodes "
                      f"from {args.hand} override it. Built by sarm_pipeline.pseudo."}
    Path(args.out).write_text(json.dumps({**meta, **segs}))

    index = pd.read_parquet(Path(cfg.cache_dir) / "index.parquet")
    lengths = dict(zip(index["episode_index"].astype(int), index["length"].astype(int)))
    test = {str(k): v for k, v in gold_segments(json.loads(Path(args.gold).read_text()), lengths, names).items()}
    meta_all = {"_notes": meta["_notes"] + f" Test episodes from {args.gold} (evaluation only)."}
    Path(args.out_all).write_text(json.dumps({**meta_all, **segs, **test}))
    print(f"[pseudo] {args.out}: {len(segs)} episodes ({len(hand)} hand); {args.out_all}: +{len(test)} test")


if __name__ == "__main__":
    main()
