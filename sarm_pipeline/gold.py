"""Score stage predictions against hand-labeled test labels (docs/GOLD.md).

    python -m sarm_pipeline.gold runs/<run>/episode_stage/predictions_test.parquet [--col stage_decoded]

Default labels: labels/test_gold_v2.json (every test episode labeled blind). Episodes with `stages_s`
get their labels rebuilt from it; `label_ok` and unlisted episodes keep the dataset labels;
`unclear` episodes are left out (and counted).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

FPS = 30


def corrected_labels(preds: pd.DataFrame, corrections: dict, n_stages: int = 5,
                     shift_s: tuple[float, ...] = (0, 0, 0, 0)) -> tuple[pd.DataFrame, list[int]]:
    """Returns preds with a `gold` column, without unclear episodes, and the list of dropped episodes.

    shift_s moves the start of stages 2..5 earlier by that many seconds. It was used with the v1 labels;
    the calibration behind it turned out to measure template labels (docs/GOLD.md), so v2 uses no shift.
    """
    preds = preds.copy()
    preds["gold"] = preds["stage_label"].to_numpy()
    dropped = []
    for key, c in corrections.items():
        if key.startswith("_"):
            continue
        ep = int(key)
        rows = preds.index[preds["episode_index"] == ep]
        if c.get("unclear"):
            dropped.append(ep)
            continue
        if "stages_s" not in c or not len(rows):
            continue
        frames = preds.loc[rows, "frame_index"].to_numpy()
        last = frames.max()
        gold = np.full(len(frames), n_stages)                       # dummy
        starts = [s for s, _ in c["stages_s"]]
        starts = [starts[0]] + [max(st - sh, starts[0]) for st, sh in zip(starts[1:], shift_s)]
        ends = starts[1:] + [c["stages_s"][-1][1]]
        for k, (s, e) in enumerate(zip(starts, ends)):
            gold[(frames >= s * FPS) & (frames < min(e * FPS, last))] = k
        preds.loc[rows, "gold"] = gold
    return preds[~preds["episode_index"].isin(dropped)], dropped


def score_gold(preds: pd.DataFrame, corrections: dict, col: str, shift_s=(0, 0, 0, 0)) -> dict:
    kept, dropped = corrected_labels(preds, corrections, shift_s=shift_s)
    reviewed = [int(k) for k, c in corrections.items() if not k.startswith("_") and not c.get("unclear")]
    per_ep = kept.assign(ok=kept[col] == kept["gold"]).groupby("episode_index").ok.mean()
    return {
        "episodes": int(kept["episode_index"].nunique()), "dropped_unclear": dropped,
        "acc_dataset_labels": float((kept[col] == kept["stage_label"]).mean()),
        "acc_corrected_labels": float((kept[col] == kept["gold"]).mean()),
        "acc_corrected_reviewed_only": float((kept[col] == kept["gold"])[kept["episode_index"].isin(reviewed)].mean()),
        "worst_episodes": per_ep.sort_values().head(5).round(3).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+")
    ap.add_argument("--col", default="stage_decoded")
    ap.add_argument("--corrections", default="labels/test_gold_v2.json")
    ap.add_argument("--shift", default="0,0,0,0",
                    help="seconds to move stage 2..5 starts earlier (v1 labels only; see docs/GOLD.md)")
    args = ap.parse_args()
    corrections = json.load(open(args.corrections))
    shift = tuple(float(x) for x in args.shift.split(","))
    for path in args.predictions:
        print(path, json.dumps(score_gold(pd.read_parquet(path), corrections, args.col, shift), indent=1))


if __name__ == "__main__":
    main()
