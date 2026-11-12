"""Per-frame progress from the folding_sarm_reward baseline (LeRobot's SARM1, base camera only).

Needs the `lerobot` package (>= 0.7, `lerobot.rewards.sarm`), which pins its own torch/transformers, so it runs
in its own environment rather than the pipeline's:

    python3 -m venv --system-site-packages ~/lr_venv && pip --python ~/lr_venv/bin/python install lerobot pandas
    ~/lr_venv/bin/python scripts/baseline_sarm1.py --root data/full_folding --repo-id lerobot/full_folding \
        --episodes-file splits/sep_full_folding.json --out runs/baseline_sarm1/progress.parquet

Same inference as lerobot's compute_rabc_weights.py (sparse head, eval-mode preprocessor, dataset stats,
centre frame of the bidirectional window), queried once per second and linearly interpolated to every frame.
We do not call that script directly: it uploads its output to the Hub by default (`--push-to-hub` is on).
For speed, run shards in parallel (`--shard 0/4` ... `--shard 3/4`), then `--merge 4`.
Output columns: episode_index, frame_index, progress (same layout as `separation score`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel
from lerobot.rewards.sarm.processor_sarm import make_sarm_pre_post_processors

MODEL = "lerobot-data-collection/folding_sarm_reward"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episodes-file", required=True, help="json with 'accepted' and 'rejected' (or 'episodes')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--step", type=int, default=30, help="query every N frames, interpolate the rest")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th episode starting at i (parallel decoding); "
                                                  "merge the shards with --merge")
    ap.add_argument("--merge", type=int, help="concatenate <out>.shard{0..N-1} into <out> and exit")
    args = ap.parse_args()

    if args.merge:
        parts = [pd.read_parquet(f"{args.out}.shard{i}") for i in range(args.merge)]
        pd.concat(parts, ignore_index=True).sort_values(["episode_index", "frame_index"]).to_parquet(args.out)
        print(f"[baseline] merged {args.merge} shards -> {args.out}")
        return
    sel = json.loads(Path(args.episodes_file).read_text())
    episodes = sorted(sel.get("episodes") or sel["accepted"] + sel["rejected"])
    i, n = (int(x) for x in args.shard.split("/"))
    episodes = episodes[i::n]
    out_path = args.out if n == 1 else f"{args.out}.shard{i}"

    model = SARMRewardModel.from_pretrained(args.model)
    model.config.device = args.device
    model.to(args.device).eval()
    cfg = model.config
    probe = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, download_videos=False)
    fps = probe.fps
    deltas = {k: [i / fps for i in cfg.observation_delta_indices] for k in (cfg.image_key, cfg.state_key)}
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=episodes, delta_timestamps=deltas,
                        download_videos=False)
    pre, _ = make_sarm_pre_post_processors(config=cfg, dataset_stats=ds.meta.stats, dataset_meta=ds.meta)
    if hasattr(pre, "eval"):
        pre.eval()
    for step in pre.steps:
        if hasattr(step, "eval"):
            step.eval()
    centre = cfg.n_obs_steps // 2

    cols = ds.hf_dataset.select_columns(["episode_index", "frame_index", "index"]).to_pandas()
    task = ds[0].get("task", "Fold the T-shirt properly")
    out = []
    for i, (ep, d) in enumerate(cols.groupby("episode_index", sort=True)):
        pos = d.index.to_numpy()                      # relative rows in the filtered dataset
        frames = d["frame_index"].to_numpy()
        q = sorted(set(range(0, len(pos), args.step)) | {len(pos) - 1})
        vals = []
        for j in q:
            s = ds[int(pos[j])]
            batch = {cfg.image_key: s[cfg.image_key], cfg.state_key: s[cfg.state_key], "task": task,
                     "index": int(d["index"].iloc[j]), "episode_index": int(ep)}
            with torch.no_grad():
                p = pre(batch)
                r = model.calculate_rewards(
                    text_embeddings=p["text_features"].to(args.device),
                    video_embeddings=p["video_features"].to(args.device),
                    state_features=p["state_features"].to(args.device) if p.get("state_features") is not None else None,
                    lengths=p.get("lengths"), return_all_frames=True, head_mode="sparse")
            vals.append(float(r[0, centre] if r.ndim == 2 else r[centre]))
        out.append(pd.DataFrame({"episode_index": int(ep), "frame_index": frames,
                                 "progress": np.interp(np.arange(len(pos)), q, vals).astype(np.float32)}))
        print(f"[baseline] {i + 1}/{len(episodes)} episode {ep}: {len(q)} queries, final {vals[-1]:.3f}", flush=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(out_path)
    print(f"[baseline] -> {out_path}")


if __name__ == "__main__":
    main()
