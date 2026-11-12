"""K-fold cross-validation for small datasets (the client set has 19 episodes): every episode is labeled by
models that never saw it, and the held-out predictions are compared with the hand labels.

    python -m sarm_pipeline.kfold split   --config configs/client_nurvai2.yaml --folds 5
    scripts/client_cv.sh                                   # trains episode-stage + SARM2 per fold (GPU box)
    python -m sarm_pipeline.kfold collect --config configs/client_nurvai2.yaml --folds 5

`split` writes splits/<name>_fold<k>.json (test = fold k, val = 2 other episodes, train = the rest).
`collect` gathers each fold's test predictions and writes reports/<name>/: per-frame tables, stage accuracy,
boundary errors, progress error, per-episode plots (hand label vs prediction) and review flags.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .evaluate import _boundary_distance, _stage_metrics


def fold_files(cfg: Config, folds: int) -> list[Path]:
    stem = Path(cfg.split.file)
    return [stem.with_name(f"{stem.stem}_fold{k}.json") for k in range(folds)]


def make_folds(cfg: Config, episodes: list[int], folds: int, seed: int, n_val: int = 2) -> list[Path]:
    rng = np.random.default_rng(seed)
    order = [int(e) for e in rng.permutation(sorted(episodes))]
    parts = [sorted(order[k::folds]) for k in range(folds)]
    paths = fold_files(cfg, folds)
    for k, path in enumerate(paths):
        rest = [e for e in order if e not in parts[k]]
        val = sorted(rest[:n_val])
        split = {"seed": seed, "fold": k, "folds": folds, "episodes": sorted(episodes),
                 "train": sorted(rest[n_val:]), "val": val, "test": parts[k]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=1))
    print(f"[kfold] {folds} folds of {len(episodes)} episodes -> {paths[0].parent}/{paths[0].stem[:-1]}*.json")
    return paths


def _boundaries(stage: np.ndarray, frames: np.ndarray, n: int) -> dict[int, float]:
    """First frame of each stage k >= 1 (stage starts), in frames; missing stages are left out."""
    out = {}
    for k in range(1, n):
        hit = frames[stage == k]
        if len(hit):
            out[k] = float(hit[0])
    return out


def review_flags(d: pd.DataFrame, top_emb: np.ndarray, n_stages: int, fps: int = 30, still: float = 0.015,
                 idle_s: int = 15) -> dict:
    """One episode's held-out predictions + its top-camera embeddings (cache rows) -> review flags.

    idle: the top camera barely changes (cosine change between frames 1 s apart, 5 s rolling median, below
          `still`) for >= idle_s seconds before the predicted last stage. The threshold was set on this data:
          it is about the 20th percentile of all 1-s changes, and the three stalls noted while hand-labeling
          (episodes 14, 15, 17) fall below it while the rest of those episodes does not.
    unfinished: the decoded stage never reaches the last stage.
    max_drawdown is reported, not flagged: per-frame progress from a model trained on ~15 episodes flickers too
    much for "progress went backwards" to mean anything yet.
    """
    secs = np.arange(0, len(d), fps)
    e = np.asarray(top_emb[secs], dtype=np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    change = pd.Series(1 - (e[1:] * e[:-1]).sum(1)).rolling(5, center=True, min_periods=1).median().to_numpy()
    stage = d["stage_decoded"].to_numpy()[secs][1:]
    quiet = (change < still) & (stage < n_stages - 1)
    longest, at, run = 0, None, 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run > longest:
            longest, at = run, i - run + 2
    sec = pd.Series(d["progress_pred"].to_numpy()[::fps]).rolling(9, center=True, min_periods=1).median().to_numpy()
    done = np.flatnonzero(sec >= 0.95)
    upto = done[0] if len(done) else len(sec)          # ignore the end-of-episode empty table (looks like the start)
    drawdown = float(np.max(np.maximum.accumulate(sec[:upto]) - sec[:upto])) if upto else 0.0
    reached = int(d["stage_decoded"].max()) >= n_stages - 1
    flag = " ".join(f for f, on in (("idle", longest >= idle_s), ("unfinished", not reached)) if on)
    return {"episode": int(d["episode_index"].iloc[0]), "length_s": round(len(d) / fps, 1),
            "longest_idle_s": int(longest), "idle_starts_s": at, "reaches_last_stage": reached,
            "max_progress": round(float(sec.max()), 3), "max_drawdown": round(drawdown, 3), "flag": flag}


def collect(cfg: Config, folds: int, es_prefix: str, sarm_prefix: str, out_dir: Path, fps: int = 30) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(cfg.stages.names)
    n = len(names)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    runs = Path(cfg.output_dir)
    es = pd.concat([pd.read_parquet(runs / f"{es_prefix}{k}" / "episode_stage" / "predictions_test.parquet")
                    for k in range(folds)], ignore_index=True)
    sarm = pd.concat([pd.read_parquet(runs / f"{sarm_prefix}{k}" / "eval_test" / "predictions.parquet")
                      for k in range(folds)], ignore_index=True)
    es.to_parquet(out_dir / "episode_stage_heldout.parquet")
    sarm.to_parquet(out_dir / "sarm2_heldout.parquet")

    result = {"episodes": int(sarm["episode_index"].nunique()), "stage_names": names}
    for name, df, col in (("episode_stage", es, "stage_decoded"), ("sarm2_stage", sarm, "stage_decoded"),
                          ("sarm2_stage_per_frame", sarm, "stage_pred")):
        near = _boundary_distance(df) <= fps
        m = _stage_metrics(df[col].to_numpy(), df["stage_label"].to_numpy(), n + 1, near)
        per_ep = df.assign(ok=df[col] == df["stage_label"]).groupby("episode_index").ok.mean()
        errs = {k: [] for k in range(1, n)}
        for _, d in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
            f = d["frame_index"].to_numpy()
            lab, pred = _boundaries(d["stage_label"].to_numpy(), f, n), _boundaries(d[col].to_numpy(), f, n)
            for k in lab:
                if k in pred:
                    errs[k].append((pred[k] - lab[k]) / fps)
        result[name] = {"accuracy": m["accuracy"], "accuracy_excl_1s_boundary": m["accuracy_excl_1s_boundary"],
                        "recall": m["recall"][:n], "worst_episodes": per_ep.sort_values().head(5).round(3).to_dict(),
                        "boundary_error_s": {names[k]: {"median_abs": float(np.median(np.abs(v))) if v else None,
                                                        "median_signed": float(np.median(v)) if v else None,
                                                        "n": len(v)} for k, v in errs.items()}}
    err = sarm["progress_pred"] - sarm["progress_label"]
    result["sarm2_progress"] = {"mae": float(err.abs().mean()),
                                "spearman": float(sarm["progress_pred"].rank().corr(sarm["progress_label"].rank())),
                                "mae_oracle_stage": float((sarm["progress_oracle"] - sarm["progress_label"]).abs().mean())}

    # Review flags: idle periods from the top camera, unfinished from the held-out stage timeline (see review_flags)
    index = pd.read_parquet(Path(cfg.cache_dir) / "index.parquet").set_index("episode_index")
    top = np.load(Path(cfg.cache_dir) / "emb_top.npy", mmap_mode="r")
    rows_of = lambda ep: top[int(index.loc[ep, "row_start"]):int(index.loc[ep, "row_start"] + index.loc[ep, "length"])]
    flags = pd.DataFrame([review_flags(d, rows_of(ep), n, fps) for ep, d in
                          sarm.sort_values(["episode_index", "frame_index"]).groupby("episode_index")]).set_index("episode")
    flags.to_csv(out_dir / "flags.csv")
    result["flags"] = flags[flags["flag"] != ""]["flag"].to_dict()

    short = [s if len(s) < 24 else s[:22] + "…" for s in names] + ["dummy"]
    for ep in sorted(sarm["episode_index"].unique()):
        d = sarm[sarm["episode_index"] == ep].sort_values("frame_index")
        e = es[es["episode_index"] == ep].sort_values("frame_index")
        t = d["frame_index"].to_numpy() / fps
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[2, 1])
        a.plot(t, d["progress_label"], color="0.55", lw=1.5, label="hand-label progress")
        a.plot(t, d["progress_pred"], color="#2a6fdb", lw=1.2, label="SARM2 (held out)")
        a.set_ylim(-0.03, 1.03)
        a.set_ylabel("progress")
        a.legend(loc="upper left", frameon=False, fontsize=8)
        a.set_title(f"client episode {ep}  (flags: {flags.loc[ep, 'flag'] or 'none'})")
        b.step(t, d["stage_label"], where="post", color="0.55", lw=2, label="hand label")
        b.step(e["frame_index"] / fps, e["stage_decoded"], where="post", color="#1b9e4b", lw=1.3, label="episode-stage")
        b.step(t, d["stage_decoded"], where="post", color="#2a6fdb", lw=1, alpha=0.7, label="SARM2 stage")
        b.set_yticks(range(len(short)), short, fontsize=7)
        b.set_xlabel("seconds")
        b.legend(loc="lower right", frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"episode_{ep}.png", dpi=100)
        plt.close(fig)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "collect"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--es-prefix", default="client_es_f")
    ap.add_argument("--sarm-prefix", default="client_sarm_f")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    if args.step == "split":
        index = pd.read_parquet(Path(cfg.cache_dir) / "index.parquet")
        labeled = {int(k) for k in json.loads(Path(cfg.labels.segments_override).read_text()) if not k.startswith("_")}
        make_folds(cfg, sorted(set(index["episode_index"].astype(int)) & labeled), args.folds, args.seed)
    else:
        out = Path(args.out or f"reports/{cfg.run_name}")
        r = collect(cfg, args.folds, args.es_prefix, args.sarm_prefix, out)
        print(json.dumps({k: v for k, v in r.items() if k != "stage_names"}, indent=1, default=float))


if __name__ == "__main__":
    main()
