"""Accepted vs rejected: how well does progress separate good demos from bad ones?

`lerobot/high_quality_folding` was selected out of `lerobot/full_folding` (the unfiltered recordings). Every
high_quality episode matches exactly one full_folding episode by length and per-episode state mean, so the
full_folding episodes recorded in the same period with the same task that were *not* selected are the rejected
set. Accepted = our test split (never trained on), looked up in full_folding; rejected = unselected episodes
in the same index range: first those stored in the same video files as labeled episodes (same
sessions), then a seeded random draw from the rest. Both go through the same code path.

    python -m sarm_pipeline.separation select --config configs/sep_full_folding.yaml
    python -m sarm_pipeline.separation embed  --config configs/sep_full_folding.yaml
    python -m sarm_pipeline.separation score  --config configs/sep_full_folding.yaml --run sarm_hand
    python -m sarm_pipeline.separation report --config configs/sep_full_folding.yaml \
        --progress runs/sarm_hand/separation/progress.parquet runs/baseline_sarm1/progress.parquet

`score` labels every frame with a trained SARM2 run (run_name's checkpoints, stats from the hqf train split).
The folding_sarm_reward baseline writes the same table with scripts/baseline_sarm1.py.
`report` turns per-frame progress into per-episode scores and their AUROC (accepted = positive).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .dataset import Cache

STATE_STAT = "stats/observation.state/mean"
TASK = "Fold the T-shirt properly"


def match_episodes(hq_root: str, full_root: str) -> dict[int, int]:
    """high_quality episode -> full_folding episode, matched on (length, rounded per-episode state mean)."""
    def table(root):
        files = sorted((Path(root) / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        return pd.concat(pd.read_parquet(f, columns=["episode_index", "length", STATE_STAT]) for f in files)

    def key(r):
        return int(r.length), tuple(np.round(np.asarray(r[STATE_STAT], dtype=float), 4))

    full = {}
    for _, r in table(full_root).iterrows():
        full.setdefault(key(r), []).append(int(r.episode_index))
    out = {}
    for _, r in table(hq_root).iterrows():
        hit = full.get(key(r), [])
        if len(hit) == 1:
            out[int(r.episode_index)] = hit[0]
    return out


def select(cfg: Config, hq_cfg: Config, n_rejected: int, seed: int) -> dict:
    from .data import fetch_metadata
    full = fetch_metadata(cfg)
    m = match_episodes(hq_cfg.dataset.root, cfg.dataset.root)
    split = json.loads(Path(hq_cfg.split.file).read_text())
    labeled = split["train"] + split["val"] + split["test"]
    accepted = sorted(m[e] for e in split["test"] if e in m)
    selected = set(m.values())
    lo, hi = min(m[e] for e in labeled if e in m), max(m[e] for e in labeled if e in m)
    eps = full.episodes
    base = f"videos/{cfg.dataset.cameras['top']}/file_index"
    session_files = set(eps.loc[[m[e] for e in labeled if e in m], base])     # sessions of all labeled episodes
    pool = [int(e) for e, r in eps.iterrows()
            if lo <= e <= hi and e not in selected and list(r["tasks"]) == [TASK]]
    same = [e for e in pool if eps.loc[e, base] in session_files]      # recorded in a labeled episode's session
    rest = [e for e in pool if eps.loc[e, base] not in session_files]
    rng = np.random.default_rng(seed)
    rejected = same[:n_rejected] + [int(e) for e in rng.choice(rest, size=max(0, min(n_rejected - len(same), len(rest))),
                                                               replace=False)]
    rejected = sorted(rejected)
    out = {"accepted": accepted, "rejected": rejected,
           "hqf_episode": {str(v): k for k, v in m.items() if v in set(accepted)},
           "_notes": f"accepted = hqf test split mapped into full_folding; rejected = {len(rejected)} of the "
                     f"{len(pool)} unselected '{TASK}' episodes in full_folding {lo}-{hi}: all {len(same)} stored "
                     f"in the same video files as labeled hqf episodes, the rest drawn at random (seed {seed}). {len(m)} hqf episodes matched to full_folding."}
    Path(cfg.separation.file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.separation.file).write_text(json.dumps(out, indent=1))
    print(f"[separation] {len(accepted)} accepted, {len(rejected)} rejected (pool {len(pool)}) -> {cfg.separation.file}")
    return out


def _episodes(cfg: Config) -> list[int]:
    s = json.loads(Path(cfg.separation.file).read_text())
    return sorted(s["accepted"] + s["rejected"])


def embed(cfg: Config) -> None:
    from .data import fetch
    from .embed import build_cache
    cfg["dataset"]["episodes"] = _episodes(cfg)
    ds, episodes = fetch(cfg)
    build_cache(cfg, ds, episodes)


def _cache_with_stats(cache_dir: str, ref: Cache) -> Cache:
    """A cache of unlabeled episodes, normalized with the training cache's state stats."""
    d = Path(cache_dir)
    index = pd.read_parquet(d / "index.parquet")
    episodes = [int(e) for e in index["episode_index"]]
    raw = np.load(d / "state.npy")
    n = len(raw)
    emb = {cam: np.ascontiguousarray(np.load(d / f"emb_{cam}.npy", mmap_mode="r")) for cam in ref.cameras}
    return Cache(episodes, dict(zip(episodes, index["length"].astype(int))),
                 dict(zip(episodes, index["row_start"].astype(int))), emb,
                 ((raw - ref.state_mean) / ref.state_std).astype(np.float32),
                 np.full(n, len(ref.stage_names) - 1, dtype=np.int64), np.zeros(n, dtype=np.float32),
                 ref.stage_names, ref.stage_emb, ref.task_emb, ref.state_mean, ref.state_std, ref.alpha)


def score(cfg: Config, hq_cfg: Config) -> Path:
    from .evaluate import frame_predictions, load_models
    from .train import run_dir, setup
    device, _, ref = setup(hq_cfg)
    cache = _cache_with_stats(cfg.cache_dir, ref)
    preds = frame_predictions(hq_cfg, cache, cache.episodes, load_models(hq_cfg, ref, device), device)
    out = run_dir(hq_cfg) / "separation"
    out.mkdir(parents=True, exist_ok=True)
    preds[["episode_index", "frame_index", "progress_pred", "stage_decoded"]].rename(
        columns={"progress_pred": "progress"}).to_parquet(out / "progress.parquet")
    print(f"[separation] {len(preds)} frames -> {out / 'progress.parquet'}")
    return out / "progress.parquet"


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random accepted episode > a random rejected one), ties count half."""
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def episode_scores(progress: pd.DataFrame, fps: int = 30) -> pd.DataFrame:
    """Per-episode summaries of a progress curve (higher = more like a good demo, except where noted)."""
    rows = []
    for ep, d in progress.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        p = d["progress"].to_numpy(dtype=float)
        f = d["frame_index"].to_numpy()
        sec = p[np.searchsorted(f, np.arange(0, f[-1] + 1, fps))]       # one value per second
        diff = np.diff(sec)
        reach = np.flatnonzero(sec >= 0.9)
        rows.append({"episode": int(ep), "length_s": (f[-1] + 1) / fps,
                     "final": float(sec[-3:].mean()), "max": float(sec.max()), "mean": float(sec.mean()),
                     "backward": float(-diff[diff < 0].sum()),                       # lower is better
                     "drops": int((diff < -0.05).sum()),                              # lower is better
                     "rank_time": float(pd.Series(sec).rank().corr(pd.Series(np.arange(len(sec))).rank())),
                     "time_to_0.9_s": float(reach[0]) if len(reach) else np.inf})     # lower is better
    return pd.DataFrame(rows).set_index("episode")


LOWER_IS_BETTER = {"backward", "drops", "time_to_0.9_s", "length_s"}


def separation_table(scores: pd.DataFrame, accepted: list[int], rejected: list[int]) -> pd.DataFrame:
    acc, rej = scores.loc[scores.index.isin(accepted)], scores.loc[scores.index.isin(rejected)]
    short = rej[rej["length_s"] <= acc["length_s"].max()]           # length-matched: drop longer-than-any-accepted
    out = {}
    for col in scores.columns:
        sign = -1 if col in LOWER_IS_BETTER else 1
        a, r, s = sign * acc[col].to_numpy(), sign * rej[col].to_numpy(), sign * short[col].to_numpy()
        a, r, s = (np.nan_to_num(x, neginf=-1e9, posinf=1e9) for x in (a, r, s))
        out[col] = {"auroc": auroc(a, r), "auroc_length_matched": auroc(a, s) if len(s) else np.nan,
                    "accepted_median": float(np.median(acc[col])), "rejected_median": float(np.median(rej[col]))}
    t = pd.DataFrame(out).T
    t.attrs.update(n_accepted=len(acc), n_rejected=len(rej), n_rejected_length_matched=len(short))
    return t


def report(cfg: Config, paths: list[str]) -> None:
    s = json.loads(Path(cfg.separation.file).read_text())
    out = Path(cfg.separation.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for p in paths:
        name = Path(p).parent.parent.name if Path(p).parent.name == "separation" else Path(p).parent.name
        scores = episode_scores(pd.read_parquet(p))
        scores["group"] = np.where(scores.index.isin(s["accepted"]), "accepted", "rejected")
        scores.to_csv(out / f"episodes_{name}.csv")
        t = separation_table(scores.drop(columns="group"), s["accepted"], s["rejected"])
        t.to_csv(out / f"auroc_{name}.csv")
        summary[name] = {"counts": t.attrs, "auroc": t.round(3).to_dict(orient="index")}
        print(f"== {name}  ({t.attrs})\n{t.round(3).to_string()}\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    _plot(paths, s, out)


def _plot(paths: list[str], s: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(paths), figsize=(6 * len(paths), 4), squeeze=False, sharey=True)
    for ax, p in zip(axes[0], paths):
        prog = pd.read_parquet(p)
        for ep, d in prog.groupby("episode_index"):
            acc = ep in s["accepted"]
            d = d.sort_values("frame_index")
            ax.plot(d["frame_index"] / 30, d["progress"], lw=0.7, alpha=0.5,
                    color="#2a6fdb" if acc else "#d1495b")
        ax.set_title(Path(p).parent.parent.name if Path(p).parent.name == "separation" else Path(p).parent.name)
        ax.set_xlabel("seconds")
    axes[0][0].set_ylabel("progress (blue accepted, red rejected)")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "embed", "score", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run", help="SARM2 run to score with (score)")
    ap.add_argument("--progress", nargs="+", help="per-frame progress parquets (report)")
    ap.add_argument("--n-rejected", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    hq_cfg = load_config(cfg.separation.train_config, [f"run_name={args.run}"] if args.run else [])
    if args.run:   # score with the run's own label settings
        stats = json.loads((Path(hq_cfg.output_dir) / args.run / "label_stats.json").read_text())
        if stats.get("segments_override"):
            hq_cfg["labels"]["segments_override"] = stats["segments_override"]
    if args.step == "select":
        select(cfg, hq_cfg, args.n_rejected, args.seed)
    elif args.step == "embed":
        embed(cfg)
    elif args.step == "score":
        score(cfg, hq_cfg)
    else:
        report(cfg, args.progress)


if __name__ == "__main__":
    main()
