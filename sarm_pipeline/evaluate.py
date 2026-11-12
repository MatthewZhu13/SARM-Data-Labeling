"""Evaluate the trained models on held-out episodes, frame by frame, the way labeling runs.

For every (strided) frame of each episode a window is built with no rewind, and we record:
  stage_alone    stage estimator from the stage run (stage/best.pt)
  stage_pred     stage estimator as fine-tuned inside the reward run (reward/best.pt)
  progress_pred  reward model using the predicted stage
  progress_oracle reward model given the true stage (ablation: how much error comes from stages)
  p_alone_k, p_ft_k  per-stage probabilities of both stage estimators
  stage_decoded, stage_alone_decoded  ordered (Viterbi) decoding over the episode, see decode.py
Outputs in <run>/eval_<split>/: predictions.parquet, metrics.json, plots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import sarm2
from .decode import ordered_viterbi
from .config import Config
from .dataset import WindowDataset
from .train import load_stage_model, run_dir, setup, to_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def predict(cfg: Config, split_name: str = "test") -> tuple[pd.DataFrame, dict, Path]:
    device, split, cache = setup(cfg)
    out_dir = run_dir(cfg) / f"eval_{split_name}"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    preds = frame_predictions(cfg, cache, split[split_name], load_models(cfg, cache, device), device)
    preds.to_parquet(out_dir / "predictions.parquet")
    metrics = compute_metrics(preds, cache.stage_names, fps=30, stride=int(cfg.eval.stride))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    plot(preds, cache.stage_names, out_dir / "plots", int(cfg.eval.plot_episodes), metrics)
    return preds, metrics, out_dir


def load_models(cfg: Config, cache, device) -> dict:
    """The run's stage estimator (stage/best.pt) and its reward model with the fine-tuned stage estimator."""
    stage_alone = load_stage_model(cfg, cache, device).eval()
    ckpt = torch.load(run_dir(cfg) / "reward" / "best.pt", map_location=device)
    emb_dim = next(iter(cache.emb.values())).shape[1]
    reward_model = sarm2.build_reward_model(cfg, emb_dim, cache.state.shape[1], len(cache.cameras)).to(device)
    reward_model.load_state_dict(ckpt["models"]["reward"])
    stage_model = load_stage_model(cfg, cache, device)
    stage_model.load_state_dict(ckpt["models"]["stage"])
    return {"alone": stage_alone, "reward": reward_model.eval(), "stage": stage_model.eval()}


@torch.no_grad()
def frame_predictions(cfg: Config, cache, episodes: list[int], models: dict, device) -> pd.DataFrame:
    """Label every (strided) frame of `episodes` from `cache`; returns the decoded per-frame table."""
    w = cfg.window
    stage_alone, reward_model, stage_model = models["alone"], models["reward"], models["stage"]
    groups = sarm2.stage_groups(cfg).to(device)
    stage_emb, task_emb = cache.stage_emb.to(device), cache.task_emb.to(device)
    records = []
    for ep in episodes:
        start, length = cache.row_start[ep], cache.length[ep]
        anchors = np.arange(start, start + length, int(cfg.eval.stride))
        ds = WindowDataset(cache, [ep], "reward", w.n_obs_steps, w.frame_gap, anchors=anchors)
        for cpu_batch in DataLoader(ds, batch_size=int(cfg.eval.batch_size)):
            batch = to_device(cpu_batch, device)
            out = sarm2.reward_forward(reward_model, stage_model, batch, stage_emb, task_emb, groups,
                                       w.n_obs_steps, finetune_stage=False)
            oracle = sarm2.reward_forward(reward_model, stage_model, batch, stage_emb, task_emb, groups,
                                          w.n_obs_steps, finetune_stage=False, oracle_stage=True)
            alone_logits, _ = stage_alone(*sarm2.stage_inputs(batch, w.n_obs_steps))
            probs = torch.softmax(out["logits"], -1)
            alone_probs = torch.softmax(alone_logits, -1).cpu().numpy()
            ft_probs = probs.cpu().numpy()
            rows = cpu_batch["row"].numpy()
            records.append(pd.DataFrame({
                "episode_index": ep,
                "frame_index": rows - start,
                "stage_label": cpu_batch["stage"].numpy(),
                "progress_label": cpu_batch["progress"][:, w.n_obs_steps].numpy(),
                "stage_pred": probs.argmax(-1).cpu().numpy(),
                "stage_conf": probs.max(-1).values.cpu().numpy(),
                "stage_alone": alone_logits.argmax(-1).cpu().numpy(),
                "progress_pred": sarm2.progress_at_anchor(out, w.n_obs_steps).cpu().numpy(),
                "progress_oracle": sarm2.progress_at_anchor(oracle, w.n_obs_steps).cpu().numpy(),
                **{f"p_alone_{k}": alone_probs[:, k] for k in range(alone_probs.shape[1])},
                **{f"p_ft_{k}": ft_probs[:, k] for k in range(ft_probs.shape[1])},
            }))
    return add_decoded(pd.concat(records, ignore_index=True), cfg.eval.get("decode", {}))


def add_decoded(preds: pd.DataFrame, decode: dict) -> pd.DataFrame:
    """Ordered decoding per episode for both stage estimators (columns stage_decoded, stage_alone_decoded)."""
    preds = preds.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    kw = {"switch_penalty": float(decode.get("switch_penalty", 5.0)), "skip_penalty": float(decode.get("skip_penalty", 5.0))}
    for src, dst in (("p_ft_", "stage_decoded"), ("p_alone_", "stage_alone_decoded")):
        cols = sorted((c for c in preds if c.startswith(src)), key=lambda c: int(c[len(src):]))
        out = np.empty(len(preds), dtype=np.int64)
        for _, idx in preds.groupby("episode_index").indices.items():
            out[idx] = ordered_viterbi(preds.loc[idx, cols].to_numpy(), **kw)
        preds[dst] = out
    return preds


def _boundary_distance(preds: pd.DataFrame) -> np.ndarray:
    """Frames from each frame to the nearest change of the labeled stage within its episode."""
    dist = np.empty(len(preds))
    for _, idx in preds.groupby("episode_index").indices.items():
        f = preds["frame_index"].to_numpy()[idx]
        lab = preds["stage_label"].to_numpy()[idx]
        change = f[1:][lab[1:] != lab[:-1]]
        dist[idx] = np.abs(f[:, None] - change[None, :]).min(1) if len(change) else np.inf
    return dist


def _stage_metrics(pred: np.ndarray, label: np.ndarray, n: int, near_boundary: np.ndarray) -> dict:
    conf = np.zeros((n, n), dtype=int)
    np.add.at(conf, (label, pred), 1)
    recall = [float(conf[k, k] / conf[k].sum()) if conf[k].sum() else None for k in range(n)]
    precision = [float(conf[k, k] / conf[:, k].sum()) if conf[:, k].sum() else None for k in range(n)]
    far = ~near_boundary
    return {"accuracy": float((pred == label).mean()),
            # labels are on whole seconds, so frames within 1 s of a labeled boundary are ambiguous
            "accuracy_excl_1s_boundary": float((pred[far] == label[far]).mean()),
            "recall": recall, "precision": precision, "confusion": conf.tolist()}


def compute_metrics(preds: pd.DataFrame, stage_names: list[str], fps: int, stride: int) -> dict:
    n = len(stage_names)
    m = {
        "n_episodes": int(preds["episode_index"].nunique()),
        "n_frames": len(preds),
        "stage_names": stage_names,
    }
    near = _boundary_distance(preds) <= fps
    for name, col in (("stage_estimator_alone", "stage_alone"), ("stage_estimator_finetuned", "stage_pred"),
                      ("stage_alone_decoded", "stage_alone_decoded"), ("stage_finetuned_decoded", "stage_decoded")):
        if col in preds:
            m[name] = _stage_metrics(preds[col].to_numpy(), preds["stage_label"].to_numpy(), n, near)
    err = preds["progress_pred"] - preds["progress_label"]
    m["progress"] = {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae_oracle_stage": float((preds["progress_oracle"] - preds["progress_label"]).abs().mean()),
        "spearman": float(preds["progress_pred"].rank().corr(preds["progress_label"].rank())),
    }

    # Monotonicity on good demos: compare predictions 1 s apart.
    hop = max(1, fps // stride)
    drops, backward, boundary_err = [], [], []
    for _, ep in preds.groupby("episode_index"):
        p = ep.sort_values("frame_index")["progress_pred"].to_numpy()[::hop]
        d = np.diff(p)
        if len(d):
            drops.append(float((d < -0.02).mean()))
            backward.append(float(-d[d < 0].sum()))
        # Boundary error: first predicted frame of each stage vs its labeled start.
        ep = ep.sort_values("frame_index")
        for k in range(1, n - 1):
            true = ep.loc[ep["stage_label"] == k, "frame_index"]
            hit = ep.loc[ep["stage_pred"] == k, "frame_index"]
            if len(true) and len(hit):
                boundary_err.append(abs(int(hit.iloc[0]) - int(true.iloc[0])) / fps)
    m["monotonicity"] = {"frac_1s_drops_gt_0.02": float(np.mean(drops)) if drops else None,
                         "mean_total_backward_per_episode": float(np.mean(backward)) if backward else None}
    m["stage_boundary_error_s"] = {"median": float(np.median(boundary_err)) if boundary_err else None,
                                   "mean": float(np.mean(boundary_err)) if boundary_err else None}
    return m


def plot(preds: pd.DataFrame, stage_names: list[str], out: Path, n_episodes: int, metrics: dict) -> None:
    short = [s if len(s) < 28 else s[:26] + "…" for s in stage_names]
    for ep_index in sorted(preds["episode_index"].unique())[:n_episodes]:
        ep = preds[preds["episode_index"] == ep_index].sort_values("frame_index")
        t = ep["frame_index"].to_numpy() / 30
        fig, (a, b) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, height_ratios=[2, 1])
        a.plot(t, ep["progress_label"], color="0.55", lw=1.5, label="label")
        a.plot(t, ep["progress_pred"], color="#2a6fdb", lw=1.6, label="predicted")
        a.plot(t, ep["progress_oracle"], color="#e07b39", lw=1, ls="--", label="predicted (true stage)")
        a.set_ylabel("progress")
        a.set_ylim(-0.03, 1.03)
        a.legend(loc="upper left", frameon=False, fontsize=8)
        a.set_title(f"episode {ep_index}")
        b.step(t, ep["stage_label"], where="post", color="0.55", lw=1.5, label="label")
        b.step(t, ep["stage_pred"], where="post", color="#2a6fdb", lw=1, alpha=0.5, label="predicted (per frame)")
        if "stage_decoded" in ep:
            b.step(t, ep["stage_decoded"], where="post", color="#1b9e4b", lw=1.5, label="predicted (ordered)")
        b.set_yticks(range(len(short)), short, fontsize=7)
        b.set_xlabel("seconds")
        b.legend(loc="lower right", frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"episode_{ep_index}.png", dpi=110)
        plt.close(fig)

    conf = np.array(metrics["stage_estimator_finetuned"]["confusion"], dtype=float)
    conf = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(short)), short, fontsize=7)
    for i in range(len(short)):
        for j in range(len(short)):
            if conf[i, j] >= 0.005:
                ax.text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if conf[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("label")
    ax.set_title("stage confusion (row-normalized)")
    fig.tight_layout()
    fig.savefig(out / "stage_confusion.png", dpi=110)
    plt.close(fig)
