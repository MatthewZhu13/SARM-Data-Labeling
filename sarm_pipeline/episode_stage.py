"""Whole-episode stage labeler (offline), an addition to SARM2's stage estimator.

SARM2's stage estimator is causal: it sees the last 5 s and nothing else, which suits online
reward but not offline labeling, where the whole episode is available. This model reads the
cached SigLIP embeddings of every camera, the joint state and the elapsed time for the whole
episode (subsampled), runs a bidirectional GRU over it and predicts the stage of every step.
Predictions are then decoded in order (decode.py) and spread back to every frame.

Hyperparameters and the checkpoint are chosen on the val split; test is scored once at the end.
Outputs in <run>/episode_stage/: best.pt, metrics.jsonl, metrics_<split>.json, predictions_<split>.parquet.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import Config
from .decode import ordered_viterbi
from .evaluate import _boundary_distance, _stage_metrics
from .train import run_dir, setup


class EpisodeStageModel(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU())
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        return self.out(h)


class _Episodes:
    """Per-episode feature sequences: [camera embeddings | state | frame/length | seconds/60]."""

    def __init__(self, cache, fps: int):
        self.c, self.fps = cache, fps
        self.feat = np.concatenate([cache.emb[cam] for cam in cache.cameras] + [cache.state.astype(np.float16)], 1)

    @property
    def dim(self) -> int:
        return self.feat.shape[1] + 2

    def steps(self, ep: int, sub: float, offset: float = 0.0) -> np.ndarray:
        return np.arange(offset, self.c.length[ep], sub).astype(np.int64)

    def sequence(self, ep: int, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = self.c.row_start[ep] + frames
        t = np.stack([frames / max(self.c.length[ep] - 1, 1), frames / self.fps / 60.0], 1)
        return np.concatenate([self.feat[rows].astype(np.float32), t.astype(np.float32)], 1), self.c.stage[rows]


def _batch(seqs: list[tuple[np.ndarray, np.ndarray]], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(x) for x, _ in seqs])
    x = torch.zeros(len(seqs), int(lengths.max()), seqs[0][0].shape[1])
    y = torch.full((len(seqs), int(lengths.max())), -100, dtype=torch.long)
    for i, (xi, yi) in enumerate(seqs):
        x[i, :len(xi)] = torch.from_numpy(xi)
        y[i, :len(yi)] = torch.from_numpy(yi.astype(np.int64))
    return x.to(device), y.to(device), lengths


@torch.no_grad()
def predict_episodes(model, data: _Episodes, episodes: list[int], sub: int, decode: dict, device) -> pd.DataFrame:
    """Stage for every frame of each episode: per-step probabilities, ordered decoding, held across the step."""
    model.eval()
    kw = {"switch_penalty": float(decode.get("switch_penalty", 5.0)), "skip_penalty": float(decode.get("skip_penalty", 5.0))}
    frames_out = []
    for ep in episodes:
        steps = data.steps(ep, sub)
        x, y, lengths = _batch([data.sequence(ep, steps)], device)
        probs = torch.softmax(model(x, lengths)[0], -1).cpu().numpy()
        decoded = ordered_viterbi(probs, **kw)
        frame = np.arange(data.c.length[ep])
        at = np.minimum(frame // sub, len(steps) - 1)
        frames_out.append(pd.DataFrame({
            "episode_index": ep, "frame_index": frame,
            "stage_label": data.c.stage[data.c.row_start[ep] + frame],
            "stage_pred": probs.argmax(-1)[at], "stage_decoded": decoded[at]}))
    return pd.concat(frames_out, ignore_index=True)


def score(preds: pd.DataFrame, n: int, fps: int) -> dict:
    near = _boundary_distance(preds) <= fps
    lab = preds["stage_label"].to_numpy()
    return {name: _stage_metrics(preds[col].to_numpy(), lab, n, near)
            for name, col in (("per_step", "stage_pred"), ("decoded", "stage_decoded"))}


def _excluded(cfg: Config, split: dict) -> set[int]:
    """Train/val episodes left out: those in `exclude_file`, and with `only_override_episodes`, every
    episode not relabeled in `labels.segments_override` (train on hand labels only)."""
    path = cfg.episode_stage.get("exclude_file")
    drop = set(json.loads(open(path).read())) if path else set()
    if cfg.episode_stage.get("only_override_episodes"):
        keep = {int(k) for k in json.loads(open(cfg.labels.segments_override).read()) if not k.startswith("_")}
        drop |= set(split["train"] + split["val"]) - keep
    return drop


def _fit(cfg: Config, data: _Episodes, n: int, train_eps: list[int], val_eps: list[int] | None, out, device,
         tag: str = "") -> EpisodeStageModel:
    """Train on train_eps. With val_eps, keep the epoch with the best decoded val accuracy; otherwise the last."""
    e = cfg.episode_stage
    fps, sub = int(cfg.get("fps", 30)), int(e.subsample)
    torch.manual_seed(int(cfg.get("seed", 42)))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    model = EpisodeStageModel(data.dim, n, int(e.hidden), int(e.layers), float(e.dropout)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(e.lr), weight_decay=float(e.weight_decay))
    epochs, bs = int(e.epochs), int(e.batch_episodes)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=float(e.lr), pct_start=0.1,
                                                total_steps=epochs * int(np.ceil(len(train_eps) / bs)))
    print(f"[episode-stage{tag}] {len(train_eps)} train / {len(val_eps or [])} val episodes, step every {sub} frames, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, device {device}", flush=True)

    best, log, t0 = -1.0, open(out / f"metrics{tag}.jsonl", "w"), time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_eps)
        for i in range(0, len(order), bs):
            seqs = []
            for ep in order[i:i + bs]:
                # speed and phase augmentation: random step size around `sub` and random offset
                s = sub * rng.uniform(0.75, 1.33)
                x, y = data.sequence(int(ep), data.steps(int(ep), s, rng.uniform(0, s)))
                x[:, :-2] += rng.normal(0, float(e.noise), x[:, :-2].shape).astype(np.float32)
                seqs.append((x, y))
            x, y, lengths = _batch(seqs, device)
            loss = F.cross_entropy(model(x, lengths).flatten(0, 1), y.flatten(), ignore_index=-100,
                                   label_smoothing=float(e.label_smoothing))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
        rec = {"epoch": epoch, "loss": loss.item(), "minutes": (time.time() - t0) / 60}
        if val_eps:
            val = score(predict_episodes(model, data, val_eps, sub, cfg.eval.get("decode", {}), device), n, fps)
            rec.update(val_acc=val["per_step"]["accuracy"], val_acc_decoded=val["decoded"]["accuracy"])
            if rec["val_acc_decoded"] > best:
                best = rec["val_acc_decoded"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg.to_dict()}, out / f"best{tag}.pt")
                rec["best"] = True
        log.write(json.dumps(rec) + "\n"); log.flush()
        if epoch % 5 == 0 or rec.get("best"):
            extra = f", val_acc {rec['val_acc']:.4f}, decoded {rec['val_acc_decoded']:.4f}" if val_eps else ""
            print(f"[episode-stage{tag}] epoch {epoch}: loss {loss.item():.4f}{extra}{' (best)' if rec.get('best') else ''}",
                  flush=True)
    if val_eps:
        model.load_state_dict(torch.load(out / f"best{tag}.pt", map_location=device)["model"])
    else:
        torch.save({"model": model.state_dict(), "epoch": epochs, "config": cfg.to_dict()}, out / f"last{tag}.pt")
    return model


def _apply_pseudo_labels(cfg: Config, cache, split: dict) -> None:
    """Replace the stage labels of train/val episodes with out-of-fold decoded predictions (self-training).

    Episodes are chosen by agreement below `pseudo_labels.max_agreement`, measured on
    `pseudo_labels.select_file` (default: `file`) against the labels that run saw; for later rounds,
    point `select_file` at the round-1 run so selection stays relative to the dataset labels.
    Test episodes are never touched.
    """
    p = cfg.episode_stage.get("pseudo_labels")
    if not p:
        return
    oof = pd.read_parquet(p["file"])
    sel = pd.read_parquet(p.get("select_file", p["file"]))
    agree = (sel.stage_decoded == sel.stage_label).groupby(sel.episode_index).mean()
    eligible = set(split["train"]) | set(split["val"])
    eps = [int(e) for e in agree[agree < float(p["max_agreement"])].index if int(e) in eligible]
    for ep in eps:
        d = oof[oof.episode_index == ep]
        cache.stage[cache.row_start[ep] + d.frame_index.to_numpy()] = d.stage_decoded.to_numpy()
    print(f"[episode-stage] pseudo-labeled {len(eps)} episodes from {p['file']}")


def train_episode_stage(cfg: Config, report_splits: tuple[str, ...] = ("val", "test")) -> dict:
    device, split, cache = setup(cfg)
    fps, sub = int(cfg.get("fps", 30)), int(cfg.episode_stage.subsample)
    out = run_dir(cfg) / "episode_stage"
    out.mkdir(parents=True, exist_ok=True)
    _apply_pseudo_labels(cfg, cache, split)
    data, n = _Episodes(cache, fps), len(cache.stage_names)
    # excluded episodes (labels judged wrong) are left out of training and of checkpoint selection;
    # test is always scored on every episode
    drop = _excluded(cfg, split)
    train_eps = [e for e in split["train"] if e not in drop]
    val_eps = [e for e in split["val"] if e not in drop]
    if drop:
        print(f"[episode-stage] excluding {len(split['train']) - len(train_eps)} train / "
              f"{len(split['val']) - len(val_eps)} val episodes")
    model = _fit(cfg, data, n, train_eps, val_eps, out, device)
    result = {"excluded_train_episodes": len(split["train"]) - len(train_eps),
              "excluded_val_episodes": len(split["val"]) - len(val_eps)}
    for name in report_splits:
        preds = predict_episodes(model, data, split[name], sub, cfg.eval.get("decode", {}), device)
        preds.to_parquet(out / f"predictions_{name}.parquet")
        m = score(preds, n, fps)
        (out / f"metrics_{name}.json").write_text(json.dumps(m, indent=1))
        result[name] = {k: {"accuracy": round(v["accuracy"], 4),
                            "accuracy_excl_1s_boundary": round(v["accuracy_excl_1s_boundary"], 4)} for k, v in m.items()}
    return result


def out_of_fold(cfg: Config, folds: int = 5) -> pd.DataFrame:
    """Predict every train+val episode with a model that never saw it (k-fold); test is left alone.

    Returns per-episode agreement between the decoded prediction and the dataset labels, used to
    find episodes whose labels are likely wrong.
    """
    device, split, cache = setup(cfg)
    _apply_pseudo_labels(cfg, cache, split)
    fps, sub = int(cfg.get("fps", 30)), int(cfg.episode_stage.subsample)
    out = run_dir(cfg) / "episode_stage_oof"
    out.mkdir(parents=True, exist_ok=True)
    data, n = _Episodes(cache, fps), len(cache.stage_names)
    pool = np.random.default_rng(int(cfg.get("seed", 42))).permutation(split["train"] + split["val"]).tolist()
    parts = [pool[k::folds] for k in range(folds)]
    preds = []
    for k, held in enumerate(parts):
        rest = [e for e in pool if e not in set(held)]
        model = _fit(cfg, data, n, rest, None, out, device, tag=f"_fold{k}")
        preds.append(predict_episodes(model, data, held, sub, cfg.eval.get("decode", {}), device))
    preds = pd.concat(preds, ignore_index=True)
    preds.to_parquet(out / "predictions_oof.parquet")
    agree = (preds.assign(ok=preds.stage_decoded == preds.stage_label)
             .groupby("episode_index").ok.mean().rename("agreement").sort_values())
    agree.to_csv(out / "agreement.csv")
    print(f"[episode-stage oof] overall agreement {float((preds.stage_decoded == preds.stage_label).mean()):.4f}; "
          f"episodes below 0.7: {int((agree < 0.7).sum())} of {len(agree)}")
    return agree


def predictions_to_segments(preds: pd.DataFrame, stage_names: list[str], col: str = "stage_decoded") -> dict:
    """Per-frame stage predictions -> {episode: [names, starts, ends]} (the cache's segments.json format)."""
    out = {}
    for ep, d in preds.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        f, s = d["frame_index"].to_numpy(), d[col].to_numpy()
        cut = np.flatnonzero(np.diff(s)) + 1
        starts, ends = np.r_[0, cut], np.r_[cut, len(s)]
        runs = [(int(s[a]), int(f[a]), int(f[b - 1]) + 1) for a, b in zip(starts, ends) if s[a] < len(stage_names)]
        out[int(ep)] = [[stage_names[k] for k, _, _ in runs], [a for _, a, _ in runs], [b for _, _, b in runs]]
    return out


def write_relabeled_segments(oof_file: str, select_file: str, max_agreement: float, stage_names: list[str],
                             out_file: str) -> int:
    """Relabeled segments for train/val episodes whose dataset-label agreement is below max_agreement."""
    oof, sel = pd.read_parquet(oof_file), pd.read_parquet(select_file)
    agree = (sel.stage_decoded == sel.stage_label).groupby(sel.episode_index).mean()
    eps = set(int(e) for e in agree[agree < max_agreement].index)
    segs = predictions_to_segments(oof[oof.episode_index.isin(eps)], stage_names)
    open(out_file, "w").write(json.dumps(segs))
    return len(segs)


@torch.no_grad()
def ensemble_predict(cfg: Config, run_names: list[str], split_name: str = "test") -> pd.DataFrame:
    """Average the per-step stage probabilities of several trained runs, then decode in order."""
    device, split, cache = setup(cfg)
    e, fps = cfg.episode_stage, int(cfg.get("fps", 30))
    sub = int(e.subsample)
    data, n = _Episodes(cache, fps), len(cache.stage_names)
    models = []
    for name in run_names:
        m = EpisodeStageModel(data.dim, n, int(e.hidden), int(e.layers), float(e.dropout)).to(device).eval()
        m.load_state_dict(torch.load(run_dir(cfg).parent / name / "episode_stage" / "best.pt", map_location=device)["model"])
        models.append(m)
    decode = cfg.eval.get("decode", {})
    kw = {"switch_penalty": float(decode.get("switch_penalty", 5.0)), "skip_penalty": float(decode.get("skip_penalty", 5.0))}
    rows = []
    for ep in split[split_name]:
        steps = data.steps(ep, sub)
        x, _, lengths = _batch([data.sequence(ep, steps)], device)
        probs = np.mean([torch.softmax(m(x, lengths)[0], -1).cpu().numpy() for m in models], 0)
        decoded = ordered_viterbi(probs, **kw)
        frame = np.arange(cache.length[ep])
        at = np.minimum(frame // sub, len(steps) - 1)
        rows.append(pd.DataFrame({"episode_index": ep, "frame_index": frame,
                                  "stage_label": cache.stage[cache.row_start[ep] + frame],
                                  "stage_pred": probs.argmax(-1)[at], "stage_decoded": decoded[at]}))
    return pd.concat(rows, ignore_index=True)
