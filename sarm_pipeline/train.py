"""Training for the two SARM2 models: the stage estimator first, then the reward model.

Both use the same loop: a fixed step budget, AdamW with linear warmup then cosine decay
(over exactly that budget), gradient clipping, validation every `val_every` steps on a
fixed set of windows, and checkpoints:
  best.pt    lowest validation loss so far (overwritten when it improves)
  latest.pt  most recent state, for resuming (`--resume`)
Metrics go to <run>/<model>/metrics.jsonl.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import sarm2
from .config import Config
from .dataset import Cache, WindowDataset, fixed_anchors, load_cache
from .embed import pick_device


def run_dir(cfg: Config) -> Path:
    return Path(cfg.output_dir) / cfg.run_name


def load_split(cfg: Config) -> dict[str, list[int]]:
    path = Path(cfg.split.file)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run the `prep` step first")
    return json.loads(path.read_text())


def setup(cfg: Config) -> tuple[torch.device, dict, Cache]:
    device = pick_device(cfg.train.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(float(cfg.train.gpu_mem_fraction))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "mps":
        # eval-mode TransformerEncoder with a padding mask uses nested tensors, which MPS lacks
        torch.backends.mha.set_fastpath_enabled(False)
    split = load_split(cfg)
    cache = load_cache(cfg.cache_dir, list(cfg.dataset.cameras), list(cfg.stages.names), cfg.labels.progress,
                       split["train"], segments_override=cfg.labels.get("segments_override"))
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    (out / "split.json").write_text(json.dumps(split, indent=1))
    (out / "label_stats.json").write_text(json.dumps({
        "progress_mode": cfg.labels.progress, "segments_override": cfg.labels.get("segments_override"),
        "stage_shares": cache.alpha.tolist(),
        "state_mean": cache.state_mean.tolist(), "state_std": cache.state_std.tolist(),
        "git_commit": _git_commit()}, indent=1))
    return device, split, cache


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def to_device(batch: dict, device: torch.device) -> dict:
    # non_blocking is only safe on CUDA; on MPS it can read the CPU buffer after it is reused
    return {k: v.to(device, non_blocking=device.type == "cuda") for k, v in batch.items()}


def make_optimizer(params, optim: Config, steps: int):
    opt = torch.optim.AdamW(params, lr=optim.lr, betas=tuple(optim.betas), eps=optim.eps,
                            weight_decay=optim.weight_decay)
    warmup = min(int(optim.warmup_steps), max(steps - 1, 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-6 / optim.lr, end_factor=1.0, total_iters=warmup),
        # opensarm's act_pri config decays over 80k steps but trains 16k; here the decay matches the budget.
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps - warmup, 1), eta_min=0.0),
    ], milestones=[warmup])
    return opt, sched


def _forever(loader):
    while True:
        yield from loader


def fit(name: str, cfg: Config, out: Path, models: dict[str, torch.nn.Module], params: list, optim: Config,
        steps: int, train_loader: DataLoader, step_fn: Callable, val_fn: Callable, device: torch.device,
        resume: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    opt, sched = make_optimizer(params, optim, steps)
    start, best, since_best = 0, float("inf"), 0
    if resume and (out / "latest.pt").exists():
        ckpt = torch.load(out / "latest.pt", map_location=device)
        for key, model in models.items():
            model.load_state_dict(ckpt["models"][key])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start, best, since_best = ckpt["step"], ckpt["best"], ckpt.get("since_best", 0)
        print(f"[{name}] resumed at step {start}, best val {best:.5f}")

    def save(path: Path, step: int):
        torch.save({"models": {k: m.state_dict() for k, m in models.items()}, "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "step": step, "best": best, "since_best": since_best,
                    "config": cfg.to_dict()}, path)

    log = open(out / "metrics.jsonl", "a")

    def write(record: dict):
        log.write(json.dumps(record) + "\n")
        log.flush()

    batches = _forever(train_loader)
    t0, step = time.time(), start
    patience = cfg.train.early_stop_patience
    for step in range(start + 1, steps + 1):
        for m in models.values():
            m.train()
        batch = to_device(next(batches), device)
        loss, logs = step_fn(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip).item()
        opt.step()
        sched.step()
        if not np.isfinite(logs["loss"]):
            raise FloatingPointError(f"[{name}] non-finite loss at step {step}: {logs}")

        if step % cfg.train.log_every == 0 or step == 1:
            rate = (step - start) / (time.time() - t0)
            rec = {"step": step, "split": "train", "lr": sched.get_last_lr()[0], "grad_norm": grad_norm,
                   "steps_per_s": rate, **logs}
            write(rec)
            print(f"[{name}] step {step}/{steps} loss {logs['loss']:.4f} lr {rec['lr']:.2e} "
                  f"{rate:.1f} it/s eta {(steps - step) / max(rate, 1e-9) / 60:.1f} min", flush=True)

        if step % cfg.train.val_every == 0 or step == steps:
            for m in models.values():
                m.eval()
            with torch.no_grad():
                val = val_fn()
            improved = val["val_loss"] < best
            if improved:
                best, since_best = val["val_loss"], 0
                save(out / "best.pt", step)
            else:
                since_best += 1
            write({"step": step, "split": "val", "best": best, **val})
            print(f"[{name}] val step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in val.items())
                  + (" (best)" if improved else ""), flush=True)
            if patience and since_best >= patience:
                print(f"[{name}] early stop: no improvement in {patience} checks")
                break

        if step % cfg.train.save_every == 0 or step == steps:
            save(out / "latest.pt", step)
    save(out / "latest.pt", step)
    log.close()
    summary = {"model": name, "steps": step, "best_val_loss": best,
               "minutes": (time.time() - t0) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _loader(ds: WindowDataset, batch_size: int, cfg: Config, shuffle: bool) -> DataLoader:
    workers = int(cfg.train.num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


def train_stage(cfg: Config, resume: bool = False) -> dict:
    device, split, cache = setup(cfg)
    w = cfg.window
    torch.manual_seed(cfg.split.seed)
    train_ds = WindowDataset(cache, split["train"], "stage", w.n_obs_steps, w.frame_gap,
                             stage_window=cfg.stage_model.window)
    val_ds = WindowDataset(cache, split["val"], "stage", w.n_obs_steps, w.frame_gap,
                           anchors=fixed_anchors(cache, split["val"], cfg.train.val_anchors, cfg.split.seed))
    emb_dim = next(iter(cache.emb.values())).shape[1]
    model = sarm2.build_stage_model(cfg, emb_dim, cache.state.shape[1], len(cache.cameras)).to(device)
    groups = sarm2.stage_groups(cfg).to(device)
    alpha_class = float(cfg.stage_model.alpha_class_loss)
    val_loader = _loader(val_ds, 256, cfg, shuffle=False)

    def step_fn(batch):
        return sarm2.stage_loss(model, batch, groups, alpha_class)

    def val_fn():
        losses, correct, n = [], 0, 0
        for batch in val_loader:
            batch = to_device(batch, device)
            loss, logs = sarm2.stage_loss(model, batch, groups, alpha_class)
            losses.append(loss.item() * len(batch["row"]))
            correct += logs["acc"] * len(batch["row"])
            n += len(batch["row"])
        return {"val_loss": sum(losses) / n, "val_acc": correct / n}

    print(f"[stage] train {len(train_ds)} anchors from {len(split['train'])} episodes, "
          f"val {len(val_ds)} anchors, device {device}")
    return fit("stage", cfg, run_dir(cfg) / "stage", {"stage": model}, list(model.parameters()),
               cfg.stage_model.optim, int(cfg.stage_model.steps),
               _loader(train_ds, cfg.stage_model.batch_size, cfg, shuffle=True), step_fn, val_fn, device, resume)


def load_stage_model(cfg: Config, cache: Cache, device: torch.device, path: Path | None = None):
    path = path or run_dir(cfg) / "stage" / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; train the stage estimator first")
    emb_dim = next(iter(cache.emb.values())).shape[1]
    model = sarm2.build_stage_model(cfg, emb_dim, cache.state.shape[1], len(cache.cameras)).to(device)
    model.load_state_dict(torch.load(path, map_location=device)["models"]["stage"])
    return model


def train_reward(cfg: Config, resume: bool = False) -> dict:
    device, split, cache = setup(cfg)
    w, rm = cfg.window, cfg.reward_model
    torch.manual_seed(cfg.split.seed)
    train_ds = WindowDataset(cache, split["train"], "reward", w.n_obs_steps, w.frame_gap,
                             max_rewind_steps=w.max_rewind_steps, rewind_prob=w.rewind_prob)
    val_ds = WindowDataset(cache, split["val"], "reward", w.n_obs_steps, w.frame_gap,
                           anchors=fixed_anchors(cache, split["val"], cfg.train.val_anchors, cfg.split.seed))
    emb_dim = next(iter(cache.emb.values())).shape[1]
    stage_model = load_stage_model(cfg, cache, device)
    reward_model = sarm2.build_reward_model(cfg, emb_dim, cache.state.shape[1], len(cache.cameras)).to(device)
    finetune = bool(rm.finetune_stage_model)
    if not finetune:
        for p in stage_model.parameters():
            p.requires_grad_(False)
    groups = sarm2.stage_groups(cfg).to(device)
    stage_emb, task_emb = cache.stage_emb.to(device), cache.task_emb.to(device)
    params = list(reward_model.parameters()) + (list(stage_model.parameters()) if finetune else [])
    val_loader = _loader(val_ds, 256, cfg, shuffle=False)

    def forward(batch, finetune_stage):
        return sarm2.reward_forward(reward_model, stage_model, batch, stage_emb, task_emb, groups,
                                    w.n_obs_steps, finetune_stage)

    def step_fn(batch):
        if not finetune:
            stage_model.eval()
        out = forward(batch, finetune)
        return sarm2.reward_loss(out, batch, groups, rm.moe.alpha_auxiliary, finetune, rm.stage_loss_weight)

    def val_fn():
        se, n_slots, abs_err, correct, n = 0.0, 0, 0.0, 0, 0
        for batch in val_loader:
            batch = to_device(batch, device)
            out = forward(batch, False)
            valid = torch.arange(out["r"].shape[1], device=device)[None] < batch["length"][:, None]
            err = (out["r"] - (1.0 - batch["progress"]))[valid]
            se += (err ** 2).sum().item()
            n_slots += valid.sum().item()
            abs_err += (sarm2.progress_at_anchor(out, w.n_obs_steps)
                        - batch["progress"][:, w.n_obs_steps]).abs().sum().item()
            correct += (out["logits"].argmax(-1) == batch["stage"]).sum().item()
            n += len(batch["row"])
        return {"val_loss": se / n_slots, "val_anchor_mae": abs_err / n, "val_stage_acc": correct / n}

    print(f"[reward] train {len(train_ds)} anchors from {len(split['train'])} episodes, "
          f"val {len(val_ds)} anchors, finetune stage model: {finetune}, device {device}")
    return fit("reward", cfg, run_dir(cfg) / "reward", {"reward": reward_model, "stage": stage_model}, params,
               rm.optim, int(rm.steps), _loader(train_ds, rm.batch_size, cfg, shuffle=True), step_fn, val_fn,
               device, resume)
