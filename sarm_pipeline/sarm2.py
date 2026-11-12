"""SARM2 models from opensarm (pinned submodule) and the losses from its training workspaces.

The model classes are imported unchanged. The loss code is copied from
third_party/opensarm/workspace/act_pri_ws.py and sarm2_ws.py (commit 9978198) so that
training matches theirs; tests/test_sarm2.py checks the copy against a hand-built reference.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import Config

OPENSARM = Path(__file__).resolve().parents[1] / "third_party" / "opensarm"


def _load(relpath: str):
    # Loaded by file path: opensarm's top-level `lerobot/` folder would shadow the real package.
    path = OPENSARM / relpath
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `git submodule update --init`")
    spec = importlib.util.spec_from_file_location(f"opensarm_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ActionTransformer = _load("models/action_estimator.py").ActionTransformer
RewardTransformer = _load("models/moe_deco_gate_reward_model.py").RewardTransformer


def stage_groups(cfg: Config) -> torch.Tensor:
    """stage index -> MoE gate group; "dummy" (last stage index) gets its own group."""
    n_stages = len(cfg.stages.names)
    mapping = [-1] * (n_stages + 1)
    groups = list(dict(cfg.stages.groups).values())
    for g, members in enumerate(groups):
        for k in members:
            mapping[k] = g
    if -1 in mapping[:-1]:
        raise ValueError(f"stages without a group: {[i for i, g in enumerate(mapping[:-1]) if g == -1]}")
    mapping[-1] = len(groups)
    return torch.tensor(mapping)


def build_stage_model(cfg: Config, emb_dim: int, state_dim: int, n_cameras: int) -> torch.nn.Module:
    m = cfg.stage_model
    return ActionTransformer(d_model=m.d_model, vis_emb_dim=emb_dim, state_dim=state_dim, n_layers=m.n_layers,
                             n_heads=m.n_heads, dropout=m.dropout, num_cameras=n_cameras,
                             num_tasks=len(cfg.stages.names) + 1,
                             num_classes=int(stage_groups(cfg).max()) + 1)


def build_reward_model(cfg: Config, emb_dim: int, state_dim: int, n_cameras: int) -> torch.nn.Module:
    m, moe = cfg.reward_model, cfg.reward_model.moe
    return RewardTransformer(d_model=m.d_model, vis_emb_dim=emb_dim, text_emb_dim=emb_dim, state_dim=state_dim,
                             n_layers=m.n_layers, n_heads=m.n_heads, dropout=m.dropout, num_cameras=n_cameras,
                             num_gates=int(stage_groups(cfg).max()) + 1, num_experts=moe.num_experts,
                             top_k=moe.top_k, gate_units=list(moe.gate_units), gate_act=moe.gate_act,
                             gate_dropout=moe.gate_dropout, expert_units=list(moe.expert_units),
                             expert_act=moe.expert_act, expert_dropout=moe.expert_dropout,
                             lambda_balance=moe.lambda_balance, lambda_entropy=moe.lambda_entropy,
                             lambda_importance=moe.lambda_importance)


def stage_loss(model, batch: dict, groups: torch.Tensor, alpha_class_loss: float) -> tuple[torch.Tensor, dict]:
    """act_pri_ws.py: cross-entropy on the stage + alpha_class_loss * cross-entropy on its group."""
    logits, class_logits = model(batch["img"], batch["state"], batch["length"])
    act_loss = F.cross_entropy(logits, batch["stage"], reduction="mean")
    class_loss = F.cross_entropy(class_logits, groups[batch["stage"]], reduction="mean")
    loss = act_loss + alpha_class_loss * class_loss
    acc = (logits.argmax(-1) == batch["stage"]).float().mean()
    return loss, {"loss": loss.item(), "act_loss": act_loss.item(), "class_loss": class_loss.item(), "acc": acc.item()}


def stage_inputs(batch: dict, n_obs_steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """sarm2_ws._slice_act_pri_inputs: the stage model sees only the history + anchor slots
    [1 : n_obs_steps + 1] of the reward window (no episode start, no rewind)."""
    img = batch["img"][:, :, 1:n_obs_steps + 1]
    state = batch["state"][:, 1:n_obs_steps + 1]
    lens = torch.full((img.shape[0],), n_obs_steps, dtype=torch.int32, device=img.device)
    return img, state, lens


def reward_forward(reward_model, stage_model, batch: dict, stage_emb: torch.Tensor, task_emb: torch.Tensor,
                   groups: torch.Tensor, n_obs_steps: int, finetune_stage: bool,
                   oracle_stage: bool = False) -> dict:
    """sarm2_ws.py training step up to the loss: stage model -> stage embedding + MoE gate -> reward model.
    Returns r = predicted steps-to-go (1 - progress) per slot, plus the stage logits."""
    s_img, s_state, s_lens = stage_inputs(batch, n_obs_steps)
    if finetune_stage:
        logits, class_logits = stage_model(s_img, s_state, s_lens)
        lang_emb = F.softmax(logits, dim=-1) @ stage_emb            # soft, keeps the path differentiable
    else:
        with torch.no_grad():
            logits, class_logits = stage_model(s_img, s_state, s_lens)
        lang_emb = stage_emb[logits.argmax(-1)]
    pred_stage = logits.argmax(-1)
    if oracle_stage:                                                # evaluation ablation: true stage
        pred_stage = batch["stage"]
        lang_emb = stage_emb[pred_stage]
    gate_idx = groups[pred_stage]
    task = task_emb.unsqueeze(0).expand(s_img.shape[0], -1)
    r, aux_loss, info = reward_model(batch["img"], lang_emb, task, batch["state"], batch["length"], gate_idx=gate_idx)
    return {"r": r, "aux_loss": aux_loss, "info": info, "logits": logits, "class_logits": class_logits}


def reward_loss(out: dict, batch: dict, groups: torch.Tensor, alpha_auxiliary: float, finetune_stage: bool,
                stage_loss_weight: float) -> tuple[torch.Tensor, dict]:
    """sarm2_ws.py: MSE on steps-to-go (padding slots included, as in opensarm) + MoE auxiliary
    loss, plus the stage cross-entropy terms when the stage model is fine-tuned jointly."""
    trg = 1.0 - batch["progress"]                                   # steps_to_go; padding -> 1
    fit_loss = F.mse_loss(out["r"], trg, reduction="mean")
    loss = fit_loss + alpha_auxiliary * out["aux_loss"]
    logs = {"fit_loss": fit_loss.item(), "aux_loss": out["aux_loss"].item(), "gate_entropy": out["info"]["gate_entropy"]}
    if finetune_stage:
        ce = F.cross_entropy(out["logits"], batch["stage"], reduction="mean")
        class_ce = F.cross_entropy(out["class_logits"], groups[batch["stage"]], reduction="mean")
        loss = loss + stage_loss_weight * (ce + class_ce)
        logs["stage_ce"] = ce.item()
    logs["loss"] = loss.item()
    return loss, logs


def progress_at_anchor(out: dict, n_obs_steps: int) -> torch.Tensor:
    """Predicted progress at the anchor slot: [start, history x (n_obs_steps - 1), anchor, rewind...]."""
    return 1.0 - out["r"][:, n_obs_steps]
