import ast
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from sarm_pipeline import sarm2
from sarm_pipeline.config import load_config
from sarm_pipeline.dataset import Cache, WindowDataset, frame_window
from sarm_pipeline.labels import episode_labels, stage_shares

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["a", "b", "c"]


def _opensarm_get_frame_indices():
    """Pull FrameGapLeRobotDataset.get_frame_indices out of opensarm's source (avoids its heavy imports)."""
    src = (ROOT / "third_party/opensarm/lerobot/common/datasets/rm_act_pri_lerobot_dataset.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "get_frame_indices":
            module = ast.Module(body=[node], type_ignores=[])
            ns = {}
            exec(compile(module, "opensarm", "exec"), ns)
            return ns["get_frame_indices"]
    raise AssertionError("get_frame_indices not found")


@pytest.mark.parametrize("stage_model", [False, True])
def test_window_matches_opensarm(stage_model):
    theirs = _opensarm_get_frame_indices()
    self = types.SimpleNamespace(stage_model=stage_model)
    for ep_start in (0, 1000):
        for idx in list(range(ep_start, ep_start + 400, 7)) + [ep_start + 2000]:
            expected = theirs(self, idx, 6, 30, ep_start, ep_start + 3000)
            assert frame_window(idx, ep_start, 6, 30, include_start=not stage_model) == expected


def test_stage_aware_progress():
    segs = (["a", "b", "c"], [10, 30, 50], [30, 50, 90])     # gap-free; frames 0-9 and 90-99 are dummy
    alpha = stage_shares([segs], NAMES)
    np.testing.assert_allclose(alpha, [0.25, 0.25, 0.5])
    stage, progress = episode_labels(100, segs, NAMES, alpha, "stage_aware")
    assert (stage[:10] == 3).all() and (stage[90:] == 3).all() and stage[10] == 0 and stage[50] == 2
    assert progress[0] == 0 and progress[9] == 0
    assert progress[30] == pytest.approx(0.25) and progress[50] == pytest.approx(0.5)
    assert progress[70] == pytest.approx(0.75) and (progress[90:] == 1).all()
    assert (np.diff(progress) >= 0).all()


def test_gap_holds_previous_stage_end():
    segs = (["a", "b"], [0, 60], [40, 100])
    _, progress = episode_labels(100, segs, ["a", "b"], np.array([0.5, 0.5]), "stage_aware")
    assert (progress[40:60] == pytest.approx(0.5)) and progress[60] == pytest.approx(0.5)


def _toy_cache(lengths=(400, 300)) -> Cache:
    n = sum(lengths)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    progress = np.concatenate([np.linspace(0, 1, n_) for n_ in lengths]).astype(np.float32)
    emb = {c: np.random.randn(n, 8).astype(np.float16) for c in ("left", "right", "top")}
    return Cache([0, 1], dict(zip([0, 1], lengths)), dict(zip([0, 1], starts.tolist())), emb,
                 np.random.randn(n, 16).astype(np.float32), np.zeros(n, dtype=np.int64), progress,
                 NAMES + ["dummy"], torch.randn(4, 8), torch.randn(8), np.zeros(16), np.ones(16), np.ones(3) / 3)


def test_reward_window_and_rewind():
    torch.manual_seed(0)
    c = _toy_cache()
    ds = WindowDataset(c, [0, 1], "reward", 6, 30, max_rewind_steps=3, rewind_prob=1.0, anchors=np.array([350]))
    item = ds[0]
    n = int(item["length"])
    assert item["img"].shape == (3, 10, 8) and 8 <= n <= 10
    assert item["progress"][0] == 0                                  # episode start
    p = item["progress"][:n].numpy()
    assert (np.diff(p[:7]) > 0).all() and (np.diff(p[6:]) < 0).all()  # forward window, then rewind goes down
    assert (item["progress"][n:] == 0).all()
    np.testing.assert_allclose(item["img"][0, 1].numpy(), c.emb["left"][200].astype(np.float32))  # 350 - 5*30


def test_no_rewind_near_episode_start_or_in_stage_mode():
    c = _toy_cache()
    ds = WindowDataset(c, [0], "reward", 6, 30, max_rewind_steps=3, rewind_prob=1.0, anchors=np.array([100]))
    assert int(ds[0]["length"]) == 7
    ds = WindowDataset(c, [0], "stage", 6, 30, max_rewind_steps=3, rewind_prob=1.0, anchors=np.array([350]))
    assert ds[0]["img"].shape[1] == 6 and int(ds[0]["length"]) == 6


def test_models_build_and_train_step():
    cfg = load_config(ROOT / "configs/smoke.yaml")
    c = _toy_cache()
    ds = WindowDataset(c, [0, 1], "reward", 6, 30, max_rewind_steps=3, rewind_prob=0.8)
    batch = {k: torch.stack([ds[i][k] for i in range(0, 600, 60)]) for k in ds[0]}
    assert sarm2.stage_groups(cfg).tolist() == [0, 2, 0, 1, 1, 3]   # the 5 folding stages + dummy
    cfg3 = cfg.__class__({**cfg, "stages": {**cfg["stages"], "names": NAMES, "groups": {"x": [0, 1], "y": [2]}}})
    stage = sarm2.build_stage_model(cfg3, 8, 16, 3)
    reward = sarm2.build_reward_model(cfg3, 8, 16, 3)
    g3 = sarm2.stage_groups(cfg3)
    out = sarm2.reward_forward(reward, stage, batch, c.stage_emb, c.task_emb, g3, 6, finetune_stage=True)
    assert out["r"].shape == (10, 10) and ((out["r"] >= 0) & (out["r"] <= 1)).all()
    loss, logs = sarm2.reward_loss(out, batch, g3, 0.2, True, 0.25)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters())  # joint fine-tuning


def test_config_extends_and_overrides():
    cfg = load_config(ROOT / "configs/smoke.yaml", ["train.val_every=7", "labels.progress=linear"])
    assert cfg.run_name == "smoke" and cfg.train.val_every == 7 and cfg.labels.progress == "linear"
    assert cfg.reward_model.optim.lr == 5.0e-5 and cfg.reward_model.optim.warmup_steps == 10
