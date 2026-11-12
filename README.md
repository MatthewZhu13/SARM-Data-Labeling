# SARM-Data-Labeling

A pipeline that takes a LeRobot v3.0 dataset, trains SARM2 (a stage estimator and a reward model) on it, and labels every frame (or selected frames) with a learned stage and a progress value.

SARM2's model code comes from [`xdofai/opensarm`](https://github.com/xdofai/opensarm), pinned as a submodule. Data loading, labels, the training loop and evaluation are ours. [`docs/SNAGS.md`](docs/SNAGS.md) lists every difference from opensarm and the reasons for it.

## Setup
```bash
git clone --recurse-submodules git@github.com:MatthewZhu13/SARM-Data-Labeling.git
cd SARM-Data-Labeling
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q tests
```
Use Python 3.10 or newer. On a machine that already has CUDA PyTorch, create the venv with `--system-site-packages`.

## Run
One command runs every step:
```bash
python -m sarm_pipeline all --config configs/hqf_pilot.yaml
```
You can also run the steps one at a time:

| Step | What it does | Output |
|---|---|---|
| `prep` | Downloads metadata and splits the stage-labeled episodes 80/10/10 by episode, with a fixed seed | `splits/*.json` |
| `fetch` | Downloads only the data parquets and videos holding those episodes | `data/<dataset>/` |
| `embed` | Decodes every video once and caches a frozen SigLIP-2 vector per frame and camera, plus state and stage boundaries | `cache/<dataset>/` |
| `train-stage` | Trains the stage estimator (cross-entropy on the anchor frame's stage) | `runs/<run>/stage/{best,latest}.pt`, `metrics.jsonl` |
| `train-reward` | Trains the reward model; it loads the stage estimator and fine-tunes it jointly | `runs/<run>/reward/...` |
| `eval` | Labels every test frame the way the labeler will, then computes metrics and plots | `runs/<run>/eval_test/` |

To change any setting without editing files, pass `--set key=value`, for example `--set labels.progress=linear train.val_every=500`. To continue an interrupted training run, add `--resume`.

Configs:
- `configs/hqf_pilot.yaml`: the full run on `lerobot/high_quality_folding` (451 stage-labeled episodes)
- `configs/local_medium.yaml`: 120 episodes, sized for a laptop
- `configs/smoke.yaml`: 6 episodes, about 60 steps per model, an end-to-end check that takes minutes

### On the GPU box
```bash
scripts/box.sh start && scripts/box.sh sync && scripts/box.sh setup
scripts/box.sh run configs/hqf_pilot.yaml        # runs in tmux on the box
scripts/box.sh log configs/hqf_pilot.yaml        # follow it; keeps the box awake during CPU-only steps
scripts/box.sh pull configs/hqf_pilot.yaml       # bring back metrics, plots, checkpoints
scripts/box.sh stop                              # the box costs money while on
```

## How it works

### Data
- `v3.py` reads v3.0 metadata. It maps every (episode, frame, camera) to (mp4 file, frame inside that file) and to the frame's row in the data parquet. Nothing is converted to v2.1.
- `embed.py` decodes each mp4 front to back once. Each frame is resized to 224×224 and encoded with frozen SigLIP-2 (`google/siglip2-base-patch16-224`), and the vectors are stored as float16. After that, training never touches video.
- `labels.py` turns stage boundaries into per-frame labels:
  - **stage**: one of the 5 folding stages, or `dummy` outside them
  - **progress**: `linear` (frame / length) or `stage_aware`, where progress = (share of the earlier stages) + (this stage's share × position within the stage). The stage shares come from the train split.

### Training windows
`dataset.py` builds windows. With `n_obs_steps: 6` and `frame_gap: 30` (opensarm's defaults):
- **reward model**: episode start frame + 5 history frames 1 s apart + anchor = 7 frames × 3 cameras, plus 16-dim joint state
- **stage estimator**: 5 history frames + anchor = 6 frames
- **rewind**: in reward-model training only, about 80% of samples get 1–3 recent frames appended in reverse order, with their lower progress, so the model learns that undoing work lowers progress. Validation and evaluation never rewind.

### Models and training
- `sarm2.py`:
  - imports opensarm's `ActionTransformer` (stage estimator) and `RewardTransformer` (MoE reward model) unchanged
  - contains the loss code copied from opensarm's workspaces
- `train.py`:
  - runs for a fixed step budget, with AdamW, warmup + cosine decay and gradient clipping
  - validates every `val_every` steps on a fixed set of windows; `best.pt` is the checkpoint with the lowest validation loss
- `evaluate.py` reports, per frame on held-out episodes:
  - stage accuracy, confusion matrix and boundary error, for the stage estimator alone and as fine-tuned
  - progress MAE and Spearman correlation, and how often progress drops on good demos
  - an oracle-stage ablation (progress error when the reward model is given the true stage)

## Keep in mind
- The progress score is a learned estimate, not ground truth. Training labels rise steadily within each stage, so the model only learns stall and regression behavior from rewind augmentation and from reading the scene. That has to be checked on real failure episodes.
- Settings are capped for a 40GB GPU (`train.gpu_mem_fraction`). The SigLIP pass runs only once, in `embed`.
