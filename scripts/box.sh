#!/usr/bin/env bash
# Run the pipeline on the GPU box (ssh hosts gpu-control / deplace-gpu, see ~/.ssh/config).
#
#   scripts/box.sh start                 power the box on (gpu-control gpu start)
#   scripts/box.sh sync                  copy this repo to the box (code, configs, splits; not data/cache/runs)
#   scripts/box.sh setup                 create the venv on the box (reuses its CUDA PyTorch)
#   scripts/box.sh run CONFIG [STEP]     start `python -m sarm_pipeline STEP` in tmux (STEP defaults to all)
#   scripts/box.sh log CONFIG            follow the run log (an open ssh session also keeps the box awake)
#   scripts/box.sh pull CONFIG           copy the run's metrics, eval outputs and checkpoints back here
#   scripts/box.sh stop                  power the box off (it costs money while on)
#
# The box powers itself off after 30 min with no ssh session and an idle GPU, so keep `log`
# open during the CPU-only steps (fetch, prep).
set -euo pipefail

HOST=${BOX_HOST:-deplace-gpu}
REMOTE_DIR=${BOX_DIR:-SARM-Data-Labeling}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

run_name() { grep -E '^run_name:' "$ROOT/$1" | awk '{print $2}'; }

case "${1:-}" in
  start) ssh gpu-control gpu start ;;
  stop)  ssh gpu-control gpu stop ;;
  sync)
    rsync -az --delete --exclude .venv --exclude data --exclude cache --exclude runs --exclude logs \
      --exclude '__pycache__' --exclude '*.egg-info' "$ROOT/" "$HOST:$REMOTE_DIR/" ;;
  setup)
    # the box has no python3-venv (ensurepip), so the venv borrows the system pip
    ssh "$HOST" "cd $REMOTE_DIR && python3 -m venv --without-pip --system-site-packages .venv \
      && .venv/bin/python -m pip install -q -e . \
      && .venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))' \
      && nproc && free -g | head -2 && df -h ~ | tail -1" ;;
  run)
    cfg=${2:?config}; step=${3:-all}; name=$(run_name "$cfg")
    ssh "$HOST" "cd $REMOTE_DIR && tmux new-session -d -s sarm_${name}_${step//-/_} bash scripts/run_logged.sh $cfg $step ${*:4}" ;;
  log)
    name=$(run_name "${2:?config}")
    ssh "$HOST" "tail -n 50 -F $REMOTE_DIR/logs/${name}.log" ;;
  pull)
    name=$(run_name "${2:?config}")
    mkdir -p "$ROOT/runs/$name"
    rsync -az "$HOST:$REMOTE_DIR/runs/$name/" "$ROOT/runs/$name/"
    rsync -az "$HOST:$REMOTE_DIR/logs/${name}.log" "$ROOT/logs/" || true ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
