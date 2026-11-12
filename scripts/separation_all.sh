#!/bin/bash
# Accepted-vs-rejected report end to end, on the GPU box (run in tmux; ~1.5 h from scratch):
#   select + download + embed the separation episodes, the folding_sarm_reward baseline (scripts/baseline_shards.sh),
#   score the SARM2 runs given as arguments, then write reports/separation/.
#     scripts/separation_all.sh sarm_hand hqf_pilot
# Safe to re-run: embedding resumes where it stopped, finished outputs are kept, and a step that is already
# running elsewhere is waited for rather than started twice.
cd ~/SARM-Data-Labeling
PY=.venv/bin/python
SEP="--config configs/sep_full_folding.yaml"
mkdir -p logs
running() { pgrep -f "$1" > /dev/null; }

[ -f splits/sep_full_folding.json ] || $PY -m sarm_pipeline.separation select $SEP
while running "sarm_pipeline.separation embed"; do sleep 30; done
$PY -m sarm_pipeline.separation embed $SEP >> logs/sep_embed.log 2>&1 || { echo "embed failed"; exit 1; }

while running "baseline_sarm1.py --model|baseline_shards.sh"; do sleep 30; done
[ -f runs/baseline_sarm1/progress.parquet ] || bash scripts/baseline_shards.sh &

PROGRESS=()
for run in "$@"; do
  while tmux has-session -t "$run" 2>/dev/null; do sleep 30; done        # still training
  [ -f runs/$run/separation/progress.parquet ] || \
    $PY -m sarm_pipeline.separation score $SEP --run "$run" --set train.gpu_mem_fraction=0.3
  PROGRESS+=(runs/$run/separation/progress.parquet)
done
wait
$PY -m sarm_pipeline.separation report $SEP --progress "${PROGRESS[@]}" runs/baseline_sarm1/progress.parquet
echo SEPARATION_DONE
