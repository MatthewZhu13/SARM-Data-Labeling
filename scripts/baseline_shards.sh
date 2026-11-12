#!/bin/bash
# folding_sarm_reward (scripts/baseline_sarm1.py) on the separation episodes, 4 shards in parallel
# (decoding is the bottleneck), then merged into runs/baseline_sarm1/progress.parquet.
cd ~/SARM-Data-Labeling
B="$HOME/lr_venv/bin/python scripts/baseline_sarm1.py --model checkpoints/folding_sarm_reward --root data/full_folding
   --repo-id lerobot/full_folding --episodes-file splits/sep_full_folding.json --out runs/baseline_sarm1/progress.parquet"
for i in 0 1 2 3; do
  [ -f runs/baseline_sarm1/progress.parquet.shard$i ] || $B --shard $i/4 > logs/baseline_shard$i.log 2>&1 &
done
wait
$B --merge 4 > logs/baseline_merge.log 2>&1
