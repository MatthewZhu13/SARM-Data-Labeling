#!/bin/bash
# Client data, 5-fold cross-validation on the GPU box (~1 h): per fold, the whole-episode stage labeler and SARM2
# (stage + reward + eval) trained on the other folds' hand labels; then the held-out comparison report.
#   scripts/client_cv.sh [config]          -> reports/client/  (run in tmux)
cd ~/SARM-Data-Labeling
PY=.venv/bin/python
CFG=${1:-configs/client_nurvai2.yaml}
FOLDS=5
mkdir -p logs
grep -q "EXIT_CODE=0" logs/client_embed.log 2>/dev/null || \
  { $PY -m sarm_pipeline embed --config $CFG > logs/client_embed.log 2>&1; echo EXIT_CODE=$? >> logs/client_embed.log; }
$PY -m sarm_pipeline.kfold split --config $CFG --folds $FOLDS
for k in $(seq 0 $((FOLDS - 1))); do
  S="split.file=splits/client_nurvai2_fold$k.json train.gpu_mem_fraction=0.18"
  (
    $PY -m sarm_pipeline episode-stage --config $CFG --set run_name=client_es_f$k $S
    for step in train-stage train-reward eval; do
      $PY -m sarm_pipeline $step --config $CFG --set run_name=client_sarm_f$k $S || break
    done
  ) > logs/client_f$k.log 2>&1 &
done
wait
$PY -m sarm_pipeline.kfold collect --config $CFG --folds $FOLDS
echo CLIENT_CV_DONE
