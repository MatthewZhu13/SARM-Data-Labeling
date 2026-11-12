#!/usr/bin/env bash
# run_logged.sh CONFIG [STEP] [extra args...]: run a pipeline step, append output to logs/<run_name>.log,
# and finish the log with EXIT_CODE=<n> so monitors can tell success from failure.
cd "$(dirname "$0")/.."
cfg=${1:?config}; step=${2:-all}; shift 2 || shift $#
name=$(grep -E '^run_name:' "$cfg" | awk '{print $2}')
mkdir -p logs
.venv/bin/python -m sarm_pipeline "$step" --config "$cfg" "$@" 2>&1 | tee -a "logs/$name.log"
code=${PIPESTATUS[0]}
echo "EXIT_CODE=$code step=$step $(date -u +%FT%TZ)" | tee -a "logs/$name.log"
exit "$code"
