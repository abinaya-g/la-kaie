#!/usr/bin/env bash
# Common metaheuristic baselines for the eigen-frame campaign (addendum in experiments/ESWA_EIGEN_CAMPAIGN.md).
# Same benchmarks, budgets, seeds and output layout as eswa_campaign.sh. Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=results/eswa_eigen/logs
mkdir -p "$LOG"
for B in CEC2022-D10 BBOB-D20 CEC2022-D20 FIR; do
  "$PY" scripts/eswa_campaign.py --bench "$B" --arms HBA MPA PSO DE GWO WOA >> "$LOG/extra_$B.log" 2>&1
done
echo "extra baselines finished $(date)" >> "$LOG/campaign.log"
