#!/usr/bin/env bash
# The eigen-frame exchange between other parent pairs (addendum, Section 10 of experiments/ESWA_EIGEN_CAMPAIGN.md).
# Same benchmarks, budgets, seeds and output layout as eswa_campaign.sh. Resumable: finished runs are skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=results/eswa_eigen/logs
mkdir -p "$LOG"
for B in BBOB-D20 FIR; do   # CEC2022 dropped, see Section 10 (2026-11-23)
  "$PY" scripts/eswa_campaign.py --bench "$B" --arms PD-R PD-EIG GW-R GW-EIG >> "$LOG/xhyb_$B.log" 2>&1
done
echo "xhyb finished $(date)" >> "$LOG/campaign.log"
