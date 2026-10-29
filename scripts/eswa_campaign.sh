#!/usr/bin/env bash
# Full eigen-frame campaign (experiments/ESWA_EIGEN_CAMPAIGN.md). Resumable: finished runs are skipped.
# Order: held-out confirmatory benchmarks first, then the rest, then S4 at the official CEC2022 D20 budget.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=results/eswa_eigen/logs
mkdir -p "$LOG"
for B in CEC2022-D10 BBOB-D20 CEC2022-D20 FIR; do
  "$PY" scripts/eswa_campaign.py --bench "$B" >> "$LOG/$B.log" 2>&1
done
"$PY" scripts/eswa_campaign.py --bench CEC2022-D20 --arms EIG50 --max-fes 1000000 \
  --out results/eswa_eigen_official >> "$LOG/S4_CEC2022-D20_1M.log" 2>&1
echo "campaign finished $(date)" >> "$LOG/campaign.log"
