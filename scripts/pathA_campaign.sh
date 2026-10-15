#!/usr/bin/env bash
# Path A campaign (experiments/MPHBS_DECONSTRUCTION_PREREG.md §4). Run only after the
# pre-registration is frozen (§12). Linux only (the runner uses fork). Resumable:
# re-running a command skips finished runs and never overwrites them.
# The campaign is normally run by .github/workflows/pathA-campaign.yml (same commands,
# split into parallel jobs); this script is the single-machine equivalent.
set -euo pipefail
cd "$(dirname "$0")/.."

# 0. integrity gate V1 (full suite) before any run
python3 -m pytest -q

run() { python3 scripts/run_experiment.py --campaign pathA "$@"; }

# 1. H1 pair: 100 runs per instance
run --benchmark CEC2022 --methods PA-SARSA PA-RND-FAIR --runs 100
run --benchmark FIR     --methods PA-SARSA PA-RND-FAIR --runs 100

# 2. other primary arms: 50 runs per instance
run --benchmark CEC2022 --methods PA-RND-SYNC PA-RND-AUTH PA-SKEL PA-MPHB --runs 50
run --benchmark FIR     --methods PA-RND-AUTH PA-SKEL PA-MPHB --runs 50

# 3. H5: CEC F10 to 100 seeds for RND-AUTH and SKEL (runs 1-50 are reused)
run --benchmark CEC2022 --methods PA-RND-AUTH PA-SKEL --instances 10 --runs 100

# 4. secondary arms: 50 runs per instance
run --benchmark CEC2022 --runs 50 --methods PA-SARSA-Ni1 PA-SARSA-Ni9 PA-RND-FAIR-Ni1 PA-RND-FAIR-Ni9 \
    PA-RND-K1 PA-RND-GreedyAcc PA-SARSA-SYNC
run --benchmark FIR --runs 50 --methods PA-SARSA-Ni1 PA-SARSA-Ni3 PA-RND-FAIR-Ni1 PA-RND-FAIR-Ni3 \
    PA-RND-K1 PA-RND-GreedyAcc

# 5. frozen analysis
python3 scripts/pathA_analysis.py
