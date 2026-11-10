"""Post hoc diagnostic for the theory section of the Path A paper: is there a learnable signal in the MPHBS mediator,
and does the SARSA table capture it?

Runs MPHBS (family MPHBS-V with default flags, which is bit-identical to the frozen MPHBS port) with a probe on every
mediator episode, on CEC2022 D=20 F1-F12 and FIR 1-8, 5 runs each, at the Path A budgets. Seeds are disjoint from all
campaigns (master 97000000). Exploratory: no hypothesis was fixed in advance.

Per run it records
  rewards           fractions of r = +1, 0, -1
  action signal     mean reward of the candidates whose coordinate d came from cell (d, rank, source), per cell;
                    between-cell heterogeneity tau (method of moments, cells with >= 30 trials) and the split-half
                    Spearman correlation of these means (first vs second half of the run)
  table             predicted noise sd of a Q entry, alpha*sigma/sqrt(beta(2-beta)), the implied separation z of two
                    cells whose mean rewards differ by tau, and the Spearman correlation between Q at mid-run and the
                    second-half action means
  updates           number of cells updated and median updates per updated cell

Output: results/analysis/pathA_theory/learning_signal.csv and summary.json
Usage:  python scripts/pathA_learning_signal.py [--runs 5] [--workers 11]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.algorithms import mphbs_variants  # noqa: E402
from lakaie.algorithms.mphbs import ALPHA_LR  # noqa: E402
from lakaie.core import CountedObjective  # noqa: E402
from lakaie.experiment import load_yaml, make_problem  # noqa: E402
from lakaie.registry import iterations_for  # noqa: E402

OUT = ROOT / "results" / "analysis" / "pathA_theory"
BENCH = {"CEC2022": ("cec2022.yaml", 20, 200000, range(1, 13)), "FIR": ("fir.yaml", 31, 150000, range(1, 9))}
MASTER = 97000000


def one(task):
    bench, inst, run = task
    cfg_file, dim, max_fe, _ = BENCH[bench]
    b = load_yaml(cfg_file)
    cfg = dict(b["methods"]["MPHBS"])
    N = int(b["population_size"])
    prob = make_problem(bench, inst, dim)
    T = iterations_for("MPHBS-V", cfg, max_fe, prob.dim, N)[0]
    obj = CountedObjective(prob, max_fe)
    rng = np.random.default_rng(MASTER + (0 if bench == "CEC2022" else 100000) + 1000 * inst + run)
    D, nsub = prob.dim, None
    st = {"r": [0, 0, 0], "act_n": None, "act_s": None, "act_s2": None, "upd_n": None, "Qmid": None}

    def probe(t, TT, ep, reward, accepted, active, rows, ch, rows_next, ch_next, Q):
        nonlocal nsub
        if st["act_n"] is None:
            nsub = Q.shape[1]
            st["act_n"] = np.zeros((2,) + Q.shape)
            st["act_s"] = np.zeros((2,) + Q.shape)
            st["upd_n"] = np.zeros(Q.shape)
        h = 0 if t < TT // 2 else 1
        if h == 1 and st["Qmid"] is None:
            st["Qmid"] = Q.copy()
        st["r"][0 if reward > 0 else (1 if reward == 0 else 2)] += 1
        for d in active:
            st["act_n"][h, d, rows_next[d], ch_next[d]] += 1
            st["act_s"][h, d, rows_next[d], ch_next[d]] += reward
            st["upd_n"][d, rows[d], ch[d]] += 1

    out = mphbs_variants.run(obj, T, prob.lb, prob.ub, prob.dim, N, rng, cfg, None, probe=probe)
    Q = out.get("_Q")
    n, s = st["act_n"], st["act_s"]
    ntot, stot = n.sum(0), s.sum(0)
    r = np.array(st["r"], float)
    pr = r / r.sum()
    mu, var_r = pr[0] - pr[2], pr[0] + pr[2] - (pr[0] - pr[2]) ** 2
    gamma = float(cfg["gamma"])
    beta = ALPHA_LR * (1 - gamma)
    sd_pred = ALPHA_LR * np.sqrt(var_r) / np.sqrt(beta * (2 - beta))
    ok = ntot >= 30
    means = stot[ok] / ntot[ok]
    tau2 = float(np.var(means, ddof=1) - np.mean(var_r / ntot[ok])) if ok.sum() > 2 else np.nan
    tau = float(np.sqrt(max(tau2, 0.0)))
    both = (n[0] >= 15) & (n[1] >= 15)
    m0, m1 = s[0][both] / n[0][both], s[1][both] / n[1][both]
    rel = float(spearmanr(m0, m1).correlation) if both.sum() > 5 else np.nan
    qv = np.nan
    if st["Qmid"] is not None:
        sel = n[1] >= 15
        if sel.sum() > 5:
            qv = float(spearmanr(st["Qmid"][sel], s[1][sel] / n[1][sel]).correlation)
    upd = st["upd_n"]
    return {"bench": bench, "instance": inst, "run": run, "T": T, "episodes": int(r.sum()),
            "p_plus": pr[0], "p_zero": pr[1], "p_minus": pr[2], "reward_sd": float(np.sqrt(var_r)),
            "Q_sd_pred": float(sd_pred), "tau_action": tau, "z_typical": float((tau / (1 - gamma)) / (np.sqrt(2) * sd_pred)),
            "split_half_rho": rel, "Qmid_validity_rho": qv, "cells": int(upd.size),
            "cells_updated": int((upd > 0).sum()), "median_updates_per_updated_cell": float(np.median(upd[upd > 0])),
            "cells_tried_30plus": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(bn, i, r) for bn, (_, _, _, inst) in BENCH.items() for i in inst for r in range(1, a.runs + 1)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, tasks))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "learning_signal.csv", index=False)
    summ = {"design": "post hoc diagnostic, scripts/pathA_learning_signal.py", "runs_per_instance": a.runs,
            "alpha": ALPHA_LR, "by_bench": {}}
    for bn, g in T.groupby("bench"):
        summ["by_bench"][bn] = {k: float(g[k].median()) for k in
                                ["p_plus", "p_zero", "p_minus", "reward_sd", "Q_sd_pred", "tau_action", "z_typical",
                                 "split_half_rho", "Qmid_validity_rho", "median_updates_per_updated_cell"]}
        summ["by_bench"][bn]["n_runs"] = int(len(g))
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
