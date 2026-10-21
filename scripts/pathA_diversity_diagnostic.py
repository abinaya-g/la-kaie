"""Path A exploratory diagnostic: does the MPA memory refresh reduce diversity?  (POST HOC, not pre-registered)

Motivation: H2b and H8 show that refreshing the MPA memory after Phase 2 (RND-SYNC, SARSA-SYNC) is harmful
on most CEC2022 functions. Proposed mechanism: with the refreshed memory, MPA agents whose Phase-1 move is
worse than their memory revert to post-mediator positions near the incumbent, so the population collapses
faster; with MPHBS's stale memory they revert to pre-mediator positions and keep more spread.

Definition (fixed and committed before computing on the full data; only one run, CEC2022 F5 run 1, was opened
beforehand to learn the trace format):
  pairs     (RND-FAIR, RND-SYNC) and (SARSA, SARSA-SYNC); CEC2022 F1-F12; seed-paired runs 1-50
  per run   D_mean  = mean over iterations of log10(pooled diversity)        (primary diversity summary)
            D_q     = log10(pooled diversity) at 25/50/75 % of the iterations
            A_mpa   = mean over iterations of the MPA move-acceptance rate (iter_mpa_success);
                      1 - A_mpa is the share of Phase-1 MPA moves reverted to memory
  per fn    two-sided paired Wilcoxon (stats.wilcoxon_p), stale minus refreshed; Holm over the 12 functions
            within each (pair, measure) family; share of seeds where the stale-memory arm is larger
  link      Spearman rank correlation over the 12 functions between the median paired difference in
            D_mean and the performance effect w of the matching contrast (H2b for RND, H8 for SARSA,
            oriented as P(stale better))
  figure    median log10 diversity per iteration over the 50 runs, RND pair, 12 functions
Outputs: results/analysis/pathA_exploratory/diversity/{summary.json, per_function.csv}, paper/figures/fig_diversity.pdf

Usage:  python scripts/pathA_diversity_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import holm, wilcoxon_p  # noqa: E402

RAW = ROOT / "results" / "raw" / "pathA" / "CEC2022"
OUT = ROOT / "results" / "analysis" / "pathA_exploratory" / "diversity"
FIG = ROOT / "paper" / "figures"
AN = ROOT / "results" / "analysis" / "pathA" / "tables"
PAIRS = {"RND": ("PA-RND-FAIR", "PA-RND-SYNC", "primary_H2b_CEC2022.csv", +1),
         # H8 table has SARSA-SYNC as anchor, so its w is P(refreshed better): flip it
         "SARSA": ("PA-SARSA", "PA-SARSA-SYNC", "secondary_H8_CEC2022.csv", -1)}
RUNS = range(1, 51)
QS = (0.25, 0.50, 0.75)
BLUE, ORANGE, INK, INK2, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#9a9893"


def run_measures(arm, inst, run):
    z = np.load(RAW / arm / f"inst{inst:02d}" / f"run{run:02d}.npz")
    d = np.log10(np.maximum(z["iter_diversity"], 1e-300))
    T = d.size
    m = {"D_mean": float(d.mean()), "A_mpa": float(z["iter_mpa_success"].mean())}
    for q in QS:
        m[f"D_{int(q * 100)}"] = float(d[max(0, round(q * T) - 1)])
    return m, d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, traj = [], {}
    for pname, (stale, fresh, table, sign) in PAIRS.items():
        perf = pd.read_csv(AN / table).set_index("instance")["w"]
        for inst in range(1, 13):
            vals = {stale: [], fresh: []}
            for arm in (stale, fresh):
                ds = []
                for r in RUNS:
                    m, d = run_measures(arm, inst, r)
                    vals[arm].append(m)
                    ds.append(d)
                if pname == "RND":
                    traj[(arm, inst)] = np.median(np.vstack(ds), axis=0)
            S, F = pd.DataFrame(vals[stale]), pd.DataFrame(vals[fresh])
            row = {"pair": pname, "instance": inst,
                   "perf_w_stale_better": float(perf[inst] if sign > 0 else 1 - perf[inst])}
            for k in S.columns:
                diff = S[k].to_numpy() - F[k].to_numpy()
                row[f"{k}_stale_median"] = float(S[k].median())
                row[f"{k}_fresh_median"] = float(F[k].median())
                row[f"{k}_median_diff"] = float(np.median(diff))
                row[f"{k}_share_stale_larger"] = float(np.mean(diff > 0))
                row[f"{k}_p"] = wilcoxon_p(S[k].to_numpy(), F[k].to_numpy())
            rows.append(row)
    T = pd.DataFrame(rows)
    measures = ["D_mean", "D_25", "D_50", "D_75", "A_mpa"]
    for pname in PAIRS:
        sel = T.pair == pname
        for k in measures:
            T.loc[sel, f"{k}_p_holm"] = holm(T.loc[sel, f"{k}_p"].to_numpy())
    T.to_csv(OUT / "per_function.csv", index=False)

    summary = {"status": "exploratory, post hoc (not in the pre-registration)",
               "definition": "see scripts/pathA_diversity_diagnostic.py docstring", "pairs": {}}
    for pname in PAIRS:
        t = T[T.pair == pname]
        rho, p_rho = stats.spearmanr(t["D_mean_median_diff"], t["perf_w_stale_better"])
        summary["pairs"][pname] = {
            "stale_more_diverse_sig": int(((t.D_mean_p_holm < 0.05) & (t.D_mean_median_diff > 0)).sum()),
            "stale_less_diverse_sig": int(((t.D_mean_p_holm < 0.05) & (t.D_mean_median_diff < 0)).sum()),
            "stale_higher_acceptance_sig": int(((t.A_mpa_p_holm < 0.05) & (t.A_mpa_median_diff > 0)).sum()),
            "stale_lower_acceptance_sig": int(((t.A_mpa_p_holm < 0.05) & (t.A_mpa_median_diff < 0)).sum()),
            "median_A_mpa_stale": float(t.A_mpa_stale_median.median()),
            "median_A_mpa_fresh": float(t.A_mpa_fresh_median.median()),
            "median_D_mean_diff": float(t.D_mean_median_diff.median()),
            "spearman_Dmean_vs_perf": {"rho": float(rho), "p": float(p_rho), "n": int(len(t))},
            "by_quantile_stale_more_diverse_sig": {
                q: int(((t[f"{q}_p_holm"] < 0.05) & (t[f"{q}_median_diff"] > 0)).sum()) for q in ("D_25", "D_50", "D_75")},
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure(traj, T)
    print(json.dumps(summary["pairs"], indent=1))


def figure(traj, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 4.6), sharex=True)
    t = T[T.pair == "RND"].set_index("instance")
    for inst, ax in zip(range(1, 13), axes.ravel()):
        for arm, col in (("PA-RND-FAIR", BLUE), ("PA-RND-SYNC", ORANGE)):
            y = traj[(arm, inst)]
            ax.plot(np.linspace(0, 1, y.size), y, color=col, lw=1.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        star = " *" if t.loc[inst, "D_mean_p_holm"] < 0.05 else ""
        ax.set_title(f"F{inst}{star}", fontsize=7, loc="left", color=INK)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("fraction of iterations")
    for ax in axes[:, 0]:
        ax.set_ylabel("log$_{10}$ diversity")
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=1.2, label="RND-FAIR (stale memory, as MPHBS)"),
                        Line2D([], [], color=ORANGE, lw=1.2, label="RND-SYNC (memory refreshed)")],
               loc="lower center", ncol=2, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(FIG / "fig_diversity.pdf")
    fig.savefig(FIG / "fig_diversity.png", dpi=200)


if __name__ == "__main__":
    main()
