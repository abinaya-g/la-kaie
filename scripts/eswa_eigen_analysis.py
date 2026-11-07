"""Confirmatory analysis of the eigen-frame campaign (experiments/ESWA_EIGEN_CAMPAIGN.md, Section 4).

H1: EIG50 vs RND, H2: EIG50 vs MPHBS, on the 28 held-out instances (CEC2022 D10 F1-F12, BBOB-D20 held-out f).
Per instance: Vargha-Delaney A = P(EIG50 < X) + 0.5 P(tie) over all 30 x 30 run pairs (errors floored at 1e-8).
Across instances: one-sided Wilcoxon signed-rank of (A - 0.5), Holm over {H1, H2} at alpha 0.05.
Secondary (no claim): per-instance two-sided paired Wilcoxon, Holm within each benchmark; paired win probability.

Usage:  python scripts/eswa_eigen_analysis.py [--root results/eswa_eigen] [--out results/analysis/eswa_eigen]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.stats import holm, load_campaign  # noqa: E402

HELD_OUT = {   # campaign folder: (benchmark name in the result files, held-out instances)
    "CEC2022-D10": ("CEC2022", list(range(1, 13))),
    "BBOB-D20": ("BBOB-D20", [2, 4, 5, 6, 7, 9, 11, 13, 14, 17, 18, 19, 20, 22, 23, 24]),
}
HYP = {"H1": ("EIG50", "RND"), "H2": ("EIG50", "MPHBS")}
RUNS, FLOOR, ALPHA = 30, 1e-8, 0.05


def vd_a(x, y):
    """P(x < y) + 0.5 P(x == y) over all pairs (lower error is better)."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(np.mean((x < y) + 0.5 * (x == y)))


def paired_p(x, y):
    d = x - y
    return 1.0 if np.all(d == 0) else float(wilcoxon(x, y, zero_method="zsplit").pvalue)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "eswa_eigen"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis" / "eswa_eigen"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for folder, (bench, insts) in HELD_OUT.items():
        d = load_campaign(Path(a.root, folder), "eswa_eigen", bench)
        d = d[(d.status == "ok") & (d.run_id <= RUNS) & d.instance.isin(insts)].copy()
        d["err"] = np.maximum(pd.to_numeric(d.best_fitness), FLOOR)
        for hid, (m, c) in HYP.items():
            for f in insts:
                X = d[(d.method == m) & (d.instance == f)].set_index("run_id").err.sort_index()
                Y = d[(d.method == c) & (d.instance == f)].set_index("run_id").err.sort_index()
                if len(X) < RUNS or len(Y) < RUNS or not X.index.equals(Y.index):
                    missing.append(f"{folder} {hid} {f}: {len(X)}/{len(Y)}")
                    continue
                cell = d[d.method.isin([m, c]) & (d.instance == f)]
                if (cell.FE > cell.max_fes).any():
                    missing.append(f"{folder} {hid} {f}: FE over budget")
                x, y = X.to_numpy(float), Y.to_numpy(float)
                rows.append({"hyp": hid, "bench": folder, "instance": f, "A": vd_a(x, y),
                             "paired_win": float(np.mean((x < y) + 0.5 * (x == y))),
                             "med_eig50": float(np.median(x)), "med_comp": float(np.median(y)),
                             "p_inst": paired_p(x, y)})
    if missing:
        print("incomplete data, no verdicts:", *missing[:20], sep="\n  ")
        return 1

    T = pd.DataFrame(rows)
    T["p_inst_holm"] = np.nan
    for (h, b), g in T.groupby(["hyp", "bench"]):
        T.loc[g.index, "p_inst_holm"] = holm(g.p_inst.to_numpy())
    T["sig"] = np.where(T.p_inst_holm < ALPHA, np.where(T.A > 0.5, "EIG50", "comparator"), "")
    T.to_csv(out / "held_out_per_instance.csv", index=False)

    summ = {"git_commit": git_hash(), "design": "experiments/ESWA_EIGEN_CAMPAIGN.md", "n_instances": {}}
    p = {}
    for hid in HYP:
        g = T[T.hyp == hid]
        dA = g.A.to_numpy() - 0.5
        p[hid] = float(wilcoxon(dA, alternative="greater").pvalue) if np.any(dA != 0) else 1.0
        summ[hid] = {"comparator": HYP[hid][1], "n": int(len(g)), "median_A": float(np.median(g.A)),
                     "mean_A": float(g.A.mean()), "n_A_gt_half": int((g.A > 0.5).sum()),
                     "n_A_lt_half": int((g.A < 0.5).sum()), "n_A_eq_half": int((g.A == 0.5).sum()),
                     "p_one_sided": p[hid],
                     "inst_sig_EIG50": int((g.sig == "EIG50").sum()),
                     "inst_sig_comparator": int((g.sig == "comparator").sum()),
                     "by_bench_median_A": {b: float(np.median(gb.A)) for b, gb in g.groupby("bench")}}
    adj = holm([p[h] for h in HYP])
    for h, ph in zip(HYP, adj):
        summ[h]["p_holm"] = float(ph)
        summ[h]["supported"] = bool(ph < ALPHA and summ[h]["median_A"] > 0.5)
    (out / "confirmatory_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    for h in HYP:
        s = summ[h]
        print(f"{h} EIG50 vs {s['comparator']}: median A {s['median_A']:.3f} (A>0.5 on {s['n_A_gt_half']}, <0.5 on "
              f"{s['n_A_lt_half']}, =0.5 on {s['n_A_eq_half']} of {s['n']}), p={s['p_one_sided']:.4g}, "
              f"Holm p={s['p_holm']:.4g} -> {'SUPPORTED' if s['supported'] else 'not supported'}; "
              f"per-instance sig {s['inst_sig_EIG50']} for / {s['inst_sig_comparator']} against; "
              f"by bench {s['by_bench_median_A']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
