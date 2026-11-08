"""Secondary analyses of the eigen-frame campaign (experiments/ESWA_EIGEN_CAMPAIGN.md, Section 5). No claim rests on them.

S1: H1/H2 statistics on all 56 instances (piloted instances labelled).
S2: rotation mechanism, ratio of median errors rotated/separable for BBOB f10/f2 and f15/f3, per arm.
S3: Friedman ranks of the five arms per benchmark and overall (per-instance median errors), plus median tables.
S3x: the same with the six common baselines of the Section 9 addendum (11 arms), plus EIG50-vs-each win/tie/loss.
S4: EIG50 vs the official EA4eig CEC2022 runs (data/external/cec2022_ea4eig) at the official budgets, unpaired.
S5: cost, runtime per run and FE used, EIG50 vs RND.

Usage:  python scripts/eswa_eigen_secondary.py [--root results/eswa_eigen] [--official results/eswa_eigen_official]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eswa_eigen_analysis import FLOOR, HELD_OUT, HYP, RUNS, git_hash, paired_p, vd_a  # noqa: E402
from lakaie.analysis.stats import holm, load_campaign  # noqa: E402

BENCHES = {"CEC2022-D10": "CEC2022", "CEC2022-D20": "CEC2022", "BBOB-D20": "BBOB-D20", "FIR": "FIR"}
ARMS = ["MPHBS", "RND", "EIG50", "CMA-ES", "L-SHADE"]
EXTRA = ["HBA", "MPA", "PSO", "DE", "GWO", "WOA"]      # Section 9 addendum, descriptive only
ALL_ARMS = ARMS + EXTRA
PILOTED = {"CEC2022-D20": set(range(1, 13)), "FIR": set(range(1, 9)), "BBOB-D20": {1, 3, 8, 10, 12, 15, 16, 21}}
ROT_PAIRS = {"ellipsoid f10/f2": (10, 2), "rastrigin f15/f3": (15, 3)}


def load(root, folder, bench):
    d = load_campaign(Path(root, folder), "eswa_eigen", bench)
    d = d[(d.status == "ok") & (d.run_id <= RUNS)].copy()
    d["err"] = np.maximum(pd.to_numeric(d.best_fitness), FLOOR)
    d["bench"] = folder
    return d


def s1(D):
    rows = []
    for folder, d in D.items():
        for hid, (m, c) in HYP.items():
            for f in sorted(d.instance.unique()):
                x = d[(d.method == m) & (d.instance == f)].sort_values("run_id").err.to_numpy(float)
                y = d[(d.method == c) & (d.instance == f)].sort_values("run_id").err.to_numpy(float)
                assert len(x) == len(y) == RUNS, (folder, hid, f)
                held = folder in HELD_OUT and f in HELD_OUT[folder][1]
                rows.append({"hyp": hid, "bench": folder, "instance": int(f), "A": vd_a(x, y),
                             "set": "held-out" if held else ("piloted" if f in PILOTED.get(folder, ()) else "other"),
                             "med_eig50": float(np.median(x)), "med_comp": float(np.median(y)),
                             "p_inst": paired_p(x, y)})
    T = pd.DataFrame(rows)
    T["p_inst_holm"] = np.nan
    for _, g in T.groupby(["hyp", "bench"]):
        T.loc[g.index, "p_inst_holm"] = holm(g.p_inst.to_numpy())
    T["sig"] = np.where(T.p_inst_holm < 0.05, np.where(T.A > 0.5, "EIG50", "comparator"), "")
    out = {}
    for hid in HYP:
        g = T[T.hyp == hid]
        dA = g.A.to_numpy() - 0.5
        out[hid] = {"n": int(len(g)), "median_A": float(np.median(g.A)),
                    "n_gt": int((dA > 0).sum()), "n_lt": int((dA < 0).sum()), "n_eq": int((dA == 0).sum()),
                    "p_wilcox": float(wilcoxon(dA, alternative="greater").pvalue),
                    "p_pratt": float(wilcoxon(dA, alternative="greater", zero_method="pratt").pvalue),
                    "sig_for": int((g.sig == "EIG50").sum()), "sig_against": int((g.sig == "comparator").sum()),
                    "by_bench": {b: {"median_A": float(np.median(gb.A)), "sig_for": int((gb.sig == "EIG50").sum()),
                                     "sig_against": int((gb.sig == "comparator").sum()), "n": int(len(gb))}
                                 for b, gb in g.groupby("bench")}}
    return T, out


def s2(d):
    out = {}
    for name, (rot, sep) in ROT_PAIRS.items():
        out[name] = {}
        for arm in ARMS:
            mr = float(d[(d.method == arm) & (d.instance == rot)].err.median())
            ms = float(d[(d.method == arm) & (d.instance == sep)].err.median())
            out[name][arm] = {"median_rotated": mr, "median_separable": ms, "ratio": mr / ms}
    return out


def s3(D, arms=ARMS):
    med, ranks, out = [], [], {}
    for folder, d in D.items():
        cnt = d[d.method.isin(arms)].groupby(["instance", "method"]).size().unstack()[arms]
        assert (cnt == RUNS).all().all(), (folder, cnt[cnt != RUNS].stack())
        M = d.groupby(["instance", "method"]).err.median().unstack()[arms]
        M.insert(0, "bench", folder)
        med.append(M.reset_index())
        R = M[arms].rank(axis=1)
        ranks.append(R)
        out[folder] = {"mean_rank": R.mean().round(3).to_dict(),
                       "friedman_p": float(friedmanchisquare(*[M[a] for a in arms]).pvalue), "n": int(len(M)),
                       "eig50_better_median": {a: int((M["EIG50"] < M[a]).sum()) for a in arms if a != "EIG50"},
                       "eig50_worse_median": {a: int((M["EIG50"] > M[a]).sum()) for a in arms if a != "EIG50"}}
    R = pd.concat(ranks)
    out["all"] = {"mean_rank": R.mean().round(3).sort_values().to_dict(), "n": int(len(R))}
    return pd.concat(med, ignore_index=True), out


def s3_pairwise(D, arms):
    """EIG50 vs each other arm: per-instance paired Wilcoxon (Holm within benchmark x comparator), win/tie/loss."""
    rows = []
    for folder, d in D.items():
        for c in arms:
            if c == "EIG50":
                continue
            ps, recs = [], []
            for f in sorted(d.instance.unique()):
                x = d[(d.method == "EIG50") & (d.instance == f)].sort_values("run_id").err.to_numpy(float)
                y = d[(d.method == c) & (d.instance == f)].sort_values("run_id").err.to_numpy(float)
                recs.append((int(f), vd_a(x, y)))
                ps.append(paired_p(x, y))
            for (f, A), p, ph in zip(recs, ps, holm(np.array(ps))):
                rows.append({"bench": folder, "comparator": c, "instance": f, "A": A, "p": p, "p_holm": ph,
                             "result": ("+" if A > 0.5 else "-") if ph < 0.05 else "="})
    T = pd.DataFrame(rows)
    wtl = T.groupby(["comparator", "bench"]).result.value_counts().unstack(fill_value=0)
    return T, wtl


def ea4eig(D, f):
    """Final errors of the 30 official EA4eig runs (row 16 of the entrant's file), floored like ours."""
    A = np.loadtxt(ROOT / "data" / "external" / "cec2022_ea4eig" / f"EA4eigN100_10_{f}_{D}.txt")
    return np.maximum(A[15], FLOOR)


def s4(d10, official):
    rows = []
    for D, d in ((10, d10[d10.method == "EIG50"]), (20, official)):
        for f, g in d.groupby("instance"):
            e, x = g.err.to_numpy(float), ea4eig(D, int(f))
            rows.append({"D": D, "instance": int(f), "max_fes": int(g.max_fes.iloc[0]), "n": len(e),
                         "mean": e.mean(), "median": np.median(e), "std": e.std(ddof=1),
                         "ea4eig_mean": x.mean(), "ea4eig_median": np.median(x), "ea4eig_std": x.std(ddof=1),
                         "A_vs_ea4eig": vd_a(e, x),
                         "p_mwu": float(mannwhitneyu(e, x).pvalue) if not np.all(e == x[0]) or not np.all(x == x[0]) else 1.0})
    T = pd.DataFrame(rows)
    T["p_mwu_holm"] = np.nan
    for _, g in T.groupby("D"):
        T.loc[g.index, "p_mwu_holm"] = holm(g.p_mwu.to_numpy())
    T["sig"] = np.where(T.p_mwu_holm < 0.05, np.where(T.A_vs_ea4eig > 0.5, "EIG50", "EA4eig"), "")
    return T


def s5(D):
    out = {}
    for folder, d in D.items():
        r = {}
        for arm in ARMS:
            g = d[d.method == arm]
            r[arm] = {"median_runtime_s": float(g.runtime_sec.median()), "median_FE": float(g.FE.median()),
                      "max_FE": int(g.FE.max()), "max_fes": int(g.max_fes.max())}
        x = d[d.method == "EIG50"].sort_values(["instance", "run_id"])
        y = d[d.method == "RND"].sort_values(["instance", "run_id"])
        r["EIG50_vs_RND_runtime_ratio_median"] = float(np.median(x.runtime_sec.to_numpy() / y.runtime_sec.to_numpy()))
        r["EIG50_vs_RND_FE_equal_share"] = float(np.mean(x.FE.to_numpy() == y.FE.to_numpy()))
        out[folder] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "eswa_eigen"))
    ap.add_argument("--official", default=str(ROOT / "results" / "eswa_eigen_official"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis" / "eswa_eigen"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = {k: load(a.root, k, b) for k, b in BENCHES.items()}
    off = load(a.official, "CEC2022-D20", "CEC2022")

    T1, r1 = s1(D)
    T1.to_csv(out / "S1_all_instances.csv", index=False)
    M3, r3 = s3(D)
    M3.to_csv(out / "S3_median_errors.csv", index=False)
    M3x, r3x = s3(D, ALL_ARMS)
    M3x.to_csv(out / "S3x_median_errors_11arms.csv", index=False)
    P3x, wtl = s3_pairwise(D, ALL_ARMS)
    P3x.to_csv(out / "S3x_eig50_pairwise.csv", index=False)
    wtl.to_csv(out / "S3x_eig50_win_tie_loss.csv")
    T4 = s4(D["CEC2022-D10"], off)
    T4.to_csv(out / "S4_eig50_official_budget.csv", index=False)
    summ = {"git_commit": git_hash(), "design": "experiments/ESWA_EIGEN_CAMPAIGN.md", "status": "secondary",
            "S1": r1, "S2": s2(D["BBOB-D20"]), "S3": r3, "S3x_11arms": r3x, "S5": s5(D)}
    (out / "secondary_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ["S1", "S2", "S3", "S3x_11arms", "S5"]}, indent=1))
    print(wtl.to_string())
    print(T4.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
