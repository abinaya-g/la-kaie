"""Confirmatory analysis H3 of the transfer study (experiments/ESWA_EIGEN_CAMPAIGN.md, Section 10).

Written and committed before the campaign finished; it is not changed after results are seen.

H3: for each parent pair, the eigen-frame exchange outperforms its axis-frame twin on the 32 instances (BBOB D = 20
f1-f24, FIR cases 1-8): median Vargha-Delaney A of (eigen vs axis) over instances exceeds 0.5.
  PSO + DE : PD-EIG vs PD-R      GWO + WOA : GW-EIG vs GW-R
Per instance A = P(eigen < axis) + 0.5 P(tie) over all 30 x 30 run pairs (errors floored at 1e-8). Across instances:
one-sided Wilcoxon signed-rank test of (A - 0.5), scipy default zero handling (zeros dropped), Holm over the two pairs
at alpha 0.05. Pratt and zero-split handling are reported alongside.
Secondary (no claim): per-instance paired two-sided Wilcoxon with Holm within benchmark and pair; the rotation pairs
BBOB f10/f2 and f15/f3; exchange acceptance rate in the first quarter of the budget on f2, f10, f11, f14.

Usage:  python scripts/eswa_xhyb_analysis.py [--root results/eswa_eigen] [--out results/analysis/eswa_xhyb]
        (--runs N and --partial only for testing the script on incomplete data)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eswa_eigen_analysis import ALPHA, FLOOR, git_hash, paired_p, vd_a  # noqa: E402
from lakaie.analysis.stats import holm, load_campaign  # noqa: E402

BENCH = {"BBOB-D20": ("BBOB-D20", list(range(1, 25))), "FIR": ("FIR", list(range(1, 9)))}
PAIRS = {"PSO+DE": ("PD-EIG", "PD-R"), "GWO+WOA": ("GW-EIG", "GW-R")}
ROT = {"ellipsoid f10/f2": (10, 2), "rastrigin f15/f3": (15, 3)}
MECH = [2, 10, 11, 14]


def load(root, folder, bench, runs, strict=True):
    d = load_campaign(Path(root, folder), "eswa_eigen", bench)
    if d.empty:
        if strict:
            sys.exit(f"no results found for {folder} under {root}")
        return pd.DataFrame(columns=["method", "instance", "run_id", "err", "FE", "max_fes", "status"])
    d = d[(d.status == "ok") & (d.run_id <= runs)].copy()
    d["err"] = np.maximum(pd.to_numeric(d.best_fitness), FLOOR)
    return d


def safe_p(x, y, strict):
    try:
        return paired_p(x, y)
    except ValueError:
        if strict:
            raise
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "eswa_eigen"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis" / "eswa_xhyb"))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--partial", action="store_true", help="test mode: allow missing instances and fewer runs")
    a = ap.parse_args()
    strict = not a.partial
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = {f: load(a.root, f, b, a.runs, strict) for f, (b, _) in BENCH.items()}

    rows = []
    for pair, (eig, ax) in PAIRS.items():
        for f, (_, insts) in BENCH.items():
            d = D[f]
            for i in insts:
                x = d[(d.method == eig) & (d.instance == i)].sort_values("run_id")
                y = d[(d.method == ax) & (d.instance == i)].sort_values("run_id")
                if len(x) != a.runs or len(y) != a.runs or not np.array_equal(x.run_id.to_numpy(), y.run_id.to_numpy()):
                    if strict:
                        sys.exit(f"incomplete: {pair} {f} instance {i}: {len(x)} / {len(y)} runs")
                    continue
                assert (x.FE.to_numpy() <= x.max_fes.to_numpy()).all() and (y.FE.to_numpy() <= y.max_fes.to_numpy()).all()
                e, c = x.err.to_numpy(float), y.err.to_numpy(float)
                rows.append({"pair": pair, "bench": f, "instance": i, "A": vd_a(e, c),
                             "med_eig": float(np.median(e)), "med_axis": float(np.median(c)),
                             "p_inst": safe_p(e, c, strict), "fe_equal": bool(np.array_equal(x.FE.to_numpy(), y.FE.to_numpy()))})
    T = pd.DataFrame(rows)
    T["p_inst_holm"] = np.nan
    for _, g in T.groupby(["pair", "bench"]):
        T.loc[g.index, "p_inst_holm"] = holm(g.p_inst.fillna(1.0).to_numpy())
    T["sig"] = np.where(T.p_inst_holm < ALPHA, np.where(T.A > 0.5, "eigen", "axis"), "")
    T.to_csv(out / "H3_per_instance.csv", index=False)

    res, ps = {}, []
    for pair in PAIRS:
        g = T[T.pair == pair]
        dA = g.A.to_numpy() - 0.5
        p = float(wilcoxon(dA, alternative="greater").pvalue) if (dA != 0).any() else 1.0
        ps.append(p)
        res[pair] = {"n": int(len(g)), "median_A": float(np.median(g.A)), "mean_A": float(np.mean(g.A)),
                     "n_A_gt_half": int((dA > 0).sum()), "n_A_lt_half": int((dA < 0).sum()),
                     "n_A_eq_half": int((dA == 0).sum()), "p_one_sided": p,
                     "p_pratt": float(wilcoxon(dA, alternative="greater", zero_method="pratt").pvalue) if (dA != 0).any() else 1.0,
                     "p_zsplit": float(wilcoxon(dA, alternative="greater", zero_method="zsplit").pvalue) if (dA != 0).any() else 1.0,
                     "inst_sig_eigen": int((g.sig == "eigen").sum()), "inst_sig_axis": int((g.sig == "axis").sum()),
                     "fe_equal_all": bool(g.fe_equal.all()),
                     "by_bench": {b: {"median_A": float(np.median(gb.A)), "n": int(len(gb)),
                                      "sig_eigen": int((gb.sig == "eigen").sum()), "sig_axis": int((gb.sig == "axis").sum())}
                                  for b, gb in g.groupby("bench")}}
    for pair, ph in zip(PAIRS, holm(np.array(ps))):
        res[pair]["p_holm"] = float(ph)
        res[pair]["supported"] = bool(ph < ALPHA and res[pair]["median_A"] > 0.5)

    rot = {}
    d = D["BBOB-D20"]
    for name, (r, s) in ROT.items():
        rot[name] = {arm: {"median_rotated": float(d[(d.method == arm) & (d.instance == r)].err.median()),
                           "median_separable": float(d[(d.method == arm) & (d.instance == s)].err.median())}
                     for arm in ("PD-R", "PD-EIG", "GW-R", "GW-EIG")}

    acc = {}
    for arm in ("PD-R", "PD-EIG", "GW-R", "GW-EIG"):
        for i in MECH:
            vals = []
            for jp in sorted(Path(a.root, "BBOB-D20", "eswa_eigen", "BBOB-D20", arm, f"inst{i:02d}").glob("run*.npz"))[:a.runs]:
                z = np.load(jp)
                m = z["iter_fe"] <= 50000
                if m.any():
                    vals.append(float(np.nanmean(z["iter_mediator_accept_rate"][m])))
            if vals:
                acc[f"{arm}/f{i}"] = float(np.median(vals))

    summ = {"git_commit": git_hash(), "design": "experiments/ESWA_EIGEN_CAMPAIGN.md Section 10",
            "status": "confirmatory (H3), rotation and mechanism secondary", "H3": res, "rotation_pairs": rot,
            "acceptance_first_quarter_median": acc}
    (out / "H3_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ["H3"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
