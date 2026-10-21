"""Exploratory BBOB extension: fixed analysis (experiments/MPHBS_BBOB_ADDENDUM.md).

Same statistics as the main study (lakaie.analysis.pathA): per function paired Wilcoxon + Holm over the 24 functions,
paired win probability, W-bar with bootstrap CI, the same verdict rule, median relative difference (descriptive), and a
descriptive breakdown by BBOB function group. Errors below 1e-8 are set to 1e-8 before any statistic.

Usage:  python scripts/pathA_bbob_analysis.py [--raw-root results/raw] [--out results/analysis/pathA_bbob]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.pathA import contrast, integrity  # noqa: E402
from lakaie.analysis.stats import load_campaign  # noqa: E402
from lakaie.experiment import load_yaml  # noqa: E402

BENCH = ["BBOB-D10", "BBOB-D20"]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def missing_cells(df, pa, functions):
    out = []
    for bench in BENCH:
        d = df.get(bench)
        for arm in pa["arms"][bench]:
            for fid in functions:
                n = 0 if d is None else int(((d.method == arm) & (d.instance == fid) & (d.run_id <= pa["runs"])).sum())
                if n < pa["runs"]:
                    out.append(f"{bench} {arm} f{fid}: {n}/{pa['runs']}")
    return out


def rel_diff(d, a, b, runs):
    per = []
    for fid in sorted(d.instance.unique()):
        A = d[(d.method == a) & (d.instance == fid) & (d.run_id <= runs)].set_index("run_id").best_fitness
        B = d[(d.method == b) & (d.instance == fid) & (d.run_id <= runs)].set_index("run_id").best_fitness
        idx = A.index.intersection(B.index)
        x, y = A.loc[idx].to_numpy(float), B.loc[idx].to_numpy(float)
        per.append(float(np.median((y - x) / np.maximum(np.abs(y), 1e-300))))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis" / "pathA_bbob"))
    ap.add_argument("--functions", nargs="*", type=int, help="tests only: restrict the completeness check")
    args = ap.parse_args()
    pa = load_yaml("pathA_bbob.yaml")
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    functions = args.functions or list(range(1, 25))

    df, integ = {}, {}
    for bench in BENCH:
        d = load_campaign(Path(args.raw_root), pa["campaign"], bench)
        if d.empty:
            continue
        integ[bench] = integrity(d, {k: v["expected"][1] for k, v in pa["arms"][bench].items()})
        d = d.copy()
        d["best_fitness"] = np.maximum(pd.to_numeric(d.best_fitness), pa["error_floor"])
        df[bench] = d
    missing = missing_cells(df, pa, functions)
    summary = {"git_commit": git_hash(), "addendum": "experiments/MPHBS_BBOB_ADDENDUM.md",
               "status": "exploratory", "integrity": integ, "missing_cells": missing[:200]}
    if not integ or not all(v["ok"] for v in integ.values()) or missing:
        summary["result"] = "integrity failure, missing runs or no data: no verdicts computed"
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json), encoding="utf-8")
        print(summary["result"])
        return 1

    dec = pa["decision"]
    summary["contrasts"], summary["groups"] = {}, {}
    for cid, (a, b) in pa["contrasts"].items():
        for bench in BENCH:
            T, s = contrast(df[bench], a, b, pa["runs"], dec["alpha"], dec["equivalence_band"],
                            dec["bootstrap_resamples"], dec["bootstrap_seed"])
            per = rel_diff(df[bench], a, b, pa["runs"])
            T["rel_diff"] = per
            s["rel_median"] = float(np.median(per))
            s["n_solved_ties"] = int(T.ties.sum())
            T.to_csv(out / "tables" / f"{cid}_{bench}.csv", index=False)
            summary["contrasts"][f"{cid}/{bench}"] = s
            for g, fids in pa["groups"].items():
                G = T[T.instance.isin(fids)]
                summary["groups"][f"{cid}/{bench}/{g}"] = {
                    "mean_w": float(G.w.mean()),
                    "sig_anchor": int((G.significant_favours == "anchor").sum()),
                    "sig_comparator": int((G.significant_favours == "comparator").sum()),
                    "median_rel": float(np.median(G.rel_diff)), "n": int(len(G))}
    summary["result"] = "ok"
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json), encoding="utf-8")
    for k, s in summary["contrasts"].items():
        print(f"{k}: W={s['wbar']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] sig {s['n_sig_anchor']}/{s['n_sig_comparator']} "
              f"rel {s['rel_median']:.2e} -> {s['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
