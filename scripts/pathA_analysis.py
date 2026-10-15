"""Path A pre-registered analysis (experiments/MPHBS_DECONSTRUCTION_PREREG.md §5, §6).

Frozen before the campaign: run unchanged on the finished `pathA` raw results.
Nothing here is tuned to the data; every number comes from configs/pathA.yaml.

Usage:  python scripts/pathA_analysis.py [--raw-root results/raw] [--out results/analysis/pathA]
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
from lakaie.analysis.pathA import authors_mirror_check, contrast, f10_escape, integrity  # noqa: E402
from lakaie.analysis.stats import friedman_block_analysis, load_campaign  # noqa: E402
from lakaie.experiment import load_yaml  # noqa: E402

REF = ROOT / "third_party" / "mphbs_reference" / "results"
BENCH = {"CEC2022": "cec2022.yaml", "FIR": "fir.yaml"}
PRIMARY_ARMS = ["PA-SARSA", "PA-RND-FAIR", "PA-RND-SYNC", "PA-RND-AUTH", "PA-SKEL", "PA-MPHB"]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(type(o))


def run_family(df, pa, level, out_tables):
    dec = pa["decision"]
    results = {}
    for hid, spec in pa["contrasts"][level].items():
        for bench in ("CEC2022", "FIR"):
            if bench not in spec:
                continue
            a, b = spec[bench]
            d = df.get(bench)
            if d is None or d.empty or not {a, b} <= set(d.method):
                results[f"{hid}/{bench}"] = {"anchor": a, "comparator": b, "verdict": "no data"}
                continue
            T, s = contrast(d, a, b, spec["runs"], dec["alpha"], dec["equivalence_band"],
                            dec["bootstrap_resamples"], dec["bootstrap_seed"])
            s["level"] = level
            results[f"{hid}/{bench}"] = s
            T.to_csv(out_tables / f"{level}_{hid}_{bench}.csv", index=False)
    return results


def v3(df, pa):
    spec = pa["v3_authors_mirror"]
    out = {}
    for bench in ("CEC2022", "FIR"):
        d = df.get(bench)
        if d is None or d.empty:
            out[bench] = {"status": "no data"}
            continue
        ours = d[(d.method == spec["arm"]) & (d.run_id <= spec["runs"])]
        ours = ours.rename(columns={"best_fitness": "value"})[["instance", "value"]]
        b = spec[bench]
        ref = pd.read_csv(REF / b["file"])
        ref = ref[ref.Alias == b["alias"]].rename(columns={b["instance_col"]: "instance",
                                                           b["value_col"]: "value"})
        T = authors_mirror_check(ours, ref[["instance", "value"]])
        out[bench] = {"n_instances": int(len(T)), "n_differ": int(T.differs.sum()) if len(T) else 0,
                      "table": T.to_dict(orient="records")}
    return out


def completeness(df, pa, instances):
    """Every primary contrast and H5 needs all its paired runs on every instance (prereg §4)."""
    need = []
    for spec in pa["contrasts"]["primary"].values():
        for bench in ("CEC2022", "FIR"):
            if bench in spec:
                need += [(bench, arm, inst, spec["runs"]) for arm in spec[bench] for inst in instances[bench]]
    h5 = pa["h5_f10_escape"]
    if h5["instance"] in instances["CEC2022"]:
        need += [("CEC2022", arm, h5["instance"], h5["runs"]) for arm in h5["arms"]]
    missing = set()
    for bench, arm, inst, runs in need:
        d = df.get(bench)
        n = 0 if d is None else int(((d.method == arm) & (d.instance == inst) & (d.run_id <= runs)).sum())
        if n < runs:
            missing.add(f"{bench} {arm} inst{inst:02d}: {n}/{runs}")
    return sorted(missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "results" / "analysis" / "pathA"))
    ap.add_argument("--instances-from-data", action="store_true",
                    help="tests only: check completeness on the instances present instead of the benchmark's")
    args = ap.parse_args()
    pa = load_yaml("pathA.yaml")
    out = Path(args.out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    df, integ = {}, {}
    for bench in ("CEC2022", "FIR"):
        d = load_campaign(Path(args.raw_root), pa["campaign"], bench)
        if d.empty:
            continue
        df[bench] = d
        expected = {k: v["expected"][1] for k, v in pa["arms"][bench].items()}
        integ[bench] = integrity(d, expected)

    if args.instances_from_data:
        instances = {b: sorted(int(i) for i in df[b].instance.unique()) if b in df else [] for b in BENCH}
    else:
        instances = {b: load_yaml(f)["instances"] for b, f in BENCH.items()}
    missing = completeness(df, pa, instances)

    summary = {"git_commit": git_hash(), "prereg": "experiments/MPHBS_DECONSTRUCTION_PREREG.md",
               "integrity": integ, "missing_primary_runs": missing}
    if not integ or not all(v["ok"] for v in integ.values()):
        summary["status"] = "integrity failure or no data: confirmatory analysis not run (prereg §6)"
    elif missing:
        summary["status"] = f"incomplete: {len(missing)} primary (arm, instance) cells lack runs; confirmatory analysis not run (prereg §4, §6)"
    else:
        summary["status"] = "ok"
        summary["primary"] = run_family(df, pa, "primary", tables)
        summary["secondary"] = run_family(df, pa, "secondary", tables)
        h5 = pa["h5_f10_escape"]
        if "CEC2022" in df:
            summary["H5_f10_escape"] = f10_escape(df["CEC2022"], h5["arms"], h5["instance"],
                                                  h5["threshold"], h5["runs"],
                                                  tuple(h5["authors_sarsa_escapes"]))
        summary["V3_authors_mirror"] = v3(df, pa)
        desc = {}
        for bench, d in df.items():
            arms = [a for a in PRIMARY_ARMS if a in set(d.method)]
            if len(arms) >= 3:
                omni, post, _, _ = friedman_block_analysis(d[d.run_id <= 50], arms, "PA-SARSA")
                desc[bench] = {"omnibus": omni, "posthoc_vs_PA-SARSA": post.to_dict(orient="records")}
        summary["descriptive_friedman"] = desc

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable))
    lines = ["| contrast | anchor | comparator | runs | sig anchor | sig comparator | W̄ [95% CI] | verdict |",
             "|---|---|---|---|---|---|---|---|"]
    for level in ("primary", "secondary"):
        for key, s in summary.get(level, {}).items():
            if "wbar" in s:
                lines.append(f"| {key} ({level}) | {s['anchor']} | {s['comparator']} | {s['runs']} | "
                             f"{s['n_sig_anchor']} | {s['n_sig_comparator']} | "
                             f"{s['wbar']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}] | {s['verdict']} |")
            else:
                lines.append(f"| {key} ({level}) | {s['anchor']} | {s['comparator']} | – | – | – | – | {s['verdict']} |")
    (out / "verdicts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"status: {summary['status']}  ->  {out / 'summary.json'}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
