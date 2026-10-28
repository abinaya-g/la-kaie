"""Exploratory pilot for the eigen-frame exchange (not part of any pre-registered campaign).

Runs a few instances of one benchmark (BBOB-D20, CEC2022 at D=20, or FIR) for a few arms and prints
median final errors.
Usage: python scripts/eswa_pilot.py --out <dir> [--bench BBOB-D20] [--runs 10] [--workers 8]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lakaie.experiment import deep_update, load_yaml, run_single

BENCHES = {   # config, default instances, seed offset
    "BBOB-D20": ("bbob_d20.yaml", [1, 3, 8, 10, 12, 15, 16, 21], 0),   # separable: 1, 3; rotated: the rest
    "CEC2022":  ("cec2022.yaml", list(range(1, 13)), 100000),
    "FIR":      ("fir.yaml", list(range(1, 9)), 200000),
}
ARMS = {
    "SARSA":   ("MPHBS",   {}),
    "RND":     ("MPHBS-V", {"mediator": "random"}),
    "EIG":     ("MPHBS-V", {"mediator": "random", "frame": "eigen"}),
    "EIG50":   ("MPHBS-V", {"mediator": "random", "frame": "eigen", "eigen_prob": 0.5}),
    "SKEL":    ("MPHBS-V", {"skip_phase2": True}),
}
MASTER = 94000000


def tasks(out, bench, runs, funcs, arms):
    cfg_file, _, offset = BENCHES[bench]
    b = load_yaml(cfg_file)
    for f in funcs:
        for label in arms:
            fam, flags = ARMS[label]
            for r in range(1, runs + 1):
                yield {"out_root": out, "campaign": "eswa_pilot", "benchmark": bench, "label": label,
                       "instance": f, "run": r, "dim": b["dimension"], "max_fes": b["max_fes"], "N": b["population_size"],
                       "family": fam, "cfg": deep_update(b["methods"]["MPHBS"], flags),
                       "seed": MASTER + offset + 1000 * f + r}


def summary(out, bench, funcs, arms):
    print("func  " + "  ".join(f"{a:>10s}" for a in arms))
    for f in funcs:
        row = []
        for a in arms:
            vals = [max(json.loads(p.read_text())["best_fitness"], 1e-8)
                    for p in Path(out, "eswa_pilot", bench, a, f"inst{f:02d}").glob("run*.json")]
            row.append(f"{np.median(vals):10.3g}" if vals else f"{'-':>10s}")
        print(f"f{f:<4d} " + "  ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", default="BBOB-D20", choices=list(BENCHES))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--funcs", type=int, nargs="*")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    funcs = a.funcs or BENCHES[a.bench][1]
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(run_single, tasks(a.out, a.bench, a.runs, funcs, a.arms)):
            if res["status"] not in ("ok", "exists"):
                print(res)
    summary(a.out, a.bench, funcs, a.arms)
