"""Eigen-frame confirmatory campaign runner (experiments/ESWA_EIGEN_CAMPAIGN.md).

Usage: python scripts/eswa_campaign.py --bench CEC2022-D10 [--arms ...] [--runs 30] [--workers 11]
       [--out results/eswa_eigen] [--funcs ...] [--max-fes N]
Results go to <out>/<bench>/eswa_eigen/<benchmark>/<arm>/instNN/runNN.json; finished runs are skipped.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lakaie.experiment import deep_update, load_yaml, run_single

BENCHES = {   # key: (config, benchmark name for make_problem, dim, MaxFE, instances, seed offset)
    "CEC2022-D10": ("cec2022.yaml", "CEC2022", 10, 200000, list(range(1, 13)), 0),        # official CEC2022
    "CEC2022-D20": ("cec2022.yaml", "CEC2022", 20, 200000, list(range(1, 13)), 100000),   # 10000 D, as Path A
    "BBOB-D20":    ("bbob_d20.yaml", "BBOB-D20", 20, 200000, list(range(1, 25)), 200000),
    "FIR":         ("fir.yaml", "FIR", 31, 150000, list(range(1, 9)), 300000),
}
ARMS = {      # label: (family, flags on the domain's MPHBS configuration; None = the method's own configuration)
    "MPHBS":   ("MPHBS",   {}),
    "RND":     ("MPHBS-V", {"mediator": "random"}),
    "EIG50":   ("MPHBS-V", {"mediator": "random", "frame": "eigen", "eigen_prob": 0.5}),
    "CMA-ES":  ("CMA-ES",  None),
    "L-SHADE": ("L-SHADE", None),
}
EXTRA = {     # common metaheuristic baselines (addendum, 2026-11-07); N = 30 like the hybrids
    "HBA": ("HBA", None), "MPA": ("MPA", None), "PSO": ("PSO", None), "DE": ("DE", None),
    "GWO": ("GWO", None), "WOA": ("WOA", None),
}
ARMS.update(EXTRA)
MAIN = ["MPHBS", "RND", "EIG50", "CMA-ES", "L-SHADE"]
MASTER = 95000000
CAMPAIGN = "eswa_eigen"


def tasks(out, bench, arms, runs, funcs, max_fes=None):
    cfg_file, name, dim, budget, _, offset = BENCHES[bench]
    b = load_yaml(cfg_file)
    budget = max_fes or budget
    for f in funcs:
        for label in arms:
            fam, flags = ARMS[label]
            for r in range(1, runs + 1):
                yield {"out_root": Path(out, bench), "campaign": CAMPAIGN, "benchmark": name, "label": label,
                       "instance": f, "run": r, "dim": dim, "max_fes": budget,
                       "N": b["population_size"], "family": fam,
                       "cfg": (dict(b["methods"].get(fam) or {}) if flags is None
                               else deep_update(b["methods"]["MPHBS"], flags)),
                       "seed": MASTER + offset + 1000 * f + r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(BENCHES))
    ap.add_argument("--arms", nargs="*", default=MAIN)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--out", default="results/eswa_eigen")
    ap.add_argument("--funcs", type=int, nargs="*")
    ap.add_argument("--max-fes", type=int)
    a = ap.parse_args()
    funcs = a.funcs or BENCHES[a.bench][4]
    with ProcessPoolExecutor(a.workers) as ex:
        for res in ex.map(run_single, tasks(a.out, a.bench, a.arms, a.runs, funcs, a.max_fes)):
            if res["status"] not in ("ok", "exists"):
                print(res, flush=True)
    print("done", a.bench, a.arms, flush=True)
