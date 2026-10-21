"""Exploratory BBOB extension (experiments/MPHBS_BBOB_ADDENDUM.md): benchmark wrapper, wiring, budgets, analysis."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cocoex = pytest.importorskip("cocoex")

from lakaie.benchmarks.bbob import BBOBProblem, instance_for_run  # noqa: E402
from lakaie.experiment import load_yaml, make_problem, pathA_arm, run_single, seed_for  # noqa: E402
from lakaie.registry import iterations_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fid", [1, 8, 15, 24])
@pytest.mark.parametrize("dim", [10, 20])
def test_error_is_zero_at_optimum_and_nonnegative(fid, dim):
    p = BBOBProblem(fid, dim, 3)
    xo = np.asarray(cocoex.BareProblem("bbob", fid, dim, 3).best_parameter(), float)
    assert abs(p(xo)[0]) <= 1e-8 * max(1.0, abs(p.optimum))
    X = np.random.default_rng(fid).uniform(-5, 5, (50, dim))
    e = p(X)
    assert e.shape == (50,) and np.all(np.isfinite(e)) and np.all(e >= -1e-9)
    assert np.all(p.lb == -5) and np.all(p.ub == 5)


def test_instance_mapping_and_problem_wiring():
    assert [instance_for_run(r) for r in (1, 15, 16, 30)] == [1, 15, 1, 15]
    p = make_problem("BBOB-D10", 7, run=16)
    assert (p.func_id, p.dim, p.instance) == (7, 10, 1)
    assert make_problem("BBOB-D20", 24, run=30).instance == 15


def _arms():
    pa = load_yaml("pathA_bbob.yaml")
    return [(b, a) for b, arms in pa["arms"].items() for a in arms]


@pytest.mark.parametrize("bench,label", _arms())
def test_bbob_arm_budgets(bench, label):
    bcfg = load_yaml("bbob_d10.yaml" if bench == "BBOB-D10" else "bbob_d20.yaml")
    fam, cfg, expected = pathA_arm(label, bcfg)
    assert iterations_for(fam, cfg, bcfg["max_fes"], bcfg["dimension"], bcfg["population_size"]) == expected


def test_seeds():
    assert seed_for("pathA_bbob", "BBOB-D20", 24, 30) == 93_424_030
    assert seed_for("pathA_bbob", "BBOB-D10", 1, 1) == 93_301_001


def test_run_single_small(tmp_path):
    bcfg = load_yaml("bbob_d10.yaml")
    fam, cfg, _ = pathA_arm("PA-RND-FAIR", bcfg)
    res = run_single({"label": "PA-RND-FAIR", "family": fam, "cfg": cfg, "benchmark": "BBOB-D10", "instance": 3,
                      "run": 17, "campaign": "pathA_bbob", "seed": 1, "max_fes": 3000, "N": 30, "dim": 10,
                      "n_checkpoints": 20, "out_root": str(tmp_path)})
    assert res["status"] == "ok"
    rec = json.loads(Path(res["path"]).read_text())
    assert rec["FE"] <= 3000 and rec["best_fitness"] >= -1e-9 and rec["instance_name"] == "f3_i2"


def test_analysis_end_to_end(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pathA_bbob_analysis
    pa = load_yaml("pathA_bbob.yaml")
    rng = np.random.default_rng(0)
    for bench, arms in pa["arms"].items():
        for arm, spec in arms.items():
            for fid in (1, 2):
                d = tmp_path / "raw" / "pathA_bbob" / bench / arm / f"inst{fid:02d}"
                d.mkdir(parents=True)
                for r in range(1, 31):
                    v = 0.0 if fid == 1 else float(rng.gamma(2, 1))      # f1 solved by everybody -> ties
                    (d / f"run{r:02d}.json").write_text(json.dumps({
                        "method": arm, "benchmark": bench, "instance": fid, "instance_name": "x", "run_id": r,
                        "seed": 0, "best_fitness": v, "FE": spec["expected"][1], "max_fes": 0, "runtime_sec": 0,
                        "status": "ok", "iterations": spec["expected"][0], "final_diversity": 0,
                        "final_stagnation": 0, "population_size": 30, "dimension": 10}))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--raw-root", str(tmp_path / "raw"), "--out", str(out),
                                      "--functions", "1", "2"])
    assert pathA_bbob_analysis.main() == 0
    s = json.loads((out / "summary.json").read_text())
    assert set(s["contrasts"]) == {"B1/BBOB-D10", "B1/BBOB-D20", "B3/BBOB-D10", "B3/BBOB-D20"}
    assert s["contrasts"]["B1/BBOB-D10"]["n_solved_ties"] == 30          # f1: floor 1e-8 makes all pairs tie
    monkeypatch.setattr(sys, "argv", ["x", "--raw-root", str(tmp_path / "raw"), "--out", str(out)])
    assert pathA_bbob_analysis.main() == 1                                # f3..f24 missing -> refuses
