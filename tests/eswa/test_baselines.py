"""External baselines (lakaie/algorithms/baselines.py).

T-BASE-BUDGET: at most max_fe evaluations, and the whole budget is used.
T-BASE-DETERMINISM: the same seed gives the same result.
T-BASE-BOUNDS: every evaluated point stays inside the bounds.
T-BASE-SPHERE: each solves a shifted sphere (GWO and WOA to a looser tolerance, see CENTRE_BIASED).
T-BASE-RUNNER: each runs through run_single on CEC2022.
"""
import json

import numpy as np
import pytest

from lakaie.algorithms.baselines import BUDGETED
from lakaie.core import CountedObjective, Recorder
from lakaie.experiment import run_single

METHODS = sorted(BUDGETED)


def _sphere(X):
    return np.sum((X - 0.7) ** 2, axis=1)


def _run(name, seed, max_fe=6000, dim=10, f=_sphere):
    lb, ub = -5 * np.ones(dim), 5 * np.ones(dim)
    obj = CountedObjective(f, max_fe)
    out = BUDGETED[name](obj, lb, ub, dim, np.random.default_rng(seed), {}, Recorder(lb, ub), max_fe)
    return out, obj


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("max_fe", [997, 6000])
def test_budget_exact(name, max_fe):
    _, obj = _run(name, 0, max_fe)
    assert obj.fe == max_fe


@pytest.mark.parametrize("name", METHODS)
def test_determinism(name):
    a, _ = _run(name, 5)
    b, _ = _run(name, 5)
    assert a["best"] == b["best"]
    assert np.array_equal(a["best_x"], b["best_x"])


@pytest.mark.parametrize("name", METHODS)
def test_bounds(name):
    seen = []

    def f(X):
        seen.append(np.array(X, copy=True))
        return np.sum((X - 4.9) ** 2, axis=1)     # optimum near the upper bound

    _run(name, 2, f=f)
    X = np.vstack(seen)
    assert X.min() >= -5 and X.max() <= 5


CENTRE_BIASED = {"GWO": 1e-5, "WOA": 1e-2}   # documented origin bias: reach ~1e-185 at 0 but not when shifted


@pytest.mark.parametrize("name", METHODS)
def test_sphere(name):
    out, _ = _run(name, 1, max_fe=40000)
    assert out["best"] < CENTRE_BIASED.get(name, 1e-8)


@pytest.mark.parametrize("name", METHODS)
def test_runner(tmp_path, name):
    task = {"out_root": tmp_path, "campaign": "t", "benchmark": "CEC2022", "label": name, "instance": 1,
            "run": 1, "dim": 10, "max_fes": 3000, "N": 30, "family": name, "cfg": {}, "seed": 11}
    res = run_single(task)
    assert res["status"] == "ok"
    rec = json.loads(open(res["path"]).read())
    assert rec["FE"] == 3000
