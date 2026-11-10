"""Exchange hybrids with other parent pairs (lakaie/algorithms/exchange_hybrids.py, design Section 10).

T-XH-RUN: every arm runs through run_single on CEC2022 and stays within the budget.
T-XH-FE: the eigen and axis variants of a pair use identical evaluation counts (the frame costs no FEs).
T-XH-MONO: the recorded global best never worsens, although PSO, GWO and WOA replace their positions.
T-XH-DET: the same seed gives the same result.
T-XH-NOFAD: a configuration with a FAD stage is rejected.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from eswa_campaign import ARMS  # noqa: E402

from lakaie.algorithms import exchange_hybrids  # noqa: E402
from lakaie.core import CountedObjective, Recorder  # noqa: E402
from lakaie.experiment import deep_update, load_yaml, run_single  # noqa: E402

XH = ["PD-R", "PD-EIG", "GW-R", "GW-EIG"]


def _cfg(label):
    fam, flags = ARMS[label]
    return deep_update(load_yaml("cec2022.yaml")["methods"]["MPHBS"], flags)


def _task(tmp_path, label, max_fes=4000, seed=7):
    return {"out_root": tmp_path, "campaign": "t", "benchmark": "CEC2022", "label": label, "instance": 4,
            "run": 1, "dim": 10, "max_fes": max_fes, "N": 30, "family": "XHYB", "cfg": _cfg(label), "seed": seed}


@pytest.mark.parametrize("label", XH)
def test_runs_within_budget(tmp_path, label):
    res = run_single(_task(tmp_path, label))
    assert res["status"] == "ok"
    rec = json.loads(open(res["path"]).read())
    assert 0 < rec["FE"] <= 4000
    assert rec["summary"]["hosts"] == ARMS[label][1]["hosts"]


@pytest.mark.parametrize("pair", [("PD-R", "PD-EIG"), ("GW-R", "GW-EIG")])
def test_eigen_costs_no_extra_fe(tmp_path, pair):
    fe = [json.loads(open(run_single(_task(tmp_path / p, p))["path"]).read())["FE"] for p in pair]
    assert fe[0] == fe[1]


def _sphere(X):
    return np.sum((np.atleast_2d(X) - 0.3) ** 2, axis=1)


@pytest.mark.parametrize("label", XH)
def test_global_best_monotone_and_deterministic(label):
    out = []
    for _ in range(2):
        lb, ub = -5 * np.ones(10), 5 * np.ones(10)
        obj = CountedObjective(_sphere, 10 ** 6)
        rec = Recorder(lb, ub)
        out.append(exchange_hybrids.run(obj, 40, lb, ub, 10, 30, np.random.default_rng(3), _cfg(label), rec))
        best = rec.as_arrays()["best"]
        assert np.all(np.diff(best) <= 0)
    assert out[0]["best"] == out[1]["best"]
    assert out[0]["best"] < 1.0


def test_fad_rejected():
    cfg = dict(_cfg("PD-R"), fad_before_p2=True)
    lb, ub = -np.ones(5), np.ones(5)
    with pytest.raises(ValueError):
        exchange_hybrids.run(CountedObjective(_sphere, 10 ** 5), 3, lb, ub, 5, 30, np.random.default_rng(0), cfg)
