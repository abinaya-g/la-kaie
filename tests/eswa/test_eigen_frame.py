"""Eigen-frame exchange (frame="eigen" in mphbs_variants).

T-EIG-IDENTITY: with an identity basis the eigen path reproduces the axis path bit for bit.
T-EIG-FE: the eigen frame costs no extra FEs.
T-EIG-BOUNDS: every evaluated point stays inside the bounds.
T-EIG-BASIS: the basis is orthonormal.
"""
import numpy as np
import pytest

from lakaie.algorithms import mphbs_variants
from lakaie.core import CountedObjective, Recorder

BC1 = dict(K=7, Ni=3, rho_sub=0.25, w_best=0.05, gamma=0.3, fad_before_p2=True, fad_after_p2=False)
AC1 = dict(K=3, Ni=9, rho_sub=0.25, w_best=0.05, gamma=0.3, fad_before_p2=False, fad_after_p2=True)
SEEDS = [0, 7]


def _rastrigin(X):
    Z = X - 0.3
    return np.sum(Z ** 2 - 10 * np.cos(2 * np.pi * Z) + 10, axis=1)


def _run(cfg, seed, T=6, dim=20, f=_rastrigin):
    lb, ub = -5 * np.ones(dim), 5 * np.ones(dim)
    obj = CountedObjective(f, None)
    rec = Recorder(lb, ub)
    out = mphbs_variants.run(obj, T, lb, ub, dim, 30, np.random.default_rng(seed), dict(cfg), rec)
    return out, rec.as_arrays(), obj.fe


@pytest.mark.parametrize("cfg", [BC1, AC1])
@pytest.mark.parametrize("mediator", ["random", "sarsa"])
@pytest.mark.parametrize("seed", SEEDS)
def test_identity_basis_reproduces_axis(monkeypatch, cfg, mediator, seed):
    ref = _run({**cfg, "mediator": mediator}, seed)
    monkeypatch.setattr(mphbs_variants, "eigen_basis", lambda P: np.eye(P.shape[1]))
    eig = _run({**cfg, "mediator": mediator, "frame": "eigen"}, seed)
    assert ref[0]["best"] == eig[0]["best"]
    assert np.array_equal(ref[0]["best_x"], eig[0]["best_x"])
    assert np.array_equal(ref[1]["best"], eig[1]["best"])
    assert ref[2] == eig[2]


@pytest.mark.parametrize("cfg", [BC1, AC1])
@pytest.mark.parametrize("prob", [1.0, 0.5])
def test_eigen_costs_no_extra_fe(cfg, prob):
    axis = _run({**cfg, "mediator": "random"}, 3)
    eig = _run({**cfg, "mediator": "random", "frame": "eigen", "eigen_prob": prob}, 3)
    assert axis[2] == eig[2]
    assert np.array_equal(axis[1]["fe"], eig[1]["fe"])


def test_evaluated_points_in_bounds():
    seen = []

    def f(X):
        seen.append(np.array(X, copy=True))
        return _rastrigin(X)

    _run({**BC1, "mediator": "random", "frame": "eigen"}, 1, f=f)
    X = np.vstack([np.atleast_2d(x) for x in seen])
    assert X.min() >= -5 and X.max() <= 5


def test_basis_orthonormal():
    P = np.random.default_rng(0).normal(size=(60, 20)) @ np.random.default_rng(1).normal(size=(20, 20))
    B = mphbs_variants.eigen_basis(P)
    assert np.allclose(B.T @ B, np.eye(20), atol=1e-10)


def test_unknown_frame_rejected():
    with pytest.raises(ValueError):
        _run({**BC1, "frame": "polar"}, 0)
