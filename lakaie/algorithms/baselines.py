"""Budget-driven external baselines for the eigen-frame campaign (experiments/ESWA_EIGEN_CAMPAIGN.md).

* ``cmaes_run``: CMA-ES (pycma) with IPOP restarts (population doubled at each restart).
* ``lshade_run``: L-SHADE (Tanabe & Fukunaga, CEC 2014), with the published settings as defaults.

Both use exactly ``max_fe`` evaluations at most: the last generation is truncated to the remaining budget.
Population sizes follow each algorithm's own defaults, not the N of the hybrids.
"""
from __future__ import annotations

import io
import contextlib
import warnings

import numpy as np

from ..core import Recorder, clip


def cmaes_run(obj, lb, ub, dim, rng, cfg=None, rec: Recorder | None = None, max_fe: int = 0):
    import cma

    cfg = cfg or {}
    sigma_frac = float(cfg.get("sigma0", 0.3))      # initial step size as a fraction of the box width
    inc = int(cfg.get("incpopsize", 2))
    lam = int(cfg.get("popsize", 4 + int(3 * np.log(dim))))
    restarts = 0
    while obj.fe < max_fe:
        x0 = rng.random(dim) * (ub - lb) + lb
        opts = {"bounds": [lb.tolist(), ub.tolist()], "popsize": lam, "verbose": -9,
                "seed": int(rng.integers(1, 2**31 - 1)), "maxfevals": np.inf}
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            es = cma.CMAEvolutionStrategy(x0, sigma_frac * float(np.max(ub - lb)), opts)
            while not es.stop() and obj.fe < max_fe:
                X = np.asarray(es.ask())
                left = max_fe - obj.fe
                if left < len(X):                     # final, truncated generation: evaluate and stop
                    obj(clip(X[:left], lb, ub))
                    break
                f = obj(clip(X, lb, ub))
                es.tell(list(X), f.tolist())
                if rec is not None:
                    rec.iteration(obj.fe, obj.best, [clip(X, lb, ub)], [f], popsize=lam, restart=restarts)
        restarts += 1
        lam *= inc
    return {"best": obj.best, "best_x": obj.best_x, "summary": {"restarts": restarts}}


def _wlehmer(s, w):
    return float(np.sum(w * s ** 2) / np.sum(w * s)) if np.sum(w * s) > 0 else 0.0


def lshade_run(obj, lb, ub, dim, rng, cfg=None, rec: Recorder | None = None, max_fe: int = 0):
    cfg = cfg or {}
    n_init = int(round(float(cfg.get("r_Ninit", 18)) * dim))
    n_min = int(cfg.get("N_min", 4))
    H = int(cfg.get("H", 6))
    arc_rate = float(cfg.get("r_arc", 2.6))
    p = float(cfg.get("p", 0.11))

    N = n_init
    X = rng.random((N, dim)) * (ub - lb) + lb
    fit = obj(X)
    A = np.empty((0, dim))
    M_CR, M_F = np.full(H, 0.5), np.full(H, 0.5)
    k = 0
    while obj.fe < max_fe:
        r = rng.integers(0, H, N)
        CR = np.where(np.isnan(M_CR[r]), 0.0, np.clip(rng.normal(np.nan_to_num(M_CR[r]), 0.1), 0, 1))
        F = np.empty(N)
        for i in range(N):
            f = -1.0
            while f <= 0:
                f = M_F[r[i]] + 0.1 * np.tan(np.pi * (rng.random() - 0.5))
            F[i] = min(f, 1.0)

        order = np.argsort(fit, kind="stable")
        n_p = max(2, int(round(p * N)))
        pbest = X[order[rng.integers(0, n_p, N)]]
        PA = np.vstack([X, A]) if len(A) else X
        r1 = np.empty(N, dtype=int)
        r2 = np.empty(N, dtype=int)
        for i in range(N):
            r1[i] = rng.integers(0, N)
            while r1[i] == i:
                r1[i] = rng.integers(0, N)
            r2[i] = rng.integers(0, len(PA))
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = rng.integers(0, len(PA))
        V = X + F[:, None] * (pbest - X) + F[:, None] * (X[r1] - PA[r2])
        V = np.where(V < lb, (lb + X) / 2, V)                 # midpoint repair, as in L-SHADE
        V = np.where(V > ub, (ub + X) / 2, V)
        mask = rng.random((N, dim)) <= CR[:, None]
        mask[np.arange(N), rng.integers(0, dim, N)] = True
        U = np.where(mask, V, X)

        left = max_fe - obj.fe
        if left < N:                                          # final, truncated generation
            obj(U[:left])
            break
        fu = obj(U)
        better = fu < fit
        improved = fu <= fit
        if better.any():
            A = np.vstack([A, X[better]])
            w = fit[better] - fu[better]
            w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
            s_cr, s_f = CR[better], F[better]
            M_CR[k] = np.nan if (np.isnan(M_CR[k]) or s_cr.max() == 0) else _wlehmer(s_cr, w)
            M_F[k] = _wlehmer(s_f, w)
            k = (k + 1) % H
        X[improved], fit[improved] = U[improved], fu[improved]

        N_next = int(round((n_min - n_init) / max_fe * obj.fe + n_init))
        if N_next < N:
            keep = np.argsort(fit, kind="stable")[:max(N_next, n_min)]
            X, fit = X[keep], fit[keep]
            N = len(X)
        a_max = int(round(arc_rate * N))
        if len(A) > a_max:
            A = A[rng.choice(len(A), a_max, replace=False)]
        if rec is not None:
            rec.iteration(obj.fe, obj.best, [X], [fit], popsize=N)
    return {"best": obj.best, "best_x": obj.best_x}


BUDGETED = {"CMA-ES": cmaes_run, "L-SHADE": lshade_run}
