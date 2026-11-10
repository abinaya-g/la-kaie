"""The MPHBS exchange between other parent algorithms (experiments/ESWA_EIGEN_CAMPAIGN.md, Section 10).

``run`` is ``mphbs_variants.run`` with a different two-population state: Phase 1 moves population A and population B
with one generation of two other algorithms; Phase 2 (the mediator, random or eigen frame) is unchanged. There is no
FAD stage (``fad_before_p2`` and ``fad_after_p2`` must be false).

cfg["hosts"]: "PSO+DE" or "GWO+WOA" (first = population A, stored as X_H; second = population B, stored as X_M).
Parent settings as in baselines.py (PSO inertia 0.9 -> 0.4, c1 = c2 = 2, |v| <= 0.2 (ub - lb); DE/rand/1/bin
F = 0.5, CR = 0.9; GWO a: 2 -> 0; WOA a: 2 -> 0, b = 1).

The best member of each population is tracked as best-ever, and the global best is only replaced by a better value,
because PSO, GWO and WOA replace their positions each generation.
"""
from __future__ import annotations

import numpy as np

from ..core import clip
from . import mphbs_variants
from .hybrid_common import HybridState


class _PSO:
    def __init__(self, X, F, lb, ub):
        self.V = np.zeros_like(X)
        self.P, self.pf = X.copy(), F.copy()
        self.vmax = 0.2 * (ub - lb)

    def generation(self, obj, rng, X, F, prog, lb, ub):
        imp = F < self.pf                                    # members written back by the mediator
        self.P[imp], self.pf[imp] = X[imp], F[imp]
        w = 0.9 - 0.5 * prog
        g = self.P[np.argmin(self.pf)]
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        self.V = np.clip(w * self.V + 2.0 * r1 * (self.P - X) + 2.0 * r2 * (g - X), -self.vmax, self.vmax)
        Xn = clip(X + self.V, lb, ub)
        Fn = obj(Xn)
        imp = Fn < self.pf
        self.P[imp], self.pf[imp] = Xn[imp], Fn[imp]
        return Xn, Fn, float(imp.mean())


class _DE:
    def __init__(self, X, F, lb, ub):
        pass

    def generation(self, obj, rng, X, F, prog, lb, ub):
        N, D = X.shape
        idx = np.array([rng.choice(np.delete(np.arange(N), i), 3, replace=False) for i in range(N)])
        V = X[idx[:, 0]] + 0.5 * (X[idx[:, 1]] - X[idx[:, 2]])
        mask = rng.random((N, D)) <= 0.9
        mask[np.arange(N), rng.integers(0, D, N)] = True
        U = clip(np.where(mask, V, X), lb, ub)
        fu = obj(U)
        imp = fu <= F
        X, F = X.copy(), F.copy()
        X[imp], F[imp] = U[imp], fu[imp]
        return X, F, float(imp.mean())


class _GWO:
    def __init__(self, X, F, lb, ub):
        o = np.argsort(F, kind="stable")[:3]
        self.L, self.lf = X[o].copy(), F[o].copy()

    def _leaders(self, X, F):
        P, pf = np.vstack([self.L, X]), np.concatenate([self.lf, F])
        o = np.argsort(pf, kind="stable")[:3]
        self.L, self.lf = P[o].copy(), pf[o].copy()

    def generation(self, obj, rng, X, F, prog, lb, ub):
        self._leaders(X, F)
        a = 2.0 * (1.0 - prog)
        Xn = np.zeros_like(X)
        for leader in self.L:
            A = 2 * a * rng.random(X.shape) - a
            C = 2 * rng.random(X.shape)
            Xn += leader - A * np.abs(C * leader - X)
        Xn = clip(Xn / len(self.L), lb, ub)
        Fn = obj(Xn)
        self._leaders(Xn, Fn)
        return Xn, Fn, float(np.mean(Fn < F))


class _WOA:
    def __init__(self, X, F, lb, ub):
        k = int(np.argmin(F))
        self.g, self.gf = X[k].copy(), float(F[k])

    def _best(self, X, F):
        k = int(np.argmin(F))
        if F[k] < self.gf:
            self.g, self.gf = X[k].copy(), float(F[k])

    def generation(self, obj, rng, X, F, prog, lb, ub):
        self._best(X, F)
        N, D = X.shape
        a, a2 = 2.0 * (1.0 - prog), -1.0 - prog
        A = 2 * a * rng.random(N) - a
        C = 2 * rng.random(N)
        p = rng.random(N)
        l = (a2 - 1) * rng.random(N) + 1
        Xr = X[rng.integers(0, N, N)]
        ref = np.where((np.abs(A) < 1)[:, None], self.g, Xr)
        enc = ref - A[:, None] * np.abs(C[:, None] * ref - X)
        spiral = np.abs(self.g - X) * np.exp(l)[:, None] * np.cos(2 * np.pi * l)[:, None] + self.g
        Xn = clip(np.where((p < 0.5)[:, None], enc, spiral), lb, ub)
        Fn = obj(Xn)
        self._best(Xn, Fn)
        return Xn, Fn, float(np.mean(Fn < F))


HOSTS = {"PSO": _PSO, "DE": _DE, "GWO": _GWO, "WOA": _WOA}
PAIRS = {"PSO+DE": ("PSO", "DE"), "GWO+WOA": ("GWO", "WOA")}


def make_state(hosts: str):
    a_name, b_name = PAIRS[hosts]

    class PairState(HybridState):
        """Two populations moved by other algorithms; attribute names follow HybridState (H = A, M = B)."""

        def __init__(self, obj, lb, ub, dim, N, rng):
            self.lb, self.ub, self.dim, self.N = lb, ub, dim, N
            self.X_H = rng.random((N, dim)) * (ub - lb) + lb
            self.X_M = rng.random((N, dim)) * (ub - lb) + lb
            self.F_H, self.F_M = obj(self.X_H), obj(self.X_M)
            self.A = HOSTS[a_name](self.X_H, self.F_H, lb, ub)
            self.B = HOSTS[b_name](self.X_M, self.F_M, lb, ub)
            self.bF_H = self.bF_M = np.inf
            self.Pg, self.Pbest = np.inf, None
            self._bests()
            self.fit_old, self.X_M_old = self.F_M.copy(), self.X_M.copy()
            self.last_hba_success = self.last_mpa_success = 0.0
            self.last_step_hba = self.last_step_mpa = 0.0

        def _bests(self):
            k = int(np.argmin(self.F_H))
            if self.F_H[k] < self.bF_H:
                self.bF_H, self.bP_H = float(self.F_H[k]), self.X_H[k].copy()
            k = int(np.argmin(self.F_M))
            if self.F_M[k] < self.bF_M:
                self.bF_M, self.bP_M = float(self.F_M[k]), self.X_M[k].copy()
            self.sync_global_keep()

        def global_from_bests(self):
            self.sync_global_keep()                          # never worsen the global best

        def sync_global_keep(self):
            for f, x in ((self.bF_M, getattr(self, "bP_M", None)), (self.bF_H, getattr(self, "bP_H", None))):
                if x is not None and f < self.Pg:
                    self.Pg, self.Pbest = f, x.copy()

        def phase1(self, obj, rng, prog_t, prog_T):
            prog = min(1.0, prog_t / max(1.0, prog_T))
            self.X_H, self.F_H, self.last_hba_success = self.A.generation(obj, rng, self.X_H, self.F_H, prog,
                                                                          self.lb, self.ub)
            self.X_M, self.F_M, self.last_mpa_success = self.B.generation(obj, rng, self.X_M, self.F_M, prog,
                                                                          self.lb, self.ub)
            self._bests()
            return 1.0

    PairState.__name__ = f"PairState_{a_name}_{b_name}"
    return PairState


def run(obj, T, lb, ub, dim, N, rng, cfg, rec=None):
    if cfg.get("fad_before_p2") or cfg.get("fad_after_p2") or cfg.get("reeval_mpa_top") or cfg.get("sync_memory"):
        raise ValueError("exchange hybrids have no FAD stage or MPA memory")
    out = mphbs_variants.run(obj, T, lb, ub, dim, N, rng, cfg, rec, state_cls=make_state(cfg["hosts"]))
    out["summary"]["hosts"] = cfg["hosts"]
    return out
