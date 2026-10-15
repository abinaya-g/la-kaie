"""MPHBS deconstruction arms (Path A, experiments/MPHBS_DECONSTRUCTION_PREREG.md).

A flag-controlled copy of ``mphbs.run``. The validated port ``mphbs.py`` is
frozen (tests/n1/test_n1_backbone_frozen.py), so the arms live here; with
default flags this module is bit-identical to ``mphbs.run``
(tests/pathA/test_pathA_arms.py::test_sarsa_unchanged).

Flags (all default to the MPHBS behaviour):

* ``mediator``: "sarsa" | "random". "random" draws (n, ch) uniformly for all K
  candidates and never touches the Q-table (= MPHBS_random_mirror.m Phase 2).
  ``random_actions: true`` is accepted as an alias for "random".
* ``skip_phase2``: no mediator at all (no f(P_best) evaluation, no episodes).
* ``relaxed_accept``: false = accept only improving candidates.
* ``sync_memory``: at the end of each iteration set the MPA marine memory
  (fit_old, X_M_old) to the current population at zero FEs.
* ``reeval_mpa_top``: at the start of each iteration re-evaluate the MPA
  population and apply memory saving, as MPHBS_random_mirror.m lines 57-66
  (N extra FEs per iteration).
* ``inject_gbest``: false = draw the two injection indices but do not write
  (test hook for T-INJ-INERT; never used by a campaign arm).
"""
from __future__ import annotations

import math

import numpy as np

from ..core import EPS, matlab_round
from .hybrid_common import HybridState
from .mphbs import (ALPHA_LR, W_NOVEL, W_RANK, closest_action, elite_plus_random, eps_decay, greedy,
                    second_best)


def _flags(cfg):
    mediator = str(cfg.get("mediator", "random" if cfg.get("random_actions", False) else "sarsa"))
    if mediator not in ("sarsa", "random"):
        raise ValueError(f"unknown mediator {mediator!r}")
    return {
        "random": mediator == "random",
        "skip_phase2": bool(cfg.get("skip_phase2", False)),
        "relaxed_accept": bool(cfg.get("relaxed_accept", True)),
        "sync_memory": bool(cfg.get("sync_memory", False)),
        "reeval_mpa_top": bool(cfg.get("reeval_mpa_top", False)),
        "inject_gbest": bool(cfg.get("inject_gbest", True)),
    }


def run(obj, T, lb, ub, dim, N, rng, cfg, rec=None):
    K, Ni = int(cfg["K"]), int(cfg["Ni"])
    rho, w_best, gamma = float(cfg["rho_sub"]), float(cfg["w_best"]), float(cfg["gamma"])
    fad_before, fad_after = bool(cfg["fad_before_p2"]), bool(cfg["fad_after_p2"])
    fl = _flags(cfg)
    random_mirror = fl["random"]

    n_blocks = 5 if dim >= 10 else 1
    block = math.ceil(dim / n_blocks)
    n_sub = min(N, max(5, matlab_round(N * rho)))
    Q = np.zeros((dim, n_sub, 2))

    s = HybridState(obj, lb, ub, dim, N, rng)
    for t in range(T):
        if fl["reeval_mpa_top"]:
            s.eval_mpa(obj)
            s.memory_saving()
        prog_global = min(1.0, t / max(1, T - 1))
        CF = s.phase1(obj, rng, t, T)
        fad_rate = np.nan
        if fad_before:
            fad_rate = s.fads_greedy(obj, rng, CF)
        s.global_from_bests()

        n_acc = n_improve_gbest = 0
        E = n_sub * Ni
        if not fl["skip_phase2"]:
            # ---------------- PHASE 2 (identical to mphbs.run unless a flag says otherwise)
            idxH = elite_plus_random(rng, s.F_H, n_sub)
            idxM = elite_plus_random(rng, s.F_M, n_sub)
            XHs, FHs = s.X_H[idxH].copy(), s.F_H[idxH].copy()
            XMs, FMs = s.X_M[idxM].copy(), s.F_M[idxM].copy()
            result = np.stack([XHs, XMs], axis=2)            # (n_sub, D, 2)

            values = s.Pbest.copy()
            ch = np.zeros(dim, dtype=int); rows = np.zeros(dim, dtype=int)
            for c in range(dim):
                ch[c], rows[c] = closest_action(values[c], result, c)
            f_curr = obj.eval1(values)
            perm = rng.permutation(dim); ptr = 0
            for ep in range(1, E + 1):
                end = ptr + block
                if end > dim:
                    part1 = perm[ptr:]
                    perm = rng.permutation(dim)
                    need = block - part1.size
                    active = np.concatenate([part1, perm[:need]])
                    ptr = need
                else:
                    active = perm[ptr:end]
                    ptr = end
                    if ptr >= dim:
                        perm = rng.permutation(dim); ptr = 0

                best_score, best_k = -np.inf, 0
                cands = []
                for k in range(1, K + 1):
                    v = values.copy(); chn = ch.copy(); rn = rows.copy()
                    for c in active:
                        if random_mirror or k == 3:
                            ck, rk = int(rng.integers(0, 2)), int(rng.integers(0, n_sub))
                        elif k == 1:
                            ck, rk = greedy(Q, c)
                        elif k == 2:
                            ck, rk = eps_decay(rng, Q, c, ep, E)
                        elif k == 4:
                            ck, rk = second_best(Q, c)
                        else:
                            ep2 = min(E, ep + int(rng.integers(1, max(1, E // 10) + 1)))
                            ck, rk = eps_decay(rng, Q, c, ep2, E)
                        v[c] = result[rk, c, ck]; chn[c] = ck; rn[c] = rk
                    score = (-W_RANK * np.mean(rn[active] + 1)
                             + W_NOVEL * np.mean(np.abs(v[active] - values[active]))
                             - w_best * np.linalg.norm(v[active] - s.Pbest[active]))
                    cands.append((v, chn, rn))
                    if score > best_score:
                        best_score, best_k = score, k - 1
                v_next, ch_next, rows_next = cands[best_k]

                val = obj.eval1(v_next)
                reward = 1.0 if val < f_curr else (0.0 if val == f_curr else -1.0)
                prog_local = (ep - 1) / max(1, E - 1)
                if val < f_curr:
                    accepted = True
                elif fl["relaxed_accept"]:
                    d_rel = (val - f_curr) / (abs(f_curr) + EPS)
                    p = 0.20 * (1 - prog_global) ** 2 * (0.35 + 0.65 * (1 - prog_local)) * math.exp(-4 * max(0.0, d_rel))
                    accepted = rng.random() < min(max(p, 0.0), 0.25)
                else:
                    accepted = False

                if not random_mirror:
                    for c in active:
                        pred = Q[c, rows[c], ch[c]]
                        nq = Q[c, rows_next[c], ch_next[c]] if accepted else pred
                        Q[c, rows[c], ch[c]] = pred + ALPHA_LR * (reward + gamma * nq - pred)

                if accepted:
                    n_acc += 1
                    values, f_curr, ch, rows = v_next, val, ch_next, rows_next
                    if f_curr < FHs.max():
                        w = int(np.argmax(FHs)); XHs[w] = values; FHs[w] = f_curr
                    if f_curr < FMs.max():
                        w = int(np.argmax(FMs)); XMs[w] = values; FMs[w] = f_curr
                    result = np.stack([XHs, XMs], axis=2)

                if val < s.Pg:
                    n_improve_gbest += 1
                    s.Pg, s.Pbest = val, v_next.copy()
                    th = idxH[int(rng.integers(0, idxH.size))]
                    tm = idxM[int(rng.integers(0, idxM.size))]
                    if fl["inject_gbest"]:
                        s.X_H[th] = s.Pbest; s.F_H[th] = s.Pg
                        s.X_M[tm] = s.Pbest; s.F_M[tm] = s.Pg

            s.X_H[idxH] = XHs; s.F_H[idxH] = FHs
            s.X_M[idxM] = XMs; s.F_M[idxM] = FMs
            k = int(np.argmin(s.F_H)); s.bF_H, s.bP_H = float(s.F_H[k]), s.X_H[k].copy()
            k = int(np.argmin(s.F_M)); s.bF_M, s.bP_M = float(s.F_M[k]), s.X_M[k].copy()
        s.sync_global_keep()

        if fad_after:
            fad_rate = s.fads_greedy(obj, rng, CF)
            s.sync_global_keep()
        if fl["sync_memory"]:
            s.fit_old, s.X_M_old = s.F_M.copy(), s.X_M.copy()
        if rec is not None:
            rec.iteration(obj.fe, s.Pg, [s.X_H, s.X_M], [s.F_H, s.F_M],
                          hba_success=s.last_hba_success, mpa_success=s.last_mpa_success,
                          fad_success=fad_rate,
                          mediator_accept_rate=(n_acc / E) if not fl["skip_phase2"] else np.nan,
                          mediator_gbest_improvements=n_improve_gbest)
    return {"best": s.Pg, "best_x": s.Pbest,
            "summary": {"q_nonzero": int(np.count_nonzero(Q)), **fl}}
