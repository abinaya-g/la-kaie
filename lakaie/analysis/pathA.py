"""Path A analysis (experiments/MPHBS_DECONSTRUCTION_PREREG.md §5), frozen before the campaign.

Pure functions; scripts/pathA_analysis.py applies them to the raw results.

* per instance: two-sided paired Wilcoxon (stats.wilcoxon_p), paired win
  probability w = (wins + 0.5 ties) / n from the anchor's view
* Holm within each (contrast, domain) family of instances
* W-bar = mean of w over instances, 95 % percentile CI from a paired bootstrap
  (run indices resampled independently within each instance)
* mechanical verdict (§5.4)
* H5: F10 escape rates (Wilson CI), McNemar exact, binomial vs the authors
* V3: Mann-Whitney vs the authors' published mirror runs, Holm within domain
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .stats import TIE_TOL, holm, wilcoxon_p

ANCHOR, COMPARATOR = "anchor superior", "comparator superior"
EQUIVALENT, MIXED, INCONCLUSIVE = "practically equivalent", "mixed", "inconclusive"


def paired_values(df: pd.DataFrame, a: str, b: str, instance, runs: int):
    """Seed-paired final values of arms a and b on one instance, runs 1..runs."""
    g = df[(df.instance == instance) & (df.run_id <= runs)]
    A = g[g.method == a].set_index("run_id")["best_fitness"]
    B = g[g.method == b].set_index("run_id")["best_fitness"]
    common = A.index.intersection(B.index).sort_values()
    return A.loc[common].to_numpy(float), B.loc[common].to_numpy(float), common.to_numpy()


def win_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 if the anchor is better (lower), 0.5 for a tie (|d| <= 1e-12), 0 otherwise."""
    d = np.asarray(a, float) - np.asarray(b, float)
    return np.where(np.abs(d) <= TIE_TOL, 0.5, np.where(d < 0, 1.0, 0.0))


def bootstrap_wbar(scores: list[np.ndarray], n_boot: int, seed: int) -> tuple[float, float, float]:
    """W-bar and its 95 % percentile CI; runs are resampled within each instance."""
    rng = np.random.default_rng(seed)
    point = float(np.mean([s.mean() for s in scores]))
    boots = np.zeros(n_boot)
    for s in scores:
        idx = rng.integers(0, s.size, size=(n_boot, s.size))
        boots += s[idx].mean(axis=1)
    boots /= len(scores)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def verdict(n_sig_anchor: int, n_sig_comparator: int, lo: float, hi: float,
            band=(0.44, 0.56)) -> str:
    """§5.4, applied in this order (mixed takes precedence)."""
    if n_sig_anchor > 0 and n_sig_comparator > 0:
        return MIXED
    if n_sig_anchor > 0 and lo > 0.5:
        return ANCHOR
    if n_sig_comparator > 0 and hi < 0.5:
        return COMPARATOR
    if n_sig_anchor == 0 and n_sig_comparator == 0 and lo >= band[0] and hi <= band[1]:
        return EQUIVALENT
    return INCONCLUSIVE


def contrast(df: pd.DataFrame, a: str, b: str, runs: int, alpha: float, band, n_boot: int,
             seed: int):
    """One (contrast, domain) family: instance table + domain summary."""
    rows, scores = [], []
    for inst in sorted(df.instance.unique()):
        x, y, rid = paired_values(df, a, b, inst, runs)
        if x.size == 0:
            continue
        w = win_scores(x, y)
        scores.append(w)
        rows.append({"instance": int(inst), "n_pairs": int(x.size),
                     "anchor_wins": int(np.sum(w == 1.0)), "ties": int(np.sum(w == 0.5)),
                     "anchor_losses": int(np.sum(w == 0.0)), "w": float(w.mean()),
                     "anchor_median": float(np.median(x)), "comparator_median": float(np.median(y)),
                     "anchor_iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
                     "comparator_iqr": float(np.subtract(*np.percentile(y, [75, 25]))),
                     "median_paired_diff": float(np.median(x - y)),
                     "p_raw": wilcoxon_p(x, y)})
    T = pd.DataFrame(rows)
    if T.empty:
        return T, {"anchor": a, "comparator": b, "runs": runs, "verdict": "no data"}
    T["p_holm"] = holm(T.p_raw.to_numpy())
    sig = T.p_holm < alpha
    # direction by the paired win probability (median difference can be 0 with a clear majority)
    T["significant_favours"] = np.where(~sig, "none", np.where(T.w > 0.5, "anchor", "comparator"))
    n_a = int((T.significant_favours == "anchor").sum())
    n_c = int((T.significant_favours == "comparator").sum())
    wbar, lo, hi = bootstrap_wbar(scores, n_boot, seed)
    summary = {"anchor": a, "comparator": b, "runs": runs, "n_instances": int(len(T)),
               "min_pairs": int(T.n_pairs.min()), "n_sig_anchor": n_a, "n_sig_comparator": n_c,
               "wbar": wbar, "ci_low": lo, "ci_high": hi,
               "verdict": verdict(n_a, n_c, lo, hi, tuple(band))}
    return T, summary


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def mcnemar_exact(x_escape: np.ndarray, y_escape: np.ndarray) -> dict:
    b = int(np.sum(x_escape & ~y_escape))
    c = int(np.sum(~x_escape & y_escape))
    p = 1.0 if b + c == 0 else float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"only_first": b, "only_second": c, "p": p}


def f10_escape(df: pd.DataFrame, arms, instance: int, threshold: float, runs: int,
               authors: tuple[int, int]) -> dict:
    g = df[(df.instance == instance) & (df.run_id <= runs)]
    out = {"threshold": threshold, "instance": instance, "arms": {}}
    esc = {}
    for a in arms:
        s = g[g.method == a].set_index("run_id")["best_fitness"].sort_index()
        e = s.to_numpy(float) < threshold
        esc[a] = s.index.to_numpy(), e
        k, n = int(e.sum()), int(e.size)
        lo, hi = wilson(k, n)
        out["arms"][a] = {"escapes": k, "n": n, "rate": k / n if n else math.nan,
                          "wilson_low": lo, "wilson_high": hi}
    if len(arms) >= 2:
        (ia, ea), (ib, eb) = esc[arms[0]], esc[arms[1]]
        common = np.intersect1d(ia, ib)
        out["mcnemar_first_two"] = mcnemar_exact(ea[np.isin(ia, common)], eb[np.isin(ib, common)])
    k_auth, n_auth = authors
    first = out["arms"].get(arms[0], {})
    if first.get("n"):
        out["binomial_vs_authors"] = {
            "authors": f"{k_auth}/{n_auth}",
            "p": float(stats.binomtest(first["escapes"], first["n"], k_auth / n_auth).pvalue)}
    return out


def authors_mirror_check(ours: pd.DataFrame, ref: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """V3: per instance two-sided Mann-Whitney (ours vs the authors' runs), Holm within domain.
    `ours` and `ref` have columns instance, value."""
    rows = []
    for inst, g in ours.groupby("instance"):
        a = g.value.to_numpy(float)
        b = ref[ref.instance == inst].value.to_numpy(float)
        if b.size == 0:
            continue
        same = np.allclose(np.r_[a, b], a[0])
        p = 1.0 if same else float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        rows.append({"instance": int(inst), "n": int(a.size), "n_authors": int(b.size),
                     "median": float(np.median(a)), "median_authors": float(np.median(b)), "p": p})
    T = pd.DataFrame(rows)
    if not T.empty:
        T["p_holm"] = holm(T.p.to_numpy())
        T["differs"] = T.p_holm < alpha
    return T


def integrity(df: pd.DataFrame, expected_fe: dict) -> dict:
    """V2/V4 on stored results: status ok, finite value, FE equal to the arm's expected total."""
    bad_status = df[df.status != "ok"]
    vals = pd.to_numeric(df.best_fitness, errors="coerce")
    bad_value = df[~np.isfinite(vals.to_numpy(float))]
    fe_exp = df.method.map(expected_fe)
    bad_fe = df[fe_exp.notna() & (df.FE != fe_exp)]
    return {"n_runs": int(len(df)), "bad_status": int(len(bad_status)),
            "bad_value": int(len(bad_value)), "bad_fe": int(len(bad_fe)),
            "ok": bool(len(bad_status) == 0 and len(bad_value) == 0 and len(bad_fe) == 0),
            "bad_paths": sorted(set(bad_status.path) | set(bad_value.path) | set(bad_fe.path))[:50]}
