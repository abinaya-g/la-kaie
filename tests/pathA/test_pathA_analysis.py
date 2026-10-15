"""Path A analysis functions and the frozen script (prereg §5, §6), on synthetic data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lakaie.analysis import pathA as A
from lakaie.experiment import load_yaml

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ decision rule (§5.4)
@pytest.mark.parametrize("na,nc,lo,hi,expected", [
    (2, 0, 0.55, 0.70, A.ANCHOR),
    (0, 3, 0.30, 0.45, A.COMPARATOR),
    (0, 0, 0.47, 0.53, A.EQUIVALENT),
    (1, 1, 0.48, 0.52, A.MIXED),          # mixed takes precedence
    (2, 0, 0.49, 0.60, A.INCONCLUSIVE),   # significant instance but CI touches 0.5
    (0, 0, 0.42, 0.53, A.INCONCLUSIVE),   # CI leaves the band
    (1, 0, 0.47, 0.53, A.INCONCLUSIVE),   # inside band but a significant instance
    (0, 0, 0.44, 0.56, A.EQUIVALENT),     # band is closed
])
def test_verdict(na, nc, lo, hi, expected):
    assert A.verdict(na, nc, lo, hi, (0.44, 0.56)) == expected


def test_win_scores_ties_and_direction():
    w = A.win_scores(np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 1.0, 3.0 + 1e-13, 4.0]))
    assert w.tolist() == [1.0, 0.0, 0.5, 0.5]


def test_bootstrap_deterministic_and_brackets_point():
    rng = np.random.default_rng(0)
    scores = [rng.integers(0, 2, 100).astype(float) for _ in range(12)]
    a = A.bootstrap_wbar(scores, 2000, 5)
    assert a == A.bootstrap_wbar(scores, 2000, 5)
    assert a[1] <= a[0] <= a[2]
    assert 0.02 < a[2] - a[1] < 0.08          # ~ 2 * 1.96 * sqrt(0.25 / 1200)


def test_wilson_known_value():
    lo, hi = A.wilson(17, 30)
    assert lo == pytest.approx(0.3920, abs=1e-3) and hi == pytest.approx(0.7262, abs=1e-3)


def test_mcnemar_exact():
    x = np.array([True] * 10 + [False] * 10)
    y = np.array([False] * 10 + [False] * 10)
    r = A.mcnemar_exact(x, y)
    assert r["only_first"] == 10 and r["only_second"] == 0 and r["p"] < 0.01


# ------------------------------------------------------------ contrast on synthetic paired data
def _df(shift_by_method, n_inst=6, runs=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for inst in range(1, n_inst + 1):
        base = rng.normal(0, 1, runs)               # common random numbers -> pairing
        for m, shift in shift_by_method.items():
            for r in range(runs):
                rows.append({"method": m, "instance": inst, "run_id": r + 1,
                             "best_fitness": 10 + base[r] + shift + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_contrast_detects_anchor_superiority():
    df = _df({"X": 0.0, "Y": 0.5})
    T, s = A.contrast(df, "X", "Y", 50, 0.05, (0.44, 0.56), 2000, 1)
    assert s["verdict"] == A.ANCHOR and s["n_sig_anchor"] == 6 and s["wbar"] > 0.95
    assert (T.significant_favours == "anchor").all() and (T.p_holm >= T.p_raw).all()


def test_contrast_equivalence_with_many_runs():
    df = _df({"X": 0.0, "Y": 0.0}, n_inst=12, runs=100, seed=3)
    _, s = A.contrast(df, "X", "Y", 100, 0.05, (0.44, 0.56), 4000, 1)
    assert s["verdict"] in (A.EQUIVALENT, A.INCONCLUSIVE)
    assert 0.44 < s["wbar"] < 0.56


def test_contrast_respects_run_limit():
    df = _df({"X": 0.0, "Y": 0.5}, runs=100)
    T, s = A.contrast(df, "X", "Y", 50, 0.05, (0.44, 0.56), 500, 1)
    assert (T.n_pairs == 50).all() and s["runs"] == 50


def test_integrity_flags_wrong_fe():
    df = pd.DataFrame({"method": ["PA-SARSA", "PA-SARSA"], "status": ["ok", "ok"],
                       "best_fitness": [1.0, 2.0], "FE": [199930, 199929], "path": ["a", "b"]})
    r = A.integrity(df, {"PA-SARSA": 199930})
    assert not r["ok"] and r["bad_fe"] == 1 and r["bad_paths"] == ["b"]


def test_f10_escape_counts():
    vals = {"P": [10.0] * 17 + [100.4] * 13, "Q": [100.4] * 26 + [5.0] * 4}
    rows = [{"method": m, "instance": 10, "run_id": i + 1, "best_fitness": v}
            for m, vs in vals.items() for i, v in enumerate(vs)]
    r = A.f10_escape(pd.DataFrame(rows), ["P", "Q"], 10, 50.0, 30, (17, 30))
    assert r["arms"]["P"]["escapes"] == 17 and r["arms"]["Q"]["escapes"] == 4
    assert r["binomial_vs_authors"]["p"] == pytest.approx(1.0)
    assert r["mcnemar_first_two"]["only_first"] == 17


# ------------------------------------------------------------ the frozen script end to end
FAKE_INSTANCES = {"CEC2022": [1, 2, 10], "FIR": [1, 2]}     # small: file I/O dominates the test time


def _write_fake_campaign(raw_root: Path, instances=FAKE_INSTANCES, max_runs=100):
    pa = load_yaml("pathA.yaml")
    rng = np.random.default_rng(42)
    for bench, insts in instances.items():
        for label, arm in pa["arms"][bench].items():
            for inst in insts:
                full = label in ("PA-SARSA", "PA-RND-FAIR") or (
                    bench == "CEC2022" and inst == pa["h5_f10_escape"]["instance"]
                    and label in pa["h5_f10_escape"]["arms"])
                runs = min(max_runs, 100 if full else 50)
                d = raw_root / "pathA" / bench / label / f"inst{inst:02d}"
                d.mkdir(parents=True, exist_ok=True)
                for r in range(1, runs + 1):
                    rec = {"method": label, "benchmark": bench, "instance": inst, "instance_name": "x",
                           "run_id": r, "seed": 0, "best_fitness": float(rng.gamma(2.0, 10.0)),
                           "FE": arm["expected"][1], "max_fes": 0, "runtime_sec": 0.0, "status": "ok",
                           "iterations": arm["expected"][0], "final_diversity": 0.0,
                           "final_stagnation": 0.0, "population_size": 30, "dimension": 20}
                    (d / f"run{r:02d}.json").write_text(json.dumps(rec))


def test_script_end_to_end(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pathA_analysis
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_fake_campaign(raw)
    monkeypatch.setattr(sys, "argv", ["pathA_analysis.py", "--raw-root", str(raw), "--out", str(out),
                                      "--instances-from-data"])
    assert pathA_analysis.main() == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["status"] == "ok"
    assert set(s["primary"]) == {"H1/CEC2022", "H1/FIR", "H2a/CEC2022", "H2a/FIR", "H2b/CEC2022",
                                 "H3/CEC2022", "H3/FIR", "H4/CEC2022", "H4/FIR"}
    assert s["primary"]["H1/CEC2022"]["runs"] == 100 and s["primary"]["H1/CEC2022"]["min_pairs"] == 100
    # independent noise for every arm -> no arm should be declared superior in H1
    assert s["primary"]["H1/CEC2022"]["verdict"] in (A.EQUIVALENT, A.INCONCLUSIVE)
    assert s["H5_f10_escape"]["arms"]["PA-SARSA"]["n"] == 100
    assert s["V3_authors_mirror"]["CEC2022"]["n_instances"] == 3
    assert s["V3_authors_mirror"]["FIR"]["n_instances"] == 2
    assert (out / "verdicts.md").exists()


def test_script_refuses_on_integrity_failure(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import pathA_analysis
    raw, out = tmp_path / "raw", tmp_path / "out"
    _write_fake_campaign(raw, {"FIR": [1]}, max_runs=2)
    p = raw / "pathA" / "FIR" / "PA-SARSA" / "inst01" / "run01.json"
    rec = json.loads(p.read_text()); rec["FE"] -= 1; p.write_text(json.dumps(rec))
    monkeypatch.setattr(sys, "argv", ["pathA_analysis.py", "--raw-root", str(raw), "--out", str(out)])
    assert pathA_analysis.main() == 1
    s = json.loads((out / "summary.json").read_text())
    assert "primary" not in s and s["integrity"]["FIR"]["bad_fe"] == 1


def test_completeness_blocks_missing_runs():
    """Missing primary runs (e.g. a failed CI shard) block the confirmatory analysis."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import pathA_analysis
    pa = load_yaml("pathA.yaml")
    rows = [{"method": arm, "instance": 1, "run_id": r}
            for arm in ("PA-SARSA", "PA-RND-FAIR") for r in range(1, 101)]
    df = {"FIR": pd.DataFrame(rows[:-1])}               # PA-RND-FAIR run 100 missing
    missing = pathA_analysis.completeness(df, pa, {"CEC2022": [], "FIR": [1]})
    assert "FIR PA-RND-FAIR inst01: 99/100" in missing
    assert "FIR PA-RND-AUTH inst01: 0/50" in missing     # an arm with no data at all
    assert not any(m.startswith("FIR PA-SARSA") for m in missing)
