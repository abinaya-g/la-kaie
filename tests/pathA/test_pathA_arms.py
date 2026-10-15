"""Path A arms (experiments/MPHBS_DECONSTRUCTION_PREREG.md §10.2).

T-SARSA-UNCHANGED, T-RND-ALIAS, T-RND-FE / T-AUTH-FE / T-SKEL-FE (all arms of
configs/pathA.yaml), T-AUTH-NOOP, T-SYNC, T-INJ-INERT, T-NO-Q.
"""
import numpy as np
import pytest

from lakaie.algorithms import mphb, mphbs, mphbs_variants
from lakaie.core import CountedObjective, Recorder
from lakaie.experiment import load_yaml, pathA_arm, resolve_method, seed_for
from lakaie.registry import iterations_for

BC1 = dict(K=7, Ni=3, rho_sub=0.25, w_best=0.05, gamma=0.3, fad_before_p2=True, fad_after_p2=False)
AC1 = dict(K=3, Ni=9, rho_sub=0.25, w_best=0.05, gamma=0.3, fad_before_p2=False, fad_after_p2=True)
CONFIGS = {"BC1": BC1, "AC1": AC1}
BENCH = {"CEC2022": "cec2022.yaml", "FIR": "fir.yaml"}
SEEDS = [0, 7, 23]

# trajectory fields that do not depend on the FE counter
TRAJ = ["best", "current_mean", "current_median", "diversity", "stagnation", "hba_success",
        "mpa_success", "fad_success", "mediator_accept_rate", "mediator_gbest_improvements"]


def _rastrigin(X):
    Z = X - 0.3
    return np.sum(Z ** 2 - 10 * np.cos(2 * np.pi * Z) + 10, axis=1)


def _run(fn, cfg, seed, T=6, dim=20):
    lb, ub = -5 * np.ones(dim), 5 * np.ones(dim)
    obj = CountedObjective(_rastrigin, None)
    rec = Recorder(lb, ub)
    out = fn(obj, T, lb, ub, dim, 30, np.random.default_rng(seed), dict(cfg), rec)
    return out, rec.as_arrays(), obj.fe


def _assert_same(a, b, fields=TRAJ, same_fe=True):
    (oa, ra, fa), (ob, rb, fb) = a, b
    assert oa["best"] == ob["best"]
    assert np.array_equal(oa["best_x"], ob["best_x"])
    for k in fields:
        assert np.array_equal(ra[k], rb[k], equal_nan=True), k
    if same_fe:
        assert fa == fb and np.array_equal(ra["fe"], rb["fe"])


# ------------------------------------------------------------ equivalence with the frozen port
@pytest.mark.parametrize("name", ["BC1", "AC1"])
@pytest.mark.parametrize("seed", SEEDS)
def test_sarsa_unchanged(name, seed):
    """T-SARSA-UNCHANGED: default flags == frozen mphbs.run, bit for bit."""
    cfg = CONFIGS[name]
    _assert_same(_run(mphbs.run, cfg, seed), _run(mphbs_variants.run, cfg, seed))


@pytest.mark.parametrize("name", ["BC1", "AC1"])
@pytest.mark.parametrize("seed", SEEDS)
def test_random_alias(name, seed):
    """T-RND-ALIAS: mediator=random == random_actions=True == frozen port's random_actions."""
    cfg = CONFIGS[name]
    ref = _run(mphbs.run, {**cfg, "random_actions": True}, seed)
    _assert_same(ref, _run(mphbs_variants.run, {**cfg, "mediator": "random"}, seed))
    _assert_same(ref, _run(mphbs_variants.run, {**cfg, "random_actions": True}, seed))


def test_unknown_mediator_rejected():
    with pytest.raises(ValueError):
        _run(mphbs_variants.run, {**BC1, "mediator": "ucb"}, 0)


# ------------------------------------------------------------ FE budgets of every configured arm
def _all_arms():
    out = []
    for bench, arms in load_yaml("pathA.yaml")["arms"].items():
        for label in arms:
            out.append((bench, label))
    return out


@pytest.mark.parametrize("bench,label", _all_arms())
def test_arm_budget_matches_config(bench, label):
    """T-RND-FE, T-AUTH-FE, T-SKEL-FE: calibrated (T, FE) == configs/pathA.yaml `expected`."""
    bcfg = load_yaml(BENCH[bench])
    family, cfg, expected = pathA_arm(label, bcfg)
    got = iterations_for(family, cfg, bcfg["max_fes"], bcfg["dimension"], bcfg["population_size"])
    assert got == expected


@pytest.mark.parametrize("bench,expected", [
    ("CEC2022", {"PA-SARSA": (1738, 199930), "PA-RND-FAIR": (1738, 199930),
                 "PA-RND-AUTH": (1378, 199870), "PA-SKEL": (2221, 199950)}),
    ("FIR", {"PA-SARSA": (919, 149857), "PA-RND-FAIR": (919, 149857),
             "PA-RND-AUTH": (776, 149828), "PA-SKEL": (1666, 150000)}),
])
def test_primary_budgets_match_authors_files(bench, expected):
    """The SARSA and RANDOM-mirror totals are those of the authors' task_results.csv."""
    arms = load_yaml("pathA.yaml")["arms"][bench]
    for label, te in expected.items():
        assert tuple(arms[label]["expected"]) == te


@pytest.mark.parametrize("cfg,per_iter", [(BC1, 145), (AC1, 193)])
def test_auth_fe_per_iteration(cfg, per_iter):
    """RND-AUTH spends 4N + 1 + E per iteration (MPHBS_random_mirror.m)."""
    _, _, fe = _run(mphbs_variants.run, {**cfg, "mediator": "random", "reeval_mpa_top": True}, 0, T=5)
    assert fe == 60 + 5 * per_iter


@pytest.mark.parametrize("cfg", [BC1, AC1])
def test_skel_has_no_phase2(cfg):
    """T-SKEL-FE: 3N per iteration, no mediator evaluations."""
    out, rec, fe = _run(mphbs_variants.run, {**cfg, "skip_phase2": True}, 0, T=5)
    assert fe == 60 + 5 * 90
    assert np.all(np.isnan(rec["mediator_accept_rate"]))
    assert np.all(rec["mediator_gbest_improvements"] == 0)


# ------------------------------------------------------------ facts F-3 and F-5 of the pre-registration
@pytest.mark.parametrize("seed", SEEDS)
def test_auth_noop_on_AC1(seed):
    """T-AUTH-NOOP (FIR config): the mirror's re-evaluation changes nothing but the FE count."""
    fair = _run(mphbs_variants.run, {**AC1, "mediator": "random"}, seed)
    auth = _run(mphbs_variants.run, {**AC1, "mediator": "random", "reeval_mpa_top": True}, seed)
    _assert_same(fair, auth, same_fe=False)
    assert auth[2] - fair[2] == 6 * 30


@pytest.mark.parametrize("seed", SEEDS)
def test_auth_equals_sync_on_BC1(seed):
    """T-AUTH-NOOP (CEC config): the mirror's block == zero-FE memory synchronisation."""
    sync = _run(mphbs_variants.run, {**BC1, "mediator": "random", "sync_memory": True}, seed)
    auth = _run(mphbs_variants.run, {**BC1, "mediator": "random", "reeval_mpa_top": True}, seed)
    _assert_same(sync, auth, same_fe=False)
    assert auth[2] - sync[2] == 6 * 30


@pytest.mark.parametrize("seed", SEEDS)
def test_sync_is_noop_on_AC1(seed):
    """T-SYNC (FIR config): the MPA memory is already synchronised after the FAD stage."""
    fair = _run(mphbs_variants.run, {**AC1, "mediator": "random"}, seed)
    sync = _run(mphbs_variants.run, {**AC1, "mediator": "random", "sync_memory": True}, seed)
    _assert_same(fair, sync)


def test_sync_changes_BC1():
    """T-SYNC (CEC config): MPHBS's MPA memory is stale after Phase 2, so syncing changes runs."""
    differs = []
    for seed in range(5):
        fair = _run(mphbs_variants.run, {**BC1, "mediator": "random"}, seed, T=15)
        sync = _run(mphbs_variants.run, {**BC1, "mediator": "random", "sync_memory": True}, seed, T=15)
        differs.append(not np.array_equal(fair[1]["current_mean"], sync[1]["current_mean"]))
    assert any(differs)


@pytest.mark.parametrize("name", ["BC1", "AC1"])
@pytest.mark.parametrize("mediator", ["sarsa", "random"])
@pytest.mark.parametrize("seed", SEEDS)
def test_injection_inert(name, mediator, seed):
    """T-INJ-INERT: the global-best injection is overwritten by subset reintegration."""
    cfg = {**CONFIGS[name], "mediator": mediator}
    _assert_same(_run(mphbs_variants.run, cfg, seed),
                 _run(mphbs_variants.run, {**cfg, "inject_gbest": False}, seed))


# ------------------------------------------------------------ remaining flags
@pytest.mark.parametrize("cfg", [BC1, AC1])
def test_random_never_touches_q(cfg):
    """T-NO-Q."""
    rnd, _, _ = _run(mphbs_variants.run, {**cfg, "mediator": "random"}, 0)
    sarsa, _, _ = _run(mphbs_variants.run, cfg, 0)
    assert rnd["summary"]["q_nonzero"] == 0
    assert sarsa["summary"]["q_nonzero"] > 0


def test_greedy_acceptance_changes_runs_and_keeps_budget():
    relaxed = _run(mphbs_variants.run, {**BC1, "mediator": "random"}, 1, T=15)
    greedy = _run(mphbs_variants.run, {**BC1, "mediator": "random", "relaxed_accept": False}, 1, T=15)
    assert relaxed[2] == greedy[2]
    assert not np.array_equal(relaxed[1]["mediator_accept_rate"], greedy[1]["mediator_accept_rate"])


@pytest.mark.parametrize("mod", [mphbs_variants])
def test_determinism(mod):
    cfg = {**BC1, "mediator": "random", "sync_memory": True}
    assert _run(mod.run, cfg, 11)[0]["best"] == _run(mod.run, cfg, 11)[0]["best"]


# ------------------------------------------------------------ wiring
def test_resolve_method_and_seeds():
    bcfg = load_yaml("cec2022.yaml")
    fam, cfg = resolve_method("PA-RND-AUTH", bcfg)
    assert fam == "MPHBS-V" and cfg["mediator"] == "random" and cfg["reeval_mpa_top"] is True
    assert cfg["K"] == 7 and cfg["fad_before_p2"] is True             # B-C1 kept
    fam, cfg = resolve_method("PA-SARSA", load_yaml("fir.yaml"))
    assert fam == "MPHBS" and cfg == load_yaml("fir.yaml")["methods"]["MPHBS"]
    fam, cfg = resolve_method("PA-MPHB", bcfg)
    assert fam == "MPHB" and mphb.run is not None
    with pytest.raises(KeyError):
        resolve_method("PA-RND-SYNC", load_yaml("fir.yaml"))
    assert seed_for("pathA", "CEC2022", 10, 1) == 92010001
    assert seed_for("pathA", "FIR", 8, 100) == 92108100
