# Path A — Does learning matter in MPHBS? Pre-registered deconstruction study

Status: **FROZEN (v1.0, 2026-10-15; §12).** The §10 code and tests are
implemented; **no campaign run had been started at freezing.** The freeze
commit is tagged `pathA-prereg-frozen`; the campaign workflow re-runs the full
test suite (V1) in the pinned environment before the first run, and records
the commit hash in every config snapshot (`git_commit`). Any later change to
this file is a deviation (§9).

Companion files: `reference/MPHBS_NOTES.md`, `MPHBS_REPRODUCTION_REPORT.md`
(Stage 0 and mediator audit), `results/SHE_STAGE1_FOLLOWUP.md` §(c),
`third_party/mphbs_reference/code/` (authors' MATLAB, commit `b3e96ca`).

---

## 0. Why this study, in one paragraph

MPHBS (Flores & Olivares, KBS 352 (2026) 117070) attributes its gains to a
SARSA-guided, dimension-wise information-exchange mediator. Its key evidence
that *learning* matters is the comparison with a "matched RANDOM mirror"
(`MPHBS_random_mirror.m`). Reading the authors' code shows that **the mirror
is not budget-matched in iterations**: it re-evaluates the MPA population at
the top of every iteration (lines 58–66), which `MPHBS_main.m` does not do.
At a fixed FE budget the mirror therefore runs **21 % fewer iterations on
CEC2022 and 16 % fewer on FIR** (§1, F-2). On CEC2022 the same block also
re-synchronises the MPA memory with the Phase-2 results, which MPHBS never
does (F-3). This study separates *learning*, *the exchange itself*, *the
evaluation budget*, *the memory synchronisation* and *the loop structure*
with budget-identical controls. Every outcome is reportable (§8). No result of
this study can "STOP" it; only integrity failures can (§9).

## 1. Facts established before pre-registration (from existing code and data)

These are inputs, not results of this study. Each is re-verified by a unit
test in §10 before any run.

| # | fact | source |
|---|---|---|
| F-1 | `MPHBS_main.m` spends **3N + 1 + E** FEs per iteration (HBA N, MPA N, evaluated FAD stage N, f(P_best) 1, E episodes). B-C1: 115/iter, T = 1738, total 199,930. A-C1: 163/iter, T = 919, total 149,857. | `MPHBS_NOTES.md` §14–15; exact match to the paper's Table 2 |
| F-2 | `MPHBS_random_mirror.m` additionally re-evaluates the whole MPA population at the start of each iteration (lines 58–66), so it spends **4N + 1 + E**: 145/iter on CEC (T = 1378, total 199,870) and 193/iter on FIR (T = 776, total 149,828). | authors' `task_results.csv`: BR-C1 = 199,870 FEs, AR-C1 = 149,828 FEs; 60 + 145·1378 = 199,870 and 60 + 193·776 = 149,828 exactly |
| F-3 | The re-evaluation itself returns known values (positions unchanged, deterministic objective, no random numbers drawn), and the memory-saving step it triggers never reverts a row (Phase 2 only ever improves rows). Its side effect is the update `fit_old ← F_MPA`, `X_MPA_old ← X_MPA`. **FIR (A-C1):** MPHBS already sets the memory after the post-Phase-2 FAD stage (`MPHBS_main.m` l. 497), so the block is a pure no-op that costs N FEs. **CEC (B-C1):** MPHBS last sets the memory *before* Phase 2 (l. 252) and never after it, so its next memory-saving step (l. 238) compares the moved MPA agents against stale pre-Phase-2 memory and can revert them to pre-Phase-2 positions, discarding Phase-2 improvements. The mirror's block removes that staleness. On CEC the mirror therefore differs in **two** ways: fewer iterations, and a synchronised memory. | code reading (l. 238–253, 493–498 of `MPHBS_main.m`; port `hybrid_common.py` `fads_greedy` / `memory_saving`); **confirmed in the port** by T-AUTH-NOOP and T-SYNC (bit-identical trajectories, 3 seeds each; staleness changes runs on B-C1) |
| F-4 | Apart from F-2, the mirror differs from MPHBS only in (n, ch) being drawn uniformly at random for all K candidates and no Q update. The Q-table and `ClosestAction` bookkeeping that the mirror drops have no behavioural effect once Q is never updated and no RNG is consumed by them. | diff of the two `.m` files; mediator audit 2026-09-29 |
| F-5 | The global-best injection inside the episode loop (MATLAB lines ~455–462; port `mphbs.py:170–173`) writes into subset rows that are **overwritten by subset reintegration** at the end of the same Phase 2. It is inert except for the two random draws it consumes. The mirror's extra `Best_F_HBA/MPA` updates (lines 218–219) are likewise overwritten by the recomputation at lines 226–227. | mediator audit §1; **confirmed in the port** by T-INJ-INERT (SARSA and random, B-C1 and A-C1, 3 seeds each) |
| F-6 | Our port's `random_actions: true` flag (`mphbs.py:72`) implements F-4 **without** F-2, i.e. it is already a budget-identical random mediator (115 / 163 FEs/iter). | `mphbs.py`; `lakaie/registry.py` T-calibration |
| F-7 | Authors' own data (our re-analysis: paired two-sided Wilcoxon, Holm over instances within domain; may differ slightly from the paper's joint Holm). **CEC2022** SARSA (BS-C1) vs RANDOM (BR-C1): SARSA better on 7/12 instances (F3, F5, F6, F7, F8, F10, F12), worse on F1; run-level 268 wins / 92 losses. **FIR** AS-C1 vs AR-C1: 0/8 significant; 134 / 106. F10 escapes (final error < 50): SARSA 17/30, RANDOM 4/30, MPHB 0/30. | `third_party/mphbs_reference/results/*_ablation_*` |
| F-8 | MPHBS also differs from MPHB in its **loop**, not only by the mediator: MPHBS evaluates the FAD stage in place with greedy acceptance (discrepancy D1), MPHB does not (D3). MPHBS > MPHB on FIR 8/8 therefore confounds mediator and loop. | `MPHBS_NOTES.md` §3, §19 |
| F-9 | Exploratory, not reused: on the `n1_unit` seeds our port escaped F10 in 10/30 runs with SARSA vs 14/30 with `random_actions` (budget-identical). | `results/reproduction/MPHBS/diagnostic_F10/` |

## 2. Research questions and hypotheses

Domains: CEC2022 (D = 20, 12 functions, 200,000 FEs, B-C1) and FIR (D = 31,
8 cases, 150,000 FEs, A-C1). Each hypothesis is tested **per domain**.

| id | question | contrast | prediction (stated in advance; a wrong prediction is still a result) |
|---|---|---|---|
| **H1** (primary) | Does the SARSA *learning* add value over a budget-identical random mediator? | SARSA vs RND-FAIR | CEC: no superiority (equivalent or inconclusive). FIR: no superiority. |
| **H2a** (primary) | Is the authors' RANDOM mirror handicapped by its budget? | RND-SYNC vs RND-AUTH on CEC (same memory, 1738 vs 1378 iterations); RND-FAIR vs RND-AUTH on FIR (identical behaviour per iteration, 919 vs 776 iterations) | the budget-identical arm is better on several CEC instances; smaller or no effect on FIR. |
| **H2b** (primary, CEC only) | Does MPHBS's stale MPA memory matter? | RND-FAIR vs RND-SYNC | no directional prediction. |
| **H3** (primary) | Does the mediator (any policy) add value over the same loop without Phase 2, at equal FEs? | SARSA vs SKEL | FIR: yes. CEC: no clear prediction. |
| **H4** (primary) | Does MPHBS's loop restructuring (evaluated greedy FAD) explain part of MPHBS > MPHB? | SKEL vs MPHB | SKEL better than MPHB on FIR. |
| H5 (secondary) | F10 escape: is the authors' 17/30 typical for SARSA, and is escape driven by learning? | escape rates, 100 seeds | SARSA rate well below 57 %, not above RND-FAIR. |
| H6 (secondary) | Dose–response: does the exchange budget E matter more than the policy? | N_i ∈ {1, 3, 9} × {SARSA, RND-FAIR} | performance changes with N_i more than with policy. |
| H7 (secondary) | Which mediator components carry the effect? | RND-K1 (no proxy screening), RND-GreedyAcc (no relaxed acceptance) vs RND-FAIR | descriptive; no directional prediction. |
| H8 (secondary, CEC) | Does a zero-cost memory synchronisation improve MPHBS itself? | SARSA-SYNC vs SARSA | no directional prediction. |

## 3. Arms

All arms use the authors' parameters for the domain (B-C1 on CEC2022, A-C1 on
FIR; `configs/cec2022.yaml`, `configs/fir.yaml`), N = 30, and T calibrated
from the FE budget by the existing FE-probe (`lakaie/core.py:calibrate_iterations`).

### 3.1 Primary arms

| arm | definition | FE/iter CEC / FIR | expected T CEC / FIR | code |
|---|---|---|---|---|
| **SARSA** (`PA-SARSA`) | unchanged MPHBS port | 115 / 163 | 1738 / 919 | frozen `mphbs.py` (validated, Stage 0) |
| **RND-FAIR** (`PA-RND-FAIR`) | MPHBS with every component's (n, ch) drawn uniformly for all K candidates and no Q update (= F-4 without F-2) | 115 / 163 | 1738 / 919 | `mphbs_variants.py`, `mediator: random` |
| **RND-SYNC** (`PA-RND-SYNC`, CEC only) | RND-FAIR plus, at the end of each iteration, `fit_old ← F_MPA`, `X_MPA_old ← X_MPA` at **zero FEs** (fitness values are already known). Reproduces the mirror's memory behaviour without its FE cost. On FIR it is identical to RND-FAIR by construction (F-3; T-SYNC), so it is not run there | 115 / – | 1738 / – | `mphbs_variants.py`, `sync_memory: true` |
| **RND-AUTH** (`PA-RND-AUTH`) | faithful port of `MPHBS_random_mirror.m`, **including** the start-of-iteration MPA re-evaluation and memory saving | 145 / 193 | 1378 / 776 | `mphbs_variants.py`, `reeval_mpa_top: true` |
| **SKEL** (`PA-SKEL`) | MPHBS loop with Phase 2 skipped (no f(P_best) evaluation, no episodes); FAD placement as in the domain config | 90 / 90 | 2221 / 1666 | `mphbs_variants.py`, `skip_phase2: true` |
| **MPHB** (`PA-MPHB`) | unchanged MPHB port (no mediator, unevaluated FAD) | 90 / 90 | 2221 / 1666 | frozen `mphb.py` (validated) |

All (T, FE) values, including those of the secondary arms, are the FE-probe
calibrations stored as `expected` in `configs/pathA.yaml` and checked by
`test_arm_budget_matches_config`. Secondary N_i arms: CEC N_i = 1 → (2019,
199,941), N_i = 9 → (1226, 199,898); FIR N_i = 1 → (1514, 149,946), N_i = 3 →
(1303, 149,905).

### 3.2 Secondary arms

| arm | definition |
|---|---|
| SARSA-Ni{1,3,9} | SARSA with N_i ∈ {1, 3, 9} (E = 8, 24, 72); the domain default is one of them and is the primary SARSA arm |
| RND-FAIR-Ni{1,3,9} | same for RND-FAIR |
| RND-K1 | RND-FAIR with K = 1 (no proxy screening: one random candidate per episode) |
| RND-GreedyAcc | RND-FAIR with the relaxed acceptance probability set to 0 (accept only improvements) |
| SARSA-SYNC (CEC only) | SARSA with the zero-FE memory synchronisation of RND-SYNC: does removing the stale memory help MPHBS itself? (H8, exploratory) |

LA-KAIE variants (FixedPolicy etc.) are **excluded**: they use a different
framework and FAD placement, and would confound every contrast.

### 3.3 The exact random mediator (RND-FAIR), normative definition

Identical to `MPHBS_main.m` Phase 2 in every line except:

1. For each episode, each of the K candidates is assembled by drawing, for
   each active dimension c in the current block, `ch ~ U{HBA, MPA}` then
   `n ~ U{1..N_sub}` (this order, as in `MPHBS_random_mirror.m` line 174),
   and setting `v[c] = result[n, c, ch]`.
2. The proxy score (rank, novelty, distance to P_best; `w_rank = 1.00`,
   `w_novel = 0.30`, `w_best` from the config) selects one of the K
   candidates, exactly as in MPHBS.
3. The Q-table is never read or written. `ClosestAction` is not called.
4. Everything else is unchanged: block schedule of active dimensions, the
   initial f(P_best) evaluation, relaxed acceptance with `min(p, 0.25)`,
   subset memory replacement, global-best injection, reintegration,
   synchronisation, FAD placement.
5. FE accounting is identical to SARSA: 3N + 1 + E per iteration.

RND-AUTH = RND-FAIR + the start-of-iteration block of
`MPHBS_random_mirror.m` lines 57–66 (bound repair, evaluate MPA population,
memory saving, `fit_old` / `X_MPA_old` update).

RND-SYNC = RND-FAIR + the `fit_old` / `X_MPA_old` update of that block,
performed at the end of each iteration without evaluating anything.

## 4. Benchmarks, outcome, seeds and run counts

* **Instances:** CEC2022 F1–F12 at D = 20 (official C code, validated);
  FIR cases 1–8 at D = 31 (validated). No other benchmark is confirmatory.
* **Outcome:** final error (CEC: f − f*; FIR: objective value) at the end of
  the budget. Two values within 1e-12 are a tie (the authors' tolerance).
* **Anytime outcome (secondary):** best-so-far error at 25 %, 50 % and 75 % of
  the budget, from the recorder's FE checkpoints.
* **Seeds:** new campaign `pathA`, master **92000000** (disjoint from all
  earlier masters in `configs/seeds.json` and `configs/reproducibility.yaml`).
  `seed = 92000000 + benchmark_offset + 1000·instance + run`, offsets as now
  (CEC 0, FIR 100000). The same seed is used for every arm on a given
  (instance, run): all comparisons are paired.
* **Run counts:**

  | arms | runs per instance |
  |---|---|
  | SARSA, RND-FAIR (the H1 pair) | **100** (runs 1–100) |
  | RND-AUTH, SKEL, MPHB; RND-SYNC (CEC only) | 50 (runs 1–50) |
  | RND-AUTH, SKEL on CEC F10 only | 100 (adds runs 51–100, for H5) |
  | all secondary arms | 50 (runs 1–50) |

  Pairing uses the common run indices: H1 uses runs 1–100; H2a, H2b, H3, H4 and
  H6–H8 use runs 1–50.

* **Sample-size rationale (H1).** The effect size is the mean paired win
  probability W̄ (§5). Its standard error is at most √(0.25 / (n·k)). With
  n = 100: CEC (k = 12) SE ≤ 0.014 (95 % CI half-width ≈ 0.028); FIR (k = 8)
  SE ≤ 0.018 (≈ 0.035). If the true W̄ is 0.5, the probability that the CI
  falls inside the equivalence band [0.44, 0.56] is ≈ 0.97 on CEC and
  ≈ 0.84 on FIR (conservative bound). With n = 50 it would be ≈ 0.67 and
  ≈ 0.34, too low for an equivalence claim, hence 100 runs for the H1 pair.

* **Compute estimate** (port: ~15 s per CEC run, ~18 s per FIR run on the
  current 4-CPU machine): primary ≈ 7,700 runs ≈ 34 CPU-h; secondary ≈ 6,600
  runs ≈ 30 CPU-h; total ≈ 16 h wall-clock on 4 workers. The runner is
  resumable; a stopped session continues the same single attempt (as in
  SHE-v2).
* **Execution environment:** GitHub-hosted Ubuntu runners, Python 3.11 with
  the pinned `requirements.txt` (the environment of all earlier campaigns),
  via `.github/workflows/pathA-campaign.yml` (started by hand). The workflow
  refuses to start while §12 is blank, runs the full test suite (V1) first,
  then runs one job per (benchmark, arm) with 4 workers (24 jobs, each under
  the 6 h job limit), and finally one analysis job on the merged raw results.
  Because every run is seeded independently, the split into jobs does not
  change any result. A failed job is re-run as a whole; the analysis refuses
  to compute verdicts while any primary (arm, instance) cell lacks runs.
  Raw results and the analysis are kept as workflow artifacts (90 days) and
  must be downloaded and committed to `results/raw/pathA/` and
  `results/analysis/pathA/`.

## 5. Statistical analysis (frozen)

All analysis code is written and committed **before** the campaign starts
and run unchanged afterwards (`scripts/pathA_analysis.py`).

### 5.1 Per instance

* Two-sided paired Wilcoxon signed-rank test, using the existing
  `lakaie/analysis/stats.py:wilcoxon_p` (zero_method "wilcox", normal
  approximation, no continuity correction).
* Paired win probability ŵ = (wins + 0.5·ties) / n, from the anchor's view
  (anchor = first arm of the contrast; a tie is |difference| ≤ 1e-12).
* Direction of a Holm-significant instance: it favours the anchor if ŵ > 0.5,
  the comparator if ŵ < 0.5.
* Median and IQR per arm.

### 5.2 Multiplicity

* **Family** = (contrast, domain): 12 tests on CEC, 8 on FIR. Holm at
  α = 0.05 within each family.
* Primary contrasts H1, H2a, H2b, H3, H4 are confirmatory. Secondary H5–H8 are reported with
  the same Holm procedure but labelled exploratory; no claim rests on them.

### 5.3 Domain-level effect

* W̄ = mean of ŵ over the instances of the domain.
* 95 % CI by paired bootstrap: resample run indices with replacement
  independently within each instance, recompute W̄; 10,000 resamples,
  percentile interval, bootstrap seed 92000001.

### 5.4 Decision rule for each primary contrast (applied mechanically)

Conditions are checked in this order; the first that holds is the verdict
(`lakaie/analysis/pathA.py:verdict`).

| verdict | condition |
|---|---|
| **mixed** | Holm-significant instances in both directions |
| **anchor superior** | ≥ 1 Holm-significant instance favouring the anchor **and** CI(W̄) lower bound > 0.5 |
| **comparator superior** | the mirror image |
| **practically equivalent** | CI(W̄) ⊂ [0.44, 0.56] (closed band) and no Holm-significant instance in either direction |
| **inconclusive** | none of the above |

If any run fails the §6 integrity checks, the script writes the integrity
report only and computes no verdict.

The ±0.06 band follows the Vargha–Delaney threshold for a negligible effect
(A < 0.56). Instance-level results are always reported in full, including
instances that disagree with the domain verdict.

### 5.5 H5 (F10 escape)

* Escape := final error < 50 (pre-registered here; F10 runs are bimodal,
  with basins at ≈ 0.3–20 and ≈ 100.3–100.6, `MPHBS_REPRODUCTION_REPORT.md`).
* Escape rate per arm with a Wilson 95 % CI (100 seeds each for SARSA,
  RND-FAIR, RND-AUTH, SKEL).
* Paired comparison SARSA vs RND-FAIR: McNemar exact test on the 100 seeds.
* Consistency with the authors: exact binomial test of our SARSA escape
  count against p = 17/30, reported alongside the CI. Not a claim about the
  authors' code; RNG streams differ (F-1, distributional reproduction).

### 5.6 Descriptive only

Friedman test and Kendall's W over the five primary arms on the per-instance
means (the authors' analysis, for comparability); convergence curves; FE
breakdown per phase; mediator accept rate and global-best improvements per
iteration (already recorded by `mphbs.py`).

## 6. Validation gates (integrity only)

These check that the arms are what §3 says. They are run before the
campaign, and they are the **only** conditions that stop it.

| gate | check | on failure |
|---|---|---|
| V1 | full test suite passes, including the new tests of §10 | fix the code, re-run V1; no data collected yet |
| V2 | FE totals per run equal §3 exactly (199,930 / 149,857 for SARSA and RND-FAIR; 199,870 / 149,828 for RND-AUTH) | fix, then repeat |
| V3 | RND-AUTH vs the authors' published BR-C1 / AR-C1 runs: two-sided Mann–Whitney per instance, Holm within domain (the Stage 0 P4 procedure) | **report, do not stop.** A failing instance limits only the H2a statement "RND-AUTH reproduces the authors' mirror" for that instance (as with F10 in Stage 0) |
| V4 | runs complete with status ok; no NaN/Inf | re-run that task with the same seed (resumable runner); a reproducible crash is a bug → fix, re-run that arm in full |

V3 uses the authors' 30 published runs; it runs on our runs 1–30 of RND-AUTH.

## 7. What is explicitly **not** done

* No tuning of any parameter for any arm. B-C1 / A-C1 were selected by the
  authors with SARSA on these same benchmarks; this favours SARSA and makes
  H1 conservative.
* No arm is added, removed or redefined after the first campaign run.
* No interim look at outcome data. The FE/integrity checks of §6 do not read
  objective values.
* No benchmark beyond CEC2022 D = 20 and FIR D = 31 is confirmatory.

## 8. How each result pattern will be reported

| H1 verdict | what the paper says |
|---|---|
| SARSA superior | independent confirmation that learning helps, with a corrected control; the budget confound (F-2) is reported as a methodological note |
| practically equivalent | the learned mediator's contribution is not distinguishable from random assembly at equal budget; gains are attributed via H3/H4/H6 |
| comparator (random) superior | learning is harmful at these settings; analysed via H6/H7 |
| inconclusive / mixed | reported as such, with instance-level detail and the realised CI |

In every case H2a/H2b quantify how much of the authors' SARSA-vs-RANDOM gap
is attributable to the budget difference and to the memory synchronisation,
and H3/H4 separate the mediator from the loop restructuring. The study is a valid contribution whatever H1 shows.

## 9. Deviations

Any deviation from this document after freezing is logged in
`logs/DEVELOPMENT_CHANGES.md` with date, reason and whether it was decided
before or after seeing outcome data, and is reported in the paper. Analyses
not listed here are labelled post hoc.

## 10. Implementation tasks and required tests

### 10.1 Code changes (implemented; default behaviour of every existing method unchanged)

The validated `mphbs.py` is hash-frozen (`tests/n1/test_n1_backbone_frozen.py`),
so it is **not** edited. The arms live in a flag-controlled copy whose
defaults are proven bit-identical to it (T-SARSA-UNCHANGED).

| file | change |
|---|---|
| `lakaie/algorithms/mphbs_variants.py` (new) | copy of `mphbs.run` with flags `mediator: sarsa \| random` (`random_actions` accepted as alias), `skip_phase2`, `relaxed_accept`, `sync_memory`, `reeval_mpa_top` (RND-AUTH), `inject_gbest` (test hook only); returns `summary.q_nonzero` and the flags |
| `lakaie/registry.py` | `MPHBS-V` family under `ITERATIVE`, so T is calibrated by the FE probe |
| `lakaie/experiment.py` | `pathA_arm()`; `resolve_method` resolves `PA-*` labels from `configs/pathA.yaml` |
| `configs/pathA.yaml` (new) | arms per domain (family, base config, flags, expected (T, FE)), contrasts, decision constants, H5 and V3 settings |
| `configs/seeds.json`, `configs/reproducibility.yaml` | master `pathA: 92000000` |
| `lakaie/analysis/pathA.py` (new) | §5 functions (win scores, bootstrap, verdict, Wilson, McNemar, V3, integrity) |
| `scripts/pathA_analysis.py` (new) | applies §5 and §6 to the raw results → `results/analysis/pathA/{summary.json, verdicts.md, tables/}` |
| `scripts/pathA_campaign.sh` (new) | the §4 campaign commands (not run) |

### 10.2 Tests (`tests/pathA/`; must pass before V1)

| test | asserts | test function(s) |
|---|---|---|
| T-SARSA-UNCHANGED | `mphbs_variants` with default flags == frozen `mphbs.run`, bit for bit (best, best_x, every recorded trajectory field, FE), B-C1 and A-C1, 3 seeds | `test_sarsa_unchanged` |
| T-RND-ALIAS | `mediator: random` == `random_actions: true` == the frozen port's `random_actions` | `test_random_alias` |
| T-RND-FE / T-AUTH-FE / T-SKEL-FE | every arm's calibrated (T, FE) equals `expected`; primary values equal the authors' files; RND-AUTH 4N+1+E per iteration; SKEL 3N and no mediator activity | `test_arm_budget_matches_config`, `test_primary_budgets_match_authors_files`, `test_auth_fe_per_iteration`, `test_skel_has_no_phase2` |
| T-AUTH-NOOP | for a fixed T, RND-AUTH gives **bit-identical** trajectories to RND-FAIR on A-C1 and to RND-SYNC on B-C1; only the FE counter differs (by N per iteration). Confirms F-3 | `test_auth_noop_on_AC1`, `test_auth_equals_sync_on_BC1` |
| T-SYNC | on A-C1, `sync_memory` changes nothing; on B-C1 it changes the trajectory for at least one seed | `test_sync_is_noop_on_AC1`, `test_sync_changes_BC1` |
| T-INJ-INERT | drawing the two injection indices without writing gives bit-identical runs. Confirms F-5 | `test_injection_inert` |
| T-NO-Q | the random mediator leaves the Q-table all zeros (SARSA does not) | `test_random_never_touches_q` |
| analysis | decision rule incl. precedence and band edges; ties; bootstrap; Wilson; McNemar; run limits; integrity; the script end to end on synthetic raw results; the script computes no verdict after an integrity failure | `tests/pathA/test_pathA_analysis.py` |

Status on 2026-10-15: all `tests/pathA` tests pass on Windows, Python
3.10.11, NumPy 2.2.6, SciPy 1.15.3, pandas 2.2.3 (not the pinned
environment). They must be re-run in the pinned campaign environment
(`requirements.txt`, Linux) as part of V1 before freezing.

If T-AUTH-NOOP, T-SYNC or T-INJ-INERT fails, F-3 / F-5 are wrong as stated. The
arms remain well defined, so the study proceeds, but §1 is corrected before
freezing.

## 11. Threats to validity (stated in advance)

* **RNG:** NumPy PCG64 vs MATLAB Mersenne Twister; all agreement with the
  authors is distributional. The unexplained F10 escape-rate difference in
  Stage 0 remains open; H5 addresses it with more seeds.
* **Scope:** two domains, one dimension each, one population size, the
  authors' final configurations only.
* **Configuration bias:** the configurations were selected for SARSA (§7).
* **Instance heterogeneity:** W̄ averages over instances of different
  difficulty; instance-level results are always shown.
* **FIR objective scale:** values differ by orders of magnitude across cases;
  all tests are rank-based and per instance, so scale does not enter.

## 12. Freeze record (to be filled when frozen)

* Frozen by commit: the commit that fills in this section. The campaign
  workflow records it as `git_commit` in every config snapshot; tag it
  `pathA-prereg-frozen`.
* Date (UTC): 2026-10-15
* Approved by: Abinaya G
* Changes from v0.1 draft: v0.2 moved the arms into `mphbs_variants.py`
  (the validated `mphbs.py` stays hash-frozen); added the RND-SYNC arm and
  split H2 into H2a/H2b after finding that the authors' mirror also
  resynchronises the MPA memory on B-C1 (F-3); added H8 (SARSA-SYNC);
  replaced the secondary N_i budgets with FE-probe calibrations; fixed the
  verdict precedence (mixed first) and the direction rule (ŵ); added the
  completeness check and GitHub Actions execution (§4). No outcome data
  existed at any point before freezing.

## 13. Optional exploratory extension (not confirmatory, run only after §5 is complete)

* COCO/BBOB (24 functions, D ∈ {10, 20}) with SARSA, RND-FAIR and SKEL, to
  check whether the H1 verdict generalises beyond CEC2022 and FIR.
* CEC2022 D = 10 (the earlier tuning set) as a second dimension.
