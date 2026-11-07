# Eigen-frame exchange: confirmatory campaign (design, fixed before any confirmatory run)

Status: FROZEN 2026-10-29 (the commit that adds `scripts/eswa_campaign.sh`), before any confirmatory run.
Launch: `bash scripts/eswa_campaign.sh`. Results: `results/eswa_eigen/<bench>/`; S4 runs go to
`results/eswa_eigen_official/CEC2022-D20/`.

## 1. Background and what is already known

- Path A (`MPHBS_DECONSTRUCTION_PREREG.md`): in the HBA-MPA hybrid with a SARSA mediator
  (Flores & Olivares 2026), the learned mediator gives no benefit over a budget-matched random choice; the
  inter-population exchange (Phase 2) drives the gains on FIR and separable functions and hurts on rotated ones.
- Exploratory pilot (`scripts/eswa_pilot.py`, results in `results/eswa_pilot/`, 10 runs, 2026-10-28):
  - BBOB-D20 f1, f3, f8, f10, f12, f15, f16, f21;
  - CEC2022 D20 F1-F12;
  - FIR 1-8.

  `eigen_prob = 0.5` (EIG50) was chosen from these pilot data. That choice is therefore not confirmatory on the
  piloted instances (Section 4).
- The eigen-coordinate idea is known: Guo & Yang 2015 (eigenvector crossover in DE), Wang et al. 2014 (CoBiDE),
  Tang et al. 2021, 2022 (inter-task transfer in eigen/aligned subspaces), Bujok & Kolenovský 2022 (EA4eig).
  See `paper/novelty_search_2026-10-28_eigen.md`.
- The contribution claimed is the finding chain plus a learning-free, rotation-invariant exchange for this class of
  hybrids. It is not a new operator.

## 2. Arms

| Arm | Family | Flags (on the per-domain MPHBS base configuration) |
|-----|--------|------------------------------------------------------|
| MPHBS | MPHBS (frozen port) | none (SARSA mediator, as published) |
| RND | MPHBS-V | `mediator: random` |
| EIG50 | MPHBS-V | `mediator: random, frame: eigen, eigen_prob: 0.5` (the proposed method) |
| CMA-ES | `lakaie/algorithms/baselines.py` | pycma 4.0.0, IPOP restarts, sigma0 = 0.3 x box width, default lambda |
| L-SHADE | `lakaie/algorithms/baselines.py` | Tanabe & Fukunaga 2014 settings |

- CMA-ES:
  - Each restart starts from a uniform random x0, with the population doubled at each restart.
  - Bounds are handled by pycma's BoundTransform; points are also clipped before evaluation.
- L-SHADE:
  - N_init = 18D, N_min = 4, H = 6, archive rate 2.6, p = 0.11.
  - Linear population-size reduction and midpoint bound repair.
- Both baselines:
  - They use their own population sizes, not N = 30.
  - The last generation is truncated so that FE = MaxFE exactly.
  - They were sanity-checked on CEC2022 D10 before the campaign (`results/eswa_validation/`).
- Runner: `scripts/eswa_campaign.py`.

- Base configurations as in Path A: B-C1 for CEC2022 and BBOB, A-C1 for FIR. N = 30. Bounds are clipped.
- SKEL (no exchange) is not re-run here; its role is covered by Path A.
- EA4eig: no Python implementation is available and MATLAB is not installed, so EA4eig is compared on published
  CEC2022 results instead (Section 5, S3).

## 3. Benchmarks and budget

| Benchmark | Instances | D | MaxFE | Runs |
|-----------|-----------|---|-------|------|
| CEC2022 | F1-F12 | 10 | 200 000 | 30 |
| CEC2022 | F1-F12 | 20 | 200 000 (10 000 D, as in Path A) | 30 |
| BBOB | f1-f24, instance ((r-1) mod 15)+1 | 20 | 200 000 | 30 |
| FIR | cases 1-8 | 31 | 150 000 | 30 |

- Error = F(x) - F*, floored at 1e-8.
- Seeds: master 95000000 + benchmark offset + 1000 * instance + run.
- Runs of different arms with the same (benchmark, instance, run) share the seed. They are paired.

## 4. Confirmatory hypotheses (held-out instances only)

Held-out set H, instances never seen in any pilot:
- CEC2022 D10 F1-F12;
- BBOB-D20 f2, f4, f5, f6, f7, f9, f11, f13, f14, f17, f18, f19, f20, f22, f23, f24.

That is 28 instances. Per instance, the effect is the Vargha-Delaney A = P(EIG50 < X) + 0.5 P(tie) over 30
paired runs.

- **H1 (EIG50 vs RND):** across the 28 held-out instances, the median A exceeds 0.5. Test: one-sided Wilcoxon
  signed-rank of (A - 0.5) over instances.
- **H2 (EIG50 vs MPHBS):** same test with X = MPHBS.
- Family-wise alpha 0.05, Holm over {H1, H2}.
- Per-instance two-sided Wilcoxon signed-rank tests (paired by run) with Holm within each benchmark are reported
  as secondary results. No claim rests on them.

## 5. Secondary analyses (reported, not confirmatory)

- **S1. All instances:** H1 and H2 repeated on all 56 instances, including the piloted ones, labelled as such.
- **S2. Rotation mechanism:** use BBOB pairs that differ only by rotation:
  - f2 vs f10 (ellipsoid);
  - f3 vs f15 (Rastrigin).

  For each arm, report the ratio of median errors (rotated / separable). Prediction: the ratio is smaller for
  EIG50 than for RND.
- **S3. Standard optimisers:** Friedman ranks and per-instance tables against CMA-ES and L-SHADE on all four
  benchmarks.
- **S4. EA4eig:** compare with EA4eig on CEC2022 D10 and D20 at the official CEC2022 budgets (200 000 for D10,
  1 000 000 for D20). Run EIG50 only, 30 runs, using the EA4eig results published for the CEC2022 competition.
  This is descriptive only, because hardware and implementation differ.
- **S5. Cost:** wall-clock overhead of the eigendecomposition per iteration, and FE identity (no extra
  evaluations).

## 6. Theory to include in the paper

- **Proposition (rotation invariance of the exchange):** for an orthogonal Q, the eigen-frame exchange applied to
  populations XQ yields (exchange of X)Q.
  - This assumes distinct eigenvalues and no bound clipping.
  - Eigenvector sign flips commute with the coordinate swap.
- **Proposition (no extra cost):** the exchange uses the same number of FEs as the axis exchange. The added cost is
  O(N D^2 + D^3) arithmetic per iteration.
- A unit test checks the first proposition numerically.

## 7. Compute estimate (from pilot timings, 11 workers)

- Each MPHBS-family arm needs about 48 CPU-hours over the four benchmarks:
  - BBOB-D20: about 96 s per run;
  - CEC2022 D20: about 91 s per run;
  - FIR: about 229 s per run.
- Totals:
  - MPHBS, RND and EIG50: about 13 h wall-clock;
  - CMA-ES and L-SHADE: about 8-9 h;
  - S4 at the official budget: about 4 h;
  - overall: about 26 h.
- Results are written per run to `results/eswa_eigen/`. A restart skips finished runs.

## 8. Deviations

Any change after freezing is logged here with date and reason.

- 2026-11-03: the design did not say how the across-instance Wilcoxon treats instances with A = 0.5 exactly
  (both arms tie on every run). The confirmatory test uses the scipy default (`zero_method="wilcox"`, zeros
  dropped), chosen before looking at its result. Sensitivity with `pratt` and `zsplit` is reported alongside:
  under those, H1 and H2 do not survive Holm (smallest p 0.028, 0.029; Holm-adjusted 0.056, 0.057).
- 2026-11-03: the "median A exceeds 0.5" condition uses the exact median over the 28 instances.
- 2026-11-03: the campaign was paused after the held-out benchmarks finished (CEC2022 D20 at 227/1800) to free
  the machine; it resumes with `bash scripts/eswa_campaign.sh`.
- 2026-11-04: the campaign finished (8 760/8 760 runs, all status ok, FE within budget).
- 2026-11-07: S2, the f10/f2 ratio is degenerate because every arm reaches the 1e-8 floor on f2; the rotated
  medians are reported alongside the ratio. S4 uses the official EA4eig entry files (30 runs per function, from
  `2022-Full-Download.zip` in P-N-Suganthan/2022-SO-BO, kept in `data/external/cec2022_ea4eig/`), compared
  unpaired (Vargha-Delaney A, Mann-Whitney U, Holm within each D). Secondary analyses
  are in `scripts/eswa_eigen_secondary.py`.

## 9. Addendum (2026-11-07, after the main campaign and its analysis): common metaheuristic baselines

Added after the confirmatory and secondary results were known, to report the comparisons that papers in this
area usually show. Descriptive only: no hypothesis is added and H1/H2 are unchanged.

- Arms: HBA and MPA (the parent algorithms, ports in `lakaie/algorithms/`), PSO (inertia 0.9 -> 0.4, c1 = c2 = 2,
  |v| <= 0.2 range), DE/rand/1/bin (F = 0.5, CR = 0.9), GWO (Mirjalili et al. 2014), WOA (Mirjalili & Lewis
  2016). N = 30 for all, as for the hybrids. Implementations in `lakaie/algorithms/baselines.py`.
- GWO and WOA have a documented bias towards the origin; the unit tests show it on a shifted sphere. CEC2022 and
  BBOB optima are shifted, so this is part of their real behaviour, not a bug.
- Same benchmarks, budgets, 30 runs and seeds as Section 3 (runs are paired with the main arms).
  Launch: `bash scripts/eswa_campaign_extra.sh`.
- Reporting: the five main arms and these six together in the S3 tables and Friedman ranks. L-SHADE, CMA-ES and
  EA4eig stay in every table as the state-of-the-art references.
