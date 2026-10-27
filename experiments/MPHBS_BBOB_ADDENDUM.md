# Path A — exploratory BBOB extension (addendum to the pre-registration §13)

Status: **FROZEN (2026-10-21), committed before any BBOB run.** The campaign workflow runs only on a commit that
contains this file with this status line. Exploratory: no claim of the paper is confirmatory on BBOB.

Purpose: check whether the confirmatory findings on CEC2022 and FIR — H1 (learning brings no benefit at equal budget)
and H3 (the mediator itself matters) — also hold on a broader, standard suite. Decided after the main results were known
(main pre-registration §13 named this extension in advance; its design details are fixed here).

## Design

| item | value |
|---|---|
| suite | COCO BBOB, functions f1–f24, via `coco-experiment==2.8.2` (`cocoex.BareProblem`) |
| dimensions | D = 10 and D = 20 (benchmarks `BBOB-D10`, `BBOB-D20`) |
| domain, budget | [-5, 5]^D; 10,000·D FEs (100,000 / 200,000); N = 30 |
| configuration | MPHBS B-C1 (the authors' configuration for numerical benchmarks): K = 7, N_i = 3, rho = 0.25, w_best = 0.05, gamma = 0.30, FAD before Phase 2 |
| arms | PA-SARSA, PA-RND-FAIR, PA-SKEL (definitions as in the main study) |
| runs | 30 per (function, dimension, arm); run r uses BBOB instance ((r − 1) mod 15) + 1 |
| seeds | campaign `pathA_bbob`, master 93,000,000; offsets BBOB-D10 300,000, BBOB-D20 400,000; seed = master + offset + 1000·function + run; identical across arms (paired) |
| outcome | final error f − f\*(instance); values below 1e-8 (COCO's final target) are set to 1e-8, so solved runs tie |

## Analysis (fixed)

- Contrasts per dimension: **B1** = SARSA vs RND-FAIR (learning), **B3** = SARSA vs SKEL (mediator).
- Same statistics as the main study (pre-registration §5): paired Wilcoxon per function, Holm over the 24 functions
  within (contrast, dimension), paired win probability, W-bar with 95 % bootstrap CI (10,000 resamples, seed 93000001),
  the same verdict rule; median relative difference reported as a descriptive magnitude.
- Descriptive breakdown by BBOB function group: separable f1–f5, low/moderate conditioning f6–f9, high conditioning
  f10–f14, multimodal with adequate global structure f15–f19, multimodal with weak global structure f20–f24.
- Integrity: every run status ok with the expected FE total; no verdict is computed if any (arm, function, dimension)
  cell lacks runs.
- Single attempt: no parameter, arm or analysis change after the campaign starts.
