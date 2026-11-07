# EA4eig official CEC 2022 results

- Source: `2022-Full-Download.zip` from https://github.com/P-N-Suganthan/2022-SO-BO (downloaded 2026-11-07),
  folder `Results_and_Ranking/CEC2080/Results/` (the EA4eig entry, Bujok & Kolenovsky, IEEE CEC 2022).
- Files `EA4eigN100_10_<F>_<D>.txt`: 17 rows x 30 runs. Rows 1-16 are the error at the 16 CEC 2022 recording
  points, row 16 is the error at the end of the budget (200 000 FE for D = 10, 1 000 000 for D = 20); row 17 is
  the FE count at which the 1e-8 threshold was reached (the budget if never).
- `Basics_*`: the entrant's summary (function, best, worst, median, mean, std). `load_data.m`: the organisers' loader.
