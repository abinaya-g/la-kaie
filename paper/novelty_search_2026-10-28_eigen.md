# Novelty search log: eigen-frame exchange (Scopus, 2026-10-28)

Question: has an exchange/recombination step *between populations* (hybrid, multi-population, cooperative or
multitask) been carried out in an eigenvector/covariance coordinate system built from the populations?

Database: Scopus (institutional sign-in), TITLE-ABS-KEY, screened on title, then abstract.

| # | Query | Hits | Relevant |
|---|-------|------|----------|
| 1 | `(eigen* OR "covariance matrix" OR "rotation*invariant" OR "coordinate system") AND ("multi-population" OR multipopulation OR "two populations" OR "hybrid metaheuristic*" OR "hybrid algorithm" OR "co-evolution*" OR coevolution*) AND (crossover OR exchange OR migration OR recombination OR "information sharing")` | 30 | Ghosh et al. 2012 (differential CMA-EA); Du et al. (multi-population covariance learning DE) |
| 2 | `(eigenvector* OR "eigen coordinate" OR "covariance matrix learning" OR "coordinate system") AND ("multi-population" OR ... OR subpopulation* OR hybrid*) AND ("differential evolution" OR metaheuristic* OR swarm OR "evolutionary algorithm*")` | 61 | Du et al. 2019 (multiscale cooperative DE); Dong et al. 2016 (hybrid EDA/DE); QLRUN (below) |
| 3 | `("covariance matrix learning" OR "eigenvector-based crossover" OR "eigen crossover" OR "eigen coordinate*" OR "rotation* invariant crossover") AND (hybrid* OR "multi-population" OR ... OR cooperat* OR "information exchange" OR "knowledge transfer")` | 18 | **Bujok & Kolenovský 2022** (EA4eig); **Adam et al. 2027** (QLRUN); Civicioglu & Besdok 2018 (A+, ESWA); Lin et al. 2023 (ESWA, adaptive coordinate systems + knowledge transfer, bilevel) |
| 4 | `(eigen* OR "covariance matrix learning" OR "principal component*" OR "subspace alignment" OR "aligned subspace") AND ("information exchange" OR "knowledge transfer" OR migration OR "dimension* exchange" OR "solution exchange") AND ("multi-population" OR ... OR multitask*) AND (metaheuristic* OR evolutionary OR swarm OR "differential evolution")` | 30 | **Tang et al. 2021** (TEVC 25(2):262-276); **Tang et al. 2022** (TCYB 52(7):6745-6758); Liang et al. 2022 (TCYB 52(4):2096-2109) |

## Closest prior work

- **Bujok, P., Kolenovský, P. (2022).** Eigen crossover in cooperative model of evolutionary algorithms applied to
  CEC 2022 single objective numerical optimisation. IEEE CEC 2022 (158 citations). EA4eig: four cooperating EAs
  (CMA-ES, CoBiDE, jSO, IDE), Eigen crossover applied in each. Cooperative/hybrid + eigen crossover; a strong
  CEC2022 baseline.
- **Tang, Z., Gong, M., Wu, Y., Qin, A.K., Tan, K.C. (2022).** A multifactorial optimization framework based on
  adaptive intertask coordinate system. IEEE TCYB. Inter-population (inter-task) knowledge transfer performed in a
  coordinate system built from the major eigenvectors of both populations' search spaces; motivated explicitly by
  "the searching ability ... strongly relies on the coordinate system".
- **Tang, Z., Gong, M., Wu, Y., Liu, W., Xie, Y. (2021).** Regularized evolutionary multitask optimization: learning
  to intertask transfer in aligned subspace. IEEE TEVC. Individuals from two populations projected onto aligned
  eigen-subspaces and recombined there.
- **Adam, A.H.A., Hassan, M.H., Kamel, S. (2027).** QLRUN: integrating Q-learning and covariance matrix learning
  into Runge-Kutta optimization. Unconventional Resources 16:100484. Q-learning operator selection plus an
  eigen-coordinate system for rotated problems in one hybrid metaheuristic.
- Background: Guo & Yang (2015) eigenvector-based crossover in DE; Wang et al. (2014) CoBiDE; Civicioglu & Besdok
  (2018) QR-based rotation-invariant crossover (ESWA).

## Conclusion

The operator itself (recombining individuals from two populations in an eigen frame of their pooled distribution)
is anticipated by the evolutionary-multitasking work of Tang et al. (2021, 2022), and eigen crossover inside a
cooperative multi-algorithm model is EA4eig (2022). An RL-plus-eigen-frame hybrid also exists (QLRUN). The eigen-frame
exchange in MPHBS would be an application of a known idea to a new host, not a new operator.
