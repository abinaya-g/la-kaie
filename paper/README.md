# Path A paper (draft)

Two-column Elsevier `elsarticle` (5p) manuscript, target *Swarm and Evolutionary Computation*.

## Build

```bash
python scripts/pathA_diversity_diagnostic.py   # post hoc diversity diagnostic (exploratory)
python scripts/paper_figures.py         # convergence, distributions, F10, dose, forest; appendix tables (~6 min)
python scripts/paper_assets.py          # numbers (\res), main tables, per-instance figure from results/analysis/pathA
cd paper && latexmk -pdf main.tex       # -> paper/main.pdf
```

Every result in the text comes from `paper/generated/results.tex` through `\res{<key>}`; the tables in
`paper/generated/` and `paper/figures/fig_instances.pdf` are generated too. Do not edit generated files: change
`scripts/paper_assets.py` and rebuild. Fig. 1 (`paper/figures/fig1_loop.tex`) is TikZ.

## Open items (shown in red as PENDING / VERIFY in the PDF)

1. Co-authors (if any) and their CRediT roles; confirm the competing-interest statement.
2. Novelty search: Scopus done on 2026-10-27 (see `novelty_search_2026-10-27.md`); closest studies added to Section 2
   and the "first HBA-MPA hybrid" claim corrected. Optional: Web of Science, and a Google Scholar cited-by check of
   MPHBS right before submission.
3. References: journal entries checked against Scopus (2026-10-27); a few classics, the CEC2022 report and the arXiv
   preprint are standard and were not checked there.
4. Zenodo release: not planned before submission (only if a reviewer asks).
5. The MPHBS authors were not contacted (author's decision); Acknowledgements thank them for the public code only.

## Notes for revision

- The authors' SARSA-vs-mirror re-analysis in Section 2.2 uses the repository's paired Wilcoxon
  (normal approximation, as in the authors' MATLAB scripts): 8 of 12 CEC2022 functions favour SARSA. The
  pre-registration's background fact F-7 says 7 of 12; that figure came from a quick check with SciPy's exact
  Wilcoxon and is background only (no confirmatory analysis depends on it).
- H1 is a small effect: keep the claim at "no benefit of learning", not "random is better".
