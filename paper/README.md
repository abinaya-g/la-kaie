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

1. Author e-mail, affiliation, co-authors, CRediT, competing interests, funding.
2. Literature check with full texts (end of Section 1): confirm no prior re-examination of MPHBS or of learned
   information-exchange mediators; add the closest studies.
3. Diversity diagnostic for the memory-refresh explanation (Section 5.4, 6.1); exploratory, from stored traces.
4. Exploratory BBOB extension (Section 6.3).
5. Verify every `refs.bib` entry against the publisher's page (written from memory).
6. Zenodo release of the repository and its DOI (Data availability).
7. Decide whether the MPHBS authors are contacted before submission (Acknowledgements).

## Notes for revision

- The authors' SARSA-vs-mirror re-analysis in Section 2.2 uses the repository's paired Wilcoxon
  (normal approximation, as in the authors' MATLAB scripts): 8 of 12 CEC2022 functions favour SARSA. The
  pre-registration's background fact F-7 says 7 of 12; that figure came from a quick check with SciPy's exact
  Wilcoxon and is background only (no confirmatory analysis depends on it).
- H1 is a small effect: keep the claim at "no benefit of learning", not "random is better".
