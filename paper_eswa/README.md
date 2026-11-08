# ESWA paper (draft): EIG-MPHBS

Single-column Elsevier `elsarticle` preprint (author-year), target *Expert Systems with Applications*.
Separate from the Path A paper in `paper/`, which it cites as a companion study.

## Build

```bash
python scripts/eswa_eigen_analysis.py      # confirmatory H1/H2
python scripts/eswa_eigen_secondary.py     # S1-S5 and the 11-arm S3x tables
python scripts/eswa_paper_assets.py        # numbers (\res), tables, figures for this paper (~3 min)
cd paper_eswa && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every number in the text comes from `generated/results.tex` through `\res{<key>}`; tables in `generated/` and
figures in `figures/` are generated. Do not edit generated files: change `scripts/eswa_paper_assets.py` and rebuild.
Arm names in the paper: EIG50 = EIG-MPHBS, RND = MPHBS-R.

## Open items (red PENDING marks in the PDF)

1. Generative-AI declaration (Elsevier requires one if AI tools were used): author's decision.
2. Data availability: campaign commit and release tag.
3. References marked `VERIFY` in `refs.bib` (EA4eig pages/DOI, QLRUN, A+): check in Scopus.
4. Status of the companion Path A paper (`g2026deconstruction`): update when it has a decision or a preprint.
5. Highlights (`highlights.txt`) are within 85 characters each; check before submission.
