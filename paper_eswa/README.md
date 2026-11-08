# ESWA paper: EIG-MPHBS

Single-column Elsevier `elsarticle` preprint (author-year), target *Expert Systems with Applications*
(double-anonymized review). Separate from the Path A paper in `paper/`, which it cites.

## Build

```bash
python scripts/eswa_eigen_analysis.py      # confirmatory H1/H2
python scripts/eswa_eigen_secondary.py     # S1-S5 and the 11-arm S3x tables
python scripts/eswa_paper_assets.py        # numbers (\res), tables, figures for this paper (~3 min)
cd paper_eswa && pdflatex main && bibtex main && pdflatex main && pdflatex main && cd ..
python scripts/eswa_make_submission.py     # anonymized submission package -> paper_eswa/submission/package/
```

Every number in the text comes from `generated/results.tex` through `\res{<key>}`; tables in `generated/` and
figures in `figures/` are generated. Do not edit generated files: change `scripts/eswa_paper_assets.py` and rebuild.
Arm names in the paper: EIG50 = EIG-MPHBS, RND = MPHBS-R. Fig. 1 is TikZ (`figures/fig_architecture_tikz.tex`),
also built standalone by `figures/fig_architecture.tex` for the artwork upload.

## Named and anonymized versions

`main.tex` is the named version. In it, lines between `%@NAMED-BEGIN` and `%@NAMED-END` (author block, repository
link, CRediT, competing interests) are removed from the anonymized version, and lines starting `%@ANON: ` are added.
The Path A citation becomes `Anonymous (2026)`. `scripts/eswa_make_submission.py` builds the anonymized manuscript
and fails if any identifying string remains in its PDF text, PDF metadata or source files.

## Submission package (`submission/package/`)

| File | Upload as |
|------|-----------|
| `manuscript_anonymized.pdf`, `manuscript_source.zip` | Manuscript (anonymized) and its LaTeX source |
| `title_page.pdf` (source `submission/title_page.tex`) | Title page (author details, acknowledgements, declarations) |
| `highlights.txt` | Highlights |
| `Figure_1.pdf` ... `Figure_10.pdf`, `Figure_A1.pdf` | Figures |
| `cover_letter.pdf` (source `submission/cover_letter.tex`) | Cover letter |

## Still to do by the author (outside this repository)

1. Declaration of competing interests: complete Elsevier's declarations tool and upload the .docx it produces.
2. Generative-AI declaration: the guide requires a statement before the references if AI tools were used in preparing
   the manuscript (not for grammar or spelling checks only).
3. Research data (Option C): deposit code and data in a repository with a DOI (e.g. a Zenodo release of the GitHub
   repository) and cite it; for review, give the link only on the title page or as an anonymized link.
4. Title page: confirm the postal address and the status of the Path A manuscript.
5. Make the GitHub repository public, or the data link on the title page will not work for the editor.
