"""Build the Journal of Heuristics submission package for the Path A paper (paper/).

The manuscript keeps the author details (the journal's guidelines ask for author information on the title page and do
not require an anonymized manuscript). Acknowledgements are on the title page only.

Output: paper/submission/package/
  manuscript.pdf, manuscript_source.zip   manuscript and its LaTeX source (main.tex, refs.bib, main.bbl,
                                          generated/, figures/)
  title_page.pdf, cover_letter.pdf
  Fig1.pdf ... Fig11.pdf                  artwork as separate files, named as the journal asks

Usage:  python scripts/pathA_make_submission.py     (rebuilds paper/main.pdf first)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SUB = PAPER / "submission"
PKG = SUB / "package"
FIGURES = ["fig1_standalone", "fig_forest", "fig_instances", "fig_convergence_cec", "fig_final_fir", "fig_f10",
           "fig_dose", "fig_diversity", "fig_bbob", "fig_convergence_fir", "fig_final_cec"]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def latex(tex, cwd, bib=False, outdir=None):
    stem = Path(tex).stem
    extra = ["-output-directory", outdir] if outdir else []
    run(["pdflatex", "-interaction=nonstopmode", *extra, tex], cwd)
    if bib:
        run(["bibtex", stem], cwd)
    run(["pdflatex", "-interaction=nonstopmode", *extra, tex], cwd)
    run(["pdflatex", "-interaction=nonstopmode", *extra, tex], cwd)
    d = Path(cwd) / (outdir or ".")
    log = (d / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    bad = [l for l in log.splitlines() if l.startswith("!") or ("undefined" in l and "Warning" in l)]
    if bad:
        sys.exit(f"LaTeX problems in {tex}: {bad[:4]}")
    return d / f"{stem}.pdf"


def main():
    if PKG.exists():
        for f in PKG.iterdir():
            f.unlink()
    PKG.mkdir(parents=True, exist_ok=True)
    shutil.copy(latex("main.tex", PAPER, bib=True), PKG / "manuscript.pdf")

    with zipfile.ZipFile(PKG / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("main.tex", "refs.bib", "main.bbl"):
            z.write(PAPER / name, name)
        for sub in ("generated", "figures"):
            for f in sorted((PAPER / sub).iterdir()):
                if f.suffix in (".tex", ".pdf") and not f.name.startswith("fig1_standalone"):
                    z.write(f, f"{sub}/{f.name}")

    latex("figures/fig1_standalone.tex", PAPER, outdir="figures")
    for i, src in enumerate(FIGURES, 1):
        shutil.copy(PAPER / "figures" / f"{src}.pdf", PKG / f"Fig{i}.pdf")

    for tex in ("title_page.tex", "cover_letter.tex"):
        shutil.copy(latex(tex, SUB), PKG / tex.replace(".tex", ".pdf"))

    print("package:", sorted(p.name for p in PKG.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
