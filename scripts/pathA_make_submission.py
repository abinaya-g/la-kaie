"""Build the Swarm and Evolutionary Computation submission package for the Path A paper (paper/).

The journal uses single-anonymized review, so the manuscript keeps the author details.

Output: paper/submission/package/
  manuscript.pdf, manuscript_source.zip   manuscript and its LaTeX source (main.tex, refs.bib, main.bbl,
                                          generated/, figures/)
  title_page.pdf, cover_letter.pdf, orcid_info.pdf, highlights.txt, highlights.docx
  Figure_1.pdf ... Figure_9.pdf, Figure_A10.pdf, Figure_A11.pdf   artwork as separate files

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
FIGURES = [("fig1_standalone", "Figure_1"), ("fig_forest", "Figure_2"), ("fig_instances", "Figure_3"),
           ("fig_convergence_cec", "Figure_4"), ("fig_final_fir", "Figure_5"), ("fig_f10", "Figure_6"),
           ("fig_dose", "Figure_7"), ("fig_diversity", "Figure_8"), ("fig_bbob", "Figure_9"),
           ("fig_convergence_fir", "Figure_A10"), ("fig_final_cec", "Figure_A11")]


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
    for src, dst in FIGURES:
        shutil.copy(PAPER / "figures" / f"{src}.pdf", PKG / f"{dst}.pdf")

    for tex in ("title_page.tex", "cover_letter.tex", "orcid_info.tex"):
        shutil.copy(latex(tex, SUB), PKG / tex.replace(".tex", ".pdf"))
    shutil.copy(SUB / "highlights.txt", PKG / "highlights.txt")
    run([sys.executable, str(ROOT / "scripts" / "eswa_highlights_docx.py"), str(PKG / "highlights.docx"),
         str(SUB / "highlights.txt")], ROOT)

    print("package:", sorted(p.name for p in PKG.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
