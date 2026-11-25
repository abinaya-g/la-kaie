"""Build the submission package of the combined manuscript (paper_combined/).

Output: paper_combined/submission/package/
  manuscript.pdf, manuscript_source.zip   manuscript and its LaTeX source (main.tex, sections/, refs.bib, main.bbl,
                                          generated/, figures/ with the files the manuscript uses)
  manuscript_source_flat.zip              the same source with every file at the top level, main.tex adjusted
  title_page.pdf, cover_letter.pdf, highlights.txt
  Fig1.pdf ... Fig8.pdf                   artwork as separate files, in the order of appearance

Usage:  python scripts/combined_make_submission.py     (rebuilds paper_combined/main.pdf first)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_combined"
SUB = PAPER / "submission"
PKG = SUB / "package"
# order of appearance in the manuscript
FIGURES = ["fig_loop_standalone", "fig_architecture", "fig_forest", "fig_dose", "fig_heldout", "fig_mechanism",
           "fig_ranks", "fig_transfer"]
USED_PDF = {"fig_forest.pdf", "fig_dose.pdf", "fig_heldout.pdf", "fig_mechanism.pdf", "fig_ranks.pdf", "fig_transfer.pdf"}
USED_TEX = {"fig1_loop.tex", "fig_architecture_tikz.tex"}
TABLES = {"tab_arms", "tab_primary", "tab_theory", "tab_bbob_groups", "tab_confirm", "tab_rotation", "tab_ranks",
          "tab_wtl", "tab_ea4eig", "tab_cost", "tab_ablation", "tab_transfer"}


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


def put(src: Path, dst: Path):
    try:
        shutil.copy(src, dst)
    except PermissionError:
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            print(f"skipped (open, unchanged): {dst.name}")
            return
        sys.exit(f"{dst.name} is open in another program and has changed; close it and run again")


def source_files():
    """(archive name, path) of every source file the manuscript uses."""
    out = [("main.tex", PAPER / "main.tex"), ("refs.bib", PAPER / "refs.bib"), ("main.bbl", PAPER / "main.bbl")]
    for i in (1, 2, 3):
        out.append((f"sections/part{i}.tex", PAPER / "sections" / f"part{i}.tex"))
    out.append(("generated/results.tex", PAPER / "generated" / "results.tex"))
    for t in sorted(TABLES):
        out.append((f"generated/{t}.tex", PAPER / "generated" / f"{t}.tex"))
    for f in sorted(USED_PDF | USED_TEX):
        out.append((f"figures/{f}", PAPER / "figures" / f))
    return out


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    put(latex("main.tex", PAPER, bib=True), PKG / "manuscript.pdf")

    # every file that the manuscript inputs must be in the source list
    tex = (PAPER / "main.tex").read_text(encoding="utf-8") + "".join(
        (PAPER / "sections" / f"part{i}.tex").read_text(encoding="utf-8") for i in (1, 2, 3))
    need = set(re.findall(r"\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    have = {n for n, _ in source_files()}
    missing = [n for n in need if n not in have and n + ".tex" not in have]
    if missing:
        sys.exit(f"source list misses: {missing}")

    with zipfile.ZipFile(PKG / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in source_files():
            z.write(path, name)

    names = set()
    with zipfile.ZipFile(PKG / "manuscript_source_flat.zip", "w", zipfile.ZIP_DEFLATED) as z:
        def flat(s):
            return re.sub(r"\{(?:sections|generated|figures)/", "{", s)
        for name, path in source_files():
            base = Path(name).name
            if base in names:
                sys.exit(f"name clash in flat zip: {base}")
            names.add(base)
            if base.endswith(".tex") and base != "results.tex" and not name.startswith(("generated", "figures")):
                z.writestr(base, flat(path.read_text(encoding="utf-8")))
            else:
                z.write(path, base)

    # artwork
    latex("figures/fig_loop_standalone.tex", PAPER, outdir="figures")
    latex("figures/fig_architecture.tex", PAPER, outdir="figures")
    for i, src in enumerate(FIGURES, 1):
        put(PAPER / "figures" / f"{src}.pdf", PKG / f"Fig{i}.pdf")

    for t in ("title_page.tex", "cover_letter.tex"):
        put(latex(t, SUB), PKG / t.replace(".tex", ".pdf"))
    put(SUB / "highlights.txt", PKG / "highlights.txt")
    print("package:", sorted(p.name for p in PKG.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
