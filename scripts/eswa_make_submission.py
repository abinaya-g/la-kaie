"""Build the ESWA submission package (double-anonymized review) from paper_eswa/.

Output: paper_eswa/submission/package/
  manuscript_anonymized.pdf      anonymized manuscript (no author, affiliation, CRediT, repository link)
  manuscript_source.zip          its LaTeX source (manuscript.tex, refs.bib, manuscript.bbl, generated/, figures/)
  title_page.pdf                 author details, acknowledgements, declarations, data link
  cover_letter.pdf
  highlights.txt
  Figure_1.pdf ... Figure_10.pdf, Figure_11.pdf   artwork as separate files

Anonymization: lines between %@NAMED-BEGIN and %@NAMED-END are dropped, lines starting with "%@ANON: " are
uncommented, the companion-paper citation is replaced by "Anonymous (2026)", and its bibliography entry is removed.
The script then scans the PDF text, PDF metadata and every source file for identifying strings and fails if any is
found.

Usage:  python scripts/eswa_make_submission.py      (run after paper_eswa/main.tex has been built)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_eswa"
SUB = PAPER / "submission"
PKG = SUB / "package"
SRC = SUB / "build_anon"
LEAKS = ["abinaya", "saveetha", "gabinaya", "github.com/abinaya", "g2026deconstruction", "Abinaya G",
         "la-kaie", "Chennai"]
FIGURES = [("fig_architecture", "Figure_1"), ("fig_heldout", "Figure_2"), ("fig_mechanism", "Figure_3"),
           ("fig_ranks", "Figure_4"), ("fig_wtl_map", "Figure_5"), ("fig_boxplots", "Figure_6"),
           ("fig_convergence", "Figure_7"), ("fig_conv_C20", "Figure_8"), ("fig_conv_FIR", "Figure_9"),
           ("fig_fir", "Figure_10"), ("fig_conv_C10", "Figure_11")]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def latex(tex, cwd, bib=False):
    stem = Path(tex).stem
    run(["pdflatex", "-interaction=nonstopmode", tex], cwd)
    if bib:
        run(["bibtex", stem], cwd)
    run(["pdflatex", "-interaction=nonstopmode", tex], cwd)
    run(["pdflatex", "-interaction=nonstopmode", tex], cwd)
    log = (Path(cwd) / f"{stem}.log").read_text(encoding="utf-8", errors="replace")
    errs = [l for l in log.splitlines() if l.startswith("!")]
    if errs or "undefined" in log.lower().replace("undefined references", "") and "Citation" in log:
        und = [l for l in log.splitlines() if "undefined" in l]
        if errs or und:
            sys.exit(f"LaTeX problems in {tex}: {errs[:3]} {und[:3]}")
    return Path(cwd) / f"{stem}.pdf"


def anonymize(text):
    out, skip = [], False
    for line in text.splitlines():
        s = line.strip()
        if s == "%@NAMED-BEGIN":
            skip = True
            continue
        if s == "%@NAMED-END":
            skip = False
            continue
        if skip:
            continue
        if s.startswith("%@ANON: "):
            line = line.replace("%@ANON: ", "", 1)
        out.append(line)
    assert not skip, "unclosed %@NAMED-BEGIN"
    return "\n".join(out).replace("g2026deconstruction", "anon2026") + "\n"


def drop_bib_entry(bib, key):
    return re.sub(r"(%[^\n]*\n)?@\w+\{" + key + r",.*?\n\}\n", "", bib, flags=re.S)


def scan(paths_texts):
    hits = []
    for where, text in paths_texts:
        low = text.lower()
        hits += [(where, k) for k in LEAKS if k.lower() in low]
    return hits


def main():
    if SRC.exists():
        shutil.rmtree(SRC)
    PKG.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True)

    # ---- anonymized manuscript source
    (SRC / "manuscript.tex").write_text(anonymize((PAPER / "main.tex").read_text(encoding="utf-8")),
                                         encoding="utf-8")
    bib = drop_bib_entry((PAPER / "refs.bib").read_text(encoding="utf-8"), "g2026deconstruction")
    (SRC / "refs.bib").write_text(bib, encoding="utf-8")
    shutil.copytree(PAPER / "generated", SRC / "generated")
    (SRC / "figures").mkdir()
    for f in (PAPER / "figures").glob("*"):
        if f.suffix == ".pdf" or f.name == "fig_architecture_tikz.tex":
            shutil.copy(f, SRC / "figures" / f.name)
    pdf = latex("manuscript.tex", SRC, bib=True)

    # ---- leak scan: PDF text, PDF metadata, every source file
    _, text = run(["pdftotext", str(pdf), "-"], SRC)
    _, info = run(["pdfinfo", str(pdf)], SRC)
    texts = [("manuscript.pdf text", text), ("manuscript.pdf metadata", info)]
    for f in SRC.rglob("*"):
        if f.is_file() and f.suffix in (".tex", ".bib", ".bbl"):
            texts.append((str(f.relative_to(SRC)), f.read_text(encoding="utf-8", errors="replace")))
    hits = scan(texts)
    if hits:
        sys.exit(f"identifying strings found: {hits}")
    shutil.copy(pdf, PKG / "manuscript_anonymized.pdf")

    # ---- source zip (what the submission system asks for)
    with zipfile.ZipFile(PKG / "manuscript_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SRC.rglob("*")):
            if f.is_file() and (f.suffix in (".tex", ".bib", ".bbl", ".pdf") and f.name != "manuscript.pdf"):
                z.write(f, f.relative_to(SRC))

    # ---- artwork as separate files
    run(["pdflatex", "-interaction=nonstopmode", "-output-directory", "figures", "figures/fig_architecture.tex"],
        PAPER)
    for src, dst in FIGURES:
        shutil.copy(PAPER / "figures" / f"{src}.pdf", PKG / f"{dst}.pdf")

    # ---- title page, cover letter, highlights
    for tex in ("title_page.tex", "cover_letter.tex"):
        shutil.copy(latex(tex, SUB), PKG / tex.replace(".tex", ".pdf"))
    shutil.copy(PAPER / "highlights.txt", PKG / "highlights.txt")

    print("package:", sorted(p.name for p in PKG.iterdir()))
    print("leak scan: clean;", len(texts), "files checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
