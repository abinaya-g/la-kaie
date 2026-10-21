"""Extra figures and appendix tables for the Path A paper, generated from the raw runs and the analysis.

Outputs (paper/figures/, paper/generated/)
  fig_convergence_cec.pdf, fig_convergence_fir.pdf   median best-so-far error vs fraction of the budget
  fig_final_cec.pdf, fig_final_fir.pdf               final-error distributions of the primary arms
  fig_f10.pdf                                        F10 escape rates with Wilson intervals (ours and the authors')
  fig_dose.pdf                                       W-bar vs mediator episodes N_i (H6)
  fig_forest.pdf                                     W-bar and 95% CI of all contrasts
  tab_meanstd_cec.tex, tab_meanstd_fir.tex           mean +- std of the final error per arm and instance
  tab_inst_<H>_<dom>.tex                             per-instance tables for H2a, H2b, H3, H4

Usage:  python scripts/paper_figures.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lakaie.analysis.pathA import wilson  # noqa: E402
from lakaie.analysis.stats import load_campaign  # noqa: E402

RAW = ROOT / "results" / "raw" / "pathA"
AN = ROOT / "results" / "analysis" / "pathA"
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
REF = ROOT / "third_party" / "mphbs_reference" / "results"

# fixed categorical order (dataviz reference palette slots 1-5) + line style / marker per arm
STYLE = {"PA-SARSA": ("#2a78d6", "-", "o", "SARSA"),
         "PA-RND-FAIR": ("#eb6834", "--", "s", "RND-FAIR"),
         "PA-SKEL": ("#1baf7a", "-.", "^", "SKEL"),
         "PA-MPHB": ("#eda100", ":", "D", "MPHB"),
         "PA-RND-AUTH": ("#e87ba4", (0, (5, 1, 1, 1)), "v", "RND-AUTH")}
INK, INK2, GREY = "#0b0b0b", "#52514e", "#9a9893"
CEC_NAMES = {1: "Zakharov", 2: "Rosenbrock", 3: "Exp. Schaffer f6", 4: "N.-c. Rastrigin", 5: "Levy",
             6: "Hybrid 1", 7: "Hybrid 2", 8: "Hybrid 3", 9: "Composition 1", 10: "Composition 2",
             11: "Composition 3", 12: "Composition 4"}
FIR_NAMES = {1: "LPF easy", 2: "LPF hard", 3: "HPF easy", 4: "HPF hard", 5: "BPF wide",
             6: "BPF narrow", 7: "BSF wide", 8: "BSF narrow"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _legend(fig, plt, arms, y=-0.005, marker=False):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=STYLE[a][0], ls=STYLE[a][1], lw=1.3,
                marker=STYLE[a][2] if marker else None, ms=4, label=STYLE[a][3]) for a in arms]
    fig.legend(handles=h, loc="lower center", ncol=len(arms), frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, y))


def _yaxis(ax, values):
    """Readable y axis: log scale with at most 4 major labels (no minor labels) when the values span at least one
    decade, otherwise a linear axis with at most 4 ticks and a shared offset (e.g. near-constant F9 errors)."""
    from matplotlib import ticker
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size and v.max() / v.min() >= 10:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        fmt = ticker.ScalarFormatter(useOffset=True, useMathText=True)
        fmt.set_powerlimits((-2, 3))
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.get_offset_text().set_fontsize(5.5)


# ------------------------------------------------------------------ convergence
def convergence(dom, arms, names, ncols, fname, runs=50):
    plt = _plt()
    insts = sorted(names)
    nrows = math.ceil(len(insts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 1.55 * nrows + 0.5))
    for inst, ax in zip(insts, axes.ravel()):
        shown = []
        for arm in arms:
            curves = []
            for r in range(1, runs + 1):
                z = np.load(RAW / dom / arm / f"inst{inst:02d}" / f"run{r:02d}.npz")
                curves.append(z["curve"])
                x = z["checkpoints"] / z["checkpoints"][-1]
            C = np.vstack(curves)
            col, ls, _, _ = STYLE[arm]
            med = np.maximum(np.median(C, axis=0), 1e-12)
            shown.append(med)
            ax.plot(x, med, color=col, ls=ls, lw=1.2)
        _yaxis(ax, np.concatenate(shown))
        ax.set_title(f"{'F' if dom == 'CEC2022' else 'Case '}{inst}: {names[inst]}", fontsize=6.8,
                     loc="left", color=INK)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5, which="major")
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(insts):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("fraction of the FE budget", fontsize=6.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("median best error" if dom == "CEC2022" else "median best objective", fontsize=6.5)
    _legend(fig, plt, arms)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.8, w_pad=0.6)
    fig.savefig(FIG / f"{fname}.pdf")
    fig.savefig(FIG / f"{fname}.png", dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ final-error distributions
def final_dist(df, dom, arms, names, ncols, fname, runs=50):
    plt = _plt()
    insts = sorted(names)
    nrows = math.ceil(len(insts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0, 1.5 * nrows + 0.5))
    d = df[df.run_id <= runs]
    for inst, ax in zip(insts, axes.ravel()):
        data = [d[(d.method == a) & (d.instance == inst)].best_fitness.to_numpy(float) for a in arms]
        pos = np.arange(len(arms))
        bp = ax.boxplot([np.maximum(v, 1e-12) for v in data], positions=pos, widths=0.6, showfliers=True,
                        patch_artist=True, medianprops={"color": INK, "lw": 1.0},
                        flierprops={"marker": ".", "ms": 2.5, "mec": GREY},
                        whiskerprops={"color": INK2, "lw": 0.7}, capprops={"color": INK2, "lw": 0.7},
                        boxprops={"lw": 0.7, "color": INK2})
        for patch, a in zip(bp["boxes"], arms):
            patch.set_facecolor(STYLE[a][0])
            patch.set_alpha(0.55)
        _yaxis(ax, np.concatenate([np.maximum(v, 1e-12) for v in data]))
        ax.set_xticks(pos)
        ax.set_xticklabels([STYLE[a][3] for a in arms], rotation=45, ha="right", fontsize=5.5)
        ax.set_title(f"{'F' if dom == 'CEC2022' else 'Case '}{inst}: {names[inst]}", fontsize=6.8,
                     loc="left", color=INK)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    for ax in axes.ravel()[len(insts):]:
        ax.set_axis_off()
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    fig.savefig(FIG / f"{fname}.pdf")
    fig.savefig(FIG / f"{fname}.png", dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ F10 escapes
def f10(summary):
    plt = _plt()
    h5 = summary["H5_f10_escape"]["arms"]
    T = pd.read_csv(REF / "CEC2022_ablation_BC1_common_seeds" / "task_results.csv")
    T = T[T.FunctionID == 10]
    rows = []
    for arm in ("PA-SARSA", "PA-RND-FAIR", "PA-RND-AUTH", "PA-SKEL"):
        a = h5[arm]
        rows.append((f"{STYLE[arm][3]} (this study)", a["escapes"], a["n"], STYLE[arm][0], STYLE[arm][2]))
    for alias, lab, arm in (("BS-C1", "SARSA (authors)", "PA-SARSA"), ("BR-C1", "RANDOM mirror (authors)", "PA-RND-AUTH"),
                            ("MPHB", "MPHB (authors)", "PA-MPHB")):
        e = T[T.Alias == alias].Error.to_numpy(float)
        rows.append((lab, int((e < 50).sum()), int(e.size), STYLE[arm][0], STYLE[arm][2]))
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    for i, (lab, k, n, col, mk) in enumerate(rows[::-1]):
        lo, hi = wilson(k, n)
        filled = "authors" not in lab
        ax.errorbar(k / n, i, xerr=[[max(0.0, k / n - lo)], [max(0.0, hi - k / n)]], fmt=mk, color=col, ms=4.5,
                    mfc=col if filled else "white", mew=1.0, elinewidth=1.0, capsize=2)
        ax.text(1.02, i, f"{k}/{n}", va="center", fontsize=6.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F10 escape rate (95% Wilson interval)")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    fig.savefig(FIG / "fig_f10.pdf")
    fig.savefig(FIG / "fig_f10.png", dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ dose-response (H6)
def dose(summary):
    plt = _plt()
    sec = summary["secondary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    spec = {"CEC2022": {1: "H6-Ni1-policy", 3: "H6-Ni3-policy", 9: "H6-Ni9-policy"},
            "FIR": {1: "H6-Ni1-policy", 3: "H6-Ni3-policy", 9: "H6-Ni9-policy"}}
    for ax, dom in zip(axes, ("CEC2022", "FIR")):
        xs = [1, 3, 9]
        for j, ni in enumerate(xs):
            c = sec[f"{spec[dom][ni]}/{dom}"]
            ax.errorbar(j - 0.08, c["wbar"], yerr=[[c["wbar"] - c["ci_low"]], [c["ci_high"] - c["wbar"]]],
                        fmt="o", color=STYLE["PA-SARSA"][0], ms=4.5, capsize=2, lw=1)
        default = 3 if dom == "CEC2022" else 9
        other = 9 if dom == "CEC2022" else 3
        for ni, key in ((1, "H6-RND-Ni1-vs-default"), (other, "H6-RND-Ni-other-vs-default")):
            c = sec[f"{key}/{dom}"]
            j = xs.index(ni)
            ax.errorbar(j + 0.08, c["wbar"], yerr=[[c["wbar"] - c["ci_low"]], [c["ci_high"] - c["wbar"]]],
                        fmt="s", color=STYLE["PA-RND-FAIR"][0], ms=4.2, capsize=2, lw=1)
        ax.axhline(0.5, color=GREY, ls=(0, (2, 2)), lw=0.8)
        ax.axhspan(0.44, 0.56, color="#f0efec", zorder=0)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"$N_i={n}$" + (" (default)" if n == default else "") for n in xs])
        ax.set_title(dom, fontsize=7.5, loc="left", fontweight="bold", color=INK)
        ax.grid(axis="y", color="#e6e5e1", lw=0.5)
    axes[0].set_ylabel(r"$\bar W$ (95% CI)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=STYLE["PA-SARSA"][0], label="SARSA vs RND-FAIR at the same $N_i$"),
                        Line2D([], [], marker="s", ls="", color=STYLE["PA-RND-FAIR"][0], label="RND-FAIR at $N_i$ vs RND-FAIR at the default"),
                        Line2D([], [], color="#d9d8d3", lw=6, label="negligible band [0.44, 0.56]")],
               loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "fig_dose.pdf")
    fig.savefig(FIG / "fig_dose.png", dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ forest of all contrasts
def forest(summary):
    plt = _plt()
    items = [(lvl, k, c) for lvl in ("primary", "secondary") for k, c in summary[lvl].items()]
    fig, ax = plt.subplots(figsize=(7.0, 0.19 * len(items) + 0.8))
    for i, (lvl, k, c) in enumerate(items[::-1]):
        col = "#2a78d6" if c["ci_low"] > 0.5 else ("#eb6834" if c["ci_high"] < 0.5 else GREY)
        mk = "o" if lvl == "primary" else "s"
        ax.errorbar(c["wbar"], i, xerr=[[c["wbar"] - c["ci_low"]], [c["ci_high"] - c["wbar"]]], fmt=mk,
                    color=col, ms=4, capsize=1.5, lw=1, mfc=col if lvl == "primary" else "white")
        ax.text(1.005, i, c["verdict"], va="center", fontsize=6, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{k}  ({STYLE.get(c['anchor'], (0, 0, 0, c['anchor']))[3]} vs "
                        f"{STYLE.get(c['comparator'], (0, 0, 0, c['comparator']))[3]})".replace("PA-", "")
                        for _, k, c in items[::-1]], fontsize=6)
    ax.axvline(0.5, color=GREY, ls=(0, (2, 2)), lw=0.8)
    ax.axvspan(0.44, 0.56, color="#f0efec", zorder=0)
    sep = len([1 for it in items if it[0] == "secondary"]) - 0.5
    ax.axhline(sep, color=INK2, lw=0.5)
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel(r"$\bar W$: mean paired win probability of the first arm (95% bootstrap CI)")
    ax.grid(axis="x", color="#e6e5e1", lw=0.5)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(FIG / "fig_forest.pdf")
    fig.savefig(FIG / "fig_forest.png", dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------ appendix tables
def _fmt(x):
    if x == 0 or not np.isfinite(x):
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    if -2 <= e <= 3:
        return f"{x:.3g}"
    return f"{x / 10 ** e:.2f}e{e}"


def meanstd(df, dom, arms, names, fname, runs=50):
    d = df[df.run_id <= runs]
    lines = []
    for inst in sorted(names):
        cells = []
        means = {a: d[(d.method == a) & (d.instance == inst)].best_fitness.mean() for a in arms}
        best = min(means.values())
        for a in arms:
            v = d[(d.method == a) & (d.instance == inst)].best_fitness.to_numpy(float)
            m, s = v.mean(), v.std(ddof=1)
            txt = f"{_fmt(m)} $\\pm$ {_fmt(s)}"
            cells.append(f"\\textbf{{{txt}}}" if means[a] == best else txt)
        lab = f"F{inst}" if dom == "CEC2022" else f"Case {inst}"
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    head = " & ".join(STYLE[a][3] if a in STYLE else a.replace("PA-", "") for a in arms)
    tex = f"""% generated by scripts/paper_figures.py -- do not edit
\\begin{{table*}}[p]
\\centering
\\caption{{Mean $\\pm$ standard deviation of the final {'error' if dom == 'CEC2022' else 'objective value'} on
{dom} (runs 1--{runs} of each arm; lowest mean in bold). Descriptive only; inference is by the paired tests of
Section~\\ref{{sec:results}}.}}
\\label{{tab:meanstd-{dom.lower()}}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{'c' * len(arms)}}}
\\toprule
 & {head} \\\\
\\midrule
{chr(10).join(lines)}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    (GEN / f"{fname}.tex").write_text(tex, encoding="utf-8")


def inst_table(hid, dom):
    T = pd.read_csv(AN / "tables" / f"primary_{hid}_{dom}.csv")
    s = json.loads((AN / "summary.json").read_text(encoding="utf-8"))["primary"][f"{hid}/{dom}"]
    a, b = (STYLE.get(x, (0, 0, 0, x.replace("PA-", "")))[3] for x in (s["anchor"], s["comparator"]))
    rows = []
    for r in T.itertuples():
        mark = {"anchor": "$^{\\uparrow}$", "comparator": "$^{\\downarrow}$"}.get(r.significant_favours, "")
        lab = f"F{r.instance}" if dom == "CEC2022" else f"Case {r.instance}"
        p = "--" if r.p_holm != r.p_holm else (f"{r.p_holm:.3f}" if r.p_holm >= 1e-3 else f"{r.p_holm:.1e}")
        rows.append(f"{lab} & {_fmt(r.anchor_median)} & {_fmt(r.comparator_median)} & "
                    f"{r.anchor_wins}/{r.ties}/{r.anchor_losses} & {r.w:.2f}{mark} & {p} \\\\")
    tex = f"""% generated by scripts/paper_figures.py -- do not edit
\\begin{{table}}[p]
\\centering
\\caption{{{hid} on {dom}: {a} (anchor) vs {b}, {s['runs']} seed-paired runs per instance. Medians of the final value;
W/T/L from the anchor's view; $\\hat w$ paired win probability of the anchor ($^\\uparrow$/$^\\downarrow$: significant in
favour of anchor/comparator); $p$ Holm-adjusted.}}
\\label{{tab:inst-{hid.lower()}-{dom.lower()}}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lccccc}}
\\toprule
 & {a} & {b} & W/T/L & $\\hat w$ & $p_\\mathrm{{Holm}}$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (GEN / f"tab_inst_{hid}_{dom}.tex").write_text(tex, encoding="utf-8")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    summary = json.loads((AN / "summary.json").read_text(encoding="utf-8"))
    four = ["PA-SARSA", "PA-RND-FAIR", "PA-SKEL", "PA-MPHB"]
    five = ["PA-SARSA", "PA-RND-FAIR", "PA-RND-AUTH", "PA-SKEL", "PA-MPHB"]
    slow = "--skip-slow" not in sys.argv          # convergence + distributions read every raw run
    if slow:
        convergence("CEC2022", four, CEC_NAMES, 4, "fig_convergence_cec")
        convergence("FIR", four, FIR_NAMES, 4, "fig_convergence_fir")
    raw = ROOT / "results" / "raw"
    cec, fir = load_campaign(raw, "pathA", "CEC2022"), load_campaign(raw, "pathA", "FIR")
    if slow:
        final_dist(cec, "CEC2022", five, CEC_NAMES, 4, "fig_final_cec")
        final_dist(fir, "FIR", five, FIR_NAMES, 4, "fig_final_fir")
    f10(summary)
    dose(summary)
    forest(summary)
    meanstd(cec, "CEC2022", ["PA-SARSA", "PA-RND-FAIR", "PA-RND-SYNC", "PA-RND-AUTH", "PA-SKEL", "PA-MPHB"],
            CEC_NAMES, "tab_meanstd_cec")
    meanstd(fir, "FIR", five, FIR_NAMES, "tab_meanstd_fir")
    for hid, dom in (("H2a", "CEC2022"), ("H2a", "FIR"), ("H2b", "CEC2022"), ("H3", "CEC2022"), ("H3", "FIR"),
                     ("H4", "CEC2022"), ("H4", "FIR")):
        inst_table(hid, dom)
    print("done")


if __name__ == "__main__":
    main()
