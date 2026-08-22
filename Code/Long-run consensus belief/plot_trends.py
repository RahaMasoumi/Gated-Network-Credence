#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_trends.py -- every b* versus prestige-bias figure, from one script.

Reads the per-realization files written by run_trends.py and produces:

  --layout main          Figure 3(a).  Erdos-Renyi only.  Rows are the two
                         regimes (Evaluative on top, Friction-averse below);
                         the left column fixes rho and varies promoter
                         prevalence x, the right column fixes x and varies rho.

  --layout topologies    Figures S3 and S4.  Columns are the three synthetic
                         topologies (ER, BA, modular at psi = 0.4); the top row
                         varies x at fixed rho and the bottom row varies rho at
                         fixed x.  One figure per regime, chosen with --regime.

  --layout polarization  Figures S5 and S6.  Same two rows, columns are the
                         modular network at psi = 0, 0.4, 1.  One figure per
                         regime.

Every panel shows the mean over realizations with +-1 standard deviation error
bars.  Colour encodes x and line style encodes rho, as in the manuscript.

Usage
-----
  python plot_trends.py --layout main
  python plot_trends.py --layout topologies   --regime E     # Figure S3
  python plot_trends.py --layout topologies   --regime F     # Figure S4
  python plot_trends.py --layout polarization --regime E     # Figure S5
  python plot_trends.py --layout polarization --regime F     # Figure S6
"""

import argparse
import glob
import json
import os
import pickle

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, NullLocator, FixedLocator, FuncFormatter


# ═══════════════════════════════════════════════════════════════════════════════
#  Options
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=["main", "topologies", "polarization"],
                    default="main")
    ap.add_argument("--regime", choices=["E", "F"], default="E",
                    help="ignored by --layout main, which shows both")
    ap.add_argument("--rho-fixed", type=float, default=-0.4,
                    help="rho held fixed in the vary-x row")
    ap.add_argument("--x-fixed", type=float, default=0.2,
                    help="x held fixed in the vary-rho row")
    ap.add_argument("--datadir", type=str, default=".",
                    help="folder containing the trends_* result directories")
    ap.add_argument("--outdir", type=str, default="trend_figures")
    ap.add_argument("--out", type=str, default=None,
                    help="output filename; default is derived from the layout")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Style  (as in the manuscript figures)
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "mathtext.fontset": "stix",
})

LINEWIDTH   = 3
SPINE_WIDTH = 1.5

MARKER_SIZE      = 7
MARKER_EDGE_LW   = 1.5
MARKER_EDGE_COL  = "black"
ERRORBAR_CAPSIZE = 5
ERRORBAR_ELW     = 3

# Colourblind-safe cool-to-warm walk: the largest x is the coolest colour and
# the smallest x the warmest, so the ordering of the curves is legible.
PALETTE_WARM_TO_COOL = ["#FF0000", "#FFA500", "#3CB371", "#1E90FF"]
DASHES_RHO = ["-", "--", "-.", ":"]      # the sole cue for rho

REGIME_TITLE = {"E": "Evaluative (E)", "F": "Friction-averse (F)"}

DIR_TEMPLATES = {
    "ER":  "trends_ER_N{N}_p{p:g}_point{pt}",
    "BA":  "trends_BA_N{N}_m{m}_point{pt}",
    "MOD": "trends_MOD_N{N}_Q0.33_psi{psi:g}_point{pt}",
}

COLUMN_TITLE = {
    "ER":  "Erdős–Rényi",
    "BA":  "Barabási–Albert",
}


def _trim(v, pos):
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s
FMT = FuncFormatter(_trim)


# ═══════════════════════════════════════════════════════════════════════════════
#  Loading
# ═══════════════════════════════════════════════════════════════════════════════
def find_dir(datadir, topology, point, psi=None, N=5000, p=0.01, m=25):
    d = DIR_TEMPLATES[topology].format(N=N, p=p, m=m, psi=psi, pt=point)
    path = os.path.join(datadir, d)
    if not os.path.isdir(path):
        hint = (f"python run_trends.py --topology {topology} --point {point}"
                + (f" --psi {psi:g}" if topology == "MOD" else ""))
        raise SystemExit(f"missing results directory {path}\n  run:  {hint}")
    return path


def load(path):
    """Returns mean, sd (both (n_rho, n_x, n_ps)) and the run metadata."""
    files = sorted(glob.glob(os.path.join(path, "realization_*.pkl")))
    if not files:
        raise SystemExit(f"no realization_*.pkl in {path}")

    stack, K, settled, reference, ids = [], [], [], None, set()
    for fn in files:
        with open(fn, "rb") as f:
            D = pickle.load(f)
        if reference is None:
            reference = D["params"]
        elif D["params"] != reference:
            raise SystemExit(
                f"{fn}: parameters differ from the other files in {path}\n"
                f"  file:  {json.dumps(D['params'], sort_keys=True)}\n"
                f"  other: {json.dumps(reference, sort_keys=True)}")
        rid = D["realization_id"]
        if rid in ids:
            raise SystemExit(f"{fn}: realization {rid} appears twice")
        ids.add(rid)
        stack.append(D["b_star"])
        K.append(D["K"])
        settled.append(D["settled"])

    arr = np.stack(stack)                       # (n_real, n_rho, n_x, n_ps)
    meta = dict(reference)
    meta["n_realizations"] = arr.shape[0]
    meta["consensus_fraction"] = float(np.mean(np.stack(K) == 1))
    meta["settled_fraction"] = float(np.mean(np.stack(settled)))
    meta["path"] = path
    # sample standard deviation across realizations (ddof = 1)
    return arr.mean(axis=0), arr.std(axis=0, ddof=1), meta


# ═══════════════════════════════════════════════════════════════════════════════
#  Panels
# ═══════════════════════════════════════════════════════════════════════════════
def style_axes(ax, ps_values, xticks, ylabel=None):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.spines[side].set_zorder(10)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlim(min(ps_values), max(ps_values))
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.tick_params(axis="both", which="major", width=SPINE_WIDTH * 0.9)
    ax.xaxis.set_major_formatter(FMT)
    ax.yaxis.set_major_formatter(FMT)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, axis="y", which="major", linewidth=0.5, alpha=0.35, zorder=0)
    ax.grid(True, axis="y", which="minor", linewidth=0.5, alpha=0.25, zorder=0)
    ax.grid(True, axis="x", which="major", linewidth=0.5, alpha=0.25, zorder=0)


def panel_vary_x(ax, mean, sd, meta, rho_fixed, color_of_x, xticks, ylabel=None):
    ps = meta["ps_values"]
    ri = meta["rho_values"].index(rho_fixed)
    for xi, x in enumerate(meta["x_values"]):
        c = color_of_x[x]
        ax.errorbar(ps, mean[ri, xi, :], yerr=sd[ri, xi, :],
                    color=c, linestyle="-", linewidth=LINEWIDTH,
                    marker="o", markersize=MARKER_SIZE, markerfacecolor=c,
                    markeredgecolor=MARKER_EDGE_COL,
                    markeredgewidth=MARKER_EDGE_LW,
                    elinewidth=ERRORBAR_ELW, capsize=ERRORBAR_CAPSIZE,
                    capthick=ERRORBAR_ELW, zorder=2)
    style_axes(ax, ps, xticks, ylabel)


def panel_vary_rho(ax, mean, sd, meta, x_fixed, color_of_x, xticks, ylabel=None):
    ps = meta["ps_values"]
    xi = meta["x_values"].index(x_fixed)
    c = color_of_x[x_fixed]
    for ri, rho in enumerate(meta["rho_values"]):
        ax.errorbar(ps, mean[ri, xi, :], yerr=sd[ri, xi, :],
                    color=c, linestyle=DASHES_RHO[ri], linewidth=LINEWIDTH,
                    marker="o", markersize=MARKER_SIZE, markerfacecolor=c,
                    markeredgecolor=MARKER_EDGE_COL,
                    markeredgewidth=MARKER_EDGE_LW,
                    elinewidth=ERRORBAR_ELW, capsize=ERRORBAR_CAPSIZE,
                    capthick=ERRORBAR_ELW, zorder=2)
    style_axes(ax, ps, xticks, ylabel)


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure assembly
# ═══════════════════════════════════════════════════════════════════════════════
def make_figure(columns, args):
    """`columns` is a list of (column_title, mean, sd, meta) with the vary-x row
    on top and the vary-rho row underneath, or, for --layout main, a list of
    (row_title, mean, sd, meta) with the two regimes as the rows."""
    n_col = len(columns)
    AX_IN, HGAP_IN, VGAP_IN = 3.456, 0.9, 0.55
    LEFT_IN, RIGHT_IN, TOP_IN, BOTTOM_IN = 1.25, 3.20, 1.10, 0.85
    ROWS = 2

    fig_w = LEFT_IN + n_col * AX_IN + (n_col - 1) * HGAP_IN + RIGHT_IN
    fig_h = BOTTOM_IN + ROWS * AX_IN + (ROWS - 1) * VGAP_IN + TOP_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    def add_axes(r, c):
        left = LEFT_IN + c * (AX_IN + HGAP_IN)
        bottom = BOTTOM_IN + (ROWS - 1 - r) * (AX_IN + VGAP_IN)
        return fig.add_axes([left / fig_w, bottom / fig_h,
                             AX_IN / fig_w, AX_IN / fig_h])

    meta0 = columns[0][3]
    x_sorted = sorted(meta0["x_values"])
    color_of_x = {x: PALETTE_WARM_TO_COOL[i] for i, x in enumerate(x_sorted)}
    ps = meta0["ps_values"]
    xticks = [ps[0]] + list(ps[2:])          # 0.1 is plotted but not ticked

    if args.layout == "main":
        # rows = regimes, columns = [vary x, vary rho]
        for r, (title, mean, sd, meta) in enumerate(columns):
            ax_left = add_axes(r, 0)
            ax_right = add_axes(r, 1)
            panel_vary_x(ax_left, mean, sd, meta, args.rho_fixed, color_of_x,
                         xticks, ylabel=r"Final Belief ($b^{*}$)")
            panel_vary_rho(ax_right, mean, sd, meta, args.x_fixed, color_of_x,
                           xticks)
            ax_left.set_title(title, loc="left")
            if r == ROWS - 1:
                ax_left.set_xlabel(r"$p_s$")
                ax_right.set_xlabel(r"$p_s$")
        n_col_eff = 2
    else:
        for c, (title, mean, sd, meta) in enumerate(columns):
            ax_top = add_axes(0, c)
            ax_bot = add_axes(1, c)
            panel_vary_x(ax_top, mean, sd, meta, args.rho_fixed, color_of_x,
                         xticks,
                         ylabel=(r"Final Belief ($b^{*}$)" if c == 0 else None))
            panel_vary_rho(ax_bot, mean, sd, meta, args.x_fixed, color_of_x,
                           xticks,
                           ylabel=(r"Final Belief ($b^{*}$)" if c == 0 else None))
            ax_top.set_title(title)
            ax_bot.set_xlabel(r"$p_s$")
        n_col_eff = n_col

    # ---- legends on the right ----
    x_handles = [
        Line2D([0], [0], color=color_of_x[x], linestyle="-",
               linewidth=LINEWIDTH, marker="o", markersize=10,
               markerfacecolor=color_of_x[x], markeredgecolor="black",
               markeredgewidth=1.5, label=fr"$x={x:g}$")
        for x in x_sorted
    ]
    rho_handles = [
        Line2D([0], [0], color=color_of_x[args.x_fixed],
               linestyle=DASHES_RHO[j], linewidth=LINEWIDTH,
               label=fr"${rho:g}$")
        for j, rho in enumerate(meta0["rho_values"])
    ]

    legend_x = (LEFT_IN + n_col_eff * AX_IN
                + (n_col_eff - 1) * HGAP_IN + 0.35) / fig_w
    fig.legend(handles=x_handles, title=r"$x$", loc="upper left",
               bbox_to_anchor=(legend_x,
                               (BOTTOM_IN + 2 * AX_IN + VGAP_IN) / fig_h + 0.05),
               frameon=False, fontsize=18, title_fontsize=18)
    fig.legend(handles=rho_handles, title=r"$\rho$", loc="upper left",
               bbox_to_anchor=(legend_x, (BOTTOM_IN + 0.95 * AX_IN) / fig_h),
               frameon=False, fontsize=18, title_fontsize=18)

    os.makedirs(args.outdir, exist_ok=True)
    if args.out is not None:
        name = args.out
    elif args.layout == "main":
        name = (f"fig3a_ER_rho{args.rho_fixed:g}_x{args.x_fixed:g}.pdf")
    else:
        name = (f"{args.layout}_regime{args.regime}"
                f"_rho{args.rho_fixed:g}_x{args.x_fixed:g}.pdf")
    out = os.path.join(args.outdir, name)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = _parse_cli()

    if args.layout == "main":
        spec = [(REGIME_TITLE[r], "ER", r, None) for r in ("E", "F")]
    elif args.layout == "topologies":
        spec = [(COLUMN_TITLE["ER"], "ER", args.regime, None),
                (COLUMN_TITLE["BA"], "BA", args.regime, None),
                (r"Modular ($\psi=0.4$)", "MOD", args.regime, 0.4)]
    else:
        spec = [(rf"Modular ($\psi={psi:g}$)", "MOD", args.regime, psi)
                for psi in (0.0, 0.4, 1.0)]

    columns = []
    for title, topo, point, psi in spec:
        path = find_dir(args.datadir, topo, point, psi)
        mean, sd, meta = load(path)
        columns.append((title, mean, sd, meta))

    out = make_figure(columns, args)

    print(f"\n  layout={args.layout}"
          + (f"  regime={args.regime} ({REGIME_TITLE[args.regime]})"
             if args.layout != "main" else "")
          + f"  ->  {out}\n")
    print(f"  {'panel':<26}{'realizations':>13}{'K = 1':>9}{'settled':>10}"
          f"{'point':>7}")
    print("  " + "-" * 65)
    for title, _, _, meta in columns:
        clean = title.replace("$", "").replace("\\", "")
        print(f"  {clean:<26}{meta['n_realizations']:>13}"
              f"{meta['consensus_fraction']:>9.3f}"
              f"{meta['settled_fraction']:>10.3f}"
              f"{meta['point']:>7}")
    bad = [t for t, _, _, m in columns if m["consensus_fraction"] < 1.0]
    if bad:
        print("\n  WARNING: not every (rho, p_s) cell reached a single reach in "
              "every realization.\n  Where K > 1 the plotted value is a network "
              "average over several reaches,\n  so 'consensus belief state' "
              "would overstate it in the caption.")
    if any(m["settled_fraction"] < 1.0 for _, _, _, m in columns):
        print("\n  WARNING: some cells had not settled by the integration "
              "horizon; raise T_HORIZON in run_trends.py.")


if __name__ == "__main__":
    main()
