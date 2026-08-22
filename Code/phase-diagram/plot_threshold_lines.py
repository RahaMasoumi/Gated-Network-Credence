#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-panel Delta b* figure with the analytic activation thresholds drawn on top.

Standalone: imports nothing from the simulation scripts.  It only reads the
merged arrays

    delta_b_results_rho<rho>/delta_b_mean.npy

and overlays four lines that are closed-form functions of rho alone, so their
positions are exact and require no fitting to the data:

    ceiling on:   T_T = -(1 - rho) / 2        (vertical)
    ceiling off:  U_T =  (1 + rho) / 2        (horizontal)
    floor on:     T_T =  (1 - rho) / 2        (vertical)
    floor off:    U_T = -(1 + rho) / 2        (horizontal)

and optionally the line on which the admissible set becomes nonempty,

    U_T = [(1 + rho) / (1 - rho)] * T_T      (--diagonal)

What is exact and what is not
-----------------------------
The lines are exact.  Their agreement with the simulated boundary is limited by
two things that have nothing to do with the analytics:

  * the threshold grid has spacing 0.05, so a boundary in the data is only
    localised to within one cell;
  * imshow's bilinear interpolation smears the boundary by about half a cell.
    Use --interpolation nearest when the point of the figure is to compare the
    lines against the data, and bilinear when the point is a smooth map.

For a numerical rather than visual comparison, read the interpolated sign
change with measure_F_edge.py and compare it with the printed line positions.

Usage
-----
  python plot_threshold_lines.py
  python plot_threshold_lines.py --interpolation nearest --diagonal
  python plot_threshold_lines.py --rhos -0.01 -0.4 -0.7 -1 --vmax 3 \
      --out fig02_panels_thresholds.pdf
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


# ═══════════════════════════════════════════════════════════════════════════════
#  Options
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[-0.01, -0.4, -0.7, -1.0])
    ap.add_argument("--indir-template", type=str,
                    default="delta_b_results_rho{rho:g}")
    ap.add_argument("--out", type=str, default="fig02_panels_thresholds.pdf")
    ap.add_argument("--vmax", type=float, default=3.0,
                    help="shared colour limit; 0 = 90th percentile pooled")
    ap.add_argument("--wspace", type=float, default=0.08)
    ap.add_argument("--interpolation", type=str, default="bilinear",
                    choices=["bilinear", "nearest"],
                    help="use nearest when comparing the lines with the data")
    ap.add_argument("--diagonal", action="store_true",
                    help="also draw U_T = [(1+rho)/(1-rho)] T_T")
    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--no-markers", action="store_true",
                    help="omit the A / E / F markers, which can sit on a line")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Style
# ═══════════════════════════════════════════════════════════════════════════════
TICK_LABELSIZE    = 14
AXIS_LABELSIZE    = 16
TITLE_SIZE        = 16
SPINE_WIDTH       = 1.5
MAJOR_TICK_WIDTH  = 1.2
MAJOR_TICK_LENGTH = 4

plt.rcParams.update({
    "font.size":        14,
    "axes.labelsize":   AXIS_LABELSIZE,
    "axes.titlesize":   TITLE_SIZE,
    "xtick.labelsize":  TICK_LABELSIZE,
    "ytick.labelsize":  TICK_LABELSIZE,
    "mathtext.fontset": "stix",
})

PANEL_SIZE  = 2.9
CBAR_WIDTH  = 0.16
CBAR_PAD    = 0.18

MARKER_SIZE     = 300
MARKER_FONTSIZE = 11
REGION_FONTSIZE = 11

LABELED_POINTS  = {'A': (-0.8, 0.8), 'E': (0.2, 0.8), 'F': (-0.8, 0.0)}
REGION_LABEL_AT = (0.25, -0.45)

# Each line gets its own hue plus a black casing, so it stays legible over the
# dark red, the dark blue, the near-white band and the black region alike.  The
# two line styles distinguish the ceiling pair from the floor pair, which keeps
# the figure readable in greyscale.
LINE_WIDTH   = 2.0
CASING_WIDTH = 4.0

THRESHOLD_LINES = [
    # (label, kind, value function of rho, colour, linestyle)
    (r"$T_T = -(1-\rho)/2$",  "v", lambda r: -(1.0 - r) / 2.0, "#00D9FF", "-"),
    (r"$U_T = (1+\rho)/2$",   "h", lambda r:  (1.0 + r) / 2.0, "#FFB000", "-"),
    (r"$T_T = (1-\rho)/2$",   "v", lambda r:  (1.0 - r) / 2.0, "#FF2D95", "--"),
    (r"$U_T = -(1+\rho)/2$",  "h", lambda r: -(1.0 + r) / 2.0, "#00E676", "--"),
]
DIAGONAL_LABEL = r"$U_T = \frac{1+\rho}{1-\rho}\,T_T$"
DIAGONAL_COLOR = "#FFFFFF"


def _casing(color, style):
    """A line drawn twice: a black casing underneath, the colour on top."""
    return dict(color=color, linestyle=style, linewidth=LINE_WIDTH,
                solid_capstyle="butt", zorder=8,
                path_effects=[pe.withStroke(linewidth=CASING_WIDTH,
                                            foreground="black")])


# ═══════════════════════════════════════════════════════════════════════════════
def load_panel(rho, template):
    path = os.path.join(template.format(rho=rho), "delta_b_mean.npy")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            f"  run:  python fig02_delta_b.py --rho {rho:g} --merge")
    return np.load(path)


def draw_lines(ax, rho, diagonal):
    """Draw the analytic thresholds on one panel, clipped to [-1, 1]^2."""
    drawn = []
    for label, kind, fn, color, style in THRESHOLD_LINES:
        v = fn(rho)
        if not (-1.0 <= v <= 1.0):
            continue                       # threshold outside the swept range
        if kind == "v":
            ax.plot([v, v], [-1, 1], **_casing(color, style))
        else:
            ax.plot([-1, 1], [v, v], **_casing(color, style))
        drawn.append((label, v))

    if diagonal:
        slope = (1.0 + rho) / (1.0 - rho)
        x = np.array([-1.0, 1.0])
        y = np.clip(slope * x, -1.0, 1.0)
        ax.plot(x, y, **_casing(DIAGONAL_COLOR, ":"))
        drawn.append((DIAGONAL_LABEL, slope))

    return drawn


def main():
    args = _parse_cli()
    panels = [(rho, load_panel(rho, args.indir_template)) for rho in args.rhos]

    shapes = {p.shape for _, p in panels}
    if len(shapes) != 1:
        raise SystemExit(f"panels have different grid shapes: {shapes}")
    n_ut, n_tt = panels[0][1].shape
    step = 2.0 / (n_tt - 1)

    if args.vmax > 0:
        vmax = args.vmax
    else:
        pooled = np.concatenate([p[np.isfinite(p)].ravel() for _, p in panels])
        vmax = float(np.percentile(np.abs(pooled), 90)) if pooled.size else 1.0
    cnorm = Normalize(vmin=-vmax, vmax=vmax, clip=True)

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="black")

    n = len(panels)
    fig_w = PANEL_SIZE * n + CBAR_PAD + CBAR_WIDTH + 1.4
    fig_h = PANEL_SIZE + 1.1 + (0.8 if not args.no_legend else 0.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    formatter = FuncFormatter(lambda x, pos: f"{x:g}")
    im = None
    table = []
    for k, (ax, (rho, data)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            np.ma.masked_invalid(data), cmap=cmap, norm=cnorm, origin="lower",
            extent=[-1, 1, -1, 1], aspect="equal",
            interpolation=args.interpolation,
        )

        ax.set_title(rf"$\rho = {rho:g}$", fontsize=TITLE_SIZE, pad=8)
        ax.set_xlabel(r"$\mathrm{T}_{\mathrm{T}}$", fontsize=AXIS_LABELSIZE)
        if k == 0:
            ax.set_ylabel(r"$\mathrm{U}_{\mathrm{T}}$", fontsize=AXIS_LABELSIZE)

        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
            spine.set_color("black")
        ax.tick_params(axis="both", which="major", width=MAJOR_TICK_WIDTH,
                       length=MAJOR_TICK_LENGTH, labelsize=TICK_LABELSIZE)

        ax.text(*REGION_LABEL_AT, r"$\mathit{fragmentation}$",
                ha="center", va="center", fontsize=REGION_FONTSIZE,
                color="white", zorder=6)

        if not args.no_markers:
            for label, (tt, ut) in LABELED_POINTS.items():
                ax.scatter(tt, ut, s=MARKER_SIZE, facecolors="#b3ff00",
                           edgecolors="black", linewidths=1.2, zorder=9)
                ax.text(tt, ut, label, ha="center", va="center",
                        fontsize=MARKER_FONTSIZE, fontweight="bold", zorder=10)

        table.append((rho, draw_lines(ax, rho, args.diagonal)))

    fig.subplots_adjust(wspace=args.wspace)
    fig.tight_layout()

    pos_first = axes[0].get_position()
    pos_last = axes[-1].get_position()
    cax = fig.add_axes([pos_last.x1 + CBAR_PAD / fig_w, pos_first.y0,
                        CBAR_WIDTH / fig_w, pos_first.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r"$\Delta b^{*} = b^{*}(p_s = 1) - b^{*}(p_s = 0)$",
                   fontsize=AXIS_LABELSIZE)
    cbar.outline.set_linewidth(SPINE_WIDTH)
    cbar.outline.set_edgecolor("black")
    cax.tick_params(axis="both", which="major", width=MAJOR_TICK_WIDTH,
                    length=MAJOR_TICK_LENGTH, labelsize=TICK_LABELSIZE)

    if not args.no_legend:
        handles = [
            Line2D([0], [0], color=c, linestyle=s, linewidth=LINE_WIDTH,
                   path_effects=[pe.withStroke(linewidth=CASING_WIDTH,
                                               foreground="black")], label=lab)
            for lab, _, _, c, s in THRESHOLD_LINES
        ]
        if args.diagonal:
            handles.append(
                Line2D([0], [0], color=DIAGONAL_COLOR, linestyle=":",
                       linewidth=LINE_WIDTH,
                       path_effects=[pe.withStroke(linewidth=CASING_WIDTH,
                                                   foreground="black")],
                       label=DIAGONAL_LABEL))
        fig.legend(handles=handles, loc="lower center",
                   ncol=len(handles), frameon=False, fontsize=13,
                   bbox_to_anchor=(0.5, -0.02),
                   handlelength=2.6, columnspacing=1.8)

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- exact line positions, for the caption and the appendix text ----
    print(f"\n  {n} panels, shared scale +-{vmax:g}, "
          f"interpolation={args.interpolation} -> {args.out}")
    print(f"  the lines are exact functions of rho; the grid spacing is "
          f"{step:.3f}, so a boundary")
    print(f"  in the DATA is localised only to within one cell, which is the "
          f"accuracy at which\n  agreement can be claimed\n")

    cols = [("T_T=-(1-rho)/2", 0), ("U_T=(1+rho)/2", 1),
            ("T_T=(1-rho)/2", 2), ("U_T=-(1+rho)/2", 3)]
    header = f"{'rho':>7}" + "".join(f"{name:>17}" for name, _ in cols)
    if args.diagonal:
        header += f"{'diag slope':>13}"
    print(header)
    print("-" * len(header))
    for rho, _ in table:
        row = f"{rho:>7g}"
        for _, idx in cols:
            v = THRESHOLD_LINES[idx][2](rho)
            off = "" if -1.0 <= v <= 1.0 else "*"
            row += f"{v:>16.3f}{off:<1}"
        if args.diagonal:
            row += f"{(1.0 + rho) / (1.0 - rho):>13.3f}"
        print(row)
    print("\n  * outside the swept range [-1, 1] and therefore not drawn")

    for rho, _ in table:
        notes = []
        if abs((1.0 + rho) / 2.0) < step / 2:
            notes.append("the two horizontal lines coincide at U_T = 0")
        if abs((1.0 - rho) / 2.0 - 1.0) < step / 2:
            notes.append("the two vertical lines sit on the panel edges")
        if notes:
            print(f"  rho={rho:>6g}: " + "; ".join(notes))


if __name__ == "__main__":
    main()
