#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-panel version of Figure 2: Delta b* over the threshold plane (T_T, U_T)
for several trust-distrust couplings rho, side by side under one shared
colorbar.

Reads the merged arrays written by fig02_delta_b.py --merge, so it does no
computation and takes a second to run.  Each panel needs

    delta_b_results_rho<rho>/delta_b_mean.npy

to exist already.  Styling (label sizes, tick geometry, spine widths, stix
mathtext) is the same as fig01_phase_diagram.py and fig02_delta_b.py.

Usage
-----
  python fig02_panels.py                                  # the four default rho
  python fig02_panels.py --rhos -0.01 -0.4 -0.7 -1
  python fig02_panels.py --vmax 3 --out fig02_panels.pdf
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from gnc_core import threshold_grid


# ═══════════════════════════════════════════════════════════════════════════════
#  Options
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[-0.01, -0.4, -0.7, -1.0],
                    help="rho values, left to right")
    ap.add_argument("--indir-template", type=str,
                    default="delta_b_results_rho{rho:g}",
                    help="folder holding delta_b_mean.npy for each rho")
    ap.add_argument("--out", type=str, default="fig02_panels_ER.pdf")
    ap.add_argument("--vmax", type=float, default=3.0,
                    help="shared colour limit; the scale is -vmax .. +vmax. "
                         "Use 0 to take the 90th percentile of |Delta b*| "
                         "pooled over all panels.")
    ap.add_argument("--wspace", type=float, default=0.08,
                    help="horizontal gap between panels, in axes widths")
    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Style  (identical to fig01_phase_diagram.py / fig02_delta_b.py)
# ═══════════════════════════════════════════════════════════════════════════════
TICK_LABELSIZE    = 14
AXIS_LABELSIZE    = 16
TITLE_SIZE        = 16
SPINE_WIDTH       = 1.5
MAJOR_TICK_WIDTH  = 1.2
MAJOR_TICK_LENGTH = 4
MINOR_TICK_WIDTH  = 1.0
MINOR_TICK_LENGTH = 3

plt.rcParams.update({
    "font.size":        14,
    "axes.labelsize":   AXIS_LABELSIZE,
    "axes.titlesize":   TITLE_SIZE,
    "xtick.labelsize":  TICK_LABELSIZE,
    "ytick.labelsize":  TICK_LABELSIZE,
    "mathtext.fontset": "stix",
})

# Panels are narrower than the single-panel figure, so the labelled markers and
# the region label are scaled down to fit.
MARKER_SIZE      = 300
MARKER_FONTSIZE  = 11
REGION_FONTSIZE  = 11
PANEL_SIZE       = 2.9      # side of one panel, in inches
CBAR_WIDTH       = 0.16     # in inches
CBAR_PAD         = 0.18     # in inches

LABELED_POINTS = {
    'A': (-0.8,  0.8),
    'E': ( 0.2,  0.8),
    'F': (-0.8,  0.0),
}
REGION_LABEL_AT = (0.25, -0.45)


# ═══════════════════════════════════════════════════════════════════════════════
def load_panel(rho, template):
    d = template.format(rho=rho)
    path = os.path.join(d, "delta_b_mean.npy")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            f"  run:  python fig02_delta_b.py --rho {rho:g} --merge\n"
            f"  (and the array job for that rho first, if not done yet)")
    return np.load(path)


def main():
    args = _parse_cli()
    panels = [(rho, load_panel(rho, args.indir_template)) for rho in args.rhos]

    shapes = {p.shape for _, p in panels}
    if len(shapes) != 1:
        raise SystemExit(f"panels have different grid shapes: {shapes}")
    n_ut, n_tt = panels[0][1].shape
    TT = threshold_grid(2.0 / (n_tt - 1))
    UT = threshold_grid(2.0 / (n_ut - 1))

    if args.vmax > 0:
        vmax = args.vmax
    else:
        pooled = np.concatenate([p[np.isfinite(p)].ravel() for _, p in panels])
        vmax = float(np.percentile(np.abs(pooled), 90)) if pooled.size else 1.0
    cnorm = Normalize(vmin=-vmax, vmax=vmax, clip=True)

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="black")

    n = len(panels)
    fig_w = PANEL_SIZE * n + CBAR_PAD + CBAR_WIDTH + 1.4   # 1.4 in for labels
    fig_h = PANEL_SIZE + 1.1                               # for xlabel + title
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes = [axes]

    formatter = FuncFormatter(lambda x, pos: f"{x:g}")
    im = None
    for k, (ax, (rho, data)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            np.ma.masked_invalid(data), cmap=cmap, norm=cnorm, origin="lower",
            extent=[TT[0], TT[-1], UT[0], UT[-1]],
            aspect="equal", interpolation="bilinear",
        )

        ax.set_title(rf"$\rho = {rho:g}$", fontsize=TITLE_SIZE, pad=8)
        ax.set_xlabel(r"$\mathrm{T}_{\mathrm{T}}$", fontsize=AXIS_LABELSIZE)
        if k == 0:
            ax.set_ylabel(r"$\mathrm{U}_{\mathrm{T}}$", fontsize=AXIS_LABELSIZE)

        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

        for spine in ax.spines.values():
            spine.set_linewidth(SPINE_WIDTH)
            spine.set_color("black")
        ax.tick_params(axis="both", which="major", width=MAJOR_TICK_WIDTH,
                       length=MAJOR_TICK_LENGTH, labelsize=TICK_LABELSIZE)
        ax.tick_params(axis="both", which="minor", width=MINOR_TICK_WIDTH,
                       length=MINOR_TICK_LENGTH)

        ax.text(*REGION_LABEL_AT, r"$\mathit{fragmentation}$",
                ha="center", va="center", fontsize=REGION_FONTSIZE,
                color="white", zorder=6)

        for label, (tt, ut) in LABELED_POINTS.items():
            ax.scatter(tt, ut, s=MARKER_SIZE, facecolors="#b3ff00",
                       edgecolors="black", linewidths=1.2, zorder=5)
            ax.text(tt, ut, label, ha="center", va="center",
                    fontsize=MARKER_FONTSIZE, fontweight="bold", zorder=6)

    fig.subplots_adjust(wspace=args.wspace)
    fig.tight_layout()

    # One shared colorbar, placed after tight_layout so the panel geometry is
    # not disturbed by it.  Its height is matched to the panels.
    pos_first = axes[0].get_position()
    pos_last = axes[-1].get_position()
    cax = fig.add_axes([
        pos_last.x1 + CBAR_PAD / fig_w,
        pos_first.y0,
        CBAR_WIDTH / fig_w,
        pos_first.height,
    ])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        r"$\Delta b^{*} = b^{*}(p_s = 1) - b^{*}(p_s = 0)$",
        fontsize=AXIS_LABELSIZE)
    cbar.outline.set_linewidth(SPINE_WIDTH)
    cbar.outline.set_edgecolor("black")
    cax.tick_params(axis="both", which="major", width=MAJOR_TICK_WIDTH,
                    length=MAJOR_TICK_LENGTH, labelsize=TICK_LABELSIZE)

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  {n} panels, shared scale +-{vmax:g} -> {args.out}")
    for rho, data in panels:
        finite = data[np.isfinite(data)]
        n_clip = int((np.abs(finite) > vmax).sum()) if finite.size else 0
        print(f"    rho={rho:>6g}  consensus cells {finite.size:4d}"
              f"  range [{finite.min():+.2f}, {finite.max():+.2f}]"
              + (f"  {n_clip} cell(s) clipped by the shared scale"
                 if n_clip else ""))


if __name__ == "__main__":
    main()
