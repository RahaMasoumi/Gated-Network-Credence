#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Delta b* = b*(p_s = 1) - b*(p_s = 0) over the threshold plane
(T_T, U_T) for an Erdos-Renyi network.

For each grid point and each p_s we build A_IN (Methods 4.1, Eq. 1), check that
the induced influence graph has a single reach (K = 1, Methods 4.2), and, if so,
integrate the gated dynamics b(t) = exp(-L t) b(0) with L = D_row - A_IN
(Eq. 3) to t = T_MAX.  Delta b* is the difference of the two consensus values.
Grid points that do not reach consensus are reported as fragmentation.

All model definitions live in gnc_core.py, shared with fig01_phase_diagram.py,
so the phase boundary of Figure 1 and the fragmentation mask of Figure 2 come
from one implementation of A_IN, L and K.  BASE_SEED is also shared, so
realization r is the same network and the same initial-belief configuration in
both figures (Methods 4.3.6, "Replication").

Usage
-----
  python fig02_delta_b.py                      # all realizations, then merge
  python fig02_delta_b.py --realization_id 7   # one realization (SLURM array)
  python fig02_delta_b.py --merge              # combine and plot
"""

import os
import glob
import json
import pickle
import time
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from gnc_core import (
    threshold_grid,
    build_realization,
    net_trust_uncertainty,
    apply_prestige_bias,
    build_A_IN,
    gated_laplacian,
    count_cabals,
    long_run_belief,
    B_MAX_DEFAULT,
)


# ====================== Command-line options ======================
def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=-0.4,
                    help="trust-distrust coupling (e.g. -0.4, -1, -0.01)")
    ap.add_argument("--outdir", type=str, default=None,
                    help="output folder; default = delta_b_results_rho<rho>")
    ap.add_argument("--ps-low", type=float, default=0.0)
    ap.add_argument("--ps-high", type=float, default=1.0)
    ap.add_argument("--prestige-norm", type=str, default="rank",
                    choices=["rank", "minmax"])
    ap.add_argument("--realization_id", type=int, default=None,
                    help="run ONE realization (for a SLURM array); "
                         "omit to run all N_REAL of them serially")
    ap.add_argument("--merge", action="store_true",
                    help="combine the per-realization files and plot")
    a, _ = ap.parse_known_args()
    return a

_CLI = _parse_cli()


# ====================== Parameters  (Methods 4.3.5, 4.3.6) ======================
N       = 5000        # Methods 4.3.6
P_EDGE  = 0.01        # Erdos-Renyi edge probability
RHO     = _CLI.rho    # Methods 4.3.5: rho in {-1, -0.7, -0.4, -0.01}
SIGMA   = 0.05        # Methods 4.3.5: sigma_eps = 0.05
B_MAX   = B_MAX_DEFAULT   # Methods 4.3.5: b_i(0) ~ U(-B_max, B_max), B_max = 5
X_PCT   = 0.2         # promoter prevalence x, fixed at 0.2 for Figures 1-2

PS_LOW  = _CLI.ps_low
PS_HIGH = _CLI.ps_high
PS_VALUES = [PS_LOW, PS_HIGH]

PRESTIGE_NORM = _CLI.prestige_norm

GRID_STEP = 0.05
TT_GRID = threshold_grid(GRID_STEP)
UT_GRID = threshold_grid(GRID_STEP)

N_REAL = 100          # Methods 4.3.6: 100 independent realizations.  Whatever
                      # value is used here must be the value reported in the
                      # manuscript.
T_MAX = 10            # integration horizon; at T_MAX = 10 the consensus cells
                      # are fully settled (max(b) - min(b) ~ 1e-17), so Delta b*
                      # is insensitive to the horizon.

# A grid point counts as fragmented when the fraction of realizations reaching
# consensus falls below this level.  0.5 is the same level as the phase boundary
# drawn in Figure 1(a), so the black region of Figure 2 and the white curve of
# Figure 1 mark the same transition.  Set to 1.0 for the stricter "fragmented if
# ANY realization fragmented" convention -- but then say so in the caption.
CONSENSUS_FRAC_MIN = 0.5

# Shared with fig01_phase_diagram.py: realization r is the same network and the
# same initial beliefs in both figures.
BASE_SEED = 20260819

COLOR_PCTL = 90
COLOR_VMAX = None

OUTDIR = (_CLI.outdir if _CLI.outdir is not None
          else f"delta_b_results_rho{RHO:g}")


def _params():
    """Recorded in every per-realization file and checked at merge time, so a
    merge can never silently mix runs made with different settings."""
    return {
        "N": N, "p": P_EDGE, "rho": RHO, "sigma": SIGMA, "B_max": B_MAX,
        "x_percent": X_PCT, "ps_low": PS_LOW, "ps_high": PS_HIGH,
        "prestige_norm": PRESTIGE_NORM, "grid_step": GRID_STEP,
        "n_grid": int(TT_GRID.size), "T_max": T_MAX, "base_seed": BASE_SEED,
    }


# ====================== One realization ======================
def run_realization(realization_id):
    # Network, initial beliefs with degree-biased positive seeding, trust and
    # distrust, and the prestige scores: all drawn by the shared builder, so
    # this realization is identical to realization `realization_id` of
    # fig01_phase_diagram.py (Methods 4.3.6, "Replication").
    R = build_realization(realization_id, N, P_EDGE, RHO, SIGMA, X_PCT, BASE_SEED)
    b0 = R.b0

    # Methods 4.3.2.  eta and the degree tie-break belong to the realization and
    # are shared by the two p_s values, so the two runs differ only in the
    # weight given to prestige and not in the underlying random scores.
    TU = {}
    for ps in PS_VALUES:
        tau, delta = apply_prestige_bias(R.tau, R.delta, R.src, R.indptr, R.deg,
                                         ps, R.eta, R.tiebreak,
                                         norm_mode=PRESTIGE_NORM)
        TU[ps] = net_trust_uncertainty(tau, delta)

    # --- threshold sweep ---
    n_tt, n_ut = TT_GRID.size, UT_GRID.size
    deltab = np.full((n_ut, n_tt), np.nan)
    consensus = np.zeros((n_ut, n_tt), dtype=bool)
    # K is recorded for each p_s separately, so that the p_s = 0 mask of this
    # figure can be checked against the phase diagram of Figure 1 cell by cell.
    K_ps = {ps: np.zeros((n_ut, n_tt), dtype=np.int32) for ps in PS_VALUES}

    for ti, TT in enumerate(TT_GRID):
        for ui, UT in enumerate(UT_GRID):
            A = {}
            for ps in PS_VALUES:
                T, U = TU[ps]
                A[ps] = build_A_IN(R.recv, R.src, T, U, TT, UT, N)
                K_ps[ps][ui, ti] = count_cabals(A[ps])   # K != 1 -> fragmentation

            if all(K_ps[ps][ui, ti] == 1 for ps in PS_VALUES):
                consensus[ui, ti] = True
                b_star = {ps: float(np.mean(long_run_belief(
                    gated_laplacian(A[ps]), b0, T_MAX))) for ps in PS_VALUES}
                deltab[ui, ti] = b_star[PS_HIGH] - b_star[PS_LOW]

    os.makedirs(OUTDIR, exist_ok=True)
    tmp = os.path.join(OUTDIR, f".realization_{realization_id}.tmp")
    final = os.path.join(OUTDIR, f"realization_{realization_id}.pkl")
    with open(tmp, "wb") as f:
        pickle.dump({"deltab": deltab, "consensus": consensus,
                     "K_ps_low": K_ps[PS_LOW], "K_ps_high": K_ps[PS_HIGH],
                     "realization_id": realization_id, "params": _params()}, f)
    os.replace(tmp, final)
    print(f"realization {realization_id}: {int(consensus.sum())} consensus cells, "
          f"{int((~consensus).sum())} fragmented -> {final}", flush=True)
    return final


# ====================== Merge ======================
def merge():
    """Cell-wise mean of Delta b* over the realizations that reached consensus,
    together with the fraction that did.  A cell is reported as fragmentation
    when that fraction falls below CONSENSUS_FRAC_MIN."""
    files = sorted(glob.glob(os.path.join(OUTDIR, "realization_*.pkl")))
    if not files:
        raise SystemExit(f"no realization_*.pkl in {OUTDIR!r}")

    n_ut, n_tt = UT_GRID.size, TT_GRID.size
    total = np.zeros((n_ut, n_tt))
    n_cons = np.zeros((n_ut, n_tt), dtype=np.int64)
    reference = _params()
    seen = set()

    for fn in files:
        with open(fn, "rb") as f:
            D = pickle.load(f)
        if D.get("params") != reference:
            raise SystemExit(
                f"{fn}: parameters differ from the current settings.\n"
                f"  file:    {json.dumps(D.get('params'), sort_keys=True)}\n"
                f"  current: {json.dumps(reference, sort_keys=True)}")
        rid = D.get("realization_id")
        if rid in seen:
            raise SystemExit(f"{fn}: realization {rid} appears twice")
        seen.add(rid)
        ok = D["consensus"]
        total[ok] += D["deltab"][ok]
        n_cons[ok] += 1

    n_files = len(files)
    frac = n_cons / n_files
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_cons > 0, total / np.maximum(n_cons, 1), np.nan)
    mean[frac < CONSENSUS_FRAC_MIN] = np.nan

    np.save(os.path.join(OUTDIR, "delta_b_mean.npy"), mean)
    np.save(os.path.join(OUTDIR, "consensus_fraction.npy"), frac)
    np.save(os.path.join(OUTDIR, "n_consensus.npy"), n_cons)
    np.save(os.path.join(OUTDIR, "TT_grid.npy"), TT_GRID)
    np.save(os.path.join(OUTDIR, "UT_grid.npy"), UT_GRID)
    with open(os.path.join(OUTDIR, "params.json"), "w") as f:
        json.dump({**reference, "n_realizations_merged": n_files,
                   "consensus_frac_min": CONSENSUS_FRAC_MIN}, f, indent=2)

    print(f"merged {n_files} realizations -> {OUTDIR}/delta_b_mean.npy")
    if n_files != N_REAL:
        print(f"  NOTE: merged {n_files} files but N_REAL = {N_REAL}")
    return mean, frac


# ====================== Plotting ======================
# Style constants and rcParams are identical to fig01_phase_diagram.py, so the
# two figures share axis label sizes, tick label sizes, spine widths and tick
# geometry.  The colorbar reuses the same sizes.
TICK_LABELSIZE    = 14
AXIS_LABELSIZE    = 16
SPINE_WIDTH       = 1.5
MAJOR_TICK_WIDTH  = 1.2
MAJOR_TICK_LENGTH = 4
MINOR_TICK_WIDTH  = 1.0
MINOR_TICK_LENGTH = 3

plt.rcParams.update({
    "font.size":        14,
    "axes.labelsize":   AXIS_LABELSIZE,
    "xtick.labelsize":  TICK_LABELSIZE,
    "ytick.labelsize":  TICK_LABELSIZE,
    "mathtext.fontset": "stix",
})

# Colorbar geometry, in axes-fraction units of the main axes.  Placed with
# ax.inset_axes so that it sits outside the axes and is ignored by
# tight_layout: the main axes therefore ends up exactly the same size as the
# one in fig01_phase_diagram.py, which is built with the same figsize, the same
# labels and the same tight_layout call.
CBAR_PAD   = 0.04
CBAR_WIDTH = 0.045

LABELED_POINTS = {
    'E': ( 0.2, 0.8),
    'F': (-0.8, 0.0),
}


def plot(deltab_mean, outfile):
    # Same figsize, same aspect and same tight_layout as
    # fig01_phase_diagram.plot_colmap, so the two plotting boxes match.
    fig, ax = plt.subplots(figsize=(5, 5))

    finite_vals = deltab_mean[np.isfinite(deltab_mean)]
    if COLOR_VMAX is not None:
        vmax = COLOR_VMAX
    elif finite_vals.size:
        vmax = np.percentile(np.abs(finite_vals), COLOR_PCTL)
    else:
        vmax = 1.0
    cnorm = Normalize(vmin=-vmax, vmax=vmax, clip=True)

    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="black")

    im = ax.imshow(
        np.ma.masked_invalid(deltab_mean),
        cmap=cmap, norm=cnorm, origin="lower",
        extent=[TT_GRID[0], TT_GRID[-1], UT_GRID[0], UT_GRID[-1]],
        aspect="auto",
        interpolation="bilinear",
    )

    ax.set_xlabel(r"$\mathrm{T}_{\mathrm{T}}$", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel(r"$\mathrm{U}_{\mathrm{T}}$", fontsize=AXIS_LABELSIZE)

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    formatter = FuncFormatter(lambda x, pos: f"{x:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("black")

    ax.tick_params(axis="both", which="major", width=MAJOR_TICK_WIDTH,
                   length=MAJOR_TICK_LENGTH, labelsize=TICK_LABELSIZE)
    ax.tick_params(axis="both", which="minor", width=MINOR_TICK_WIDTH,
                   length=MINOR_TICK_LENGTH)

    ax.text(0.0, -0.6, r"$\mathrm{fragmentation}$",
            ha="center", va="center", fontsize=18, color="white", zorder=6)

    for label, (tt, ut) in LABELED_POINTS.items():
        ax.scatter(tt, ut, s=800, facecolors="#b3ff00", edgecolors="black",
                   linewidths=1.5, zorder=5)
        ax.text(tt, ut, label, ha="center", va="center",
                fontsize=16, fontweight="bold", zorder=6)

    # tight_layout is called before the colorbar is attached, so the main axes
    # geometry is fixed by the same inputs as in fig01_phase_diagram.py.
    plt.tight_layout()

    cax = ax.inset_axes([1.0 + CBAR_PAD, 0.0, CBAR_WIDTH, 1.0])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        rf"$\Delta b^{{*}} = b^{{*}}(p_s={PS_HIGH:g}) - b^{{*}}(p_s={PS_LOW:g})$",
        fontsize=AXIS_LABELSIZE)
    cbar.outline.set_linewidth(SPINE_WIDTH)
    cbar.outline.set_edgecolor("black")
    cax.tick_params(axis="both", which="major", width=MAJOR_TICK_WIDTH,
                    length=MAJOR_TICK_LENGTH, labelsize=TICK_LABELSIZE)
    cax.tick_params(axis="both", which="minor", width=MINOR_TICK_WIDTH,
                    length=MINOR_TICK_LENGTH)

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {outfile}")


if __name__ == "__main__":
    t0 = time.time()
    figfile = os.path.join(OUTDIR, f"delta_b_colormap_rho{RHO:g}.pdf")
    if _CLI.merge:
        mean, frac = merge()
        plot(mean, figfile)
    elif _CLI.realization_id is not None:
        run_realization(_CLI.realization_id)
    else:
        print(f"rho = {RHO:g}   ->   writing to '{OUTDIR}'")
        print(f"prestige: rank-association permutation "
              f"(norm='{PRESTIGE_NORM}'), ps: {PS_LOW:g} -> {PS_HIGH:g}")
        for rid in range(N_REAL):
            run_realization(rid)
        mean, frac = merge()
        plot(mean, figfile)
    print(f"Done in {time.time() - t0:.1f}s")