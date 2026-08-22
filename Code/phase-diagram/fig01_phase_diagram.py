#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1(a) (2): consensus / fragmentation phase diagram of Gated Network
Credence on an Erdos-Renyi graph.

For each threshold pair (T_T, U_T) we build the masked adjacency matrix A_IN
(Methods 4.1, Eq. 1),

    A_IN[i, j] = 1  iff  A[i, j] = 1 and T_ij >= T_T and U_ij <= U_T,

with row = receiver and column = source, and count K, the number of reaches of
the induced influence graph (equivalently, the number of cabals; Methods 4.2).
Global consensus arises if and only if K = 1.  The criterion is purely
topological, so no belief trajectory has to be integrated here.

All model definitions live in gnc_core.py, shared with fig02_delta_b.py, and
both scripts obtain their realizations from gnc_core.build_realization, so
realization r is the same network with the same trust-distrust values in both
figures.

Per grid point we run N_REAL independent realizations and store the FRACTION
that reached consensus; that fraction is the colormap intensity and the phase
boundary is its 0.5 level.

Usage
-----
  python fig01_phase_diagram.py                      # all realizations, merge, plot
  python fig01_phase_diagram.py --realization_id 7   # one realization (SLURM array)
  python fig01_phase_diagram.py --merge              # combine and plot
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
from matplotlib.ticker import FuncFormatter

from gnc_core import (
    threshold_grid,
    build_realization,
    apply_prestige_bias,
    net_trust_uncertainty,
    build_A_IN,
    count_cabals,
    B_MAX_DEFAULT,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Command-line options
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=float, default=0.0,
                    help="prestige bias; Figure 1 is the p_s = 0 condition")
    ap.add_argument("--rho", type=float, default=-0.4,
                    help="trust-distrust coupling; Figure 1 uses -0.4")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--realization_id", type=int, default=None,
                    help="run ONE realization (for a SLURM array); "
                         "omit to run all N_REAL of them serially")
    ap.add_argument("--merge", action="store_true",
                    help="combine the per-realization files and plot")
    a, _ = ap.parse_known_args()
    return a

_CLI = _parse_cli()


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameters  (Methods 4.3.5, 4.3.6)
# ═══════════════════════════════════════════════════════════════════════════════
N       = 5000        # Methods 4.3.6
P_EDGE  = 0.01        # Erdos-Renyi edge probability
RHO     = _CLI.rho    # trust-distrust coupling
SIGMA   = 0.05        # Methods 4.3.5: sigma_eps = 0.05
PS      = _CLI.ps     # prestige bias
X_PCT   = 0.2         # promoter prevalence; unused by this figure, kept so the
                      # realization is identical to the one used for Figure 2

GRID_STEP = 0.05      # -> 41 x 41
TT_GRID = threshold_grid(GRID_STEP)
UT_GRID = threshold_grid(GRID_STEP)

N_REAL = 100          # Methods 4.3.6: 100 independent realizations per topology
                      # and parameter combination.  Whatever value is used here
                      # must be the value reported in the manuscript.

PRESTIGE_NORM = "rank"

# Shared with fig02_delta_b.py: realization r is the same network in both.
BASE_SEED = 20260819

OUTDIR = (_CLI.outdir if _CLI.outdir is not None
          else f"phase_diagram_rho{RHO:g}_ps{PS:g}")

DRAW_BOUNDARY_CONTOUR = True
BOUNDARY_LEVEL        = 0.5


def _params():
    """Recorded in every per-realization file and checked at merge time."""
    return {
        "N": N, "p": P_EDGE, "rho": RHO, "sigma": SIGMA, "ps": PS,
        "B_max": B_MAX_DEFAULT, "x_percent": X_PCT,
        "prestige_norm": PRESTIGE_NORM, "grid_step": GRID_STEP,
        "n_grid": int(TT_GRID.size), "base_seed": BASE_SEED,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  One realization
# ═══════════════════════════════════════════════════════════════════════════════
def run_realization(realization_id, write=True):
    """K over the whole threshold grid, for one realization."""
    R = build_realization(realization_id, N, P_EDGE, RHO, SIGMA, X_PCT, BASE_SEED)

    # Methods 4.3.2
    tau, delta = apply_prestige_bias(R.tau, R.delta, R.src, R.indptr, R.deg,
                                     PS, R.eta, R.tiebreak,
                                     norm_mode=PRESTIGE_NORM)
    T, U = net_trust_uncertainty(tau, delta)

    K = np.zeros((UT_GRID.size, TT_GRID.size), dtype=np.int32)
    for ui, UT in enumerate(UT_GRID):
        for ti, TT in enumerate(TT_GRID):
            K[ui, ti] = count_cabals(build_A_IN(R.recv, R.src, T, U, TT, UT, N))
    consensus = (K == 1)

    if not write:
        return consensus, K

    os.makedirs(OUTDIR, exist_ok=True)
    tmp = os.path.join(OUTDIR, f".realization_{realization_id}.tmp")
    final = os.path.join(OUTDIR, f"realization_{realization_id}.pkl")
    with open(tmp, "wb") as f:
        pickle.dump({"consensus": consensus, "K": K,
                     "realization_id": realization_id, "params": _params()}, f)
    os.replace(tmp, final)
    print(f"realization {realization_id}: {int(consensus.sum())} consensus cells "
          f"of {consensus.size} -> {final}", flush=True)
    return final


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge
# ═══════════════════════════════════════════════════════════════════════════════
def merge():
    files = sorted(glob.glob(os.path.join(OUTDIR, "realization_*.pkl")))
    if not files:
        raise SystemExit(f"no realization_*.pkl in {OUTDIR!r}")

    reference = _params()
    n_consensus = np.zeros((UT_GRID.size, TT_GRID.size), dtype=np.int64)
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
        n_consensus += D["consensus"]

    n_files = len(files)
    frac = n_consensus / n_files

    np.save(os.path.join(OUTDIR, "consensus_fraction.npy"), frac)
    np.save(os.path.join(OUTDIR, "n_consensus.npy"), n_consensus)
    np.save(os.path.join(OUTDIR, "TT_grid.npy"), TT_GRID)
    np.save(os.path.join(OUTDIR, "UT_grid.npy"), UT_GRID)
    np.savetxt(os.path.join(OUTDIR, "consensus_fraction.txt"), frac, fmt="%.4f")
    with open(os.path.join(OUTDIR, "params.json"), "w") as f:
        json.dump({**reference, "n_realizations_merged": n_files}, f, indent=2)

    print(f"merged {n_files} realizations -> {OUTDIR}/consensus_fraction.npy")
    if n_files != N_REAL:
        print(f"  NOTE: merged {n_files} files but N_REAL = {N_REAL}")
    return frac


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════
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

POINTS = {
    'A': (-0.8,  0.8),
    'E': ( 0.2,  0.8),
    'F': (-0.8,  0.0),
    'G': ( 0.0,  0.0),
    'B': ( 0.8,  0.8),
    'C': (-0.8, -0.8),
    'D': ( 0.8, -0.8),
}


def plot_colmap(colmap, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))

    # colmap in [0, 1]: 1 = consensus in every realization -> blue end of
    # coolwarm_r, 0 = fragmented in every realization -> red end, 0.5 -> white.
    im = ax.imshow(
        colmap, cmap="coolwarm_r", interpolation="gaussian",
        extent=[-1, 1, -1, 1], origin="lower", aspect="auto",
        vmin=0.0, vmax=1.0,
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

    plt.figtext(0.40, 0.25, r"$\mathrm{fragmentation}$", fontsize=18, color="white")
    plt.figtext(0.32, 0.68, r"$\mathrm{consensus}$",     fontsize=18, color="white")

    if DRAW_BOUNDARY_CONTOUR:
        ax.contour(TT_GRID, UT_GRID, colmap, levels=[BOUNDARY_LEVEL],
                   colors="white", linewidths=1.5)

    for label, (tt, ut) in POINTS.items():
        ax.scatter(tt, ut, s=800, facecolors="#b3ff00", edgecolors="black",
                   linewidths=1.5, zorder=5)
        ax.text(tt, ut, label, ha="center", va="center",
                fontsize=16, fontweight="bold", zorder=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figure -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    figfile = os.path.join(OUTDIR, f"phase_diagram_rho{RHO:g}_ps{PS:g}.pdf")

    if _CLI.merge:
        plot_colmap(merge(), figfile)
    elif _CLI.realization_id is not None:
        run_realization(_CLI.realization_id)
    else:
        print(f"Phase diagram (K = number of reaches): N={N}, p={P_EDGE}, "
              f"rho={RHO}, sigma={SIGMA}, ps={PS}, step={GRID_STEP}, "
              f"N_REAL={N_REAL}, base_seed={BASE_SEED}")
        for rid in range(N_REAL):
            run_realization(rid)
        plot_colmap(merge(), figfile)
    print(f"Done in {time.time() - t0:.1f}s")
