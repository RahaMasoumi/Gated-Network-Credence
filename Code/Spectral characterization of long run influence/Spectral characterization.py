#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4 -- spectral characterisation of long-run influence after filtering.
ER main panels with BA and modular insets.

Quantity plotted
----------------
gamma_bar, the normalised left null vector of the directed Laplacian
L = D_row - A_IN of the filtered influence graph.  Eq. (4) gives

    lim_t b(t) = sum_m (gamma_bar_m . b(0)) gamma_m,

so gamma_bar_i is the coefficient multiplying agent i's initial belief in the
limiting belief: its direct long-run contribution (Methods 4.2).  Because it is
a property of the filtered graph alone, it does not depend on b(0), so neither
the initial beliefs, nor the degree-biased seeding, nor the modular
polarization psi enter this figure at all.

Consistency with the rest of the repository
------------------------------------------
Every model quantity comes from gnc_core.py, shared with the phase-diagram, the
Delta b*, and the b*-versus-p_s scripts.  In particular:

  * the same index convention, A_IN[i, j] = 1 iff j can influence i, and the
    same unnormalised Laplacian L = D_row - A_IN of Eq. (3);
  * the same reach count: gnc_core.gamma_bar returns K alongside gamma, and K
    agrees with gnc_core.count_cabals by construction;
  * the same topologies, so realization r of this figure is the same graph as
    realization r of every other figure (BASE_SEED = 20260819, spawned into
    independent sub-streams);
  * tau sampled once with only eps rejected, per Methods 4.3.1.  This matters
    here: rho = -1 is one of the two couplings shown, and at rho = -1 the mean
    of delta sits on the boundary for tau near 0 or 1, so resampling tau along
    with eps would distort its marginal at exactly the coupling the figure
    contrasts;
  * prestige confined to modules on the modular network, per Methods 4.3.5.
    Applying it across the whole graph would give this figure a different
    prestige rule from the modular panels of Figures S5 and S6.

The plotting section below is unchanged from the previous version.
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import NullLocator, MaxNLocator, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import (
    AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea,
)

from gnc_core import (
    erdos_renyi_undirected,
    barabasi_albert_undirected,
    modular_undirected,
    potential_relationships,
    sample_tau_delta,
    net_trust_uncertainty,
    apply_prestige_bias,
    build_A_IN,
    gamma_bar,
)


# ================= Reproducibility & model parameters =================
# Shared with every other script in the repository, so realization r is the
# same network here as there (Methods 4.3.6, "Replication").
BASE_SEED = 20260819

N     = 5000
SIGMA = 0.05                 # Methods 4.3.5: sigma_eps = 0.05

ER_P        = 0.01           # Methods 4.3.6
BA_M        = 25
MOD_HALF    = 2500
MOD_BA_M    = 21
CROSS_EDGES = 20_857         # => modularity ~ 0.333, <k> = 50.0

rho_values = [-1.0, -0.01]   # the two couplings contrasted in Figure 4
ps_values  = [0.4, 1]        # the two prestige-bias levels
TT_UT_points = {"E": (0.2, 0.8), "F": (-0.8, 0.0)}   # Evaluative, Friction-averse

PRESTIGE_NORM = "rank"       # rank-association permutation, Methods 4.3.2

# Stable integer per topology for the seed spawn.  A hash() of the name would
# not do: Python randomises string hashing per process unless PYTHONHASHSEED is
# fixed, so the figure would not be reproducible between runs.
TOPOLOGY_INDEX = {"ER": 0, "BA": 1, "Modular": 2}

OUTDIR = "output_block_design"


def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realization_id", type=int, default=0,
                    help="which realization to show; Figure 4 is a single "
                         "realization, and 0 is the one used in the paper")
    return ap.parse_args()


# ================= Smoothing knobs =================
NBINS               = 22
DENSE_N             = 420
MEDIAN_SMOOTH_SIGMA = 1.6
# The +/-1 SD band must use the SAME kernel as the mean line, otherwise the band
# is not centred on the curve that is drawn and the caption claim "one standard
# deviation around the bin mean" is not literally true.  Set to 1.9 to restore
# the old (wider, slightly off-centre) band.
RIBBON_SMOOTH_SIGMA = MEDIAN_SMOOTH_SIGMA
MEDIAN_LW           = 1.7


# ================= Global text style =================
FONT_SIZE     = 12
INSET_TICK_FS = 9
SPINE_WIDTH   = 1
TICK_WIDTH    = 0.7

plt.rcParams.update({
    "font.size":        FONT_SIZE + 2,
    "axes.labelsize":   FONT_SIZE + 2,
    "axes.titlesize":   FONT_SIZE + 2,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "xtick.labelsize":  FONT_SIZE + 2,
    "ytick.labelsize":  FONT_SIZE + 2,
    "legend.fontsize":  FONT_SIZE + 2,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "svg.fonttype":     "none",
})


# ================= Visual encoding =================
ER_COLOR    = "#B7410E"       # ER hero colour (rust / burnt orange)
SPREAD_MODE = "ribbon"        # "ribbon" | "errorbar"

NET_COLOR = {
    "ER":      ER_COLOR,
    "BA":      "#0072B2",
    "Modular": "#CC79A7",
}
PS_LS = {0.4: "-", 1: (0, (5, 3))}

RIBBON_FILL_ALPHA = 0.2
RIBBON_LIGHTEN    = 0.55
RIBBON_N_STD      = 1.0

# ---- Panel headings -------------------------------------------------------
# NOTE: the submitted Fig. 4 image reads "Friction-averse (F)", but the
# caption says "Friction-averse (F)" twice, and so does the rest of the paper.
# Pick one and keep it consistent; "Friction-averse" is the coherent reading.
BLOCK_TITLES = ["Evaluative (E)", "Friction-averse (F)"]
# BLOCK_TITLES = ["Evaluative (E)", "Friction-averse (F)"]

RHO_TITLE_DY   = 0.30   # inches above the panel top for the rho sub-titles
BLOCK_TITLE_DY = 0.66   # inches above the panel top for the regime titles

ERRBAR_N_MAIN  = 7
ERRBAR_N_INSET = 5
ERRBAR_CAP     = 1.6
ERRBAR_ELW     = 0.8
ERRBAR_DODGE   = 0.010
ERRBAR_COLOR   = "#404040"


def _lighten(color, frac):
    r, g, b = to_rgb(color)
    return (r + (1.0 - r) * frac,
            g + (1.0 - g) * frac,
            b + (1.0 - b) * frac)


INSET_POS = {
    (0, 0): [0.08, 0.56, 0.40, 0.35],
    (0, 1): [0.08, 0.56, 0.40, 0.35],
    (1, 0): [0.08, 0.56, 0.40, 0.35],
    (1, 1): [0.08, 0.56, 0.40, 0.35],
}



# ================= Networks and model, all from gnc_core =================
def build_realization(realization_id):
    """One realization of each topology, drawn exactly as in run_trends.py.

    Returns {name: (recv, src, indptr, deg, community_or_None)} together with
    the raw pre-filtering degrees used as the horizontal axis of the figure.
    """
    ss = np.random.SeedSequence([BASE_SEED, realization_id])
    ss_er, ss_ba, ss_mod = ss.spawn(3)

    nets = {}

    u, v, deg = erdos_renyi_undirected(N, ER_P, np.random.default_rng(ss_er))
    nets["ER"] = (*potential_relationships(u, v, N), deg, None)

    u, v, deg = barabasi_albert_undirected(N, BA_M, np.random.default_rng(ss_ba))
    nets["BA"] = (*potential_relationships(u, v, N), deg, None)

    u, v, deg, community = modular_undirected(
        MOD_HALF, MOD_BA_M, MOD_BA_M, CROSS_EDGES,
        np.random.default_rng(ss_mod))
    nets["Modular"] = (*potential_relationships(u, v, N), deg, community)

    deg_raw = {nm: d.astype(int) for nm, (_, _, _, d, _) in nets.items()}
    return nets, deg_raw


def influence_for_condition(net, rho, ps, TT, UT, rng):
    """gamma_bar, K and n_scc for one (topology, rho, p_s, gate point).

    tau, eps and the prestige scores belong to the (topology, rho) draw and are
    shared by the two p_s values, so moving from p_s = 0.4 to p_s = 1 changes
    only the prestige weighting.
    """
    recv, src, indptr, deg, community = net
    tau, delta = sample_tau_delta(recv.size, rho, SIGMA, rng)
    eta = rng.random(recv.size)
    tiebreak = rng.random(recv.size)
    t, d = apply_prestige_bias(tau, delta, src, indptr, deg, ps,
                               eta, tiebreak, norm_mode=PRESTIGE_NORM,
                               group=community)
    T, U = net_trust_uncertainty(t, d)
    A_IN = build_A_IN(recv, src, T, U, TT, UT, N)
    return gamma_bar(A_IN)

# ================= Binning / smoothing helpers =================
def _interp_nan(a):
    a  = np.asarray(a, dtype=float)
    x  = np.arange(len(a))
    ok = np.isfinite(a)
    if ok.sum() == 0:
        return a
    if ok.sum() == 1:
        out = a.copy(); out[~ok] = a[ok][0]; return out
    out = a.copy()
    out[~ok] = np.interp(x[~ok], x[ok], a[ok])
    return out


def _binned_stat(x, y, nbins, x_range, stat):
    lo, hi = x_range
    edges  = np.linspace(lo, hi, nbins + 1)
    xc     = 0.5 * (edges[:-1] + edges[1:])
    out    = np.full(nbins, np.nan)
    for k in range(nbins):
        m = (x >= edges[k]) & (x < edges[k + 1])
        if m.any():
            if stat == "mean":
                out[k] = np.mean(y[m])
            elif stat == "std":
                out[k] = np.std(y[m])
            else:
                raise ValueError(stat)
    return xc, out


def _smooth_densify(xc, y, sigma_, n_dense, x_extent=None):
    y_filled = _interp_nan(y)
    y_smooth = gaussian_filter1d(y_filled, sigma=sigma_, mode="nearest")
    if x_extent is not None:
        lo, hi = x_extent
        if lo < xc[0]:
            xc       = np.concatenate([[lo], xc])
            y_smooth = np.concatenate([[y_smooth[0]], y_smooth])
        if hi > xc[-1]:
            xc       = np.concatenate([xc, [hi]])
            y_smooth = np.concatenate([y_smooth, [y_smooth[-1]]])
    x_dense = np.linspace(xc[0], xc[-1], n_dense)
    y_dense = PchipInterpolator(xc, y_smooth)(x_dense)
    return x_dense, y_dense


def _nice_three_ticks(lo, hi):
    span = hi - lo
    if   span > 400: step = 100
    elif span > 150: step = 50
    elif span >  75: step = 25
    elif span >  30: step = 10
    else:            step = 5
    t_lo  = int(np.ceil(lo  / step) * step)
    t_hi  = int(np.floor(hi / step) * step)
    t_mid = (t_lo + t_hi) // 2
    return [t_lo, t_mid, t_hi]


# ================= Unified curve-plotting function =================
def plot_curve(ax, x, y, ps, x_range, color, mode=None,
               median_lw=None, n_err=ERRBAR_N_MAIN, dodge_frac=0.0,
               cap=ERRBAR_CAP, elw=ERRBAR_ELW, err_alpha=0.9,
               fill_alpha=RIBBON_FILL_ALPHA, lighten=RIBBON_LIGHTEN,
               n_std=RIBBON_N_STD):
    if mode is None:
        mode = SPREAD_MODE
    ls = PS_LS[ps]
    if median_lw is None:
        median_lw = MEDIAN_LW

    xc, mean_raw = _binned_stat(x, y, NBINS, x_range, "mean")
    _,  std_raw  = _binned_stat(x, y, NBINS, x_range, "std")

    okm = np.isfinite(mean_raw)
    if okm.sum() < 2:
        return
    xd_m, mean_d = _smooth_densify(xc[okm], mean_raw[okm],
                                   MEDIAN_SMOOTH_SIGMA, DENSE_N,
                                   x_extent=x_range)

    if mode == "ribbon":
        lo_raw = mean_raw - n_std * std_raw
        hi_raw = mean_raw + n_std * std_raw
        ok = np.isfinite(lo_raw) & np.isfinite(hi_raw)
        if ok.sum() >= 2:
            xd, lo_d = _smooth_densify(xc[ok], lo_raw[ok],
                                       RIBBON_SMOOTH_SIGMA, DENSE_N,
                                       x_extent=x_range)
            _,  hi_d = _smooth_densify(xc[ok], hi_raw[ok],
                                       RIBBON_SMOOTH_SIGMA, DENSE_N,
                                       x_extent=x_range)
            lo_d, hi_d = np.minimum(lo_d, hi_d), np.maximum(lo_d, hi_d)
            lo_d = np.clip(lo_d, 0.0, None)
            ax.fill_between(xd, lo_d, hi_d,
                            facecolor=_lighten(color, lighten),
                            alpha=fill_alpha, linewidth=0, zorder=1)
    else:
        lo, hi = x_range
        pad = 0.06 * (hi - lo)
        xs  = np.linspace(lo + pad, hi - pad, n_err)
        oks = np.isfinite(std_raw)
        std_at  = (np.interp(xs, xc[oks], std_raw[oks])
                   if oks.sum() >= 2 else np.zeros_like(xs))
        mean_at = np.interp(xs, xd_m, mean_d)
        xs_d    = xs + dodge_frac * (hi - lo)
        low     = np.clip(mean_at - n_std * std_at, 0.0, None)
        ax.errorbar(xs_d, mean_at, yerr=[mean_at - low, n_std * std_at],
                    fmt="none", ecolor=ERRBAR_COLOR, elinewidth=elw,
                    capsize=cap, capthick=elw, alpha=err_alpha, zorder=6)

    ax.plot(xd_m, mean_d, color=color, lw=median_lw, ls=ls,
            zorder=7, solid_capstyle="round", dash_capstyle="round")


# ================= Axis styling =================
def _make_sci_formatter():
    f = ScalarFormatter(useMathText=True)
    f.set_powerlimits((0, 0))
    f.set_scientific(True)
    return f


def style_ax(ax, inset=False):
    lw   = SPINE_WIDTH * 0.7 if inset else SPINE_WIDTH
    tlw  = TICK_WIDTH * 0.7 if inset else TICK_WIDTH
    ftsz = INSET_TICK_FS if inset else FONT_SIZE + 2

    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_zorder(10)

    ax.tick_params(axis="both", which="major",
                   width=tlw, pad=2 if inset else 3, labelsize=ftsz)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(False)

    ax.yaxis.set_major_formatter(_make_sci_formatter())
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3 if inset else 5))
    if not inset:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
    ax.yaxis.get_offset_text().set_size(ftsz - 1)


# ================= Legend builder =================
def _legend_line_entry(color, ls, label, fontsize, lw=MEDIAN_LW,
                       icon_w=34, icon_h=10, sep=5):
    da = DrawingArea(icon_w, icon_h, 0, 0)
    line = Line2D([2, icon_w - 2], [icon_h / 2, icon_h / 2],
                  color=color, ls=ls, lw=lw,
                  solid_capstyle="round", dash_capstyle="round")
    da.add_artist(line)
    ta = TextArea(label, textprops={"fontsize": fontsize, "color": "black"})
    return HPacker(children=[da, ta], align="center", pad=0, sep=sep)


def build_side_legend(fig, left_x, centre_y=0.5, fontsize=None):
    if fontsize is None:
        fontsize = FONT_SIZE + 2
    net_group = VPacker(align="left", pad=0, sep=6, children=[
        _legend_line_entry(NET_COLOR["ER"],      "-", "Erdős–Rényi",     fontsize),
        _legend_line_entry(NET_COLOR["BA"],      "-", "Barabási–Albert", fontsize),
        _legend_line_entry(NET_COLOR["Modular"], "-", "Modular",         fontsize),
    ])
    ps_group = VPacker(align="left", pad=0, sep=6, children=[
        _legend_line_entry("0.20", "-",  r"$p_s = 0.4$", fontsize),
        _legend_line_entry("0.20", "--", r"$p_s = 1$",  fontsize),
    ])
    combined = VPacker(align="left", pad=0, sep=16, children=[net_group, ps_group])
    return AnchoredOffsetbox(
        loc="center left", child=combined, pad=0.4, frameon=False,
        bbox_to_anchor=(left_x, centre_y), bbox_transform=fig.transFigure,
        borderpad=0,
    )


# ================= Build & save the figure =================
def save_figure(results, deg_raw, x_ranges, out_pdf=None, out_png=None):
    point_list   = list(TT_UT_points.items())          # [("E",...), ("F",...)]
    block_titles = BLOCK_TITLES

    S             = 2.5
    GAP_TIGHT     = 0.12
    GROUP_GAP     = 0.85
    LEFT_MARGIN   = 0.95
    RIGHT_MARGIN  = 2.80
    TOP_MARGIN    = 1.05
    BOTTOM_MARGIN = 0.85

    fig_w = LEFT_MARGIN + 4 * S + 2 * GAP_TIGHT + GROUP_GAP + RIGHT_MARGIN
    fig_h = TOP_MARGIN + S + BOTTOM_MARGIN

    def fx(v): return v / fig_w
    def fy(v): return v / fig_h

    col1 = LEFT_MARGIN
    col2 = col1 + S + GAP_TIGHT
    col3 = col2 + S + GROUP_GAP
    col4 = col3 + S + GAP_TIGHT
    col_lefts = [col1, col2, col3, col4]
    row_bot   = BOTTOM_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))
    axs = {}
    for b in range(2):
        for rg in range(2):
            c = b * 2 + rg
            axs[(b, rg)] = fig.add_axes(
                [fx(col_lefts[c]), fy(row_bot), fx(S), fy(S)])

    inset_pos  = INSET_POS[(0, 0)]
    inset_axes = {}

    for b, (pt, _) in enumerate(point_list):
        for rg, rho in enumerate(rho_values):
            ax = axs[(b, rg)]
            for pk, ps in enumerate(ps_values):
                w = results[("ER", rho, ps, pt)]
                # sign from the index, not a hard-coded ps value, so the dodge
                # keeps working when ps_values changes
                dodge = -ERRBAR_DODGE if pk == 0 else ERRBAR_DODGE
                plot_curve(ax, deg_raw["ER"], w, ps,
                           x_ranges["ER"], NET_COLOR["ER"],
                           n_err=ERRBAR_N_MAIN, dodge_frac=dodge)
            ax.set_xlim(*x_ranges["ER"])
            ax.margins(x=0, y=0)
            style_ax(ax)

            ax_ins = ax.inset_axes(inset_pos)
            for nm in ("BA", "Modular"):
                net_sign = -1 if nm == "BA" else 1
                for pk, ps in enumerate(ps_values):
                    ps_sign = -1 if pk == 0 else 1
                    w = results[(nm, rho, ps, pt)]
                    dodge = ps_sign * (ERRBAR_DODGE + 0.004) + net_sign * 0.006
                    plot_curve(ax_ins, deg_raw[nm], w, ps,
                               x_ranges["BA_Mod"], NET_COLOR[nm],
                               median_lw=MEDIAN_LW * 0.7,
                               n_err=ERRBAR_N_INSET, cap=1.5, elw=0.8,
                               dodge_frac=dodge, err_alpha=0.85)
            ax_ins.set_xlim(*x_ranges["BA_Mod"])
            ax_ins.margins(x=0, y=0)
            ax_ins.set_facecolor("white")
            style_ax(ax_ins, inset=True)
            ax_ins.set_xticks(_nice_three_ticks(*x_ranges["BA_Mod"]))
            ax_ins.yaxis.tick_right()
            ax_ins.yaxis.set_label_position("right")
            ax_ins.yaxis.get_offset_text().set_ha("left")
            ax_ins.set_title("")
            inset_axes[(b, rg)] = ax_ins

    unified_ymax = max(axs[(b, rg)].get_ylim()[1]
                       for b in range(2) for rg in range(2))
    for b in range(2):
        for rg in range(2):
            axs[(b, rg)].set_ylim(0, unified_ymax)
    for b in range(2):
        axs[(b, 1)].tick_params(labelleft=False)
        axs[(b, 1)].yaxis.get_offset_text().set_visible(False)

    for b in range(2):
        ymax = max(inset_axes[(b, rg)].get_ylim()[1] for rg in range(2))
        for rg in range(2):
            inset_axes[(b, rg)].set_ylim(0, ymax)

    # -- Headings.  IMPORTANT: every one of these artists must end up in
    #    bbox_extra_artists below.  When bbox_extra_artists is passed
    #    explicitly, bbox_inches="tight" measures ONLY the axes plus the listed
    #    artists, so any fig.text left out of the list is drawn and then cropped
    #    away.  That is why the rho and regime titles were missing.
    title_artists = []
    panel_top = row_bot + S

    # rho sub-title above each of the four panels
    for b in range(2):
        for rg in range(2):
            c = b * 2 + rg
            cx = fx(col_lefts[c] + S / 2)
            title_artists.append(fig.text(
                cx, fy(panel_top + RHO_TITLE_DY),
                fr"$\rho={rho_values[rg]:g}$",
                ha="center", va="bottom", fontsize=FONT_SIZE + 2))

    # regime title centred over each block's two panels
    for b in range(2):
        c0 = b * 2
        block_cx = fx((col_lefts[c0] + col_lefts[c0 + 1] + S) / 2.0)
        title_artists.append(fig.text(
            block_cx, fy(panel_top + BLOCK_TITLE_DY), block_titles[b],
            ha="center", va="bottom", fontsize=FONT_SIZE + 2))

    for b in range(2):
        axs[(b, 0)].set_ylabel(
            r"$\overline{\gamma}$",
            fontsize=FONT_SIZE + 2, rotation=0, labelpad=6,
            ha="right", va="center",
        )

    panel_row_centre_x = fx((col1 + col4 + S) / 2.0)
    xlabel_artist = fig.text(
        panel_row_centre_x, fy(BOTTOM_MARGIN * 0.28),
        r"In-degree before filtering ($k_{\rm in}^{\rm before}$)",
        ha="center", va="center", fontsize=FONT_SIZE + 2, zorder=100,
    )

    right_edge_in = col4 + S
    legend_box = build_side_legend(
        fig,
        left_x=fx(right_edge_in + 0.25),
        centre_y=fy(row_bot + S / 2.0),
    )
    fig.add_artist(legend_box)

    if out_pdf is None:
        out_pdf = os.path.join(OUTDIR, "block_design_gamma_bar_safe.pdf")
    if out_png is None:
        out_png = os.path.join(OUTDIR, "block_design_gamma_bar_safe.png")
    extras = [legend_box, xlabel_artist, *title_artists]
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05, dpi=300,
                bbox_extra_artists=extras)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=200,
                bbox_extra_artists=extras)
    plt.close(fig)
    print(f"  Saved -> {out_pdf}")
    print(f"  Saved -> {out_png}")



# ================= Main =================
def main():
    args = _parse_cli()
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"Figure 4: gamma_bar after filtering   realization "
          f"{args.realization_id}, base seed {BASE_SEED}")
    print(f"Prestige: rank-association permutation (norm='{PRESTIGE_NORM}'), "
          f"confined to modules on the modular network")
    print("Building networks ...")
    nets, deg_raw = build_realization(args.realization_id)
    for nm, (recv, _, _, deg, community) in nets.items():
        print(f"  {nm:<8s}  nodes={N}  edges={recv.size // 2}  "
              f"<k>={deg.mean():.1f}"
              + ("  [2 communities]" if community is not None else ""))

    er_lo, er_hi = int(deg_raw["ER"].min()), int(deg_raw["ER"].max())
    bm_concat = np.concatenate([deg_raw["BA"], deg_raw["Modular"]])
    bm_lo = int(bm_concat.min())
    bm_hi = int(np.quantile(bm_concat, 0.99))
    x_ranges = {"ER": (er_lo, er_hi), "BA_Mod": (bm_lo, bm_hi)}
    print(f"  ER     x-range: ({er_lo}, {er_hi})")
    print(f"  BA+Mod x-range: ({bm_lo}, {bm_hi})")

    print("\nRunning conditions (K = number of reaches; consensus <=> K == 1) ...")
    results = {}
    n_frag = 0
    for nm, net in nets.items():
        for rho in rho_values:
            for ps in ps_values:
                # one stream per (topology, rho, p_s); tau, eps and eta are
                # drawn inside, so the two p_s runs of a given rho differ only
                # in the prestige weighting
                rng = np.random.default_rng(np.random.SeedSequence(
                    [BASE_SEED, args.realization_id, TOPOLOGY_INDEX[nm],
                     int(round(-rho * 1000)), int(round(ps * 1000))]))
                for pt, (TT, UT) in TT_UT_points.items():
                    g, K, n_scc = influence_for_condition(net, rho, ps,
                                                          TT, UT, rng)
                    results[(nm, rho, ps, pt)] = g
                    if K != 1:
                        n_frag += 1
                    flag = "single reach" if K == 1 else f"FRAGMENTED (K={K})"
                    strong = "  [strongly connected]" if n_scc == 1 else ""
                    print(f"  {nm:<8s} rho={rho:6g} ps={ps:4g} {pt}: "
                          f"K={K:<4d} n_scc={n_scc:<5d} -> {flag}{strong}")

    if n_frag:
        print(f"\n  NOTE: {n_frag} of {len(results)} conditions have K > 1. "
              f"There gamma is the\n  reach-specific coefficient of Eq. (4), "
              f"positive on each cabal and zero on\n  common-part nodes, and "
              f"its entries sum to K rather than to 1. That is the\n  "
              f"quantity Methods 4.2 defines for a fragmented graph, but the "
              f"caption should\n  say so rather than calling it a consensus "
              f"weight.")

    print("\nPlotting ...")
    save_figure(results, deg_raw, x_ranges)
    print("Done.")


if __name__ == "__main__":
    main()
