#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_trends.py -- the (rho, x, p_s) sweep behind Figure 3(a) and Figures S3-S6.

One script for all three synthetic topologies (Methods 4.3.6) and for both
gating regimes.  For each realization it computes the long-run mean belief

    b*(rho, x, p_s)

on a fixed threshold pair (T_T, U_T), i.e. at one of the labelled points of the
phase diagram.

Everything model-related comes from gnc_core.py, which is shared with the
phase-diagram and Delta b* scripts, so all figures in the repository use one
definition of A_IN, of the Laplacian, and of the consensus criterion.

Design (Methods 4.3.6, "Replication")
-------------------------------------
Per realization the network, the initial beliefs, tau, the noise eps, and the
prestige scores eta are each drawn once from their own independent stream.
Consequently:

  * varying p_s changes only the prestige weighting, not the underlying trust
    values or random scores, so the p_s axis of every figure is a PAIRED
    comparison within a realization;
  * varying x changes only b(0), through the seeding rule;
  * varying rho reuses the same tau and the same eps wherever the rejection
    step accepts them, so the rho curves are paired as well.

The alternative -- an independent trust draw per (rho, p_s, x) -- makes the
curves noisier for reasons unrelated to the mechanism and breaks the stated
design that the same networks and initial beliefs are reused across parameter
combinations.

Long-run belief
---------------
A_IN depends on (rho, p_s) only, and x enters solely through b(0), so the
propagator is built once per (rho, p_s) and applied to every x:

    mean(b(T)) = [exp(-L^T T) 1] . b(0) / N          (Eq. 3)

evaluated with expm_multiply -- exact, with no time step and no stability
condition.  K is recorded for every (rho, p_s) so that the "consensus belief
state" of the captions is a checked claim and not an assumption; a second
evaluation at 2T certifies that the value has settled.

Usage
-----
  python run_trends.py --topology ER  --point E --realization_id 0
  python run_trends.py --topology BA  --point F --realization_id 0
  python run_trends.py --topology MOD --point E --psi 0.4 --realization_id 0
  python run_trends.py --topology ER  --point E            # all realizations
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

from gnc_core import (
    B_MAX_DEFAULT,
    erdos_renyi_undirected,
    barabasi_albert_undirected,
    modular_undirected,
    potential_relationships,
    sample_tau_delta,
    net_trust_uncertainty,
    apply_prestige_bias,
    degree_biased_seeding,
    polarize_beliefs,
    build_A_IN,
    gated_laplacian,
    count_cabals,
    mean_belief_weights,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Labelled points of the phase diagram (Figure 1(a))
# ═══════════════════════════════════════════════════════════════════════════════
# The regime names are those of the manuscript.  Note in particular that
# E = (0.2, 0.8) is Evaluative and F = (-0.8, 0.0) is Friction-averse; the two
# must not be swapped, since every caption in the paper is keyed to them.
POINTS = {
    "A": (-0.8,  0.8),   # Accommodating
    "E": ( 0.2,  0.8),   # Evaluative
    "F": (-0.8,  0.0),   # Friction-averse
    "G": ( 0.0,  0.0),   # Guarded  (near the phase boundary: see the warning
                         #           printed at run time, K > 1 is common here
                         #           and "consensus belief" would not apply)
    "B": ( 0.8,  0.8),
    "C": (-0.8, -0.8),
    "D": ( 0.8, -0.8),
}
REGIME_NAME = {"A": "Accommodating", "E": "Evaluative",
               "F": "Friction-averse", "G": "Guarded"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameters  (Methods 4.3.5, 4.3.6)
# ═══════════════════════════════════════════════════════════════════════════════
N        = 5000
P_EDGE   = 0.01          # ER
M_BA     = 25            # BA links per arriving node
N_COMM   = 2500          # modular: nodes per community
M_MOD    = 21            # modular: BA parameter of each community
N_INTER  = 20857         # modular: cross-community edges -> Q ~ 0.333

SIGMA    = 0.05                       # Methods 4.3.5: sigma_eps = 0.05
B_MAX    = B_MAX_DEFAULT              # b_i(0) ~ U(-B_max, B_max), B_max = 5
RHO_VALUES = [-1.0, -0.7, -0.4, -0.01]
PS_VALUES  = [0.05, 0.10, 0.40, 0.60, 0.80, 1.00]
X_VALUES   = [0.05, 0.10, 0.15, 0.20]

N_REAL = 100             # Methods 4.3.6
T_HORIZON = 10.0         # integration horizon; convergence is verified per cell
SETTLED_TOL = 1e-6       # |mean(T) - mean(2T)| below this counts as settled

PRESTIGE_NORM = "rank"
BASE_SEED = 20260819     # shared with every other script in the repository


def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=["ER", "BA", "MOD"], required=True)
    ap.add_argument("--point", choices=sorted(POINTS), default="E",
                    help="labelled point of the phase diagram; E = Evaluative, "
                         "F = Friction-averse")
    ap.add_argument("--psi", type=float, default=None,
                    help="inter-community polarization; MOD only "
                         "(Methods 4.3.5 uses 0, 0.4, 1)")
    ap.add_argument("--realization_id", type=int, default=None,
                    help="run ONE realization (for a SLURM array); omit to run "
                         "all N_REAL of them serially")
    ap.add_argument("--outdir", type=str, default=None)
    a = ap.parse_args()
    if a.topology == "MOD" and a.psi is None:
        ap.error("--psi is required for --topology MOD")
    if a.topology != "MOD" and a.psi is not None:
        ap.error("--psi applies to --topology MOD only")
    return a


_CLI = _parse_cli()
TOPOLOGY = _CLI.topology
POINT = _CLI.point
PSI = _CLI.psi
TT, UT = POINTS[POINT]


def default_outdir():
    if TOPOLOGY == "ER":
        tag = f"ER_N{N}_p{P_EDGE:g}"
    elif TOPOLOGY == "BA":
        tag = f"BA_N{N}_m{M_BA}"
    else:
        tag = f"MOD_N{N}_Q0.33_psi{PSI:g}"
    return f"trends_{tag}_point{POINT}"


OUTDIR = _CLI.outdir if _CLI.outdir is not None else default_outdir()


def params():
    return {
        "topology": TOPOLOGY, "point": POINT, "regime": REGIME_NAME.get(POINT),
        "TT": TT, "UT": UT, "N": N,
        "p": P_EDGE if TOPOLOGY == "ER" else None,
        "m_ba": M_BA if TOPOLOGY == "BA" else None,
        "n_comm": N_COMM if TOPOLOGY == "MOD" else None,
        "m_mod": M_MOD if TOPOLOGY == "MOD" else None,
        "n_inter": N_INTER if TOPOLOGY == "MOD" else None,
        "psi": PSI, "sigma": SIGMA, "B_max": B_MAX,
        "rho_values": RHO_VALUES, "ps_values": PS_VALUES, "x_values": X_VALUES,
        "T_horizon": T_HORIZON, "prestige_norm": PRESTIGE_NORM,
        "base_seed": BASE_SEED,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  One realization
# ═══════════════════════════════════════════════════════════════════════════════
def build_topology(rng):
    """Returns (u, v, deg, community_or_None)."""
    if TOPOLOGY == "ER":
        u, v, deg = erdos_renyi_undirected(N, P_EDGE, rng)
        return u, v, deg, None
    if TOPOLOGY == "BA":
        u, v, deg = barabasi_albert_undirected(N, M_BA, rng)
        return u, v, deg, None
    u, v, deg, community = modular_undirected(N_COMM, M_MOD, M_MOD, N_INTER, rng)
    return u, v, deg, community


def run_realization(realization_id, verbose=True):
    ss = np.random.SeedSequence([BASE_SEED, realization_id])
    ss_graph, ss_beliefs, ss_trust, ss_prestige = ss.spawn(4)

    u, v, deg, community = build_topology(np.random.default_rng(ss_graph))
    recv, src, indptr = potential_relationships(u, v, N)
    n_pairs = recv.size

    # --- initial beliefs: Methods 4.3.5, then 4.3.4, then 4.3.3 -------------
    rng_b = np.random.default_rng(ss_beliefs)
    b_init = rng_b.uniform(-B_MAX, B_MAX, N)          # i.i.d., not stratified
    if TOPOLOGY == "MOD":
        b_init = polarize_beliefs(b_init, community, PSI, rng_b)
        seed_pool = np.flatnonzero(community == 0)    # promoters in one community
    else:
        seed_pool = None
    b0 = {x: degree_biased_seeding(b_init, deg, x, subset=seed_pool)
          for x in X_VALUES}

    # --- trust and distrust: Eq. (5) ---------------------------------------
    # tau and eps are drawn once for the realization and reused for every rho
    # wherever the rejection step accepts them, so the rho comparison is paired.
    rng_t = np.random.default_rng(ss_trust)
    tau_base = rng_t.uniform(0.0, 1.0, n_pairs)

    # --- prestige scores: Methods 4.3.2 ------------------------------------
    rng_p = np.random.default_rng(ss_prestige)
    eta = rng_p.random(n_pairs)
    tiebreak = rng_p.random(n_pairs)

    n_rho, n_ps, n_x = len(RHO_VALUES), len(PS_VALUES), len(X_VALUES)
    b_star = np.full((n_rho, n_x, n_ps), np.nan)
    K = np.zeros((n_rho, n_ps), dtype=np.int64)
    settled = np.zeros((n_rho, n_ps), dtype=bool)
    n_admitted = np.zeros((n_rho, n_ps), dtype=np.int64)

    for ri, rho in enumerate(RHO_VALUES):
        # delta from the shared tau; only the rejected entries consume fresh
        # randomness, so most relationships are identical across rho
        tau, delta = sample_tau_delta_shared(tau_base, rho, SIGMA, rng_t)

        for pi, ps in enumerate(PS_VALUES):
            t, d = apply_prestige_bias(tau, delta, src, indptr, deg, ps,
                                       eta, tiebreak,
                                       norm_mode=PRESTIGE_NORM,
                                       group=community)
            T, U = net_trust_uncertainty(t, d)
            A_IN = build_A_IN(recv, src, T, U, TT, UT, N)
            n_admitted[ri, pi] = A_IN.nnz
            K[ri, pi] = count_cabals(A_IN)

            L = gated_laplacian(A_IN)
            w, w2 = mean_belief_weights(L, T_HORIZON, also_at=2.0 * T_HORIZON)
            drift = 0.0
            for xi, x in enumerate(X_VALUES):
                m1 = float(w @ b0[x]) / N
                b_star[ri, xi, pi] = m1
                drift = max(drift, abs(float(w2 @ b0[x]) / N - m1))
            settled[ri, pi] = drift < SETTLED_TOL

    os.makedirs(OUTDIR, exist_ok=True)
    tmp = os.path.join(OUTDIR, f".realization_{realization_id}.tmp")
    final = os.path.join(OUTDIR, f"realization_{realization_id}.pkl")
    with open(tmp, "wb") as f:
        pickle.dump({"b_star": b_star, "K": K, "settled": settled,
                     "n_admitted": n_admitted,
                     "realization_id": realization_id, "params": params()}, f)
    os.replace(tmp, final)

    if verbose:
        n_frag = int((K != 1).sum())
        n_unsettled = int((~settled).sum())
        msg = (f"realization {realization_id}: {K.size} (rho, p_s) cells, "
               f"{n_frag} with K != 1, {n_unsettled} not settled -> {final}")
        print(msg, flush=True)
        if n_frag:
            print(f"  WARNING: {n_frag} cells have K > 1, so b* there is a "
                  f"network average over several reaches, not a consensus "
                  f"value. Point {POINT} may be outside the consensus region "
                  f"for this topology.", flush=True)
    return final


def sample_tau_delta_shared(tau_base, rho, sigma, rng):
    """Eq. (5) with tau supplied rather than redrawn.

    Methods 4.3.1: tau is sampled once and only eps is rejected and resampled
    until delta lands in [0, 1].  Passing tau in from outside lets the same tau
    serve every rho, which pairs the rho comparison; the rejection step still
    consumes fresh eps per rho, as it must, since the acceptance region depends
    on rho.
    """
    tau = np.asarray(tau_base, dtype=float)
    mu = rho * (tau - 0.5) + 0.5
    delta = np.empty(tau.size)
    pending = np.arange(tau.size)
    while pending.size:
        cand = mu[pending] + rng.normal(0.0, sigma, pending.size)
        ok = (cand >= 0.0) & (cand <= 1.0)
        delta[pending[ok]] = cand[ok]
        pending = pending[~ok]
    return tau, delta


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()
    label = (f"{TOPOLOGY}" + (f" psi={PSI:g}" if TOPOLOGY == "MOD" else "")
             + f"  point {POINT} = ({TT:g}, {UT:g})"
             + (f"  [{REGIME_NAME[POINT]}]" if POINT in REGIME_NAME else ""))
    print(f"{label}  ->  '{OUTDIR}'")
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "params.json"), "w") as f:
        json.dump(params(), f, indent=2)

    if _CLI.realization_id is not None:
        run_realization(_CLI.realization_id)
    else:
        for rid in range(N_REAL):
            run_realization(rid)
    print(f"Done in {time.time() - t0:.1f}s")
