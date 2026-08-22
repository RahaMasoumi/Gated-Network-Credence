#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_reproducibility.py -- checks that the two figure scripts implement the
model described in the manuscript and agree with each other.

Run it before submitting the array jobs (it uses a reduced N so it finishes in
about a minute), and again after the merge to check the produced arrays:

    python verify_reproducibility.py                 # checks 1-6, small N
    python verify_reproducibility.py --check-outputs \\
        --fig1-dir phase_diagram_rho-0.4_ps0 \\
        --fig2-dir delta_b_results_rho-0.4        # check 7, on the real output

Checks
------
1. gnc_core self-tests: K against a networkx condensation, the left null vector
   of L, the seeding permutation, the prestige permutation, the marginal of tau.
2. Index convention: A_IN[i, j] = 1 exactly for the admitted (receiver, source)
   pairs, and L = D_row - A_IN has zero row sums.
3. Realization identity: fig01 and fig02 draw the same network, the same trust
   and distrust values, and the same prestige scores for the same realization
   id, and re-running gives bit-identical arrays.
4. Cross-figure agreement: on the same realization, the K computed by fig01 at
   p_s = 0 equals the K computed by fig02 for its p_s = 0 branch, cell by cell.
5. Consensus semantics: on cells where K = 1, exp(-L t) b(0) really does settle
   on a single common value, and on cells where K > 1 it does not.
6. Delta b* identity: b*(p_s) recomputed independently from the left null vector
   of L, gamma_bar . b(0) (Eq. 4), matches the value obtained by integration.
7. Produced outputs: the merged consensus fraction of Figure 1 equals the
   p_s = 0 consensus fraction implied by the Figure 2 run, and Delta b* is
   finite exactly where the fraction reaches the reported threshold.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp

import gnc_core as core


PASS = "  [ok] "
FAIL = "  [FAIL] "
_failures = []


def check(name, ok, detail=""):
    print((PASS if ok else FAIL) + name + (f"  {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


# ---------------------------------------------------------------------------
def check_1_core():
    print("\n1. gnc_core self-tests")
    try:
        core._selftest()
        check("all model-level invariants", True)
    except AssertionError as e:
        check("all model-level invariants", False, str(e))


def check_2_convention(N=300, p=0.05):
    print("\n2. index convention and Laplacian (Methods 4.1, Eqs. 1-3)")
    R = core.build_realization(0, N, p, -0.4, 0.05, 0.2, 20260819)
    T, U = core.net_trust_uncertainty(R.tau, R.delta)
    TT, UT = -0.2, 0.3
    A_IN = core.build_A_IN(R.recv, R.src, T, U, TT, UT, N)

    admitted = (T >= TT) & (U <= UT)
    want = set(zip(R.recv[admitted].tolist(), R.src[admitted].tolist()))
    got = set(zip(*[a.tolist() for a in A_IN.nonzero()]))
    check("A_IN[i, j] = 1 exactly for admitted (receiver i, source j) pairs",
          want == got, f"{len(want)} admitted relationships")

    # every stored pair must be a potential relationship of the base graph
    base = set(zip(R.recv.tolist(), R.src.tolist()))
    check("A_IN is a submatrix of the potential graph A", got <= base)

    L = core.gated_laplacian(A_IN)
    check("L = D_row - A_IN has zero row sums",
          np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0.0))
    k_row = np.asarray(A_IN.sum(axis=1)).ravel()
    check("diag(L) = number of admitted sources of each receiver",
          np.allclose(L.diagonal(), k_row))
    check("off-diagonal of L is -A_IN",
          (L + A_IN - sp.diags(k_row)).nnz == 0)


def check_3_identity(N=300, p=0.05, rid=5):
    print("\n3. realization identity across scripts and across runs")
    import fig01_phase_diagram as f1
    import fig02_delta_b as f2

    a = core.build_realization(rid, N, p, -0.4, 0.05, 0.2, 20260819)
    b = core.build_realization(rid, N, p, -0.4, 0.05, 0.2, 20260819)
    check("build_realization is deterministic",
          all(np.array_equal(x, y) for x, y in zip(a[2:], b[2:])))

    check("fig01 and fig02 use the same BASE_SEED",
          f1.BASE_SEED == f2.BASE_SEED, f"{f1.BASE_SEED}")
    check("fig01 and fig02 use the same N, p, sigma, x, prestige norm",
          (f1.N, f1.P_EDGE, f1.SIGMA, f1.X_PCT, f1.PRESTIGE_NORM) ==
          (f2.N, f2.P_EDGE, f2.SIGMA, f2.X_PCT, f2.PRESTIGE_NORM))
    check("fig01 and fig02 use the same threshold grid",
          np.array_equal(f1.TT_GRID, f2.TT_GRID) and
          np.array_equal(f1.UT_GRID, f2.UT_GRID), f"{f1.TT_GRID.size} points")
    check("fig01 p_s equals fig02 ps_low (so the two K masks are comparable)",
          f1.PS == f2.PS_LOW, f"ps={f1.PS}, ps_low={f2.PS_LOW}")


def check_4_cross_figure(N=300, p=0.05, rid=2, step=0.2):
    print("\n4. cross-figure agreement of K on the same realization")
    import fig01_phase_diagram as f1
    import fig02_delta_b as f2

    for m in (f1, f2):
        m.N, m.P_EDGE = N, p
        m.TT_GRID = m.UT_GRID = core.threshold_grid(step)
        m.GRID_STEP = step
    f2.PS_VALUES = [f2.PS_LOW, f2.PS_HIGH]
    f1.OUTDIR = "_verify_f1"
    f2.OUTDIR = "_verify_f2"

    K1 = f1.run_realization(rid, write=False)[1]

    R = core.build_realization(rid, N, p, f2.RHO, f2.SIGMA, f2.X_PCT, f2.BASE_SEED)
    tau, delta = core.apply_prestige_bias(R.tau, R.delta, R.src, R.indptr, R.deg,
                                          f2.PS_LOW, R.eta, R.tiebreak,
                                          norm_mode=f2.PRESTIGE_NORM)
    T, U = core.net_trust_uncertainty(tau, delta)
    K2 = np.zeros_like(K1)
    for ui, UT in enumerate(f2.UT_GRID):
        for ti, TT in enumerate(f2.TT_GRID):
            K2[ui, ti] = core.count_cabals(
                core.build_A_IN(R.recv, R.src, T, U, TT, UT, N))

    check("K(fig01, p_s=0) == K(fig02, p_s=0) on every grid cell",
          np.array_equal(K1, K2),
          f"{K1.size} cells, {(K1 == 1).sum()} with K = 1")
    return K1


def check_5_and_6_dynamics(N=300, p=0.05, rid=2):
    print("\n5. consensus semantics and 6. Delta b* via the left null vector")
    R = core.build_realization(rid, N, p, -0.4, 0.05, 0.2, 20260819)
    tau, delta = core.apply_prestige_bias(R.tau, R.delta, R.src, R.indptr,
                                          R.deg, 0.0, R.eta, R.tiebreak)
    T, U = core.net_trust_uncertainty(tau, delta)

    tested_consensus = tested_fragmented = 0
    spread_ok = frag_ok = True
    null_ok = True
    max_null_err = 0.0

    for TT in np.linspace(-1, 0.6, 9):
        for UT in np.linspace(-0.6, 1, 9):
            A_IN = core.build_A_IN(R.recv, R.src, T, U, TT, UT, N)
            K = core.count_cabals(A_IN)
            L = core.gated_laplacian(A_IN)
            b = core.long_run_belief(L, R.b0, 2000.0)
            if K == 1 and tested_consensus < 6:
                tested_consensus += 1
                spread_ok &= (b.max() - b.min() < 1e-8)
                # Eq. (4) with a single reach: b* = gamma_bar . b(0)
                w, V = np.linalg.eig(L.toarray().T)
                g = np.real(V[:, np.argmin(np.abs(w))])
                g = g / g.sum()
                null_ok &= bool((g > -1e-9).all())
                max_null_err = max(max_null_err,
                                   abs(float(g @ R.b0) - float(b.mean())))
            elif K > 1 and tested_fragmented < 6:
                tested_fragmented += 1
                frag_ok &= (b.max() - b.min() > 1e-6)

    check("K = 1 cells settle on one common belief",
          spread_ok, f"{tested_consensus} cells, spread < 1e-8")
    check("K > 1 cells do not",
          frag_ok, f"{tested_fragmented} cells")
    check("gamma_bar is nonnegative and sums to one (Methods 4.2)", null_ok)
    check("b* from integration == gamma_bar . b(0)  (Eq. 4)",
          max_null_err < 1e-6, f"max |difference| = {max_null_err:.2e}")


def check_7_outputs(fig1_dir, fig2_dir):
    print("\n7. produced outputs")
    f1_frac_path = os.path.join(fig1_dir, "consensus_fraction.npy")
    if not os.path.exists(f1_frac_path):
        check("Figure 1 merged output present", False, f1_frac_path)
        return
    frac1 = np.load(f1_frac_path)
    params1 = json.load(open(os.path.join(fig1_dir, "params.json")))

    mean = np.load(os.path.join(fig2_dir, "delta_b_mean.npy"))
    params2 = json.load(open(os.path.join(fig2_dir, "params.json")))

    check("both figures merged the same number of realizations",
          params1["n_realizations_merged"] == params2["n_realizations_merged"],
          f"{params1['n_realizations_merged']} vs {params2['n_realizations_merged']}")
    check("both figures used the same base seed, N, p, sigma",
          all(params1[k] == params2[k] for k in ("base_seed", "N", "p", "sigma")))
    check("both figures used the same rho",
          params1["rho"] == params2["rho"], f"rho = {params1['rho']}")

    # Figure 2 stores K per p_s, so the p_s = 0 consensus fraction of the
    # Figure 2 run can be rebuilt and compared with Figure 1 cell by cell.
    files = sorted(glob.glob(os.path.join(fig2_dir, "realization_*.pkl")))
    ids2, acc = [], np.zeros_like(frac1, dtype=np.int64)
    for fn in files:
        D = pickle.load(open(fn, "rb"))
        ids2.append(D["realization_id"])
        acc += (D["K_ps_low"] == 1)
    frac2_ps0 = acc / len(files)

    ids1 = sorted(pickle.load(open(fn, "rb"))["realization_id"]
                  for fn in glob.glob(os.path.join(fig1_dir, "realization_*.pkl")))
    check("the two figures used the same realization ids",
          sorted(ids2) == ids1, f"{len(ids1)} ids")
    check("no realization id missing from Figure 1",
          ids1 == list(range(len(ids1))), f"max id = {max(ids1)}")
    check("no realization id missing from Figure 2",
          sorted(ids2) == list(range(len(ids2))), f"max id = {max(ids2)}")

    if sorted(ids2) == ids1:
        check("Figure 1 consensus fraction == Figure 2 p_s = 0 fraction",
              np.allclose(frac1, frac2_ps0),
              f"max |difference| = {np.abs(frac1 - frac2_ps0).max():.3g}")

    thr = params2["consensus_frac_min"]
    frac_both = np.load(os.path.join(fig2_dir, "consensus_fraction.npy"))
    check(f"Delta b* is finite exactly where the fraction reaches {thr}",
          np.array_equal(np.isfinite(mean), frac_both >= thr),
          f"{int(np.isfinite(mean).sum())} finite cells")
    check("the fragmented region of Figure 2 contains that of Figure 1",
          bool(((frac1 < 0.5) <= ~np.isfinite(mean)).all()),
          "Figure 2 needs consensus at both p_s, so it can only be larger")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-outputs", action="store_true")
    ap.add_argument("--fig1-dir", default="phase_diagram_rho-0.4_ps0")
    ap.add_argument("--fig2-dir", default="delta_b_results_rho-0.4")
    args, _ = ap.parse_known_args()

    if args.check_outputs:
        check_7_outputs(args.fig1_dir, args.fig2_dir)
    else:
        check_1_core()
        check_2_convention()
        check_3_identity()
        check_4_cross_figure()
        check_5_and_6_dynamics()

    print()
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED:")
        for f in _failures:
            print("  -", f)
        sys.exit(1)
    print("all checks passed")
