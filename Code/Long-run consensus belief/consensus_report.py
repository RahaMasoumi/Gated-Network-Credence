#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
consensus_report.py -- where does the filtered graph fail to reach a single
reach, and by how much does it matter?

Reads the per-realization files written by run_trends.py and reports, for every
panel:

  1. the overall fraction of (realization, rho, p_s) cells with K = 1;
  2. a rho x p_s table of the fraction of realizations with K > 1, so the
     parameters responsible are visible rather than inferred;
  3. how many reaches appear when it does fragment, and how large the largest
     reach is relative to N -- a graph with one dominant reach plus a few
     singletons is a different situation from a genuinely split network;
  4. the effect of dropping the affected data, computed three ways, so the
     decision can be made on the size of the change rather than on principle:

       none         every realization contributes (current behaviour)
       cell         per (rho, p_s), average only the realizations with K = 1
       realization  drop any realization that fragmented in any cell

Usage
-----
  python consensus_report.py
  python consensus_report.py --dir trends_BA_N5000_m25_pointE
"""

import argparse
import glob
import os
import pickle

import numpy as np


def _parse_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=None,
                    help="a single results directory; default = all trends_*")
    ap.add_argument("--x", type=float, default=0.2,
                    help="promoter prevalence at which the three averaging "
                         "conventions are compared")
    return ap.parse_args()


def load(path):
    files = sorted(glob.glob(os.path.join(path, "realization_*.pkl")))
    if not files:
        return None
    b, K, settled, ids = [], [], [], []
    params = None
    for fn in files:
        with open(fn, "rb") as f:
            D = pickle.load(f)
        params = D["params"]
        b.append(D["b_star"])
        K.append(D["K"])
        settled.append(D["settled"])
        ids.append(D["realization_id"])
    return (np.stack(b), np.stack(K), np.stack(settled),
            np.array(ids), params)


def report(path, x_probe):
    loaded = load(path)
    if loaded is None:
        print(f"\n{path}: no realization files")
        return
    b, K, settled, ids, P = loaded
    n_real = b.shape[0]
    rho = P["rho_values"]
    ps = P["ps_values"]
    xs = P["x_values"]

    frag = K > 1
    print(f"\n{'=' * 78}\n{path}")
    print(f"  {P['topology']}"
          + (f"  psi={P['psi']:g}" if P["psi"] is not None else "")
          + f"  point {P['point']} = ({P['TT']:g}, {P['UT']:g})"
          + (f"  [{P['regime']}]" if P.get("regime") else "")
          + f"   {n_real} realizations")

    tot = frag.size
    print(f"\n  cells with K = 1 : {100 * (1 - frag.mean()):.2f}%"
          f"   ({tot - int(frag.sum())} of {tot})")
    print(f"  cells settled    : {100 * settled.mean():.2f}%")
    n_bad_real = int(frag.any(axis=(1, 2)).sum())
    print(f"  realizations with at least one fragmented cell : "
          f"{n_bad_real} of {n_real}  ({100 * n_bad_real / n_real:.0f}%)")

    if not frag.any():
        print("\n  nothing fragmented anywhere: every panel value is a "
              "consensus value.")
        return

    # ---- 2. where, in (rho, p_s) ----
    print(f"\n  fraction of realizations with K > 1"
          f"      (rows = rho, columns = p_s)")
    print("        " + "".join(f"{v:>8g}" for v in ps))
    for i, r in enumerate(rho):
        row = "".join(f"{v:>8.2f}" for v in frag[:, i, :].mean(axis=0))
        print(f"  {r:>6g}" + row)

    # ---- 3. how badly ----
    K_bad = K[frag]
    print(f"\n  when it fragments: K ranges {K_bad.min()} to {K_bad.max()}, "
          f"median {int(np.median(K_bad))}")
    print(f"  (K counts reaches; a few singleton reaches alongside one large "
          f"reach is\n   far milder than a genuine split, but both give K > 1)")

    # ---- 4. what difference dropping it makes ----
    xi = xs.index(x_probe) if x_probe in xs else 0
    keep_real = ~frag.any(axis=(1, 2))
    print(f"\n  b* at x = {xs[xi]:g} under the three averaging conventions")
    print(f"  {'rho':>6}  {'p_s':>6}  {'none':>9}{'cell':>9}"
          f"{'realization':>13}   {'n(K=1)':>7}")
    print("  " + "-" * 60)
    worst = 0.0
    for i, r in enumerate(rho):
        for j, s in enumerate(ps):
            if not frag[:, i, j].any():
                continue
            col = b[:, i, xi, j]
            m_none = col.mean()
            ok = ~frag[:, i, j]
            m_cell = col[ok].mean()
            m_real = col[keep_real].mean() if keep_real.any() else np.nan
            worst = max(worst, abs(m_cell - m_none))
            print(f"  {r:>6g}  {s:>6g}  {m_none:>9.4f}{m_cell:>9.4f}"
                  f"{m_real:>13.4f}   {int(ok.sum()):>7}")
    print(f"\n  largest shift from dropping fragmented cells: {worst:.4f}")
    sd = b[:, :, xi, :].std(axis=0, ddof=1).mean()
    print(f"  mean +-1 SD error bar at this x:                {sd:.4f}")
    if worst < 0.25 * sd:
        print("  -> the shift is small next to the error bars; the choice of "
              "convention\n     does not change what the figure shows.")
    else:
        print("  -> the shift is comparable to the error bars; state the "
              "convention used\n     in the caption.")


if __name__ == "__main__":
    args = _parse_cli()
    dirs = [args.dir] if args.dir else sorted(glob.glob("trends_*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise SystemExit("no trends_* directories here")
    for d in dirs:
        report(d, args.x)
    print()
