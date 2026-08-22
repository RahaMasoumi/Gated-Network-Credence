#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gnc_core.py -- shared implementation of the Gated Network Credence model.

Every convention here follows the manuscript literally; the section/equation
number that each function implements is given in its docstring.  Both figure
scripts (fig01_phase_diagram.py and fig02_delta_b.py) import from this module,
so the two figures are guaranteed to use the same definitions of A_IN, of the
Laplacian, and of the consensus criterion.

INDEX CONVENTION (Methods 4.1)
------------------------------
    A[i, j] = 1   <=>   agent i regards agent j as a potential source
                        of influence,  i.e.  j can influence i.

    row index  = RECEIVER of influence
    column index = SOURCE of influence

The influence digraph therefore has an arc  j -> i  whenever A_IN[i, j] = 1;
its arc-adjacency matrix is A_IN.T .  This distinction matters only for
reachability (cabals / reaches); the Laplacian is built directly from A_IN.
"""

from collections import namedtuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import expm_multiply
from scipy.stats import rankdata

# Methods 4.3.5: b_i(0) ~ U(-B_max, B_max) with B_max = 5.
B_MAX_DEFAULT = 5.0

__all__ = [
    "B_MAX_DEFAULT",
    "Realization",
    "build_realization",
    "threshold_grid",
    "erdos_renyi_undirected",
    "potential_relationships",
    "sample_tau_delta",
    "net_trust_uncertainty",
    "apply_prestige_bias",
    "degree_biased_seeding",
    "build_A_IN",
    "gated_laplacian",
    "count_cabals",
    "gamma_bar",
    "left_null_cabal",
    "long_run_belief",
]


# ---------------------------------------------------------------------------
# Threshold grid
# ---------------------------------------------------------------------------
def threshold_grid(step=0.05):
    """Both thresholds are swept over [-1, 1] (Results, after Eq. 1).

    linspace (not arange) so that the endpoints are exactly -1 and +1 and the
    number of points does not depend on binary rounding of `step`.
    """
    n = int(round(2.0 / step)) + 1
    return np.linspace(-1.0, 1.0, n)


# ---------------------------------------------------------------------------
# Network topology  (Methods 4.3.6, "Synthetic networks")
# ---------------------------------------------------------------------------
def erdos_renyi_undirected(N, p, rng):
    """G(N, p): each unordered pair connected independently with probability p
    (Methods 4.3.6, "Synthetic networks").

    Returns the undirected edge list (u, v) with u < v, and the degree vector.
    Sampled one row of the upper triangle at a time: exact Bernoulli per pair,
    but with O(N) peak memory instead of materialising all N(N-1)/2 pair
    indices (5 MB rather than 313 MB at N = 5000).
    """
    us, vs = [], []
    for i in range(N - 1):
        hit = np.flatnonzero(rng.random(N - i - 1) < p)
        if hit.size:
            us.append(np.full(hit.size, i, dtype=np.int64))
            vs.append(hit + i + 1)
    if us:
        u = np.concatenate(us)
        v = np.concatenate(vs)
    else:
        u = np.empty(0, dtype=np.int64)
        v = np.empty(0, dtype=np.int64)
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.float64)
    return u, v, deg


def barabasi_albert_undirected(N, m, rng):
    """Barabasi-Albert preferential attachment, m links per arriving node
    (Methods 4.3.6: m = 25 for the BA topology).

    networkx is used for the growth process; the rng only supplies its seed, so
    the graph is a deterministic function of the stream.
    """
    import networkx as nx
    seed = int(rng.integers(0, 2**31 - 1))
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    E = np.array(G.edges(), dtype=np.int64)
    u, v = E[:, 0], E[:, 1]
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.float64)
    return u, v, deg


def modular_undirected(n_comm, m1, m2, n_inter, rng):
    """Methods 4.3.6: two equal-size communities, each an independent BA graph,
    plus randomly selected cross-community edges tuned to modularity Q ~ 0.33.

    Community 1 is [0, n_comm) and community 2 is [n_comm, 2 n_comm).  Returns
    the edge list, the degrees, and the community label of every node.
    """
    import networkx as nx
    N = 2 * n_comm
    s1 = int(rng.integers(0, 2**31 - 1))
    s2 = int(rng.integers(0, 2**31 - 1))
    E1 = np.array(nx.barabasi_albert_graph(n_comm, m1, seed=s1).edges(),
                  dtype=np.int64)
    E2 = np.array(nx.barabasi_albert_graph(n_comm, m2, seed=s2).edges(),
                  dtype=np.int64) + n_comm

    # distinct cross-community pairs, drawn uniformly without replacement
    codes = np.empty(0, dtype=np.int64)
    while codes.size < n_inter:
        draw = (rng.integers(0, n_comm, 2 * n_inter) * n_comm
                + rng.integers(0, n_comm, 2 * n_inter))
        codes = np.unique(np.concatenate([codes, draw]))
    codes = rng.permutation(codes)[:n_inter]
    Ex = np.column_stack([codes // n_comm, codes % n_comm + n_comm])

    E = np.vstack([E1, E2, Ex])
    u, v = E[:, 0], E[:, 1]
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.float64)
    community = np.zeros(N, dtype=np.int64)
    community[n_comm:] = 1
    return u, v, deg, community


def potential_relationships(u, v, N):
    """Methods 4.3.6, "Directed relationships": every undirected edge {i, j} is
    represented by the two potential directed relationships (i, j) and (j, i).

    Returns (recv, src, indptr), sorted so that all relationships of a given
    receiver are contiguous:  recv[indptr[i]:indptr[i+1]] == i, and
    src[indptr[i]:indptr[i+1]] is N(i), the set of potential sources of i.
    """
    recv = np.concatenate([u, v])
    src = np.concatenate([v, u])
    order = np.argsort(recv, kind="stable")
    recv, src = recv[order], src[order]
    indptr = np.searchsorted(recv, np.arange(N + 1))
    return recv, src, indptr


# ---------------------------------------------------------------------------
# Trust-distrust coupling  (Methods 4.3.1, Eq. 5)
# ---------------------------------------------------------------------------
def sample_tau_delta(n_relationships, rho, sigma, rng):
    """Eq. (5):  delta_ij = rho * (tau_ij - 1/2) + 1/2 + eps,  eps ~ N(0, sigma^2).

    Manuscript wording: "For each directed edge (i, j), we first sample
    tau_ij ~ U(0, 1).  We then sample eps and compute delta_ij from Eq. (5).
    Because delta_ij is required to remain in [0, 1], any draw of eps that
    yields delta_ij outside [0, 1] is rejected and resampled until a valid
    value is obtained."

    tau is therefore drawn ONCE and held fixed; only eps is resampled.  This
    keeps the marginal of tau exactly U(0, 1).  (Resampling the pair (tau, eps)
    jointly would instead down-weight the tau values whose acceptance
    probability is low -- a visible effect at rho = -1, where tau near 0 or 1
    puts the mean of delta on the boundary.)
    """
    tau = rng.uniform(0.0, 1.0, n_relationships)
    mu = rho * (tau - 0.5) + 0.5
    delta = np.empty(n_relationships)
    pending = np.arange(n_relationships)
    while pending.size:
        cand = mu[pending] + rng.normal(0.0, sigma, pending.size)
        ok = (cand >= 0.0) & (cand <= 1.0)
        delta[pending[ok]] = cand[ok]
        pending = pending[~ok]
    return tau, delta


def net_trust_uncertainty(tau, delta):
    """Methods 4.1:  T_ij = tau_ij - delta_ij,  U_ij = tau_ij + delta_ij - 1."""
    return tau - delta, tau + delta - 1.0


# ---------------------------------------------------------------------------
# Prestige-biased trust assignment  (Methods 4.3.2)
# ---------------------------------------------------------------------------
def apply_prestige_bias(tau, delta, src, indptr, deg, ps, eta, tiebreak,
                        norm_mode="rank", group=None):
    """Methods 4.3.2.  For each receiver i, sources j in N(i) are ranked by

        s_ij = ps * ktilde_j + (1 - ps) * eta_ij,

    with ktilde_j the normalised PRE-FILTERING degree of source j among i's
    potential sources.  The sampled (tau, delta) pairs of receiver i are sorted
    by tau in descending order and assigned to the sources sorted by s_ij in
    descending order.

    Because the procedure only permutes the existing tau and delta values
    within N(i), the receiver-specific marginals are preserved exactly and the
    pairing of tau with its own delta is preserved.

    `eta` and `tiebreak` are supplied by the caller and drawn ONCE per
    realization, so that the eta_ij of a relationship is a property of the
    realization and is shared across the ps values being compared.  ps then
    changes only the weight given to prestige, not the underlying random
    scores.  ps = 0 gives assignment independent of source degree; ps = 1 gives
    maximal alignment with prestige, with degree ties resolved by `tiebreak`.

    norm_mode = "rank" uses tie-corrected (average) ranks rescaled to [0, 1],
    i.e. prestige enters as a rank association, consistent with the
    manuscript's description of the mechanism as a rank association between
    source prominence and received trust.

    `group`, when given, is a per-node label and the permutation is confined to
    the sources sharing the receiver's label: Methods 4.3.5, "Prestige bias is
    applied within modules in the modular simulations, reflecting the
    assumption that prestige is evaluated primarily within community boundaries
    rather than across weak inter-community ties."  Cross-group relationships
    keep their independently sampled (tau, delta) and are still gated normally,
    so inter-community influence exists but carries no prestige signal.
    """
    tau_out = tau.copy()          # cross-group pairs keep their sampled values
    delta_out = delta.copy()

    for i in range(indptr.size - 1):
        a, b = indptr[i], indptr[i + 1]
        if b == a:
            continue

        j = src[a:b]
        if group is None:
            sel = slice(None)
            slots = np.arange(a, b)
        else:
            same = group[j] == group[i]
            if not same.any():
                continue
            sel = same
            slots = np.arange(a, b)[same]

        t = tau[a:b][sel]
        d = delta[a:b][sel]
        k = deg[j[sel]]
        m = t.size
        if m == 0:
            continue

        if norm_mode == "rank":
            r = rankdata(k, method="average")
            lo, hi = r.min(), r.max()
            ktil = np.zeros(m) if hi == lo else (r - lo) / (hi - lo)
        elif norm_mode == "minmax":
            lo, hi = k.min(), k.max()
            ktil = np.zeros(m) if hi == lo else (k - lo) / (hi - lo)
        else:
            raise ValueError(f"unknown norm_mode {norm_mode!r}")

        s = ps * ktil + (1.0 - ps) * eta[a:b][sel]
        src_order = np.lexsort((tiebreak[a:b][sel], -s))  # most prestigious first
        pair_order = np.argsort(-t, kind="stable")        # highest trust first
        tau_out[slots[src_order]] = t[pair_order]
        delta_out[slots[src_order]] = d[pair_order]

    return tau_out, delta_out


# ---------------------------------------------------------------------------
# Degree-biased positive seeding  (Methods 4.3.3)
# ---------------------------------------------------------------------------
def degree_biased_seeding(b_init, deg, x, subset=None):
    """Methods 4.3.3.  H_x = nodes in the top fraction x of the degree ranking;
    P_x = nodes in the top fraction x ranked by initial belief value b_i(0)
    (most positive, not largest in absolute value).  H_x is sorted in
    descending degree, P_x in descending belief value, and beliefs are
    reassigned pairwise so that the highest-degree node in H_x ends up holding
    the most positive belief in P_x, the second-highest-degree node the
    second-most-positive belief, and so on.

    Implemented as a sequence of transpositions of belief values, so the
    overall multiset of initial beliefs is preserved exactly (only their
    allocation to agents changes).  When H_x and P_x overlap, the swap at step
    m exchanges with whichever agent currently holds the m-th most positive
    original value; this is what makes the composition of swaps a permutation
    and gives b[H_x[m]] = (m-th most positive original belief) for every m.

    `subset`, when given, restricts both H_x and P_x to those agents and takes
    the fraction x of the subset: Methods 4.3.5, "positive promoters are seeded
    within one community" in the modular simulations.
    """
    N = b_init.size
    b = b_init.copy()
    pool = np.arange(N) if subset is None else np.asarray(subset)
    h = int(pool.size * x)
    if h == 0:
        return b

    H = pool[np.argsort(-deg[pool], kind="stable")][:h]      # ties -> low id
    P = pool[np.argsort(-b_init[pool], kind="stable")][:h]   # most positive

    holder = np.arange(N)   # holder[k] = agent currently holding agent k's original value
    owner = np.arange(N)    # owner[a]  = agent whose original value agent a now holds
    for m in range(h):
        hub = H[m]
        donor = holder[P[m]]
        if donor == hub:
            continue
        b[hub], b[donor] = b[donor], b[hub]
        o_hub, o_donor = owner[hub], owner[donor]
        holder[o_hub], holder[o_donor] = donor, hub
        owner[hub], owner[donor] = o_donor, o_hub
    return b


# ---------------------------------------------------------------------------
# Inter-community polarization  (Methods 4.3.4)
# ---------------------------------------------------------------------------
def polarization(b, community):
    """Methods 4.3.4:  P(b) = (2/N) ( sum_{C1} b_i - sum_{C2} b_j ).

    C1 (label 0) is the positive pole, so P > 0 means the positive beliefs sit
    in C1.  With |C1| = |C2| = N/2 this equals mean(C1) - mean(C2)."""
    c1 = community == 0
    return 2.0 * (b[c1].sum() - b[~c1].sum()) / b.size


def polarize_beliefs(b_init, community, psi, rng, random_draw_cap=200_000):
    """Methods 4.3.4.  Beliefs are exchanged between the two communities until
    the polarization reaches

        P_target = P0 + psi * (Pmax - P0),

    where Pmax is attained by giving the N/2 largest values to C1 and the N/2
    smallest to C2.  Only existing belief values are exchanged, so the global
    distribution of initial beliefs is preserved exactly.

    Phase 1 is the procedure as written: a pair (i, j) is drawn uniformly from
    C1 x C2 and swapped whenever b_i < b_j, which raises P by 4 (b_j - b_i) / N.

    Phase 2 exists because the procedure as written does not terminate at
    psi = 1: once one discordant pair is left, the chance of drawing it is
    ~1/(N/2)^2 per draw.  After `random_draw_cap` draws the remaining distance
    is therefore closed deterministically, by exchanging the smallest remaining
    C1 belief with the largest remaining C2 belief -- the most productive
    admissible swaps -- so psi = 1 lands exactly on the maximally polarized
    configuration.  For psi <~ 0.9 phase 1 alone reaches the target in a few
    thousand draws and phase 2 never runs.
    """
    b = np.asarray(b_init, dtype=float).copy()
    if psi <= 0.0:
        return b

    c1 = np.flatnonzero(community == 0)
    c2 = np.flatnonzero(community == 1)
    n = b.size

    sv = np.sort(b_init)
    b_max = np.empty_like(b)
    b_max[c2] = sv[:c2.size]                 # smallest values to C2
    b_max[c1] = sv[c2.size:]                 # largest values to C1
    P0 = polarization(b_init, community)
    P_target = P0 + psi * (polarization(b_max, community) - P0)
    if P0 >= P_target:
        return b

    P = P0
    I = rng.integers(0, c1.size, random_draw_cap)
    J = rng.integers(0, c2.size, random_draw_cap)
    for step in range(random_draw_cap):
        i, j = c1[I[step]], c2[J[step]]
        if b[i] < b[j]:
            b[i], b[j] = b[j], b[i]
            P += 4.0 * (b[i] - b[j]) / n
            if P >= P_target:
                return b

    order1 = c1[np.argsort(b[c1])]            # C1, smallest first
    order2 = c2[np.argsort(-b[c2])]           # C2, largest first
    for i, j in zip(order1, order2):
        if b[i] >= b[j]:
            break
        b[i], b[j] = b[j], b[i]
        P += 4.0 * (b[i] - b[j]) / n
        if P >= P_target:
            break
    return b


# ---------------------------------------------------------------------------
# Long-run mean belief  (Eq. 3-4)
# ---------------------------------------------------------------------------
def mean_belief_weights(L, T, also_at=None):
    """Weights w with  mean(b(T)) = w . b(0) / N  for every b(0).

    Since b(T) = exp(-L T) b(0),

        mean(b(T)) = 1^T exp(-L T) b(0) / N = [exp(-L^T T) 1]^T b(0) / N,

    so one matrix exponential per Laplacian serves every initial condition.
    Exact: the propagator is evaluated with expm_multiply, not time-stepped, so
    there is no step size and no stability condition to check.

    With `also_at` given, the weights at both horizons are returned as a pair,
    obtained from a single expm_multiply over a two-point time grid.  Comparing
    them is how a caller certifies that the value has settled.
    """
    ones = np.ones(L.shape[0])
    A = -L.T.tocsc()
    if also_at is None:
        return expm_multiply(A, ones, start=0.0, stop=float(T),
                             num=2, endpoint=True)[-1]
    W = expm_multiply(A, ones, start=float(T), stop=float(also_at),
                      num=2, endpoint=True)
    return W[0], W[1]


# ---------------------------------------------------------------------------
# One realization: every random quantity of the model, on independent streams
# ---------------------------------------------------------------------------
Realization = namedtuple(
    "Realization",
    "id N deg u v recv src indptr b_init b0 tau delta eta tiebreak")


def build_realization(realization_id, N, p, rho, sigma, x, base_seed):
    """Draw one realization of the model.

    Methods 4.3.6, "Replication": each realization redraws the network, the
    initial beliefs, the trust-distrust values, and the prestige-based
    assignment of relational evaluations; the same networks and initial belief
    configurations are reused across parameter combinations.

    Each of those four quantities is drawn from its own independent stream,
    spawned from SeedSequence([base_seed, realization_id]).  Consequently
    realization r is bit-identical in every script and for every threshold or
    p_s setting, regardless of what else a given script happens to draw.  Both
    figure scripts obtain their realizations only through this function, so the
    reuse is guaranteed by construction rather than by convention; see
    verify_reproducibility.py.
    """
    ss = np.random.SeedSequence([base_seed, realization_id])
    ss_graph, ss_beliefs, ss_trust, ss_prestige = ss.spawn(4)

    u, v, deg = erdos_renyi_undirected(N, p, np.random.default_rng(ss_graph))
    recv, src, indptr = potential_relationships(u, v, N)

    # Methods 4.3.5 / 4.3.3
    rng_b = np.random.default_rng(ss_beliefs)
    b_init = rng_b.uniform(-B_MAX_DEFAULT, B_MAX_DEFAULT, N)
    b0 = degree_biased_seeding(b_init, deg, x)

    # Methods 4.3.1, Eq. (5)
    tau, delta = sample_tau_delta(recv.size, rho, sigma,
                                  np.random.default_rng(ss_trust))

    # Methods 4.3.2: eta_ij and the degree tie-break are properties of the
    # realization, so a comparison across p_s changes only the weight given to
    # prestige and not the underlying random scores.
    rng_p = np.random.default_rng(ss_prestige)
    eta = rng_p.random(recv.size)
    tiebreak = rng_p.random(recv.size)

    return Realization(realization_id, N, deg, u, v, recv, src, indptr,
                       b_init, b0, tau, delta, eta, tiebreak)


# ---------------------------------------------------------------------------
# Gate, masked adjacency, Laplacian  (Methods 4.1, Eqs. 1-3)
# ---------------------------------------------------------------------------
def build_A_IN(recv, src, T, U, TT, UT, N):
    """Methods 4.1.  Retaining only the admissible relationships defines

        A_IN[i, j] = 1  iff  A[i, j] = 1 and T_ij >= TT and U_ij <= UT,

    i.e. row = receiver i, column = admitted source j.  Eq. (1) is the gate.
    """
    admitted = (T >= TT) & (U <= UT)
    i = recv[admitted]
    j = src[admitted]
    return sp.csr_matrix((np.ones(i.size), (i, j)), shape=(N, N))


def gated_laplacian(A_IN):
    """Eq. (3):  L = D_row - A_IN,  with D_row = diag(k_row_1, ..., k_row_N)
    and k_row_i = sum_j A_IN[i, j] the number of admitted sources of receiver i.

    Unnormalised, and generally non-Hermitian (Methods 4.1, "Modeling note:
    additive social pull").
    """
    k_row = np.asarray(A_IN.sum(axis=1)).ravel()
    return (sp.diags(k_row) - A_IN).tocsr()


def count_cabals(A_IN):
    """K = the number of reaches of the graph induced by A_IN (Methods 4.2).

    Reachability is defined in the influence direction: source j points to
    receiver i whenever A_IN[i, j] = 1.  The arc-adjacency matrix of that
    influence digraph is therefore

        M = A_IN.T ,      M[j, i] = 1  for the arc  j -> i.

    Each reach contains exactly one cabal: a strongly connected component that
    receives no influence from outside itself, i.e. a source SCC of M -- an SCC
    with no incoming inter-SCC arc.  The number of reaches equals the number of
    cabals, so K is obtained by counting them.

    Global consensus arises if and only if K == 1; K > 1 is fragmentation.
    """
    N = A_IN.shape[0]
    if A_IN.nnz == 0:
        return N                      # no admissible ties -> N singleton reaches

    M = A_IN.T.tocsr()
    n_scc, label = connected_components(M, directed=True, connection="strong")

    Mc = M.tocoo()
    inter = label[Mc.row] != label[Mc.col]        # inter-SCC influence arcs
    has_incoming = np.zeros(n_scc, dtype=bool)
    has_incoming[label[Mc.col][inter]] = True     # the receiving SCC is influenced
    return n_scc - int(has_incoming.sum())


def left_null_cabal(A_sub, tol=1e-13, max_iter=20000, warn=True):
    """Left null vector of L = D_row - A_sub for a STRONGLY CONNECTED subgraph,
    normalised to be nonnegative and to sum to one.

    Method: left-null(D - A) = pi / d, where pi is the stationary distribution
    of the row-stochastic walk S = D^{-1} A (Veerman & Kummel, Cor. 4.6: each
    receiver averages the sources it retains).  Verifying the identity: the
    condition L^T w = 0 reads d_j w_j = sum_i A_ij w_i for every j, and
    substituting w = pi / d turns it into pi_j = (S^T pi)_j, which is exactly
    stationarity.

    pi is obtained by power iteration on the lazy walk (I + S)/2, which is
    irreducible and aperiodic on a strongly connected subgraph, so the
    stationary distribution is unique and convergence is geometric.  Sparse
    mat-vecs only, so this avoids the LU fill-in of a direct solve.
    """
    n = A_sub.shape[0]
    if n == 1:
        return np.array([1.0])
    d = np.asarray(A_sub.sum(axis=1)).ravel()
    if d.min() <= 0:
        raise ValueError("left_null_cabal: subgraph has a receiver with no "
                         "admitted source, so it is not strongly connected")
    dinv = 1.0 / d
    S = sp.diags(dinv) @ A_sub
    PT = (0.5 * (sp.eye(n, format="csr") + S)).transpose().tocsr()

    pi = np.full(n, 1.0 / n)
    converged = False
    for _ in range(max_iter):
        nxt = PT @ pi
        tot = nxt.sum()
        if tot <= 0:
            break
        nxt /= tot
        delta = np.abs(nxt - pi).sum()
        pi = nxt
        if delta < tol:
            converged = True
            break
    if warn and not converged:
        import warnings
        warnings.warn(
            f"left_null_cabal: power iteration did not reach tol={tol:g} in "
            f"{max_iter} iterations on a cabal of {n} nodes; the returned "
            f"vector may not be the left null vector.", RuntimeWarning)

    w = np.clip(pi * dinv, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def gamma_bar(A_IN, warn=True):
    """Per-agent direct long-run influence, Eq. (4), computed exactly.

    Returns (gamma, K, n_scc):

      gamma[i]  the coefficient multiplying agent i's initial belief in the
                limiting belief of its reach: gamma_bar_m(i) on the cabal B_m,
                and zero on common-part nodes, per Eq. (4) and Methods 4.2.
                The entries of each cabal sum to one, so sum(gamma) = K.
      K         the number of reaches, equal to the number of cabals.
                Consensus iff K == 1, in which case gamma . b(0) is the
                consensus value.
      n_scc     the number of strongly connected components.  n_scc == 1 means
                the single cabal spans every agent, so gamma is strictly
                positive everywhere.

    No eigenvalue tolerance and no regularisation are involved: the cabals are
    identified combinatorially and the null vector on each is the unique
    stationary distribution of a strongly connected walk.
    """
    A_IN = sp.csr_matrix(A_IN, dtype=float)
    n = A_IN.shape[0]
    M = A_IN.T.tocsr()                       # influence digraph, arcs j -> i
    n_scc, label = connected_components(M, directed=True, connection="strong")

    Mc = M.tocoo()
    inter = label[Mc.row] != label[Mc.col]
    has_incoming = np.zeros(n_scc, dtype=bool)
    has_incoming[label[Mc.col][inter]] = True
    cabals = np.flatnonzero(~has_incoming)

    gamma = np.zeros(n)
    for m in cabals:
        nodes = np.flatnonzero(label == m)
        gamma[nodes] = left_null_cabal(A_IN[nodes][:, nodes].tocsr(), warn=warn)
    return gamma, int(cabals.size), int(n_scc)


def long_run_belief(L, b0, T_max):
    """Integrate Eq. (3) to time T_max:  b(t) = exp(-L t) b(0)."""
    return expm_multiply(-L.tocsc() * T_max, b0)


# ---------------------------------------------------------------------------
# Self-tests  (run:  python gnc_core.py)
# ---------------------------------------------------------------------------
def _selftest():
    import networkx as nx

    rng = np.random.default_rng(0)

    # 1. count_cabals against a networkx condensation, on random digraphs.
    for trial in range(300):
        n = int(rng.integers(3, 12))
        A_IN = (rng.random((n, n)) < 0.18).astype(float)
        np.fill_diagonal(A_IN, 0.0)
        A_sp = sp.csr_matrix(A_IN)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(n):
                if A_IN[i, j]:
                    G.add_edge(j, i)              # influence arc j -> i
        C = nx.condensation(G)
        k_ref = sum(1 for v in C.nodes if C.in_degree(v) == 0)
        assert count_cabals(A_sp) == k_ref, (trial, count_cabals(A_sp), k_ref)

    # 2. Consensus <=> a single reach: on a consensus graph exp(-L t) b0 must
    #    converge to one common value, and L must have a nonnegative left null
    #    vector summing to one.
    n = 60
    G = nx.gnp_random_graph(n, 0.12, seed=3, directed=True)
    A_IN = sp.csr_matrix(nx.to_numpy_array(G).T)      # A_IN[i, j] = arc j -> i
    if count_cabals(A_IN) == 1:
        L = gated_laplacian(A_IN)
        b = long_run_belief(L, rng.uniform(-5, 5, n), 400.0)
        assert b.max() - b.min() < 1e-9, b.max() - b.min()
        w, V = np.linalg.eig(L.toarray().T)
        g = np.real(V[:, np.argmin(np.abs(w))])
        g = g / g.sum()
        assert (g > -1e-10).all()
        assert abs(float(g @ np.ones(n)) - 1.0) < 1e-10

    # 3. Row sums of L vanish (each admitted tie enters with the same additive
    #    coefficient and the diagonal balances it).
    assert np.allclose(np.asarray(gated_laplacian(A_IN).sum(axis=1)).ravel(), 0.0)

    # 4. degree_biased_seeding preserves the belief multiset and puts the h most
    #    positive values on the h highest-degree agents in degree order.
    for trial in range(300):
        n = int(rng.integers(5, 40))
        deg = rng.integers(0, 8, n).astype(float)
        b0 = rng.uniform(-5, 5, n)
        x = float(rng.choice([0.05, 0.1, 0.2, 0.4]))
        b = degree_biased_seeding(b0, deg, x)
        assert np.allclose(np.sort(b), np.sort(b0))
        h = int(n * x)
        H = np.argsort(-deg, kind="stable")[:h]
        assert np.allclose(b[H], np.sort(b0)[::-1][:h])

    # 5. apply_prestige_bias preserves each receiver's marginals and the
    #    (tau, delta) pairing, and ps = 1 orders trust by source degree.
    u, v, deg = erdos_renyi_undirected(300, 0.05, rng)
    recv, src, indptr = potential_relationships(u, v, 300)
    tau, delta = sample_tau_delta(recv.size, -0.4, 0.05, rng)
    eta = rng.random(recv.size)
    tie = rng.random(recv.size)
    for ps in (0.0, 0.4, 1.0):
        t2, d2 = apply_prestige_bias(tau, delta, src, indptr, deg, ps, eta, tie)
        for i in range(300):
            a, b_ = indptr[i], indptr[i + 1]
            if b_ - a < 2:
                continue
            assert np.allclose(np.sort(tau[a:b_]), np.sort(t2[a:b_]))
            assert np.allclose(np.sort(delta[a:b_]), np.sort(d2[a:b_]))
            k = np.argsort(tau[a:b_])
            k2 = np.argsort(t2[a:b_])
            assert np.allclose(delta[a:b_][k], d2[a:b_][k2])
            # trust must be non-increasing along the prestige-score ranking
            k_ = deg[src[a:b_]]
            r_ = rankdata(k_, method="average")
            ktil = (np.zeros(b_ - a) if r_.max() == r_.min()
                    else (r_ - r_.min()) / (r_.max() - r_.min()))
            s_ = ps * ktil + (1.0 - ps) * eta[a:b_]
            assert np.all(np.diff(t2[a:b_][np.lexsort((tie[a:b_], -s_))]) <= 1e-12)

    # 6. tau keeps its U(0, 1) marginal at rho = -1, where rejection is heaviest.
    tau, delta = sample_tau_delta(400_000, -1.0, 0.05, rng)
    assert (delta >= 0).all() and (delta <= 1).all()
    assert abs(tau.mean() - 0.5) < 5e-3, tau.mean()
    counts = np.histogram(tau, bins=10, range=(0, 1))[0]
    assert counts.max() / counts.min() < 1.05, counts

    print("gnc_core: all self-tests passed")


if __name__ == "__main__":
    _selftest()
