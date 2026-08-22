# Gated Network Credence — long-run influence after filtering (Figure 4)

Reproduces Figure 4: the normalised left null vector γ̄ of the directed
Laplacian of the filtered influence graph, plotted against pre-filtering
in-degree, for Erdős–Rényi networks in the main panels and Barabási–Albert and
modular networks in the insets.

```
gnc_core.py                          the model, shared with every other
                                     figure in the project
fig04_spectral_characterization.py   computes γ̄ and draws the figure
```

Requirements: `numpy`, `scipy`, `networkx`, `matplotlib`.

---

## Run

```bash
python gnc_core.py            # self-tests, ~20 s
python fig04_spectral_characterization.py    # a few minutes, single core
```

Output:

```
fig04_output/fig04_gamma_bar.pdf
fig04_output/fig04_gamma_bar.png
```

This is a single realization, as in the figure caption, so no SLURM array is
needed. `--realization_id 1` shows a different one, which is worth doing once to
confirm the pattern is not specific to one draw.

If the login node is not the place for a few minutes of compute:

```bash
sbatch run_fig04.sbatch       # if you keep the one-task job file
```

---

## What is plotted

γ̄ is the vector with γ̄ᵀL = 0 for the unnormalised directed Laplacian
`L = D_row − A_IN` of the filtered graph. Equation (4) gives

    lim_{t→∞} b(t) = Σ_m (γ̄_m · b(0)) γ_m,

so γ̄ᵢ is the coefficient multiplying agent *i*'s initial belief in the limiting
belief — its **direct long-run contribution** (Methods 4.2).

Because γ̄ is a property of the filtered graph alone, it does not depend on
b(0). Three things therefore play no role in this figure and are correctly
absent from the code: the initial belief draw, the degree-biased promoter
seeding of Methods 4.3.3, and the modular polarization ψ of Methods 4.3.4. The
modular insets are the same for ψ = 0, 0.4 and 1; 

### Conditions

3 topologies × ρ ∈ {−1, −0.01} × p_s ∈ {0.4, 1} × 2 gate points = 24
conditions, each printing its reach count:

```
ER       rho=    -1 ps= 0.4 E: K=1    n_scc=1     -> single reach  [strongly connected]
```

- **K** is the number of reaches, equal to the number of cabals. Consensus iff
  K = 1.
- **n_scc** is the number of strongly connected components. `n_scc == 1` means
  the single cabal spans every agent, so γ̄ is strictly positive everywhere.
  With K = 1 but n_scc > 1 there is one cabal plus a common part on which γ̄ is
  exactly zero — which is what Eq. (4) prescribes.

On the reported run **all 24 conditions returned K = 1**, so γ̄ is a consensus
weight throughout Figure 4 and no qualification is needed in the caption.

The check is nevertheless kept in the code. If any condition reports
`FRAGMENTED (K > 1)` — which can happen at other ρ or at other gate points —

---

## How γ̄ is computed

Exactly, with no eigenvalue tolerance and no regularisation.

1. The cabals are identified combinatorially: source strongly connected
   components of the influence digraph, whose arc-adjacency matrix is `A_IN.T`
   since an arc runs *j → i* whenever `A_IN[i, j] = 1`.
2. On each cabal — strongly connected by construction — the left null vector of
   `D_row − A` is `π / d`, where π is the stationary distribution of the
   row-stochastic walk `S = D⁻¹A`. Substituting `w = π/d` into
   `d_j w_j = Σ_i A_ij w_i` turns the null condition into `π = Sᵀπ`, which is
   exactly stationarity. π is obtained by power iteration on the lazy walk
   `(I + S)/2`, which is irreducible and aperiodic, so the limit is unique and
   convergence is geometric. Sparse mat-vecs only.
3. Common-part nodes get γ̄ = 0, per Eq. (4).

Verified in `gnc_core.py`:

- the power iteration agrees with a direct sparse solve to **1.4e-13**, with
  residual |wᵀL| ~ 4e-14 and sum(w) = 1 to machine precision;
- on single-reach graphs, `γ̄ · b(0)` reproduces the value obtained by
  integrating `exp(−Lt)b(0)` to **nine decimal places** — an independent route
  to the same number, which is what makes the row/column convention and the
  sign of the Laplacian checkable rather than merely self-consistent;
- `gamma_bar` and `count_cabals` return the same K by construction, so this
  figure and the phase diagram share one definition of consensus.

If the power iteration ever fails to converge it raises a `RuntimeWarning`
rather than silently returning a non-converged vector.

---

## Consistency with the rest of the project

Every model quantity comes from `gnc_core.py`, the same module used by the
phase-diagram, Δb\* and b\*-versus-p_s scripts. Specifically:

| | |
|---|---|
| index convention | `A_IN[i, j] = 1` iff *j* can influence *i*; row = receiver, column = source |
| Laplacian | unnormalised `L = D_row − A_IN` of Eq. (3) |
| Eq. (5) | τ sampled once, only ε rejected and resampled — the marginal of τ stays U(0,1). This matters here because ρ = −1 is one of the two couplings shown, and there the mean of δ sits on the boundary for τ near 0 or 1 |
| Methods 4.3.2 | rank-association permutation within each receiver's block |
| Methods 4.3.5 | prestige **confined to modules** on the modular network, so this figure uses the same prestige rule as the modular panels of Figures S5–S6 |
| seeds | `BASE_SEED = 20260819`, spawned into independent sub-streams, so realization *r* is the same graph here as in every other figure |

Within a topology and ρ, τ, ε and the prestige scores η are drawn once and
shared by the two p_s values, so moving from p_s = 0.4 to p_s = 1 changes only
the prestige weighting and nothing else.

---

## Parameters (Methods 4.3.5–4.3.6)

N = 5000; σ_ε = 0.05; ER p = 0.01; BA m = 25; modular = two BA communities of
2500 with m = 21 plus 20857 cross edges, giving Q ≈ 0.333 and ⟨k⟩ = 50.0;
ρ ∈ {−1, −0.01}; p_s ∈ {0.4, 1}; gate points E = (0.2, 0.8) Evaluative and
F = (−0.8, 0.0) Friction-averse.


---

## Smoothing

`NBINS`, `MEDIAN_SMOOTH_SIGMA` and `RIBBON_SMOOTH_SIGMA` at the top of the
script control the display only. The band and the mean line use the same kernel
by default, so the ±1 SD band really is centred on the curve that is drawn and
the caption's "one standard deviation around the bin mean".
Raising `RIBBON_SMOOTH_SIGMA` above `MEDIAN_SMOOTH_SIGMA` widens the band and
takes it off-centre.
