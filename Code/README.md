# Gated Network Credence — Code

Simulation and analysis code for *"Regimes of Influence under Trust–Distrust
Gating"*. One folder per analysis. Each folder is self-contained: it ships its
own copy of the model module `gnc_core.py`, so it can be run without reference
to the others, and each has its own README with the exact commands, SLURM
recipes, runtimes and diagnostics.

Start with the project [README](../README.md) for the model itself.

---

## Folders

| Folder | Produces | What it computes |
| --- | --- | --- |
| [`phase-diagram/`](phase-diagram/) | Fig. 1(a)②, Fig. 2, Fig. S8 | consensus/fragmentation over the `(T_T, U_T)` plane, and `Δb*` across that plane |
| [`Long-run consensus belief/`](Long-run%20consensus%20belief/) | Fig. 3(a), Fig. S3–S6 | long-run collective belief `b*` as a function of prestige bias `p_s`, across `ρ`, `x`, three topologies and three polarization levels |
| [`Spectral characterization of long run influence/`](Spectral%20characterization%20of%20long%20run%20influence/) | Fig. 4 | the normalised left null vector `γ̄` of the filtered Laplacian, against pre-filtering in-degree |

## Figure → script

| Figure | Folder | Script |
| --- | --- | --- |
| Fig. 1(a)② | `phase-diagram/` | `fig01_phase_diagram.py` |
| Fig. 2 | `phase-diagram/` | `fig02_delta_b.py` (compute + merge) |
| Fig. S8 | `phase-diagram/` | `fig02_panels.py` (draws only; needs Fig. 2 merged first) |
| Fig. 3(a) | `Long-run consensus belief/` | `run_trends.py`, then `plot_trends.py --layout main` |
| Fig. S3–S4 | `Long-run consensus belief/` | `plot_trends.py --layout topologies --regime E\|F` |
| Fig. S5–S6 | `Long-run consensus belief/` | `plot_trends.py --layout polarization --regime E\|F` |
| Fig. 4 | `Spectral characterization of long run influence/` | `Spectral characterization.py` |
| analytic threshold overlay | `phase-diagram/` | `plot_threshold_lines.py` (draws only; reads merged `Δb*` arrays) |

<!-- TODO — analyses not yet in this repository:
     Fig. 1(b–c), S2 : belief trajectories at p_s = 0 and p_s = 1
     Fig. 3(b)       : in-degree retention after filtering
     Fig. 5          : Bitcoin-OTC reputation network
     Fig. 6          : U.S. state-legislator follow network -->

---

## Shared conventions

Every folder uses the same index convention, the same Laplacian, the same
definition of consensus, and the same seeds. These are documented once here and
assumed by each folder's README.

### The gate and the influence matrix (Methods 4.1–4.2)

Row = receiver, column = source:

    A_IN[i, j] = 1   <=>   A[i, j] = 1 and T_ij >= T_T and U_ij <= U_T

so `A_IN[i, j] = 1` means agent *j* can influence agent *i*. The dynamics use
the unnormalised directed Laplacian `L = D_row − A_IN` of Eq. (3), with
`k_row_i` the number of admitted sources of receiver *i*.

The influence digraph has an arc *j → i* whenever `A_IN[i, j] = 1`, so its
arc-adjacency matrix is `A_IN.T`. `count_cabals` counts its source strongly
connected components, which is *K*, the number of reaches.

**Consensus iff K = 1.** Every folder reports *K* rather than assuming it.

### How direction enters

Each topology is generated **undirected** (Methods 4.3.6), then every undirected
edge `{i, j}` is represented as the two potential directed relationships
`(i, j)` and `(j, i)`, stored receiver-major. The two directions receive
independently sampled `(τ, δ)`, the prestige permutation acts within each
receiver's block, and the gate is applied to each direction separately. The
potential graph is bidirectional but `A_IN` is not — on ER roughly two thirds of
admitted relationships are one-way, which is what makes the Laplacian
non-Hermitian and the reach structure meaningful.

### Seeding

Realization *r* is `SeedSequence([BASE_SEED, r])` with `BASE_SEED = 20260819`,
spawned into independent sub-streams for the network, the initial beliefs, the
trust–distrust values, and the prestige scores. Realizations are obtained only
through `gnc_core.build_realization`, so realization *r* is bit-identical across
scripts and across parameter settings.

Consequently the curves are **paired comparisons**: moving along `p_s` changes
only the prestige weighting, moving along `x` changes only `b(0)`, and moving
along `ρ` reuses the same `τ` and `ε` wherever the rejection step accepts them.

### Merge safety

Every `.pkl` carries the full parameter dict. The merge steps refuse to combine
files whose parameters differ, or to count a realization id twice, so a
partially failed SLURM array cannot silently produce a figure.

---

## Requirements

Python 3.10 or newer, with `numpy`, `scipy`, `networkx`, `matplotlib`.

Run the self-tests before anything else — they are the fastest check that the
environment is sound:

```bash
python gnc_core.py          # ~20 s, in any of the three folders
```

`phase-diagram/` additionally ships `verify_reproducibility.py`, 24 consistency
checks at reduced `N` (~40 s).

---

## Running on a cluster

`phase-diagram/` and `Long-run consensus belief/` include SLURM job files.
Before submitting, open each `.sbatch` and check the interpreter line against
`which python`:

```bash
PY=$HOME/.conda/envs/myenv/bin/python
```

If the conda environment has another name, every array task fails instantly with
"No such file or directory".

The jobs are single-threaded. The `*_NUM_THREADS=1` exports in the sbatch files
matter — without them BLAS oversubscribes the single allocated core.

`Spectral characterization of long run influence/` runs in a few minutes on one
core and needs no array.

<!-- Note: two folder names contain spaces, which requires quoting in every
     shell command:  cd "Long-run consensus belief"
     Consider renaming to long-run-consensus-belief and spectral-influence. -->

---

## Parameters

Shared across folders (Methods 4.3.5–4.3.6): `N = 5000`; `σ_ε = 0.05`;
`B_max = 5`; ER `p = 0.01`; BA `m = 25`; modular = two BA communities of 2500
with `m = 21` plus 20,857 cross edges, giving `Q ≈ 0.333` and `⟨k⟩ = 50.0`;
integration horizon `T = 10`; 100 realizations per panel.

Which values of `ρ`, `p_s`, `x` and `ψ` a given analysis sweeps is stated in its
own README.

Threshold points are drawn from a single table so the manuscript letter and the
`(T_T, U_T)` pair cannot disagree:

| point | `(T_T, U_T)` | regime |
| --- | --- | --- |
| A | (−0.8, 0.8) | Accommodating |
| E | (0.2, 0.8) | Evaluative |
| F | (−0.8, 0.0) | Friction-averse |
| G | (0.0, 0.0) | Guarded — on the phase boundary; available but not used |
