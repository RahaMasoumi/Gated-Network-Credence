# Regimes of Influence under Trust–Distrust Gating

[![DOI](https://zenodo.org/badge/1341484743.svg)](https://doi.org/10.5281/zenodo.22063433)

Code and data for the paper:

> Razieh Masoumi, Ahana Biswas and Yu-Ru Lin,
> *"Regimes of Influence under Trust–Distrust Gating"* (2026).
> School of Computing and Information, University of Pittsburgh.
> Preprint: [arXiv](https://arxiv.org/abs/2606.24095)
---

## Contents

```
Gated-Network-Credence/
├── Code/                  simulation and analysis code, one folder per analysis
│   ├── phase-diagram/                                    Fig. 1(a)②, 2, S8
│   ├── Long-run consensus belief/                         Fig. 3(a), S3–S6
│   └── Spectral characterization of long run influence/   Fig. 4
├── Data/                  empirical networks and their preprocessing notebooks
├── LICENSE                MIT
└── README.md              this file
```

Two READMEs sit below this one and carry the detail:

| File | Covers |
| --- | --- |
| [`Code/README.md`](Code/README.md) | shared model conventions, figure → folder map, how to run each analysis |
| [`Data/README.md`](Data/README.md) | both empirical datasets: source, schema, provenance, reproducibility |

Each folder under `Code/` also has its own README with the exact commands,
SLURM recipes, runtimes and diagnostics for that analysis.

---

## Model

**Gated Network Credence** represents each directed relationship with a separate
trust assessment `tau_ij` and distrust assessment `delta_ij`, mapped to a
two-dimensional credence space:

```
T_ij = tau_ij - delta_ij          (net trust)
U_ij = tau_ij + delta_ij - 1      (trust–distrust conflict, "uncertainty")
```

Influence from source *j* to receiver *i* is admitted only when `T_ij >= T_T`
and `U_ij <= U_T`. The surviving ties form an effective influence graph whose
directed Laplacian governs the long-run belief state. The two thresholds define
four regimes:

| Regime | Label | `T_T` | `U_T` |
| --- | --- | --- | --- |
| Accommodating | **A** | permissive | permissive |
| Evaluative | **E** | selective | permissive |
| Friction-averse | **F** | permissive | selective |
| Guarded | **G** | selective | selective |

Representative threshold points used throughout the code:

| Point | `(T_T, U_T)` | Regime |
| --- | --- | --- |
| A | (−0.8, 0.8) | Accommodating |
| E | (0.2, 0.8) | Evaluative |
| F | (−0.8, 0.0) | Friction-averse |
| G | (0.0, 0.0) | Guarded (near phase boundary) |
| B, C, D | (0.8, 0.8), (−0.8, −0.8), (0.8, −0.8) | fragmented corners |

---

## Which figure comes from where

| Figure | Folder | Script |
| --- | --- | --- |
| Fig. 1(a)② — consensus/fragmentation phase diagram | `Code/phase-diagram/` | `fig01_phase_diagram.py` |
| Fig. 2 — `Δb*` across the threshold plane | `Code/phase-diagram/` | `fig02_delta_b.py` |
| Fig. S8 — multi-`ρ` `Δb*` panels | `Code/phase-diagram/` | `fig02_panels.py` |
| Fig. 3(a), S3–S6 — `b*` vs prestige bias | `Code/Long-run consensus belief/` | `run_trends.py`, `plot_trends.py` |
| Fig. 4 — spectral long-run influence `γ̄` | `Code/Spectral characterization of long run influence/` | `Spectral characterization.py` |
| Analytic activation thresholds overlaid on `Δb*` | `Code/phase-diagram/` | `plot_threshold_lines.py` |

<!-- TODO — not yet in this repository:
     Fig. 1(b–c), S2 : belief trajectories at p_s = 0 and p_s = 1
     Fig. 3(b)       : in-degree retention after filtering
     Fig. 5          : Bitcoin-OTC reputation network
     Fig. 6          : U.S. state-legislator follow network
     Add each as a new folder under Code/ following the same layout. -->

---

## Requirements

Python 3.10 or newer.

Core dependencies: `numpy`, `scipy`, `matplotlib`, `networkx`.
The `Data/` preprocessing notebooks additionally require `pandas` and `jupyter`.

<!-- TODO: commit a requirements.txt pinned to what was actually run:
     pip freeze | grep -iE 'numpy|scipy|matplotlib|networkx|pandas' > requirements.txt
     then replace the list above with:  pip install -r requirements.txt -->

---

## Getting started

Every analysis folder ships `gnc_core.py`, which self-tests when run directly.
Start there — it is the fastest way to confirm the environment is sound:

```bash
cd Code/phase-diagram
python gnc_core.py                  # ~20 s of self-tests
python fig01_phase_diagram.py --rho -0.4 --ps 0
```

Each folder's README gives the full set of commands, including SLURM recipes for
the analyses that need a cluster. Output goes to a subfolder created on first run.

All synthetic results are seeded (`BASE_SEED = 20260819`) and reproduce the
published panels.

---

## Parameters

| Symbol | Meaning | Values used |
| --- | --- | --- |
| `rho` | trust–distrust coupling strength | −1, −0.7, −0.4, −0.01 |
| `sigma` | Gaussian noise s.d. on distrust | 0.05 |
| `p_s` | prestige bias (trust–degree alignment) | 0.05, 0.10, 0.40, 0.60, 0.80, 1 |
| `x` | promoter prevalence (top-degree fraction seeded positive) | 0.05, 0.10, 0.15, 0.20 |
| `psi` | inter-community polarization (modular networks only) | 0, 0.4, 1 |
| `B_max` | initial-belief support, `b_i(0) ~ U(-B_max, B_max)` | 5 |
| `T_T`, `U_T` | net-trust and uncertainty thresholds | swept over [−1, 1] |

Network topologies, all `N = 5000`: Erdős–Rényi (`p = 0.01`); Barabási–Albert
(`m = 25`); modular (two communities of 2500, each BA with `m = 21`, plus random
cross-community edges giving modularity `Q ≈ 0.33`). Each undirected edge becomes
two independently evaluated directed relationships, so the effective influence
graph is generally asymmetric.

Synthetic results average over 100 independent realizations per parameter
combination; the same 100 networks and initial-belief configurations are reused
across parameter values. Error bars are ±1 s.d.

---

## Data

Full documentation — schema, provenance, and what is and is not exactly
reproducible — is in [`Data/README.md`](Data/README.md). In brief:

**Bitcoin-OTC reputation network** — 5,881 users, 35,592 observed directed rating
ties. Scores in [−10, 10] are rescaled to [0, 1] and used as the mean of the
edge-specific trust distribution; distrust is generated through the
trust–distrust coupling, since independent distrust is not observed here.

> S. Kumar, F. Spezzano, V. S. Subrahmanian and C. Faloutsos, "Edge weight
> prediction in weighted signed networks," *ICDM 2016*, pp. 221–230.
> https://doi.org/10.1109/ICDM.2016.0033

Source: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html

**U.S. state-legislator follow network** — 3,152 accounts, 107,785 directed
follow ties on X, from the political-elite dataset in Biswas et al. (2025) and
Biswas & Lin (2026). Trust is operationalized as structural affinity (cosine
similarity between node2vec embeddings of the full directed follower graph);
distrust as ideological divergence in audience composition. Initial beliefs
derive from Shor–McCarty ideology scores, sign reversed so the larger seeded
community is positively oriented.

Raw platform data are not redistributed here. Processed networks and derived
trust/distrust attributes are included where redistribution is permitted.

---

@unpublished{masoumi2026gated,
  title  = {Regimes of Influence under Trust--Distrust Gating},
  author = {Masoumi, Razieh and Biswas, Ahana and Lin, Yu-Ru},
  year   = {2026},
  note   = {Working paper}
}
```

## License

MIT License. See [https://github.com/RahaMasoumi/Gated-Network-Credence/blob/main/LICENSE](LICENSE).

## Contact

Please open a GitHub issue, or contact the corresponding author.
