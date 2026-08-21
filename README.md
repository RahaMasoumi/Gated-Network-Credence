# Regimes of Influence under Trust–Distrust Gating

<!-- Paste the Zenodo badge here after your first GitHub release. -->
[![DOI](https://zenodo.org/badge/DOI/ I should make a DOI wth Zenodo )

Code and data for the paper:

Razieh Masoumi, Ahana Biswas and Yu-Ru Lin, "Regimes of Influence under Trust–Distrust Gating" (2026).
School of Computing and Information, University of Pittsburgh.

## Model

**Gated Network Credence** represents each directed relationship with a separate trust assessment
`tau_ij` and distrust assessment `delta_ij`, mapped to a two-dimensional credence space:

```
T_ij = tau_ij - delta_ij          (net trust)
U_ij = tau_ij + delta_ij - 1      (trust–distrust conflict, "uncertainty")
```

Influence from source *j* to receiver *i* is admitted only when `T_ij >= T_T` and `U_ij <= U_T`.
The surviving ties form an effective influence graph whose directed Laplacian governs the long-run
belief state. The two thresholds define four regimes:

| Regime | Label | `T_T` | `U_T` |
| --- | --- | --- | --- |
| Accommodating | **A** | permissive | permissive |
| Evaluative | **E** | selective | permissive |
| Friction-averse | **F** | permissive | selective |
| Guarded | **G** | selective | selective |

## Contents

Each script is standalone and reproduces one figure from the paper. Run any of them directly; no
installation or package setup is needed beyond the dependencies below.

| Script | Figure |
| --- | --- |
| `fig01_phase_diagram.py` | Fig. 1(a)② — consensus/fragmentation phase diagram |
| `fig01bc_S2_trajectories` | Fig. 1(b–c), Fig. S2 — belief trajectories at `p_s = 0` and `p_s = 1` |
| `TODO` | Fig. 2 — `Δb*` across the threshold plane |
| `TODO` | Fig. 3(a) — `b*` vs prestige bias |
| `TODO` | Fig. 3(b) — in-degree retention after filtering |
| `TODO` | Fig. 4 — spectral long-run influence `γ̄` |
| `TODO` | Fig. 5 — Bitcoin-OTC reputation network |
| `TODO` | Fig. 6 — U.S. state-legislator follow network |

Parameters are set in the block at the top of each script. Output goes to a subfolder created on
first run.

## Requirements

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

<!-- Pin to what you actually ran: pip freeze | grep -iE 'numpy|scipy|matplotlib|networkx' -->
Core dependencies: `numpy`, `scipy`, `matplotlib`, `networkx`.
The legislator-network scripts additionally require `node2vec` (or `gensim`).

## Running

```bash
python fig01_phase_diagram.py
```

All scripts are seeded (`SEED = 42`) and reproduce the published panels as committed.


<!-- Time it once and fill the TODO in. Reviewers who see no output for a long
     stretch will assume the script has hung. -->

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

Representative threshold points:

| Point | `(T_T, U_T)` | Regime |
| --- | --- | --- |
| A | (−0.8, 0.8) | Accommodating |
| E | (0.2, 0.8) | Evaluative |
| F | (−0.8, 0.0) | Friction-averse |
| G | (0.0, 0.0) | Guarded (near phase boundary) |
| B, C, D | (0.8, 0.8), (−0.8, −0.8), (0.8, −0.8) | fragmented corners |

Network topologies, all `N = 5000`: Erdős–Rényi (`p = 0.01`); Barabási–Albert (`m = 25`); modular
(two communities of 2500, each BA with `m = 21`, plus random cross-community edges giving
modularity `Q ≈ 0.33`). Each undirected edge becomes two independently evaluated directed
relationships, so the effective influence graph is generally asymmetric.

Synthetic results average over 100 independent realizations per parameter combination; the same 100
networks and initial-belief configurations are reused across parameter values. Error bars are ±1 s.d.

## Data

**Bitcoin-OTC reputation network** — 5,881 users, 35,592 directed rating ties. Observed scores in
[−10, 10] are rescaled to [0, 1] and used as the mean of the edge-specific trust distribution;
distrust is generated through the trust–distrust coupling, since independent distrust is not
observed in this dataset.

> S. Kumar, F. Spezzano, V. S. Subrahmanian and C. Faloutsos, "Edge weight prediction in weighted
> signed networks," *ICDM 2016*, pp. 221–230. https://doi.org/10.1109/ICDM.2016.0033

Source: https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html <!-- verify this URL resolves -->

**U.S. state-legislator follow network** — 3,152 accounts, 107,785 directed follow ties on X, from
the political-elite dataset in Biswas et al. (2025) and Biswas & Lin (2026). Trust is
operationalized as structural affinity (cosine similarity between node2vec embeddings of the full
directed follower graph); distrust as ideological divergence in audience composition. Initial
beliefs derive from Shor–McCarty ideology scores, sign reversed so the larger seeded community is
positively oriented.

Raw platform data are not redistributed here. Processed networks and derived trust/distrust
attributes are included where redistribution is permitted.

<! -->

## Citation

```bibtex
@article{masoumi2026gated,
  title   = {Regimes of Influence under Trust--Distrust Gating},
  author  = {Masoumi, Razieh and Biswas, Ahana and Lin, Yu-Ru},
  journal = {TODO},
  year    = {2026},
  doi     = {TODO}
}
```

`CITATION.cff` is included; GitHub renders it as a "Cite this repository" button and Zenodo reads it
when minting the DOI.

## License

MIT License. See [LICENSE](LICENSE). <!-- Zenodo requires a LICENSE file before archiving. -->

## Contact

Please open a GitHub issue, or contact the corresponding author.

