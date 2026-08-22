# # Gated Network Credence— Empirical networks

Two empirical signed networks prepared for opinion-dynamics simulations in which every
directed link carries a trust and a distrust weight. Each dataset is documented
separately below: the source data it comes from, the preprocessing applied to it, and
the properties a downstream analysis can rely on.

| Dataset | 
| --- | 
| A — U.S. State Legislators' Follow Network |
| B — Bitcoin OTC Signed Trust Network | 

---

## Dataset A — U.S. State Legislators' Follow Network — `Legislators_data_processed`

A directed, attribute-annotated edge list of a two-party (Democratic / Republican)
communication network in which every edge carries a **trust** and a **distrust**
weight, and every node carries a **party label** and a scalar **belief** value.

The released file carries the empirical network in full: every observed edge and every
trust, distrust, party, and belief value is preserved exactly as recorded. Two
transparent preprocessing steps sit on top of it — node IDs are relabeled so that the
two parties occupy contiguous index blocks, and 121 edges (0.11% of the edge list) are
supplied to nodes with no recorded outgoing links so that the network is well posed
for out-Laplacian dynamics. Both steps are documented in Section 3, reproducible from
the notebook in this repository, and individually traceable: the supplied edges are the
final 121 rows of the file, so the network exactly as observed is recovered by dropping
them.

---

## 1. Files

### Data

| File | Rows (edges) | Description |
| --- | --- | --- |
| `dataset_all_edges_coded_RD.csv` | 107,664 | **Source data.** The empirical edge list, original node IDs. |
| `Legislators_data_processed.csv` | 107,785 | **Analysis dataset.** Same network with party-blocked node IDs, plus the 121 supplied edges (Section 3). |

### Code

| Notebook | Purpose |
| --- | --- |
| `legislator-trust-network-preprocessing.ipynb` | Builds the analysis dataset from the source data, end to end, with validation. |

Running the notebook also writes three provenance files that are not shipped, since
they are byproducts rather than data: `dataset_all_edges_coded_RD_reindexed.csv` (the
Step 1 output), `node_reindex_map_full.csv` (old → new node ID map, needed to join back
to original IDs), and `added_edges.csv` (the 121 supplied edges with the
diagnostics behind each choice).

---

## 2. Schema

`Legislators_data_processed.csv` — one row per directed edge, no header ambiguity,
no missing values.

| Column | Type | Description |
| --- | --- | --- |
| `source` | int | Sender node ID, `0`–`3151`. |
| `target` | int | Receiver node ID, `0`–`3151`. |
| `source_party` | str | `Democratic` or `Republican`. |
| `target_party` | str | `Democratic` or `Republican`. |
| `source_belief` | float | Sender's belief score, range `[-3.048, 3.857]`. |
| `target_belief` | float | Receiver's belief score, same scale. |
| `trust` | float | Edge trust weight, range `[0.066, 0.991]`. |
| `distrust` | float | Edge distrust weight, range `[0.000, 0.971]`. |

**Graph summary.** 3,152 nodes (1,661 Democratic, 1,491 Republican); 107,785 directed
edges; no self-loops; no duplicated `(source, target)` pairs; weakly connected.

---

## 3. Provenance

```
dataset_all_edges_coded_RD.csv          107,664 edges / 3,152 nodes
        │
        │  Step 1 — party-blocked node reindexing (bijective relabeling)
        ▼
dataset_all_edges_coded_RD_reindexed.csv 107,664 edges  (graph unchanged)
        │
        │  Step 2 — sink resolution: +121 outgoing edges from out-degree-0 nodes
        ▼
Legislators_data_processed.csv             107,785 edges
```

### Step 1 — Node reindexing

Node IDs are relabeled so that the two parties occupy contiguous index blocks:

* Democratic nodes → `0 … 1660`
* Republican nodes → `1661 … 3151`

Within each block, nodes are ordered by ascending original ID. Column order is also
normalized to `source, target, source_party, target_party, source_belief,
target_belief, trust, distrust`.

### Step 2 — Sink resolution 

**The problem.** In the empirical network, 121 nodes have out-degree 0. In the
condensation (SCC-DAG) these appear as 121 singleton *sink* components. For
out-Laplacian dynamics of the form, every sink component is an
absorbing terminal set, so the system has 121 independent invariant subspaces and
global consensus is structurally impossible — regardless of the trust/distrust values.

**The intervention.** For each sink node `u`, exactly **one** outgoing edge `u → v`
is added into the dominant SCC (the *anchor*), chosen as follows:

1. **Target selection.** `v` is the anchor node of the *same party* as `u` whose
   belief is closest to `u`'s. (A closest-belief-overall fallback exists in the code
   but was never triggered: a same-party candidate was found for all 121 cases.
   Realized `|belief(u) − belief(v)| ≤ 0.007`.)
2. **Weight assignment.** `(trust, distrust)` is set to the **median of the
   corresponding party-pair distribution in the source data**, so added edges do not
   shift the marginal weight distributions:
   * D→D: `trust = 0.816818`, `distrust = 0.039430`
   * R→R: `trust = 0.881276`, `distrust = 0.055391`
3. **Admissibility filter.** Weights must satisfy `trust − distrust ≥ TT`,
   `trust + distrust − 1 ≤ UT`, and `0 ≤ trust, distrust ≤ 1`, with `TT = 0.2`,
   `UT = 0.8`. Where medians fail the filter, `nearest_feasible_trust_distrust()`
   applies a minimal L2 correction. **All 121 median pairs passed unmodified**, so no
   correction was applied in this release.

**Magnitude.** 121 of 107,785 edges = **0.112%**. Every added edge is listed in
`added_edges.csv` with its source, target, both beliefs, the party-pair
medians used, whether the medians passed the filter, the final weights, and the
selection rule applied. The unmodified network is recoverable exactly by dropping
these 121 rows (they are the final 121 rows of the file, and the first 107,664 rows
are byte-identical to the Step 1 output).

---

## 4. Reproducing the artifact

Both steps run end to end from the source file in a single notebook:

```bash
jupyter nbconvert --execute --inplace \
  legislator-trust-network-preprocessing.ipynb
```

Inputs: `dataset_all_edges_coded_RD.csv`.
Outputs: `dataset_all_edges_coded_RD_reindexed.csv`, `node_reindex_map_full.csv`,
`Legislators_data_processed.csv`, `added_edges.csv`.

Requires `pandas`, `numpy`, `networkx`. The pipeline is **fully deterministic** — no
random seed is involved: the anchor is the unique largest SCC, target selection is an
argmin over beliefs with ties broken on the lower node id, and components are
iterated in sorted order.

---

## Dataset B — Bitcoin OTC Signed Trust Network — `Bitcoin-OTC-trust_threshold-with-both-community-labels-scaled`

A directed signed edge list of the Bitcoin OTC marketplace rating network, with a
minimum out-degree floor, a `[0, 1]`-scaled sign, and two community labelings attached
to both endpoints of every edge.

All 35,592 observed ratings are preserved verbatim. The remaining 41,022 edges are
neutral placeholders added so that every node has out-degree at least 10, and are
identifiable as exactly the rows with `Sign = 0`, which does not occur in the source
data.

---

## 5. Files

| File | Rows (edges) | Description |
| --- | --- | --- |
| `Bitcoin-OTC-trust.csv` | 35,592 | **Source data.** Observed signed ratings, columns `From, To, Sign`. |
| `Bitcoin-OTC-trust_threshold-with-both-community-labels-scaled.csv` | 76,614 | **Analysis dataset.** |

Notebooks implementing the steps: `degree-distribution.ipynb` (out-degree
augmentation) and `connected_components.ipynb` (component filter, min–max scaling).

---

## 6. Schema

| Column | Type | Description |
| --- | --- | --- |
| `source`, `target` | int | Node IDs, `1`–`6005` (5,881 distinct, non-contiguous). Named `From`, `To` in the source data. |
| `Sign` | int | Rating in `[-10, 10]`. `0` marks an added neutral edge. |
| `source_com_label`, `target_com_label` | int | First community labeling, `{0, 1}`, sizes 4,613 / 1,268. |
| `Sign_scaled` | float | `Sign` min–max scaled to `[0, 1]`, equal to `(Sign + 10) / 20`, so `0.5` is neutral. |
| `source_com_label_new`, `target_com_label_new` | int | Second community labeling, `{0, 1}`, sizes 2,566 / 3,315. |

Both labelings are node-consistent: every node carries one label across all rows in
which it appears, as source or as target. They agree on 73% of nodes after matching
group orientation.

`Sign_scaled` is the raw scaled rating, not the model's trust variable. Downstream work
reshapes it (beta-spread of trust, distrust generated as
`δ = −ρ(τ − 0.5) + ε + 0.5`) outside this file.

**Graph summary.** 5,881 nodes; 76,614 directed edges (35,592 observed, 41,022
neutral); no self-loops; no duplicated `(source, target)` pairs; minimum out-degree 10;
strongly connected; reciprocity 0.37.

---

## 7. Provenance

```
Bitcoin-OTC-trust.csv                    35,592 edges / 5,881 nodes
        │
        │  Step 1 — out-degree augmentation: +41,022 neutral edges (Sign = 0)
        ▼
Bitcoin-OTC-trust_threshold.csv           76,614 edges
        │
        │  Step 2 — largest weakly connected component (no-op here)
        │  Step 3 — min–max scaling of Sign into [0, 1]
        │  Step 4 — join two community labelings onto both endpoints
        ▼
…-with-both-community-labels-scaled.csv   76,614 edges
```

### Step 1 — Out-degree augmentation

`augment_network(df, threshold=10)` in `degree-distribution.ipynb`. For each node with
out-degree below 10, neutral edges (`Sign = 0`) are drawn to uniformly random
non-neighbours, excluding self-loops and duplicates, until the node reaches out-degree
10. The direction is always outgoing from the deficient node; in-degree is never
targeted. Nodes already at or above 10 are untouched, so this is a floor, not a quota.

**Why.** In the observed network 1,067 of 5,881 nodes have out-degree 0, and the
condensation has 1,144 strongly connected components of which 1,082 are sinks. Under
the trust/distrust gate — an edge survives only if `T = trust − distrust ≥ T_T` and
`U = trust + distrust − 1 ≤ U_T` — the loose setting (`T_T = −0.8`, `U_T = 0`) leaves
roughly a third of nodes isolated and the dynamics never converge. The neutral value
assigned to the added links is `(trust, distrust) = (0.5, 0.5)`, giving `T = 0` and
`U = 0`, so they clear that gate while asserting neither trust nor distrust. Under the
loose gate they cut sink components from 970 to 9; under the strict gate
(`T_T = 0.2`) they all fail and change nothing.

**Choice of 10.** Not justified in the code. Connectivity alone is reached earlier: a
threshold of 5 already gives a single strongly connected component in most draws
(16,239 added edges), and 2 cuts the component count from 1,144 to about 12. Ten adds
roughly 2.5x the edges connectivity requires.

### Step 2 — Component filter

The largest weakly connected component is retained (`connected_components.ipynb`).
After Step 1 the network is already connected, so nothing is removed and the
accompanying `*-removed_nodes.csv` and `*-removed_edges.csv` are empty.

### Step 3 — Scaling

`MinMaxScaler` on `Sign` over the observed range `[-10, 10]`, verified to equal
`(Sign + 10) / 20` to floating-point precision.

### Step 4 — Community labels

Two independent two-community partitions of the same 76,614-edge graph, one from Gephi
(resolution 1.4, modularity 0.467) and one from networkx, joined onto both endpoints.
The `*_new` columns carry the 2,566 / 3,315 split used in the simulations. Both are
structurally meaningful: modularity 0.188 and 0.191 respectively, against 0.001 for a
size-matched random split.

---

## 8. Reproducing the artifact

Unlike Dataset A, this file is **not reproducible edge for edge.** `augment_network`
draws targets with `np.random.choice` and no seed, so each run yields a different set of
neutral edges: the `Bitcoin-OTC-trust_threshold.csv` in the folder is a different run of
the same procedure, sharing only 59 of its 41,022 neutral edges with the analysis
dataset. Re-running reproduces the construction and all summary statistics, not the
draw. The two community labelings likewise come from external Gephi and networkx runs.

Treat `Bitcoin-OTC-trust_threshold-with-both-community-labels-scaled.csv` as the fixed
artifact, and recover the observed network with `Sign != 0` when a result should not
depend on the placeholders — 54% of the edges are neutral.

Even after augmentation, 9 sink components survive the loose gate, so a sink-resolution
step of the kind used for Dataset A is still needed before out-Laplacian dynamics can
reach a single consensus.
