# U.S. State Legislators’ Follow Network — `Legislators_data_processed`

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
they are byproducts rather than data: `dataset_all_edges_coded_RD_2_reindexed.csv` (the
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

Node attributes are stored redundantly on every incident edge and are internally
consistent (each node ID maps to exactly one party and one belief value).

**Graph summary.** 3,152 nodes (1,661 Democratic, 1,491 Republican); 107,785 directed
edges; no self-loops; no duplicated `(source, target)` pairs; weakly connected.
Edge composition: 52,522 D→D, 31,532 R→R, 11,182 D→R, 12,428 R→D (source-data counts).

---

## 3. Provenance

```
dataset_all_edges_coded_RD_2.csv          107,664 edges / 3,152 nodes
        │
        │  Step 1 — party-blocked node reindexing (bijective relabeling)
        ▼
dataset_all_edges_coded_RD_2_reindexed.csv 107,664 edges  (graph unchanged)
        │
        │  Step 2 — sink resolution: +121 outgoing edges from out-degree-0 nodes
        ▼
dataset_with_un_sink_links.csv             107,785 edges
```

### Step 1 — Node reindexing

Node IDs are relabeled so that the two parties occupy contiguous index blocks:

* Democratic nodes → `0 … 1660`
* Republican nodes → `1661 … 3151`

Within each block, nodes are ordered by ascending original ID. Column order is also
normalized to `source, target, source_party, target_party, source_belief,
target_belief, trust, distrust`.

**Rationale.** Contiguous party blocks make the adjacency and Laplacian matrices
block-structured, so within-party and cross-party blocks can be sliced directly
without an auxiliary index lookup.

**Guarantee.** This step is a pure relabeling. Verified by reconstructing the map
and re-applying it to the source file: the resulting edge set and all attribute
values are identical, row for row, to `dataset_all_edges_coded_RD_2_reindexed.csv`.
No edge, node, or weight is added, removed, or modified.

### Step 2 — Sink resolution (the only substantive modification)

**The problem.** In the empirical network, 121 nodes have out-degree 0. In the
condensation (SCC-DAG) these appear as 121 singleton *sink* components, alongside one
dominant strongly connected component of 2,922 nodes; 230 SCCs in total. For
out-Laplacian dynamics of the form `ẋ = −L_out x`, every sink component is an
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
4. **Safeguards.** Self-loops and duplicate edges are skipped by construction.

**Effect.** Sink SCCs: **121 → 1**. Dominant SCC: 2,922 → 3,043 nodes. Total SCCs:
230 → 109 (the remainder are non-terminal source/intermediate components, which are
harmless for the dynamics). Added edges: 55 D→D and 66 R→R — no cross-party edge was
introduced, so cross-party statistics are untouched.

**Magnitude.** 121 of 107,785 edges = **0.112%**. Every added edge is listed in
`un_sink_edges_added.csv` with its source, target, both beliefs, the party-pair
medians used, whether the medians passed the filter, the final weights, and the
selection rule applied. The unmodified network is recoverable exactly by dropping
these 121 rows (they are the final 121 rows of the file, and the first 107,664 rows
are byte-identical to the Step 1 output).

---

## 4. Reproducing the artifact

Both steps run end to end from the source file in a single notebook:

```bash
jupyter nbconvert --execute --inplace \
  legislator-trust-network-preprocessing-and-sink-resolution.ipynb
```

Inputs: `dataset_all_edges_coded_RD_2.csv`.
Outputs: `dataset_all_edges_coded_RD_2_reindexed.csv`, `node_reindex_map_full.csv`,
`dataset_with_un_sink_links.csv`, `un_sink_edges_added.csv`.

Requires `pandas`, `numpy`, `networkx`. The pipeline is **fully deterministic** — no
random seed is involved: the anchor is the unique largest SCC, target selection is an
argmin over beliefs with ties broken on the lower node id, and components are
iterated in sorted order.

Row order in the regenerated file is canonicalised to `(source, target)`. The shipped
`dataset_with_un_sink_links.csv` carries a different row order (same-party edges first,
then cross-party); this is a permutation of rows with no graph meaning, so the two
should be compared as multisets, which is what the notebook's final cell does.

### Validation checks performed

These run as assertions inside the pipeline notebook, so a replaced upstream file
fails loudly rather than silently.

* Step 1 relabeling is bijective; edge set and attributes invariant under the map.
* First 107,664 rows of the release are identical to the Step 1 output.
* Added-edge count (121) matches the sink-SCC count and the row count of
  `un_sink_edges_added.csv`.
* Post-intervention graph: 1 sink SCC, no self-loops, no duplicate edges, no NaNs.
* Node → (party, belief) mapping is single-valued across all 107,785 rows.

---

## 5. Intended use and limitations

**Appropriate uses.** Consensus and opinion dynamics on signed/weighted directed
graphs, spectral analysis of out-Laplacians, party-block community analysis, and
comparisons of trust versus distrust structure across the four directional
party pairs.

**Please note.**

* The 121 added edges are a **modeling intervention, not observed data.** Any
  reported result that depends on the local out-neighborhood of those 121 nodes
  should be checked against the source data. A robustness appendix
  reporting both is the recommended practice.
* The intervention is deliberately *minimal and conservative* (one edge per sink,
  same party, nearest belief, median weights). It is nevertheless a **structural
  assumption**: it presumes that an observed absence of outgoing links reflects
  incomplete observation rather than genuine one-way behavior. Where that assumption
  is untenable for a research question, the appropriate baseline is the source data.
* Because added edges are same-party and median-weighted, they leave cross-party edge
  counts and the party-pair weight marginals essentially unchanged — but they do
  change SCC structure, reachability, and the spectrum of `L_out` by design. That is
  the point of the step, and it should be stated explicitly wherever the dataset is
  used.
* `trust` and `distrust` are independent weights, not complements; `trust + distrust`
  is not constrained to 1.
* Belief values are on an unnormalized empirical scale (`[-3.048, 3.857]`) and are
  not centered or standardized in this release.
