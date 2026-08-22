# Gated Network Credence — b\* versus prestige bias (Figure 3(a) and S3–S6)

Simulation code for the long-run collective belief b\* as a function of prestige
bias p_s, across trust–distrust coupling ρ, promoter prevalence x, three
synthetic topologies, and three levels of inter-community polarization ψ.

One model definition, one generator, one plotter. Every function in
`gnc_core.py` names the manuscript section or equation it implements.

```
gnc_core.py             the model: topologies, Eq. (5), prestige assignment,
                        seeding, polarization, A_IN, Laplacian, reach count
run_trends.py           the (rho, x, p_s) sweep — one script for all topologies
plot_trends.py          all five figures — one script, three layouts
consensus_report.py     diagnostics: where the filtered graph has one reach
run_trends.sbatch       SLURM array covering every panel
plot_all_trends.sh      produces all five figures at once
```

Requirements: `numpy`, `scipy`, `networkx`, `matplotlib`.

---

## Quick start

```bash
python gnc_core.py                                                # self-tests, ~20 s
python run_trends.py --topology ER --point E --realization_id 0    # ~25 s
sbatch run_trends.sbatch                                          # 10 tasks
python consensus_report.py                                        # diagnostics
bash plot_all_trends.sh                                           # the five figures
```

---

## Conventions (Methods 4.1–4.2)

Row = receiver, column = source:

    A_IN[i, j] = 1   <=>   A[i, j] = 1 and T_ij >= T_T and U_ij <= U_T

so `A_IN[i, j] = 1` means agent *j* can influence agent *i*. The dynamics use
the unnormalised directed Laplacian `L = D_row − A_IN` of Eq. (3), with
`k_row_i` the number of admitted sources of receiver *i*. The influence digraph
has an arc *j → i* whenever `A_IN[i, j] = 1`, so its arc-adjacency matrix is
`A_IN.T`; `count_cabals` counts its source SCCs, which is *K*, the number of
reaches. Consensus iff *K* = 1.

### How the networks are built, and where direction enters

Each topology is first generated as an **undirected** graph (Methods 4.3.6):

| topology | construction |
|---|---|
| ER | independent Bernoulli(p) on every unordered pair, p = 0.01 |
| BA | preferential attachment, m = 25 |
| Modular | two independent BA graphs of 2500 nodes each, m = 21, plus 20857 distinct random cross-community edges → Q = 0.333, ⟨k⟩ = 50.0 |

Direction is then introduced by representing every undirected edge {i, j} as the
two potential directed relationships (i, j) and (j, i), stored receiver-major so
that `src[indptr[i]:indptr[i+1]]` is N(i), the set of potential sources of i.
The two directions receive independently sampled (τ, δ), the prestige
permutation acts within each receiver's block, and the gate is applied to each
direction separately. The potential graph is bidirectional but `A_IN` is not:
on ER roughly two thirds of admitted relationships are one-way, which is what
makes the Laplacian non-Hermitian and the reach/cabal structure meaningful.

### Long-run belief

`A_IN` depends on (ρ, p_s) only, and x enters solely through b(0), so the
propagator is built once per (ρ, p_s) and applied to every x:

    mean(b(T)) = [exp(-L^T T) 1] . b(0) / N

---

## Design: the curves are paired comparisons

Per realization the network, the initial beliefs, τ, the noise ε and the
prestige scores η are each drawn once, from independent sub-streams of
`SeedSequence([BASE_SEED, realization_id])`. Consequently:

- moving along **p_s** changes only the prestige weighting, not the trust
  values or the random scores;
- moving along **x** changes only b(0), through the seeding rule;
- moving along **ρ** reuses the same τ and ε wherever the rejection step
  accepts them.

---

## Which figure needs which command

### Data

Each SLURM task is one (topology, point, ψ) combination running all 100
realizations. Task ids:

| task | | task | |
|---|---|---|---|
| 0 | ER, E | 5 | MOD, E, ψ=0.4 |
| 1 | ER, F | 6 | MOD, E, ψ=1 |
| 2 | BA, E | 7 | MOD, F, ψ=0 |
| 3 | BA, F | 8 | MOD, F, ψ=0.4 |
| 4 | MOD, E, ψ=0 | 9 | MOD, F, ψ=1 |

```bash
sbatch run_trends.sbatch                    # everything
sbatch --array=2 run_trends.sbatch          # one task, e.g. after a failure
```

Without SLURM, drop `--realization_id` and the script runs all 100 serially:

```bash
python run_trends.py --topology MOD --point F --psi 1.0
```

Results land in `trends_<topology>_..._point<X>/`, one `realization_<id>.pkl`
per realization plus `params.json`.

### Figures

```bash
python plot_trends.py --layout main                      # Figure 3(a)
python plot_trends.py --layout topologies   --regime E   # Figure S3
python plot_trends.py --layout topologies   --regime F   # Figure S4
python plot_trends.py --layout polarization --regime E   # Figure S5
python plot_trends.py --layout polarization --regime F   # Figure S6
```

or all five with `bash plot_all_trends.sh`. Output goes to `trend_figures/`.

In every layout the top row fixes ρ = −0.4 and varies x, the bottom row fixes
x = 0.2 and varies ρ; colour encodes x and line style encodes ρ. `--layout main`
instead puts the two regimes on the two rows, as in Figure 3(a). Change the held
values with `--rho-fixed` and `--x-fixed`.

---

## Points of the phase diagram

`--point` takes the manuscript letter and the threshold pair is derived from a
single table, so the two cannot disagree:

| point | (T_T, U_T) | regime |
|---|---|---|
| A | (−0.8, 0.8) | Accommodating |
| **E** | **(0.2, 0.8)** | **Evaluative** |
| **F** | **(−0.8, 0.0)** | **Friction-averse** |
| G | (0.0, 0.0) | Guarded |

E and F are the two reported in Figure 3(a) and S3–S6. Point G sits on the
phase boundary: there the filtered graph has hundreds of reaches and the
"consensus belief" framing does not apply, so it is available but not used.

---

## Consensus diagnostics

Figure 3(a) and S3–S6 are captioned "long-run consensus belief state b\*", which
is a claim about *K*. `run_trends.py` records *K* for every (realization, ρ, p_s)
cell and `consensus_report.py` reports it, so the claim is checked rather than
assumed.

Measured over 100 realizations per panel:

| panel | cells with K = 1 |
|---|---|
| ER, point E | 100.00% |
| ER, point F | 100.00% |
| BA, point F | 100.00% |
| Modular, points E and F, ψ = 0, 0.4, 1 | 100.00% |
| **BA, point E** | **98.25%** |

Nine of the ten panels are exact: every plotted value is a consensus value.

**The one exception is benign.** On BA at the Evaluative point, 1.75% of cells
have K > 1, and `K ranges 2 to 3, median 2` — one reach containing almost every
agent, plus one or two isolated single nodes. At N = 5000 a single detached node
raises *K* from 1 to 2 while contributing 1/5000 of the network average, which
is why it barely moves the number. Averaging only over the cells with a single
reach shifts b\* by **at most 0.0010**, against a ±1 SD error bar of **0.0381** —
about one fortieth of the statistical uncertainty, and invisible on the figure.
Dropping the affected realizations entirely gives the same answer to three
decimal places. No exclusion is therefore applied: the figures average over all
100 realizations.

The pattern is informative rather than a nuisance:

```
fraction of realizations with K > 1   (rows = rho, columns = p_s)
          0.05     0.1     0.4     0.6     0.8       1
    -1    0.00    0.00    0.00    0.00    0.00    0.00
  -0.7    0.00    0.00    0.00    0.01    0.01    0.01
  -0.4    0.00    0.00    0.00    0.01    0.01    0.01
 -0.01    0.01    0.01    0.04    0.10    0.10    0.10
```


To reproduce the numbers:

```bash
python consensus_report.py                                # all panels
python consensus_report.py --dir trends_BA_N5000_m25_pointE
```

`plot_trends.py` also prints the K = 1 fraction per panel and warns if it is
below 1, so a figure cannot be produced from fragmented data without saying so.

### A note on `settled`

`settled` reports whether mean(b(T)) and mean(b(2T)) agree to within 1e-6, i.e.
whether the integration horizon was long enough. It is 100% everywhere except
the three modular Evaluative panels, where it is 99.4–99.5%: a few cells have a
small spectral gap and are still approaching their limit at T = 10. Since K = 1
there the limit exists and is merely reached more slowly; raising `T_HORIZON` to
20 removes the flag and moves b\* by the same ~0.001 order.

---

## Parameters (Methods 4.3.5–4.3.6)

N = 5000; σ_ε = 0.05; B_max = 5; ρ ∈ {−1, −0.7, −0.4, −0.01};
p_s ∈ {0.05, 0.10, 0.40, 0.60, 0.80, 1}; x ∈ {0.05, 0.10, 0.15, 0.20};
ψ ∈ {0, 0.4, 1} for the modular network; 100 realizations per panel;
integration horizon T = 10.

Initial beliefs are i.i.d. U(−B_max, B_max), as stated in Methods 4.3.5. This
matters for the error bars: forcing exactly half the agents negative would halve
the standard deviation of mean b(0) — a factor 1.99, measured — and the bars are
±1 SD across realizations.

---

## Reproducibility

Realization *r* is `SeedSequence([BASE_SEED, r])` with `BASE_SEED = 20260819`,
spawned into four independent streams. Every `.pkl` carries the full parameter
dict; `plot_trends.py` refuses to combine files whose parameters differ, or to
count a realization id twice, so a partially failed array cannot silently
produce a figure.

Runtime on one core at N = 5000: ~25 s per realization for ER, so a task is
about 40 minutes; BA is denser and takes roughly twice as long. Peak memory is
well under 2 GB. The jobs are single-threaded — the `*_NUM_THREADS=1` exports in
the sbatch file matter, since otherwise BLAS oversubscribes the single allocated
core.
