# Regimes of Influence under Trust–Distrust Gating — simulation code


```
gnc_core.py                  the model: sampling, gate, A_IN, Laplacian, cabal count
fig01_phase_diagram.py       Figure 1(a)②  — consensus / fragmentation phase diagram
fig02_delta_b.py             Figure 2      — Δb* over the (T_T, U_T) plane, one rho
fig02_panels.py              Figure S8     — several \rho side by side, shared colorbar
verify_reproducibility.py    24 consistency checks
run_fig01.sbatch             SLURM job for Figure 1
run_fig02.sbatch             SLURM array for Figure 2
merge_and_verify.sh          merge, plot everything, verify
```

Requirements: `numpy`, `scipy`, `matplotlib`, and `networkx` (self-tests only).

---

## Before anything else

Two commands, about one minute, on the login node. Nothing should be submitted
until both print `all checks passed`.

```bash
python gnc_core.py                # 6 groups of self-tests, ~20 s
python verify_reproducibility.py  # 15 checks at reduced N, ~40 s
```

Then open both sbatch files and check the interpreter line against
`which python`:

```bash
PY=$HOME/.conda/envs/myenv/bin/python
```

If the conda environment has another name, every array task fails instantly
with "No such file or directory".

---

## Recipe A — Figure 1(a)②, the phase diagram

Consensus/fragmentation over the (T_T, U_T) plane at ρ = −0.4, p_s = 0. One
realization takes 9 s.

**On the cluster**

```bash
sbatch --export=ALL,RHO_ONLY=-0.4 --array=0 run_fig01.sbatch
```

**Without SLURM** 

```bash
python fig01_phase_diagram.py --rho -0.4 --ps 0
```

Either way the script runs all 100 realizations, Result:

```
phase_diagram_rho-0.4_ps0/phase_diagram_rho-0.4_ps0.pdf   <- the figure
phase_diagram_rho-0.4_ps0/consensus_fraction.npy          <- the underlying array
```

The colormap is the fraction of realizations reaching K = 1; the white curve is
its 0.5 level.

For other ρ, change `RHO_ONLY` or `--rho`. For all four at once:

```bash
sbatch run_fig01.sbatch           # array 0-3, one task per rho
```

---

## Recipe B — Figure 2, the Δb\* map for one ρ

Δb\* = b\*(p_s=1) − b\*(p_s=0). One realization takes 10–20 min, so this is one
SLURM task per realization: 100 tasks per ρ.

**Step 1 — compute.** For a single ρ, `RHO_ONLY` makes the task id the
realization id, so the array is always `0-99` and you never have to work out
which slice of the array a given ρ occupies:

```bash
sbatch --export=ALL,RHO_ONLY=-0.4 --array=0-99%60 run_fig02.sbatch
```

**Step 2 — wait**, then confirm all 100 files landed:

```bash
squeue -M htc -u $USER | grep gnc                          # empty when done
ls delta_b_results_rho-0.4/realization_*.pkl | wc -l       # want 100
```

**Step 3 — merge and plot.** Unlike Figure 1, the array tasks do *not* draw the
figure: each writes one `.pkl` and exits, This step is mandatory.

```bash
python fig02_delta_b.py --rho -0.4 --merge
```

Result:

```
delta_b_results_rho-0.4/delta_b_colormap_rho-0.4.pdf   <- the figure
delta_b_results_rho-0.4/delta_b_mean.npy               <- Δb*, NaN where fragmented
delta_b_results_rho-0.4/consensus_fraction.npy         <- fraction reaching consensus
delta_b_results_rho-0.4/n_consensus.npy                <- realizations per cell
```

Repeat with `RHO_ONLY=-0.7`, `-0.01`, `-1` for the other couplings, or submit
all four ρ in one go with `sbatch run_fig02.sbatch` (array 0-399, mapping in
the file header).

**Without SLURM**: `python fig02_delta_b.py --rho -0.4` runs all 100
realizations serially, merges and plots 

---

## Recipe C — Figure S8, the multi-panel Δb\* figure

This script draws only; it reads the `delta_b_mean.npy` that Recipe B produced.
So **Recipe B must have been completed for every ρ you want as a panel.**

```bash
python fig02_panels.py
```

Default panels are ρ = −0.01, −0.4, −0.7, −1, left to right, under one shared
colorbar fixed at ±3, with points A, E and F marked and ρ above each panel.
Output: `fig02_panels_ER.pdf`.

To choose the panels, the scale, or the spacing:

```bash
python fig02_panels.py --rhos -0.01 -0.4 -0.7 -1 \
                       --vmax 3 --wspace 0.08 --out fig02_panels_ER.pdf
python fig02_panels.py --vmax 0        # scale from the 90th percentile instead
```

The script prints, per panel, how many consensus cells it has and its actual
Δb\* range, and warns when the shared scale clips values:

```
rho=  -0.7  consensus cells  556  range [-3.26, +3.50]  38 cell(s) clipped by the shared scale
```

If you see that line, either raise `--vmax` or use `--vmax 0`, and say in the
caption which was used.

If a panel is missing the script stops and tells you exactly what to run:

```
missing delta_b_results_rho-0.7/delta_b_mean.npy
  run:  python fig02_delta_b.py --rho -0.7 --merge
```

---


---

## Everything at once

After both arrays have finished for all four ρ:

```bash
bash merge_and_verify.sh
```

This counts the realization files per ρ, merges and plots both figures for each
ρ, builds the multi-panel figure, and then runs all 24 checks. It is the
recommended final step even if you already merged by hand, because it catches a
partially failed array.

---

## Conventions (Methods 4.1–4.2)

Row = receiver, column = source:

    A_IN[i, j] = 1   <=>   A[i, j] = 1 and T_ij >= T_T and U_ij <= U_T

so `A_IN[i, j] = 1` means agent *j* can influence agent *i*. The Laplacian is
 `L = D_row − A_IN` of Eq. (3), 
`count_cabals` counts the source SCCs of that digraph, which is *K*, the number
of reaches. Consensus iff *K* = 1.

Parameters, all from Methods 4.3.5–4.3.6: N = 5000, p = 0.01, σ_ε = 0.05,
B_max = 5, x = 0.2, 41×41 threshold grid at spacing 0.05, 100 realizations,
integration horizon T_max = 10.

---

## Reproducibility

Realization *r* is `SeedSequence([BASE_SEED, r])`, spawned into four independent
streams for the network, the initial beliefs, the trust–distrust values, and
the prestige scores. Both figure scripts obtain realizations only through
`gnc_core.build_realization`, so realization *r* is bit-identical across scripts
and across parameter settings (Methods 4.3.6, "Replication").

Every `.pkl` carries the full parameter dict, and `merge()` refuses to combine
files whose parameters differ from the current settings, or to count a
realization id twice. A partially failed array cannot silently produce a figure.



## Two conventions to state in the captions

**Where the phase boundary sits.** `consensus_fraction.npy` is a fraction in
[0, 1] and the drawn boundary is its 0.5 level. Figure 2 masks a cell as
fragmentation when that fraction falls below `CONSENSUS_FRAC_MIN`
(`fig02_delta_b.py`, default 0.5), so both figures mark the same transition.

**Figure 2 needs consensus at both p_s.** Δb\* is defined only when the same
realization reaches consensus at p_s = 0 *and* at p_s = 1, 

---

## Re-styling without recomputing

These change only how the merged arrays are drawn, so editing them and
re-running the merge or the panel script is enough — the array jobs do not have
to be repeated.

- `COLOR_VMAX` in `fig02_delta_b.py`: `None` scales the colorbar to the 90th
  percentile of |Δb\*|; set `3.0` to lock the range at ±3.
- `CONSENSUS_FRAC_MIN` in `fig02_delta_b.py`: the consensus fraction below which
  a cell is reported as fragmentation. `n_consensus.npy` stores the per-cell
  counts, so any threshold can be applied afterwards.
- `--vmax`, `--wspace`, `--rhos` on `fig02_panels.py`.

---

## Troubleshooting

**Some tasks failed.** Find them, resubmit only those, merge again. Existing
`.pkl` files are untouched by a resubmission.

```bash
sacct -M htc -j <JOBID> --format=JobID,State,Elapsed,MaxRSS -X | grep -v COMPLETED
sbatch --export=ALL,RHO_ONLY=-0.4 --array=14,77,90 run_fig02.sbatch
python fig02_delta_b.py --rho -0.4 --merge
```


## Runtime and resources

Measured on one core at N = 5000 with the 41×41 grid:

| | per realization | per ρ | peak RSS |
|---|---|---|---|
| Figure 1 | 9 s | ~15 min (one task) | 165 MB |
| Figure 2 | 10–20 min | ~20 CPU-hours (100 tasks) | 175 MB |

`--mem=2g` is ample. The jobs are single-threaded, so the `*_NUM_THREADS=1`
exports in the sbatch files matter — without them BLAS oversubscribes the
single allocated core. On Pitt CRC htc nodes a wave of 60 Figure 2 tasks
finished in a few minutes, so the whole 400-task array is well under an hour of
wall time when the queue is free.
