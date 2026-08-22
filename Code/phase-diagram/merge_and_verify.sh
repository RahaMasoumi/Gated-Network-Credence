#!/bin/bash
# ---------------------------------------------------------------------------
# Run on the login node after the arrays finish.  Merging is a few seconds per
# rho, so no job is needed.  Figure 1 already merged itself inside its own
# array task; this re-merges it so that a partially failed array is caught.
#
#   bash merge_and_verify.sh
# ---------------------------------------------------------------------------
set -euo pipefail

module purge
module load python/ondemand-jupyter-python3.11
PY=$HOME/.conda/envs/myenv/bin/python

export OMP_NUM_THREADS=1

echo "=== expected number of realization files per rho ==="
for r in -0.01 -0.4 -0.7 -1; do
    n1=$(ls "phase_diagram_rho${r}_ps0"/realization_*.pkl 2>/dev/null | wc -l)
    n2=$(ls "delta_b_results_rho${r}"/realization_*.pkl 2>/dev/null | wc -l)
    echo "  rho=${r}:  fig1 ${n1}/100   fig2 ${n2}/100"
done

echo
echo "=== merge and plot ==="
for r in -0.01 -0.4 -0.7 -1; do
    echo "--- rho = ${r} ---"
    "$PY" fig01_phase_diagram.py --rho "${r}" --ps 0 --merge
    "$PY" fig02_delta_b.py      --rho "${r}" --merge
done

echo
echo "=== model-level checks ==="
"$PY" verify_reproducibility.py

echo
echo "=== multi-panel figure ==="
"$PY" fig02_panels.py

echo
echo "=== output-level checks ==="
for r in -0.01 -0.4 -0.7 -1; do
    echo "--- rho = ${r} ---"
    "$PY" verify_reproducibility.py --check-outputs \
        --fig1-dir "phase_diagram_rho${r}_ps0" \
        --fig2-dir "delta_b_results_rho${r}"
done
