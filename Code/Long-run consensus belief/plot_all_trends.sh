#!/bin/bash
# All five b* trend figures, from the merged per-realization files.
# Seconds on the login node; no job needed.
set -euo pipefail
module purge
module load python/ondemand-jupyter-python3.11
PY=$HOME/.conda/envs/myenv/bin/python
export OMP_NUM_THREADS=1

echo "=== realization counts ==="
for d in trends_*; do
    [[ -d "$d" ]] && echo "  $d: $(ls "$d"/realization_*.pkl 2>/dev/null | wc -l)/100"
done

echo
echo "=== Figure 3(a) ==="
"$PY" plot_trends.py --layout main
echo "=== Figures S3 / S4 ==="
"$PY" plot_trends.py --layout topologies   --regime E
"$PY" plot_trends.py --layout topologies   --regime F
echo "=== Figures S5 / S6 ==="
"$PY" plot_trends.py --layout polarization --regime E
"$PY" plot_trends.py --layout polarization --regime F
