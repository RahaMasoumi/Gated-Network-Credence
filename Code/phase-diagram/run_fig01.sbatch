#!/bin/bash
#SBATCH --job-name=gnc_fig1
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=01:00:00
#SBATCH --array=0-3                    # one task per rho value
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=RAM587@pitt.edu
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# Figure 1(a)(2): consensus / fragmentation phase diagram.
#
# One realization takes about 9 s at N = 5000 on a 41 x 41 grid, so all 100
# realizations of one rho run inside a single task (~15 min) and no realization
# array is needed.  Task id -> rho:  0 -> -0.01, 1 -> -0.4, 2 -> -0.7, 3 -> -1.
# The manuscript figure is rho = -0.4; the other two are for the appendix and
# for the cross-check against Figure 2 at the same rho.
#
# Each task merges and plots its own rho at the end, so nothing else is needed.
# Measured peak RSS: ~165 MB.
# ---------------------------------------------------------------------------
set -euo pipefail

RHO_LIST=(-0.01 -0.4 -0.7 -1)
if [[ -n "${RHO_ONLY:-}" ]]; then
    RHO=${RHO_ONLY}          # sbatch --export=ALL,RHO_ONLY=-0.7 --array=0 run_fig01.sbatch
else
    RHO=${RHO_LIST[${SLURM_ARRAY_TASK_ID}]}
fi
PS=0                                   # Figure 1 is the p_s = 0 condition

mkdir -p logs
module purge
module load python/ondemand-jupyter-python3.11
PY=$HOME/.conda/envs/myenv/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
echo "task=${SLURM_ARRAY_TASK_ID}  rho=${RHO}  ps=${PS}  host=$(hostname)"
echo "start $(date)"

"$PY" fig01_phase_diagram.py --rho "${RHO}" --ps "${PS}"

echo "end   $(date)"
