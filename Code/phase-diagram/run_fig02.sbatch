#!/bin/bash
#SBATCH --job-name=gnc_fig2
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=02:00:00
#SBATCH --array=0-399%60               # 4 rho x 100 realizations, 60 at a time
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=RAM587@pitt.edu
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# Figure 2: Delta b* = b*(p_s = 1) - b*(p_s = 0) over the (T_T, U_T) plane.
#
# One realization takes 10-20 min at N = 5000 on a 41 x 41 grid (dominated by
# expm_multiply on the consensus cells), so one task = one realization.
#
# Task id -> (rho, realization), when the whole set is submitted:
#     0-99    -> rho = -0.01
#     100-199 -> rho = -0.4     <- the manuscript figure
#     200-299 -> rho = -0.7
#     300-399 -> rho = -1
#
# To run a SINGLE rho without having to work out which slice of the array it
# occupies, set RHO_ONLY and use array 0-99; the task id is then the
# realization id directly:
#
#     sbatch --export=ALL,RHO_ONLY=-0.7 --array=0-99%60 run_fig02.sbatch
#
# Measured peak RSS: ~175 MB.  The %60 throttle keeps the array from filling
# the queue; remove it if the cluster is idle.
#
# After the array finishes, run merge_and_verify.sh on the login node.
# ---------------------------------------------------------------------------
set -euo pipefail

RHO_LIST=(-0.01 -0.4 -0.7 -1)
NREAL=100

if [[ -n "${RHO_ONLY:-}" ]]; then
    # single-rho mode: task id == realization id
    RHO=${RHO_ONLY}
    REALIZATION=${SLURM_ARRAY_TASK_ID}
else
    RHO=${RHO_LIST[$((SLURM_ARRAY_TASK_ID / NREAL))]}
    REALIZATION=$((SLURM_ARRAY_TASK_ID % NREAL))
fi

mkdir -p logs
module purge
module load python/ondemand-jupyter-python3.11
PY=$HOME/.conda/envs/myenv/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
echo "task=${SLURM_ARRAY_TASK_ID}  rho=${RHO}  realization=${REALIZATION}  host=$(hostname)"
echo "start $(date)"

"$PY" fig02_delta_b.py --rho "${RHO}" --realization_id "${REALIZATION}"

echo "end   $(date)"
