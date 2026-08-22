#!/bin/bash
#SBATCH --job-name=gnc_trends
#SBATCH --cluster=htc
#SBATCH --partition=htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-user=RAM587@pitt.edu
#SBATCH --mail-type=FAIL
# ---------------------------------------------------------------------------
# The (rho, x, p_s) sweep behind Figure 3(a) and Figures S3-S6.
#
# One task = one (topology, point, psi) combination, running all 100
# realizations serially.  Measured on one core at N = 5000: ~32 s per
# realization for ER, so a task is roughly one hour; BA and modular are denser
# and can take longer, hence the 4 h wall clock.
#
#   task  topology  point  psi     figure it feeds
#   ----  --------  -----  ------  ---------------------------------
#     0   ER        E      -       Figure 3(a) top, S3 column 1
#     1   ER        F      -       Figure 3(a) bottom, S4 column 1
#     2   BA        E      -       S3 column 2
#     3   BA        F      -       S4 column 2
#     4   MOD       E      0.0     S5 column 1
#     5   MOD       E      0.4     S3 column 3, S5 column 2
#     6   MOD       E      1.0     S5 column 3
#     7   MOD       F      0.0     S6 column 1
#     8   MOD       F      0.4     S4 column 3, S6 column 2
#     9   MOD       F      1.0     S6 column 3
#
# To rerun one combination, submit just its id, e.g.  --array=5
# ---------------------------------------------------------------------------
set -euo pipefail

TOPO=(ER  ER  BA  BA  MOD MOD MOD MOD MOD MOD)
PT=(  E   F   E   F   E   E   E   F   F   F  )
PSI=( -   -   -   -   0.0 0.4 1.0 0.0 0.4 1.0)

i=${SLURM_ARRAY_TASK_ID}
ARGS="--topology ${TOPO[$i]} --point ${PT[$i]}"
if [[ "${PSI[$i]}" != "-" ]]; then
    ARGS="${ARGS} --psi ${PSI[$i]}"
fi

mkdir -p logs
module purge
module load python/ondemand-jupyter-python3.11
PY=$HOME/.conda/envs/myenv/bin/python

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"
echo "task=${i}  ${ARGS}  host=$(hostname)"
echo "start $(date)"

"$PY" run_trends.py ${ARGS}

echo "end   $(date)"
