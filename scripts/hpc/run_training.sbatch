#!/bin/bash
#
# Generic SLURM job script for running bandgap-benchmark training workloads.
# The actual configuration and training mode are provided via exported
# environment variables when submitting the job with sbatch.
#
# Required environment variables supplied by the submitter:
#   CONFIG   - Path to the YAML config (relative to repo root)
# Optional environment variables:
#   RUN_MODE - Either "pretrain" (default) or empty for finetune runs
#   DEVICES  - Value passed to `--devices` (defaults to 1)
#
# Example submission:
#   sbatch --export=CONFIG=configs/pretrain/pretrained_cgcnn.yaml,RUN_MODE=pretrain \
#          scripts/hpc/run_training.sbatch

# SLURM settings expected by the user
# Adjust the directives below if your cluster requires different defaults.
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --job-name=bandgap-train
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=haolin.wang@sheffield.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=outputs/%x_%j.txt

set -euo pipefail

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG env variable is required when submitting the job."
  exit 1
fi

# Default values if not provided.
RUN_MODE="${RUN_MODE:-pretrain}"
DEVICES="${DEVICES:-1}"

module load Anaconda3/2019.07
module load CUDA/12.4.0

source activate pykale

echo "Begin: $(date)"

cd /users/ac1hw/projects/bandgap-benchmark/

CMD=(python main.py --cfg "${CONFIG}" --devices "${DEVICES}")

if [[ "${RUN_MODE}" == "pretrain" ]]; then
  CMD+=(--pretrain)
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"

echo "Done: $(date)"
