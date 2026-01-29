#!/bin/bash
#
# Helper script to submit multiple bandgap-benchmark training jobs to SLURM.
# Edit the CONFIGS array below to list every (mode, config, job-name) tuple you
# would like to launch. Jobs are submitted sequentially with sbatch.
#
# Usage on the HPC login node:
#   cd /users/ac1hw/projects/bandgap-benchmark
#   ./scripts/hpc/submit_jobs.sh
#
# Optional environment variables when invoking the script:
#   DEVICES   - Passed through to Lightning's `--devices` flag (default: 1)
#   RUN_MODE  - Default mode for entries that omit an explicit mode (default: pretrain)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_training.sbatch"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: Expected sbatch template at ${SBATCH_SCRIPT}."
  exit 1
fi

mkdir -p outputs

DEFAULT_MODE="${RUN_MODE:-pretrain}"
DEVICES="${DEVICES:-1}"

# Each entry: "<mode> <config_path> <job_name>"
#   mode        - Either 'pretrain' or 'finetune'. Defaults to ${DEFAULT_MODE} if set to '-'.
#   config_path - Path to the YAML config relative to the repository root.
#   job_name    - Job name and log prefix (avoid spaces).
CONFIGS=(
  "pretrain configs/pretrain/pretrained_cgcnn.yaml cgcnn_pretrain"
  "pretrain configs/pretrain/pretrained_chgnet.yaml chgnet_pretrain"
  "pretrain configs/pretrain/pretrained_alignn.yaml alignn_pretrain"
  "pretrain configs/pretrain/pretrained_alignn_prop.yaml alignn_prop_pretrain"
  "pretrain configs/pretrain/pretrained_cartnet.yaml cartnet_pretrain"
  "pretrain configs/pretrain/leftnet/pretrained_z.yaml leftnetz_pretrain"
  "pretrain configs/pretrain/leftnet/pretrained_prop.yaml leftnetprop_pretrain"
  "pretrain configs/pretrain/split_json_chemsys_k0.yaml chemsys_k0_pretrain"
)

for entry in "${CONFIGS[@]}"; do
  read -r mode cfg_path job_name <<<"${entry}"

  if [[ "${mode}" == "-" ]]; then
    mode="${DEFAULT_MODE}"
  fi

  if [[ ! -f "${cfg_path}" ]]; then
    echo "WARNING: Skipping ${cfg_path} (file not found)."
    continue
  fi

  # Compose sbatch-specific overrides. Job names must be <= 128 characters.
  sbatch --job-name="${job_name}" \
         --output="outputs/${job_name}_%j.txt" \
         --export=CONFIG="${cfg_path}",RUN_MODE="${mode}",DEVICES="${DEVICES}" \
         "${SBATCH_SCRIPT}"
done
