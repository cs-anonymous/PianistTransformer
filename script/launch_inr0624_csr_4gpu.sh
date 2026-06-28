#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIGS=(
  "configs/inr0624_csr_cine_sb0.json"
  "configs/inr0624_csr_cine_sb1.json"
  "configs/inr0624_csr_sine_sb0.json"
  "configs/inr0624_csr_sine_sb1.json"
)

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#GPUS[@]} -ge ${#CONFIGS[@]} ]] || {
  echo "Need at least ${#CONFIGS[@]} GPUs in CUDA_VISIBLE_DEVICES, got: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}" >&2
  exit 1
}

mkdir -p results/inr_csr_pipeline/launcher_logs

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu="${GPUS[$idx]}"
  name="$(basename "${config}" .json)"
  log="results/inr_csr_pipeline/launcher_logs/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "launch ${name} on GPU ${gpu}; log=${log}"
  setsid env CUDA_VISIBLE_DEVICES="${gpu}" CONFIG="${config}" \
    bash script/run_inr_csr_pipeline.sh >"${log}" 2>&1 < /dev/null &
done

wait
