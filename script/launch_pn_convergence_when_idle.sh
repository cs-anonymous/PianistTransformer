#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU_LIST="${GPU_LIST:-0,1,2}"
OUT_ROOT="${OUT_ROOT:-results/pn_convergence_constant60_$(date '+%Y%m%d_%H%M%S')}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
mkdir -p "${OUT_ROOT}/logs"
while true; do
  busy=$(GPU_LIST="${GPU_LIST}" python - <<'PY'
import os, subprocess
ids=[x.strip() for x in os.environ['GPU_LIST'].split(',')]
out=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
busy=0
for line in out.splitlines():
    fields=[x.strip() for x in line.split(',')]
    if fields[0] in ids and (int(fields[1]) > 5 or int(fields[2]) > 1000):
        busy += 1
print(busy)
PY
)
  if [[ "${busy}" == "0" ]]; then
    exec env GPU_LIST="${GPU_LIST}" OUT_ROOT="${OUT_ROOT}" bash script/run_pn_convergence_experiment.sh
  fi
  echo "waiting for GPUs ${GPU_LIST}; busy=${busy}" >> "${OUT_ROOT}/logs/launcher.log"
  sleep "${WAIT_SECONDS}"
done
