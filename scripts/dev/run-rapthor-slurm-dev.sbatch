#!/bin/bash
#SBATCH --job-name=rapthor-dev
#SBATCH --output=rapthor-dev-%j.out
#SBATCH --error=rapthor-dev-%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
RAPTHOR_PARSET="${RAPTHOR_PARSET:-$REPO_DIR/rapthor.parset}"
RAPTHOR_ENV_SCRIPT="${RAPTHOR_ENV_SCRIPT:-$REPO_DIR/scripts/dev/aws-setup-env.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/runs/${SLURM_JOB_NAME:-rapthor-dev}-${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)}"

PREFECT_HOST="${PREFECT_HOST:-${SLURM_SUBMIT_HOST:-127.0.0.1}}"
PREFECT_PORT="${PREFECT_PORT:-4200}"
DASK_SCHEDULER_PORT="${DASK_SCHEDULER_PORT:-8786}"
DASK_DASHBOARD_PORT="${DASK_DASHBOARD_PORT:-8787}"

mkdir -p "$OUTPUT_DIR"

if [[ -n "$RAPTHOR_ENV_SCRIPT" ]]; then
    # shellcheck source=/dev/null
    source "$RAPTHOR_ENV_SCRIPT"
fi

export PREFECT_API_URL="${PREFECT_API_URL:-http://${PREFECT_HOST}:${PREFECT_PORT}/api}"
curl -fsS "$PREFECT_API_URL/health" >/dev/null

NODE_COUNT="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}"
WORKER_THREADS="${RAPTHOR_DASK_WORKER_THREADS:-1}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cleanup() {
    local children
    children="$(jobs -pr)"
    if [[ -n "$children" ]]; then
        kill $children 2>/dev/null || true
    fi
}
trap cleanup EXIT

mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
FIRST_NODE_HOSTNAME="${SLURM_HOSTS[0]}"
FIRST_NODE_IP="$(srun --nodes=1 --ntasks=1 --nodelist="$FIRST_NODE_HOSTNAME" hostname -I | awk '{print $1}')"
SCHEDULER_FILE="$OUTPUT_DIR/dask-scheduler.json"
export DASK_SCHEDULER="tcp://${FIRST_NODE_IP}:${DASK_SCHEDULER_PORT}"
export RAPTHOR_DASK_EXPECTED_WORKERS="$NODE_COUNT"

srun --nodes=1 --ntasks=1 --nodelist="$FIRST_NODE_HOSTNAME" \
    dask scheduler \
        --host "$FIRST_NODE_IP" \
        --port "$DASK_SCHEDULER_PORT" \
        --dashboard-address ":$DASK_DASHBOARD_PORT" \
        --scheduler-file "$SCHEDULER_FILE" \
        >"$OUTPUT_DIR/dask-scheduler.log" 2>&1 &

for _ in $(seq 1 60); do
    if [[ -s "$SCHEDULER_FILE" ]]; then
        break
    fi
    sleep 2
done
[[ -s "$SCHEDULER_FILE" ]]

srun --nodes="$NODE_COUNT" --ntasks="$NODE_COUNT" --ntasks-per-node=1 \
    dask worker \
        --scheduler-file "$SCHEDULER_FILE" \
        --nthreads "$WORKER_THREADS" \
        --no-dashboard \
        >"$OUTPUT_DIR/dask-workers.log" 2>&1 &

python - <<'PY'
import os
import sys
import time
from dask.distributed import Client

expected = int(os.environ["RAPTHOR_DASK_EXPECTED_WORKERS"])
client = Client(os.environ["DASK_SCHEDULER"], timeout="60s")
try:
    for _ in range(60):
        workers = client.scheduler_info().get("workers", {})
        if len(workers) >= expected:
            break
        time.sleep(2)
    else:
        sys.exit(f"Expected {expected} Dask workers, found {len(workers)}")
finally:
    client.close()
PY

rapthor "$RAPTHOR_PARSET"
