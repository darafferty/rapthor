#!/usr/bin/env bash
# Start an isolated local Prefect server for interactive Rapthor testing.

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "ERROR: Execute this script; do not source it:" >&2
    echo "  scripts/dev/start-prefect-server.sh" >&2
    return 2
fi

set -euo pipefail

PREFECT_HOST="${PREFECT_HOST:-0.0.0.0}"
PREFECT_PORT="${PREFECT_PORT:-4200}"
PREFECT_CONNECT_HOST="${PREFECT_CONNECT_HOST:-127.0.0.1}"
PREFECT_HOME_CREATED=false
SERVER_PID=""

usage() {
    cat <<'EOF'
Usage: scripts/dev/start-prefect-server.sh [OPTIONS]

Start an isolated Prefect server for interactive Rapthor testing.

Options:
  --host HOST          Server bind host (default: 0.0.0.0)
  --port PORT          Server and dashboard port (default: 4200)
  --connect-host HOST  Host used by local clients (default: 127.0.0.1)
  -h, --help           Show this help message

The same defaults can be overridden with PREFECT_HOST, PREFECT_PORT, and
PREFECT_CONNECT_HOST.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            [[ $# -ge 2 ]] || { echo "ERROR: --host requires a value." >&2; exit 2; }
            PREFECT_HOST="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "ERROR: --port requires a value." >&2; exit 2; }
            PREFECT_PORT="$2"
            shift 2
            ;;
        --connect-host)
            [[ $# -ge 2 ]] || {
                echo "ERROR: --connect-host requires a value." >&2
                exit 2
            }
            PREFECT_CONNECT_HOST="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$PREFECT_PORT" =~ ^[0-9]+$ ]] || ((PREFECT_PORT < 1 || PREFECT_PORT > 65535)); then
    echo "ERROR: --port must be an integer between 1 and 65535." >&2
    exit 2
fi

if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    echo "ERROR: Python is required to check the Prefect API." >&2
    exit 1
fi

check_port_available() {
    "$PYTHON" - "$PREFECT_HOST" "$PREFECT_PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
family = socket.AF_INET6 if ":" in host else socket.AF_INET
with socket.socket(family, socket.SOCK_STREAM) as listener:
    listener.bind((host, port))
PY
}

if ! check_port_available >/dev/null 2>&1; then
    echo "ERROR: Port ${PREFECT_PORT} is already in use on ${PREFECT_HOST}." >&2
    echo "Choose another port, for example: $0 --port 14200" >&2
    exit 2
fi

if [[ -z "${PREFECT_HOME:-}" ]]; then
    TEMP_ROOT="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
    mkdir -p "$TEMP_ROOT"
    PREFECT_HOME="$(mktemp -d "$TEMP_ROOT/rapthor-prefect-${USER:-user}-XXXXXX")"
    export PREFECT_HOME
    PREFECT_HOME_CREATED=true
fi

export PREFECT_SERVER_ANALYTICS_ENABLED=false
export PREFECT_UI_API_URL=/api
export DO_NOT_TRACK=1

API_URL="http://${PREFECT_CONNECT_HOST}:${PREFECT_PORT}/api"
HEALTH_URL="${API_URL}/health"
DASHBOARD_URL="http://${PREFECT_CONNECT_HOST}:${PREFECT_PORT}"
export PREFECT_API_URL="$API_URL"

cleanup() {
    exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [[ "$PREFECT_HOME_CREATED" == true ]]; then
        rm -rf -- "$PREFECT_HOME"
    fi
    exit "$exit_code"
}

check_health() {
    "$PYTHON" - "$HEALTH_URL" <<'PY'
import sys
from urllib.request import urlopen

with urlopen(sys.argv[1], timeout=1) as response:
    if not 200 <= response.status < 300:
        raise SystemExit(1)
PY
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Starting Prefect server on ${PREFECT_HOST}:${PREFECT_PORT}"
echo "Prefect server state: ${PREFECT_HOME}"
prefect server start --host "$PREFECT_HOST" --port "$PREFECT_PORT" &
SERVER_PID=$!

echo "Waiting for Prefect API at ${HEALTH_URL}"
healthy=false
for _ in {1..60}; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        wait "$SERVER_PID"
    fi
    if check_health >/dev/null 2>&1; then
        healthy=true
        break
    fi
    sleep 1
done

if [[ "$healthy" != true ]]; then
    echo "ERROR: Prefect API did not become healthy at ${HEALTH_URL}" >&2
    exit 1
fi

cat <<EOF

Prefect is ready.

Dashboard:
  ${DASHBOARD_URL}

In another terminal, run Rapthor normally:
  export PREFECT_API_URL=${API_URL}
  rapthor rapthor.parset

Press Ctrl+C here to stop the Prefect server.
EOF

wait "$SERVER_PID"
