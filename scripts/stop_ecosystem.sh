#!/usr/bin/env bash
# ==============================================================================
# Kitchome RAG Intelligence Platform - Ecosystem Shutdown Script
# ==============================================================================
# Gracefully terminates all 5 ecosystem processes:
#  - Port 8080: Auth Service
#  - Port 8000: FastAPI Backend
#  - Port 8501: Streamlit UI
#  - Python: Ingestion Controller Daemon
#  - Python: Ingestion Worker Swarm
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${REPO_ROOT}/.ecosystem.pids"

echo "================================================================="
echo "  Shutting Down Kitchome RAG Ecosystem Services"
echo "================================================================="

kill_port() {
    local name="$1"
    local port="$2"
    local pids=$(lsof -ti :${port} 2>/dev/null || true)
    if [ -n "${pids}" ]; then
        local pids_clean=$(echo "${pids}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
        echo "Stopping ${name} on port ${port} (PID: ${pids_clean})..."
        kill -15 ${pids} 2>/dev/null || true
        sleep 0.5
        # Force kill if still hanging
        local remaining=$(lsof -ti :${port} 2>/dev/null || true)
        if [ -n "${remaining}" ]; then
            kill -9 ${remaining} 2>/dev/null || true
        fi
        echo "  -> ${name} stopped."
    else
        echo "  ${name} (port ${port}) is not running."
    fi
}

kill_pattern() {
    local name="$1"
    local pattern="$2"
    local pids=$(pgrep -f "${pattern}" 2>/dev/null || true)
    if [ -n "${pids}" ]; then
        local pids_clean=$(echo "${pids}" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
        echo "Stopping ${name} matching '${pattern}' (PID: ${pids_clean})..."
        kill -15 ${pids} 2>/dev/null || true
        sleep 0.5
        local remaining=$(pgrep -f "${pattern}" 2>/dev/null || true)
        if [ -n "${remaining}" ]; then
            kill -9 ${remaining} 2>/dev/null || true
        fi
        echo "  -> ${name} stopped."
    else
        echo "  ${name} is not running."
    fi
}

# 1. Kill by port
kill_port "FastAPI Backend" 8000
kill_port "Streamlit UI" 8501
kill_port "Auth Service" 8080

# 2. Kill by process command
kill_pattern "Controller Daemon" "src.ingestion.run_controller"
kill_pattern "Worker Swarm" "src.ingestion.run_workers"

# 3. Clean up PID files if present
for pfile in "${PID_FILE}" "${SCRIPT_DIR}/.ecosystem.pids"; do
    if [ -f "${pfile}" ]; then
        while IFS= read -r pid; do
            if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
                kill -9 "${pid}" 2>/dev/null || true
            fi
        done < "${pfile}"
        rm -f "${pfile}"
    fi
done

echo "================================================================="
echo "  All Kitchome Ecosystem services have been stopped."
echo "================================================================="
