#!/usr/bin/env bash
# ==============================================================================
# Kitchome RAG Intelligence Platform - Ecosystem Multi-Terminal Launcher
# ==============================================================================
# Launches all 5 core services in dedicated macOS Terminal windows (or background):
#  1. Auth Service (Spring Boot on Port 8080)
#  2. FastAPI Backend (Uvicorn on Port 8000 with metrics & tracing)
#  3. Streamlit UI (Frontend on Port 8501)
#  4. Ingestion Controller Daemon (Queue & lease supervisor)
#  5. Ingestion Worker Swarm (Distributed chunker & embedder workers)
# ==============================================================================

set -eo pipefail

# Resolve canonical script directory even if invoked through symlink
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

# Correctly resolve REPO_ROOT (handles both repo root and scripts/ locations)
if [ -f "${SCRIPT_DIR}/src/api/main.py" ]; then
    REPO_ROOT="${SCRIPT_DIR}"
elif [ -f "${SCRIPT_DIR}/../src/api/main.py" ]; then
    REPO_ROOT="$(cd -P "${SCRIPT_DIR}/.." && pwd)"
else
    REPO_ROOT="$(cd -P "${SCRIPT_DIR}/.." && pwd)"
fi

# Ensure Pyenv, Java 17, virtualenv, and user binaries are accessible
export JAVA_HOME="$(/usr/libexec/java_home -v 17 2>/dev/null || echo "$JAVA_HOME")"

if [ -d "${REPO_ROOT}/.venv" ]; then
    PYTHON_EXEC="${REPO_ROOT}/.venv/bin/python3"
    FULL_PATH="${REPO_ROOT}/.venv/bin:${JAVA_HOME}/bin:$HOME/.pyenv/shims:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
else
    PYTHON_EXEC="python3"
    FULL_PATH="${JAVA_HOME}/bin:$HOME/.pyenv/shims:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
fi
export PATH="$FULL_PATH:$PATH"

# Resolve Auth Service path and Maven wrapper
AUTH_DIR="${REPO_ROOT}/../Complete-custom-auth/auth-service"
if [ ! -d "${AUTH_DIR}" ]; then
    AUTH_DIR="${REPO_ROOT}/../Complete-custom-auth"
fi

if [ -f "${AUTH_DIR}/mvnw" ]; then
    MVN_CMD="./mvnw"
elif [ -f "${AUTH_DIR}/../mvnw" ]; then
    MVN_CMD="../mvnw"
elif command -v mvn &>/dev/null; then
    MVN_CMD="mvn"
else
    MVN_CMD="../mvnw"
fi

mkdir -p "${REPO_ROOT}/logs"
PID_FILE="${REPO_ROOT}/.ecosystem.pids"

MODE="terminal"
if [[ "${1:-}" == "--background" || "${1:-}" == "-b" ]]; then
    MODE="background"
elif [[ "${1:-}" == "--status" || "${1:-}" == "-s" ]]; then
    MODE="status"
fi

# ------------------------------------------------------------------------------
# Status Checker Mode
# ------------------------------------------------------------------------------
if [ "${MODE}" == "status" ]; then
    echo "================================================================="
    echo "  Kitchome Ecosystem Port & Service Status"
    echo "================================================================="
    printf "%-25s %-10s %-15s\n" "Service" "Port" "Status"
    echo "-----------------------------------------------------------------"
    for item in "FastAPI:8000" "Streamlit:8501" "Auth-Service:8080"; do
        NAME="${item%%:*}"
        PORT="${item##*:}"
        PID=$(lsof -ti :${PORT} 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)
        if [ -n "${PID}" ]; then
            printf "%-25s %-10s \033[0;32mRUNNING\033[0m (PID %s)\n" "${NAME}" "${PORT}" "${PID}"
        else
            printf "%-25s %-10s \033[0;31mSTOPPED\033[0m\n" "${NAME}" "${PORT}"
        fi
    done
    
    CONTROLLER_PID=$(pgrep -f "src.ingestion.run_controller" 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)
    if [ -n "${CONTROLLER_PID}" ]; then
        printf "%-25s %-10s \033[0;32mRUNNING\033[0m (PID %s)\n" "Controller Daemon" "N/A" "${CONTROLLER_PID}"
    else
        printf "%-25s %-10s \033[0;31mSTOPPED\033[0m\n" "Controller Daemon" "N/A"
    fi

    WORKER_PID=$(pgrep -f "src.ingestion.run_workers" 2>/dev/null | tr '\n' ',' | sed 's/,$//' || true)
    if [ -n "${WORKER_PID}" ]; then
        printf "%-25s %-10s \033[0;32mRUNNING\033[0m (PID %s)\n" "Worker Swarm" "N/A" "${WORKER_PID}"
    else
        printf "%-25s %-10s \033[0;31mSTOPPED\033[0m\n" "Worker Swarm" "N/A"
    fi
    echo "================================================================="
    exit 0
fi

# ------------------------------------------------------------------------------
# Define Commands for Each Service (Terminal Window Mode)
# ------------------------------------------------------------------------------

# 1. Auth Service
AUTH_CMD="cd '${AUTH_DIR}' && export JAVA_HOME='${JAVA_HOME}' && export PATH=\"${JAVA_HOME}/bin:${FULL_PATH}:\$PATH\" && set -a && ([ -f ../.env.local ] && source ../.env.local || ([ -f .env.local ] && source .env.local || true)) && set +a && echo -e '\\033]0;[1/5] Kitchome Auth Service (8080)\\007' && echo -e '\\033[1;36m=== [1/5] Starting Auth Service (Port 8080) ===\\033[0m' && ${MVN_CMD} spring-boot:run"

# 2. FastAPI Backend
FASTAPI_CMD="cd '${REPO_ROOT}' && export PATH=\"${FULL_PATH}:\$PATH\" && ([ -d .venv ] && source .venv/bin/activate || true) && echo -e '\\033]0;[2/5] Kitchome FastAPI (8000)\\007' && echo -e '\\033[1;32m=== [2/5] Starting FastAPI Backend (Port 8000) ===\\033[0m' && \"${PYTHON_EXEC}\" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

# 3. Streamlit UI
STREAMLIT_CMD="cd '${REPO_ROOT}' && export PATH=\"${FULL_PATH}:\$PATH\" && ([ -d .venv ] && source .venv/bin/activate || true) && echo -e '\\033]0;[3/5] Kitchome Streamlit UI (8501)\\007' && echo -e '\\033[1;35m=== [3/5] Starting Streamlit UI (Port 8501) ===\\033[0m' && \"${PYTHON_EXEC}\" -m streamlit run src/ui/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"

# 4. Ingestion Controller Daemon
CONTROLLER_CMD="cd '${REPO_ROOT}' && export PATH=\"${FULL_PATH}:\$PATH\" && ([ -d .venv ] && source .venv/bin/activate || true) && echo -e '\\033]0;[4/5] Kitchome Controller Daemon\\007' && echo -e '\\033[1;33m=== [4/5] Starting Ingestion Controller Daemon ===\\033[0m' && \"${PYTHON_EXEC}\" -m src.ingestion.run_controller --poll-interval 5.0"

# 5. Ingestion Worker Swarm
WORKER_CMD="cd '${REPO_ROOT}' && export PATH=\"${FULL_PATH}:\$PATH\" && ([ -d .venv ] && source .venv/bin/activate || true) && echo -e '\\033]0;[5/5] Kitchome Worker Swarm\\007' && echo -e '\\033[1;34m=== [5/5] Starting Ingestion Worker Swarm ===\\033[0m' && \"${PYTHON_EXEC}\" -m src.ingestion.run_workers --role all --mode thread"

# ------------------------------------------------------------------------------
# Terminal Mode: Spin Dedicated macOS Terminal Windows via AppleScript
# ------------------------------------------------------------------------------
if [ "${MODE}" == "terminal" ]; then
    echo "================================================================="
    echo "  Spinning 5 Dedicated Terminal Windows for Kitchome Ecosystem"
    echo "================================================================="
    echo "  1. 🔐 Auth Service          -> Port 8080 (Spring Boot)"
    echo "  2. 🚀 FastAPI Backend       -> Port 8000 (Uvicorn)"
    echo "  3. 🎨 Streamlit UI          -> Port 8501 (Browser Interface)"
    echo "  4. ⚙️ Ingestion Controller   -> Queue & Lease Supervisor"
    echo "  5. 🐝 Ingestion Workers     -> Chunker & Embedder Swarm"
    echo "================================================================="

    launch_terminal_window() {
        local title="$1"
        local cmd="$2"
        osascript - "$cmd" <<'EOF'
on run argv
    set cmdToRun to item 1 of argv
    tell application "Terminal"
        activate
        do script cmdToRun
    end tell
end run
EOF
    }

    # 1. Auth Service
    PID_8080=$(lsof -ti :8080 2>/dev/null || true)
    if [ -n "${PID_8080}" ]; then
        echo "  [1/5] 🔐 Auth Service is already RUNNING on port 8080 (PID ${PID_8080})."
    else
        echo "Launching [1/5] Auth Service Terminal..."
        launch_terminal_window "Auth Service" "${AUTH_CMD}"
        sleep 1
    fi

    # 2. FastAPI Backend
    PID_8000=$(lsof -ti :8000 2>/dev/null || true)
    if [ -n "${PID_8000}" ]; then
        echo "  [2/5] 🚀 FastAPI Backend is already RUNNING on port 8000 (PID ${PID_8000})."
    else
        echo "Launching [2/5] FastAPI Backend Terminal..."
        launch_terminal_window "FastAPI Backend" "${FASTAPI_CMD}"
        sleep 1
    fi

    # 3. Streamlit UI
    PID_8501=$(lsof -ti :8501 2>/dev/null || true)
    if [ -n "${PID_8501}" ]; then
        echo "  [3/5] 🎨 Streamlit UI is already RUNNING on port 8501 (PID ${PID_8501})."
    else
        echo "Launching [3/5] Streamlit UI Terminal..."
        launch_terminal_window "Streamlit UI" "${STREAMLIT_CMD}"
        sleep 1
    fi

    # 4. Ingestion Controller Daemon
    PID_CTRL=$(pgrep -f "src.ingestion.run_controller" 2>/dev/null || true)
    if [ -n "${PID_CTRL}" ]; then
        echo "  [4/5] ⚙️ Ingestion Controller is already RUNNING (PID ${PID_CTRL})."
    else
        echo "Launching [4/5] Ingestion Controller Terminal..."
        launch_terminal_window "Ingestion Controller" "${CONTROLLER_CMD}"
        sleep 1
    fi

    # 5. Ingestion Worker Swarm
    PID_WRK=$(pgrep -f "src.ingestion.run_workers" 2>/dev/null || true)
    if [ -n "${PID_WRK}" ]; then
        echo "  [5/5] 🐝 Ingestion Worker Swarm is already RUNNING (PID ${PID_WRK})."
    else
        echo "Launching [5/5] Ingestion Worker Terminal..."
        launch_terminal_window "Ingestion Worker" "${WORKER_CMD}"
    fi

    echo ""
    echo "================================================================="
    echo "  All services ready!"
    echo "  - Streamlit UI:   http://localhost:8501"
    echo "  - FastAPI Docs:   http://localhost:8000/docs"
    echo "  - Auth Service:   http://localhost:8080"
    echo "  - Metrics Feed:   http://localhost:8000/metrics"
    echo "  - Grafana (LAN):  http://192.168.0.117:3000"
    echo ""
    echo "  To stop all services anytime, run: ./scripts/stop_ecosystem.sh"
    echo "================================================================="
    exit 0
fi

# ------------------------------------------------------------------------------
# Background / Headless Mode
# ------------------------------------------------------------------------------
if [ "${MODE}" == "background" ]; then
    echo "Launching all 5 services in background..."
    > "${PID_FILE}"

    # 1. Auth Service
    PID_8080=$(lsof -ti :8080 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "${PID_8080}" ]; then
        echo "  [Auth Service] Already running on port 8080 (PID ${PID_8080})"
        echo "${PID_8080}" >> "${PID_FILE}"
    else
        (cd "${AUTH_DIR}" && export JAVA_HOME="${JAVA_HOME}" && export PATH="${JAVA_HOME}/bin:${FULL_PATH}:$PATH" && set -a && ([ -f ../.env.local ] && source ../.env.local || ([ -f .env.local ] && source .env.local || true)) && set +a && ${MVN_CMD} spring-boot:run) > "${REPO_ROOT}/logs/auth_service.log" 2>&1 &
        echo $! >> "${PID_FILE}"
        echo "  [Auth Service] Started (PID $!, log: logs/auth_service.log)"
    fi

    # 2. FastAPI
    PID_8000=$(lsof -ti :8000 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "${PID_8000}" ]; then
        echo "  [FastAPI] Already running on port 8000 (PID ${PID_8000})"
        echo "${PID_8000}" >> "${PID_FILE}"
    else
        (cd "${REPO_ROOT}" && export PATH="${FULL_PATH}:$PATH" && ([ -d .venv ] && source .venv/bin/activate || true) && "${PYTHON_EXEC}" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000) > "${REPO_ROOT}/logs/fastapi.log" 2>&1 &
        echo $! >> "${PID_FILE}"
        echo "  [FastAPI] Started (PID $!, log: logs/fastapi.log)"
    fi

    # 3. Streamlit
    PID_8501=$(lsof -ti :8501 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "${PID_8501}" ]; then
        echo "  [Streamlit] Already running on port 8501 (PID ${PID_8501})"
        echo "${PID_8501}" >> "${PID_FILE}"
    else
        (cd "${REPO_ROOT}" && export PATH="${FULL_PATH}:$PATH" && ([ -d .venv ] && source .venv/bin/activate || true) && "${PYTHON_EXEC}" -m streamlit run src/ui/app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false) > "${REPO_ROOT}/logs/streamlit.log" 2>&1 &
        echo $! >> "${PID_FILE}"
        echo "  [Streamlit] Started (PID $!, log: logs/streamlit.log)"
    fi

    # 4. Controller
    PID_CTRL=$(pgrep -f "src.ingestion.run_controller" 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "${PID_CTRL}" ]; then
        echo "  [Controller] Already running (PID ${PID_CTRL})"
        echo "${PID_CTRL}" >> "${PID_FILE}"
    else
        (cd "${REPO_ROOT}" && export PATH="${FULL_PATH}:$PATH" && ([ -d .venv ] && source .venv/bin/activate || true) && "${PYTHON_EXEC}" -m src.ingestion.run_controller --poll-interval 5.0) > "${REPO_ROOT}/logs/controller.log" 2>&1 &
        echo $! >> "${PID_FILE}"
        echo "  [Controller] Started (PID $!, log: logs/controller.log)"
    fi

    # 5. Worker
    PID_WRK=$(pgrep -f "src.ingestion.run_workers" 2>/dev/null | tr '\n' ' ' || true)
    if [ -n "${PID_WRK}" ]; then
        echo "  [Worker] Already running (PID ${PID_WRK})"
        echo "${PID_WRK}" >> "${PID_FILE}"
    else
        (cd "${REPO_ROOT}" && export PATH="${FULL_PATH}:$PATH" && ([ -d .venv ] && source .venv/bin/activate || true) && "${PYTHON_EXEC}" -m src.ingestion.run_workers --role all --mode thread) > "${REPO_ROOT}/logs/worker.log" 2>&1 &
        echo $! >> "${PID_FILE}"
        echo "  [Worker] Started (PID $!, log: logs/worker.log)"
    fi

    echo ""
    echo "All services running in background. Stop with: ./scripts/stop_ecosystem.sh"
fi
