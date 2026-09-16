#!/usr/bin/env bash
# ==============================================================================
# Kitchome RAG - Deploy Observability Stack to 192.168.0.117
# ==============================================================================
set -euo pipefail

TARGET_HOST="192.168.0.117"
TARGET_DIR="~/kitchome_observability"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================="
echo "  Deploying LGTM Observability Stack to ${TARGET_HOST}"
echo "================================================================="

# Detect user or allow override
REMOTE_USER="${1:-${USER:-vboxuser}}"

echo "[1/3] Testing SSH connectivity to ${REMOTE_USER}@${TARGET_HOST}..."
if ssh -o BatchMode=yes -o ConnectTimeout=3 "${REMOTE_USER}@${TARGET_HOST}" echo "Connected" 2>/dev/null; then
    echo "       SSH connection verified."
    echo "[2/3] Syncing configuration and docker-compose.yml to ${TARGET_HOST}:${TARGET_DIR}..."
    ssh "${REMOTE_USER}@${TARGET_HOST}" "mkdir -p ${TARGET_DIR}"
    rsync -avz --exclude '.git' "${LOCAL_DIR}/" "${REMOTE_USER}@${TARGET_HOST}:${TARGET_DIR}/"

    echo "[3/3] Launching Docker Compose stack on ${TARGET_HOST}..."
    ssh "${REMOTE_USER}@${TARGET_HOST}" "cd ${TARGET_DIR} && docker compose up -d"

    echo ""
    echo "================================================================="
    echo "  Observability Stack Successfully Started on ${TARGET_HOST}!"
    echo "================================================================="
    echo "  Grafana Dashboard:  http://${TARGET_HOST}:3000 (admin / admin)"
    echo "  Prometheus TSDB:    http://${TARGET_HOST}:9090"
    echo "  Loki Log Receiver:  http://${TARGET_HOST}:3100"
    echo "  Tempo Traces:       http://${TARGET_HOST}:3200"
    echo "================================================================="
else
    echo "Notice: Automatic passwordless SSH to ${REMOTE_USER}@${TARGET_HOST} is not configured or requires manual password."
    echo ""
    echo "Manual Deployment Steps:"
    echo "1. Copy this directory to ${TARGET_HOST}:"
    echo "   scp -r ${LOCAL_DIR} ${REMOTE_USER}@${TARGET_HOST}:${TARGET_DIR}"
    echo ""
    echo "2. SSH to ${TARGET_HOST} and launch the stack:"
    echo "   ssh ${REMOTE_USER}@${TARGET_HOST}"
    echo "   cd ${TARGET_DIR} && docker compose up -d"
    echo ""
    echo "3. Open Grafana in your browser:"
    echo "   http://${TARGET_HOST}:3000 (admin / admin)"
fi
