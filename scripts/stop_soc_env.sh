#!/usr/bin/env bash
# ==============================================================================
# Script: stop_soc_env.sh
# Purpose: Gracefully stop SOC Simulation Environment container services
#          without removing containers, images, volumes, or persistent data.
# ==============================================================================
set -Eeuo pipefail

# Robust project root resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WAZUH_DIR="${PROJECT_ROOT}/wazuh-docker/single-node"
WAZUH_COMPOSE="${WAZUH_DIR}/docker-compose.yml"
CALDERA_DIR="${PROJECT_ROOT}/caldera"
CALDERA_COMPOSE="${CALDERA_DIR}/docker-compose.yml"

echo "=================================================================="
echo "         Stopping SOC Simulation Environment Infrastructure      "
echo "=================================================================="
echo "Project Root: ${PROJECT_ROOT}"

# 1. Stop Caldera Compose project
if [[ -f "${CALDERA_COMPOSE}" ]]; then
    echo ""
    echo "[1/2] Stopping Caldera Compose stack..."
    docker compose --project-directory "${CALDERA_DIR}" -f "${CALDERA_COMPOSE}" stop
    echo "  [OK] Caldera containers stopped (data and state preserved)."
else
    echo "  [WARN] Caldera compose file not found at: ${CALDERA_COMPOSE}"
fi

# 2. Stop Wazuh Compose stack
if [[ -f "${WAZUH_COMPOSE}" ]]; then
    echo ""
    echo "[2/2] Stopping Wazuh Compose stack (Single-Node)..."
    docker compose --project-directory "${WAZUH_DIR}" -f "${WAZUH_COMPOSE}" stop
    echo "  [OK] Wazuh containers stopped (volumes and state preserved)."
else
    echo "  [WARN] Wazuh compose file not found at: ${WAZUH_COMPOSE}"
fi

echo ""
echo "=================================================================="
echo "                         Notice                                   "
echo "=================================================================="
echo "- Containers were stopped using 'docker compose stop'."
echo "- No containers, networks, images, or volumes were removed."
echo "- Docker daemon remains running."
echo "- Host systemd services ('wazuh-agent' and 'osqueryd') remain"
echo "  installed and active on the host unless stopped separately via:"
echo "    sudo systemctl stop wazuh-agent osqueryd"
echo "=================================================================="
echo "[SUCCESS] SOC Simulation Environment containers cleanly stopped."
exit 0
