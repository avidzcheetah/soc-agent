#!/usr/bin/env bash
# ==============================================================================
# Script: start_soc_env.sh
# Purpose: Start and verify the SOC Simulation Environment infrastructure
#          (Wazuh SIEM Cluster, Host Wazuh Agent, OSQuery, MITRE Caldera)
# ==============================================================================
set -Eeuo pipefail

# Robust project root resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WAZUH_DIR="${PROJECT_ROOT}/wazuh-docker/single-node"
WAZUH_COMPOSE="${WAZUH_DIR}/docker-compose.yml"
CALDERA_DIR="${PROJECT_ROOT}/caldera"
CALDERA_COMPOSE="${CALDERA_DIR}/docker-compose.yml"

WAZUH_CONTAINERS=(
    "single-node-wazuh.indexer-1"
    "single-node-wazuh.manager-1"
    "single-node-wazuh.dashboard-1"
)
CALDERA_CONTAINER="caldera"

# Tracking component status for summary
STATUS_DOCKER="FAIL"
STATUS_WAZUH_STACK="FAIL"
STATUS_WAZUH_AGENT="FAIL"
STATUS_OSQUERYD="FAIL"
STATUS_CALDERA="FAIL"
STATUS_DASHBOARD_EP="FAIL"
STATUS_CALDERA_EP="FAIL"

print_header() {
    echo "=================================================================="
    echo "         Starting SOC Simulation Environment Infrastructure      "
    echo "=================================================================="
    echo "Project Root: ${PROJECT_ROOT}"
}

print_summary() {
    echo ""
    echo "=================================================================="
    echo "            SOC Simulation Environment Verification Summary       "
    echo "=================================================================="
    echo "  Docker Daemon:          [${STATUS_DOCKER}]"
    echo "  Wazuh Containers:       [${STATUS_WAZUH_STACK}]"
    echo "  Host Wazuh Agent:       [${STATUS_WAZUH_AGENT}]"
    echo "  Host OSQuery (osqueryd):[${STATUS_OSQUERYD}]"
    echo "  Caldera Container:      [${STATUS_CALDERA}]"
    echo "  Wazuh Dashboard (443):  [${STATUS_DASHBOARD_EP}]"
    echo "  Caldera Web UI (8888):  [${STATUS_CALDERA_EP}]"
    echo "=================================================================="
}

trap 'echo ""; echo "[ERROR] Script interrupted or failed on line $LINENO."; print_summary' ERR

print_header

# ------------------------------------------------------------------------------
# 1. Check Docker Daemon
# ------------------------------------------------------------------------------
echo ""
echo "[1/7] Checking Docker daemon status..."
if docker info >/dev/null 2>&1; then
    echo "  [OK] Docker daemon is running and accessible."
    STATUS_DOCKER="PASS"
else
    echo "  [*] Docker is not running. Attempting to start service..."
    if command -v systemctl >/dev/null 2>&1; then
        systemctl start docker 2>/dev/null || sudo systemctl start docker 2>/dev/null || true
    fi

    if docker info >/dev/null 2>&1; then
        echo "  [OK] Docker service started successfully."
        STATUS_DOCKER="PASS"
    else
        echo "  [FAIL] Docker daemon is not active or user lacks docker permissions."
        echo "  Diagnostic commands:"
        echo "    sudo systemctl status docker --no-pager"
        echo "    sudo journalctl -u docker -n 30 --no-pager"
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 2. Start Wazuh Stack (Single-Node Compose)
# ------------------------------------------------------------------------------
echo ""
echo "[2/7] Starting Wazuh Compose stack (Single-Node)..."
if [[ ! -f "${WAZUH_COMPOSE}" ]]; then
    echo "  [FAIL] Wazuh compose file not found at: ${WAZUH_COMPOSE}"
    exit 1
fi

docker compose --project-directory "${WAZUH_DIR}" -f "${WAZUH_COMPOSE}" up -d

echo "  [*] Waiting for Wazuh containers to enter 'running' state..."
MAX_WAIT=90
for c in "${WAZUH_CONTAINERS[@]}"; do
    elapsed=0
    c_status="unknown"
    while [[ $elapsed -lt $MAX_WAIT ]]; do
        if docker inspect -f '{{.State.Status}}' "$c" >/dev/null 2>&1; then
            c_status="$(docker inspect -f '{{.State.Status}}' "$c")"
            if [[ "$c_status" == "running" ]]; then
                break
            fi
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [[ "$c_status" == "running" ]]; then
        echo "  [OK] Container '$c' is running."
    else
        echo "  [FAIL] Container '$c' failed to reach 'running' state (current: $c_status)."
        echo "  Diagnostic commands:"
        echo "    docker compose -f ${WAZUH_COMPOSE} ps"
        echo "    docker logs --tail 50 $c"
        exit 1
    fi
done
STATUS_WAZUH_STACK="PASS"

# ------------------------------------------------------------------------------
# 3. Verify & Start Host Systemd Services (wazuh-agent & osqueryd)
# ------------------------------------------------------------------------------
echo ""
echo "[3/7] Verifying host Wazuh Agent (wazuh-agent)..."
if systemctl is-active --quiet wazuh-agent; then
    echo "  [OK] Host systemd service 'wazuh-agent' is active."
    STATUS_WAZUH_AGENT="PASS"
else
    echo "  [*] 'wazuh-agent' is inactive. Attempting to start..."
    systemctl start wazuh-agent 2>/dev/null || sudo systemctl start wazuh-agent 2>/dev/null || true
    if systemctl is-active --quiet wazuh-agent; then
        echo "  [OK] Successfully started 'wazuh-agent'."
        STATUS_WAZUH_AGENT="PASS"
    else
        echo "  [FAIL] 'wazuh-agent' service is not running."
        echo "  Diagnostic commands:"
        echo "    sudo systemctl status wazuh-agent --no-pager"
        echo "    sudo journalctl -u wazuh-agent -n 30 --no-pager"
        exit 1
    fi
fi

echo ""
echo "[4/7] Verifying host OSQuery daemon (osqueryd)..."
if systemctl is-active --quiet osqueryd; then
    echo "  [OK] Host systemd service 'osqueryd' is active."
    STATUS_OSQUERYD="PASS"
else
    echo "  [*] 'osqueryd' is inactive. Attempting to start..."
    systemctl start osqueryd 2>/dev/null || sudo systemctl start osqueryd 2>/dev/null || true
    if systemctl is-active --quiet osqueryd; then
        echo "  [OK] Successfully started 'osqueryd'."
        STATUS_OSQUERYD="PASS"
    else
        echo "  [FAIL] 'osqueryd' service is not running."
        echo "  Diagnostic commands:"
        echo "    sudo systemctl status osqueryd --no-pager"
        echo "    sudo journalctl -u osqueryd -n 30 --no-pager"
        exit 1
    fi
fi

# ------------------------------------------------------------------------------
# 5. Start Caldera Compose Stack
# ------------------------------------------------------------------------------
echo ""
echo "[5/7] Starting Caldera Compose stack..."
if [[ ! -f "${CALDERA_COMPOSE}" ]]; then
    echo "  [FAIL] Caldera compose file not found at: ${CALDERA_COMPOSE}"
    exit 1
fi

docker compose --project-directory "${CALDERA_DIR}" -f "${CALDERA_COMPOSE}" up -d

echo "  [*] Waiting for Caldera container to enter and remain in 'running' state..."
elapsed=0
c_status="unknown"
MAX_WAIT=60
while [[ $elapsed -lt $MAX_WAIT ]]; do
    if docker inspect -f '{{.State.Status}}' "${CALDERA_CONTAINER}" >/dev/null 2>&1; then
        c_status="$(docker inspect -f '{{.State.Status}}' "${CALDERA_CONTAINER}")"
        if [[ "$c_status" == "running" ]]; then
            break
        fi
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [[ "$c_status" == "running" ]]; then
    # Brief stabilization pause to ensure container does not crash immediately
    sleep 3
    c_status="$(docker inspect -f '{{.State.Status}}' "${CALDERA_CONTAINER}")"
    if [[ "$c_status" == "running" ]]; then
        echo "  [OK] Caldera container is running and stable."
        STATUS_CALDERA="PASS"
    else
        echo "  [FAIL] Caldera container crashed shortly after start (status: $c_status)."
        echo "  Diagnostic commands:"
        echo "    docker compose -f ${CALDERA_COMPOSE} ps"
        echo "    docker logs --tail 50 ${CALDERA_CONTAINER}"
        exit 1
    fi
else
    echo "  [FAIL] Caldera container failed to reach 'running' state (current: $c_status)."
    echo "  Diagnostic commands:"
    echo "    docker compose -f ${CALDERA_COMPOSE} ps"
    echo "    docker logs --tail 50 ${CALDERA_CONTAINER}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 6. Check Wazuh Dashboard Endpoint
# ------------------------------------------------------------------------------
echo ""
echo "[6/7] Checking Wazuh Dashboard endpoint (https://localhost:443)..."
elapsed=0
MAX_WAIT=60
dash_code=""
while [[ $elapsed -lt $MAX_WAIT ]]; do
    dash_code="$(curl -k -s -o /dev/null -w "%{http_code}" --connect-timeout 4 --max-time 6 https://localhost:443 || true)"
    if [[ "$dash_code" == "200" || "$dash_code" == "302" || "$dash_code" == "301" ]]; then
        break
    fi
    sleep 3
    elapsed=$((elapsed + 3))
done

if [[ "$dash_code" == "200" || "$dash_code" == "302" || "$dash_code" == "301" ]]; then
    echo "  [OK] Wazuh Dashboard responded with HTTP ${dash_code}."
    STATUS_DASHBOARD_EP="PASS"
else
    echo "  [FAIL] Wazuh Dashboard did not respond with expected HTTP status (got: '${dash_code:-none}')."
    echo "  Diagnostic commands:"
    echo "    curl -k -I https://localhost:443"
    echo "    docker logs --tail 50 single-node-wazuh.dashboard-1"
    exit 1
fi

# ------------------------------------------------------------------------------
# 7. Check Caldera Web Endpoint
# ------------------------------------------------------------------------------
echo ""
echo "[7/7] Checking Caldera web endpoint (http://localhost:8888)..."
elapsed=0
MAX_WAIT=45
caldera_code=""
while [[ $elapsed -lt $MAX_WAIT ]]; do
    caldera_code="$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 4 --max-time 6 http://localhost:8888 || true)"
    if [[ "$caldera_code" == "200" || "$caldera_code" == "302" || "$caldera_code" == "301" ]]; then
        break
    fi
    sleep 3
    elapsed=$((elapsed + 3))
done

if [[ "$caldera_code" == "200" || "$caldera_code" == "302" || "$caldera_code" == "301" ]]; then
    echo "  [OK] Caldera web endpoint responded with HTTP ${caldera_code}."
    STATUS_CALDERA_EP="PASS"
else
    echo "  [FAIL] Caldera web endpoint did not respond on port 8888 (got: '${caldera_code:-none}')."
    echo "  Diagnostic commands:"
    echo "    curl -I http://localhost:8888"
    echo "    docker logs --tail 50 ${CALDERA_CONTAINER}"
    exit 1
fi

print_summary
echo "[SUCCESS] All SOC Simulation Environment infrastructure components are active and verified!"
exit 0
