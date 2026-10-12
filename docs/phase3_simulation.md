# SOC Simulation Environment (`soc_env`) Infrastructure & System Analysis

This document captures the current hardware, system configuration, and active services powering the **SOC Simulation Environment (`soc_env`)**.

## System Specifications

### OS
```
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.4 LTS
Release:        24.04
Codename:       noble
```

### CPU
```
CPU(s):          8
Model name:      11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
NUMA node0:      CPU(s) 0-7
```

### RAM
```
total        used        free      shared  buff/cache   available
23Gi        5.9Gi       483Mi       849Mi        17Gi        17Gi
Swap: 8.0Gi
```

### Storage
```
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdb4       294G  102G  178G  37% /
```

### GPU
```
No NVIDIA GPU detected (CPU-based inference)
```

### Docker & Docker Compose
```
Docker version 29.1.3, build 29.1.3-0ubuntu3~24.04.2
Docker Compose version 2.40.3+ds1-0ubuntu1~24.04.1
Status: Installed and operational
```

### Project Path
```
/home/avidz/Research/soc-agent
```

---

## Active Deployed Services & Verification Status

| Service / Container | Deployment Mode | Ports / Binding | Status |
| :--- | :--- | :--- | :---: |
| **Wazuh Indexer** (`single-node-wazuh.indexer-1`) | Docker (`wazuh-docker/single-node`) | `9200/tcp` | Active (Up) |
| **Wazuh Manager** (`single-node-wazuh.manager-1`) | Docker (`wazuh-docker/single-node`) | `1514`, `1515`, `55000` | Active (Up) |
| **Wazuh Dashboard** (`single-node-wazuh.dashboard-1`) | Docker (`wazuh-docker/single-node`) | `443:5601/tcp` | Active (Up, HTTP 302) |
| **Wazuh Agent** (`wazuh-agent`) | Host systemd service | Agent ID: `001 (meow)` | Active (Enrolled) |
| **OSQuery Daemon** (`osqueryd`) | Host systemd service (v5.23.1) | BPF socket filter (`IN (2, 10)`) | Active (Running) |
| **MITRE Caldera** (`caldera`) | Docker (`caldera/`) | `8888/tcp` | Active (Up, HTTP 200) |

---

## Lifecycle Automation

The testbed lifecycle is automated via dedicated operational scripts:
- **Start & Healthcheck**: `scripts/start_soc_env.sh` (validates all 7 components, exits nonzero on failure)
- **Graceful Shutdown**: `scripts/stop_soc_env.sh` (halts containers with `docker compose stop`, preserves volumes and systemd daemons)

---

## Technical Constraints & Operational Considerations

1. **CPU Inference:** Inference for SecBERT (frozen state encoder) and PPO+ATGP policy runs on the 11th Gen i7 CPU. Single-sample forward passes take ~15–30 ms, which is well within operational tolerances for incident response decision cycles.
2. **Resource Allocation:** Total RAM usage with the Wazuh single-node stack and Caldera active remains well within the 24 GB budget (~6 GB used, ~17 GB available). The Indexer JVM heap is configured at 1 GB (`-Xms1g -Xmx1g`).
3. **Data Integrity:** Host agents (`wazuh-agent`, `osqueryd`) remain persistently installed on the host OS and send events directly to the containerized Wazuh Manager. Containers use named volumes (`wazuh-indexer-data`, `wazuh_logs`, etc.) ensuring all logs and alerts survive stop/start cycles.
