# SOC Simulation Environment: Research Web Dashboard

**Project:** Optimizing Cybersecurity Incident Response Actions Using a Transformer-Based Agent  
**Institution:** Department of Computer Engineering, University of Jaffna  
**Architecture:** Fine-tuned SecBERT (768d state representation) + PPO Policy Agent + Gymnasium SOC Environment  

This web application serves as a dedicated visualization, monitoring, investigation, and evaluation interface for the autonomous Security Operations Center (SOC) research platform.

---

## 1. System & Page Connectivity Matrix

In accordance with strict scientific integrity rules, the interface explicitly distinguishes verified historical benchmarks from planned live daemons:

| Route | Page Name | Primary Data Source | Connection State | Notes |
|:---|:---|:---|:---:|:---|
| `/overview` | **Overview** | FastAPI `/api/health`, `/api/evaluations` | **Active / Verified** | Displays verified system status, 10-step interactive architecture pipeline, latest test evaluation metrics, and recent activity feed. |
| `/telemetry` | **Telemetry & Incidents** | `data/processed/*.csv` (CISSM & rcATT) | **Active / Verified** | Ingests 15,390 curated incidents with search, pagination, CISA phase filters, and source filters. Wazuh/OSQuery live daemons labeled as Phase 3 Planned. |
| `/incidents/[id]` | **Incident Detail** | FastAPI `/api/incidents/{id}` | **Active / Verified** | Shows raw alert text, tokenizable prepared text, target context, and authoritative ground-truth action from `data/action_space.csv`. |
| `/decisions/[id]` | **AI Decision Inspector** | FastAPI `/api/decisions/{id}` | **Active / Verified** | 5-section inspection: Metadata, Prepared Input, Frozen 768d Embedding inspector, PPO categorical action distribution, Critic value $V(s)$, and software-simulated effect. |
| `/simulations` | **Simulations** | FastAPI `/api/simulations` | **Active / Verified** | Chronological step timelines, cumulative reward curves, JSON/CSV exports, and safe contextual bandit rollout runner. Live Caldera attacks are disabled. |
| `/evaluation` | **Policy Evaluation** | Evaluation artifacts from test split | **Active / Verified** | Controlled comparison of SecBERT (EXP_002G) vs PPO (v1.0 Core) vs PPO+ATGP (Planned). Interactive confusion matrix, per-class metrics table, failure case analysis. |
| `/research-results` | **Research Results** | `docs/experiment_index.md`, `phase1_report.md` | **Active / Verified** | Full archive of Phase 1 SecBERT ablation experiments (EXP_001 to EXP_002G), negative results (EXP_002A, EXP_002E), PPO milestone audit score (95/100). |
| `/system-health` | **System Health** | FastAPI `/api/system/status` | **Active / Verified** | Technical monitoring of Docker, Wazuh, OSQuery, Caldera (Phase 3 Planned), data pipeline (15,390 records), and model checkpoint readiness. |

---

## 2. Technology Stack

- **Frontend:** Next.js 14 (App Router), React 18, TypeScript (strict mode), Tailwind CSS, Lucide React, Recharts.
- **Backend Adapter:** FastAPI, Uvicorn, Python standard libraries (`csv`, `json`, `math`).
- **Data Layers:** `data/action_space.csv`, `data/class_weights.json`, `data/processed/train.csv`, `val.csv`, `test.csv`.

---

## 3. Quick Startup Guide

The backend adapter and frontend dashboard are independently startable and communicate over HTTP.

### Step 1: Start the Python Backend API Adapter

Open a terminal in the project root (`soc-agent-main`):

```bash
# On Windows (PowerShell):
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload

# On Ubuntu Linux:
python3 -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`. You can inspect the OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### Step 2: Start the Next.js Dashboard Frontend

Open a second terminal inside `soc-agent-main/dashboard`:

```bash
cd dashboard

# Install dependencies if not already installed:
npm install

# Start the Next.js development server:
npm run dev
```

The web dashboard will be available at `http://localhost:3000`.

### Production Build & Launch

To run the optimized production build:

```bash
cd dashboard
npm run build
npm run start
```

---

## 4. Running Verification Test Suites

### Backend Unit Tests
Verifies all 17 API endpoints, schema validation, pagination, and error responses:

```bash
# From soc-agent-main root:
python -m unittest tests/test_api_server.py
```

### Environment Unit Tests
Verifies Gymnasium SOCEnvironment reset, step, reward calculation, and seed reproducibility:

```bash
# From soc-agent-main root:
python -m unittest tests/test_environment.py
```

---

## 5. Security and Operational Boundaries

1. **Software Simulation Only:** All mitigation action effects (e.g. `isolate_host`, `kill_process`, `block_source_ip`) are software-simulated within the Gymnasium contextual bandit. The dashboard never sends destructive commands to real network adapters, host firewalls, or running processes.
2. **Zero Shell Endpoints:** The API does not expose any unrestricted shell execution or Docker daemon control endpoints.
3. **No Secrets in Frontend:** The application consumes no API tokens, database passwords, or private keys. All client communications target read-only or benchmark execution endpoints.
