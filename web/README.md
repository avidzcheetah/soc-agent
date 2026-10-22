# SOC Simulation Environment & Web Control Panel

**OPTIMIZING CYBERSECURITY INCIDENT RESPONSE ACTIONS USING A TRANSFORMER-BASED AGENT**

This directory (`web/`) contains the full-stack visualization, monitoring, investigation, and evaluation interface for the autonomous SOC incident response research platform developed at the **Department of Computer Engineering, University of Jaffna**.

---

## Architecture Overview

```
                      +-----------------------------------+
                      |      Next.js 14 Web Frontend      |
                      |   (React Flow, Recharts, Tailwind)|
                      +-----------------+-----------------+
                                        | HTTP / SSE Stream
                                        v
                      +-----------------+-----------------+
                      |     FastAPI Simulation API        |
                      |  (Real-Time Engine & SSE Adapter) |
                      +-----------------+-----------------+
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
  +--------------+--------------+               +--------------+--------------+
  |    Gymnasium SOC Replay     |               |    Frozen SecBERT Encoder   |
  |  (Contextual Bandit Engine) |               |  (768-dim State Vectors)    |
  +--------------+--------------+               +--------------+--------------+
                 |                                             |
                 +----------------------+----------------------+
                                        |
                                        v
                         +--------------+--------------+
                         |      PPO Incident Policy    |
                         | (20-Class Action Selection) |
                         +-----------------------------+
```

---

## Directory Structure

```
web/
├── api/                             # FastAPI simulation backend & adapters
│   ├── server.py                    # REST endpoints & SSE event broadcaster
│   ├── simulation_engine.py         # Gymnasium replay engine, scenario filter, persistence
│   ├── requirements.txt             # Python backend dependencies
│   └── __init__.py
├── dashboard/                       # Next.js 14 frontend application
│   ├── app/                         # App router (16 interactive routes)
│   │   ├── results/                 # Evaluation reports & PPO vs PPO+ATGP comparison
│   │   ├── simulations/             # Live execution panel & trajectory inspector
│   │   ├── architecture/            # 8-stage React Flow live architecture
│   │   ├── telemetry/               # Wazuh, OSQuery, Caldera telemetry logs
│   │   ├── action-space/            # 20 CISA mitigation actions mapping
│   │   └── decisions/               # Single-incident deep dive & evaluation
│   ├── components/                  # UI components, React Flow nodes, charts
│   ├── lib/                         # API client & formatting utilities
│   ├── types/                       # TypeScript definitions
│   └── package.json                 # Frontend dependencies
├── tests/                           # Integration & verification test suite (51 tests)
│   ├── test_api_server.py           # API contracts, health, telemetry, scenarios
│   ├── test_simulation_engine.py    # Seed reproducibility, reward logic (+1.0/-1.0)
│   ├── test_architecture_visualization.py # 8-stage pipeline, sequence tracker
│   └── test_results_and_comparison.py # Persistence, delta metrics, JSON/CSV export
├── data/
│   └── simulations/                 # Persisted benchmark comparison runs (JSON)
└── README.md                        # This documentation
```

---

## Quickstart Guide

### Prerequisites
- Python 3.10+ with project dependencies installed
- Node.js 18+ and `npm`

### Step 1: Start the FastAPI Backend
From the repository root or `web/api`:
```bash
# Option A: From repository root
python -m uvicorn web.api.server:app --host 127.0.0.1 --port 8000 --reload

# Option B: From inside web/api
cd web/api
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```
Verify the API is running at:
- Swagger Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Step 2: Start the Next.js Frontend
In a separate terminal:
```bash
cd web/dashboard
npm install
npm run dev
```
Open your browser to:
- **Web Dashboard**: [http://localhost:3000](http://localhost:3000)

---

## Core Features

1. **Live 8-Stage Architecture Pipeline:**
   Visualizes the live flow across: Benchmark Incidents &rarr; SOC Environment &rarr; Preprocessing &rarr; Frozen SecBERT Encoder &rarr; PPO Policy &rarr; Simulated Response &rarr; Reward Calculator &rarr; Event Log.
2. **Three Validated Attack Scenarios:**
   - Host Compromise & Ransomware Containment (`scenario-ransomware`)
   - Adversary C2 Beaconing & Lateral Movement (`scenario-c2`)
   - Exploitation of Vulnerable Service & Persistence (`scenario-vulnerability`)
3. **PPO vs. PPO+ATGP Policy Comparison:**
   Rigorous initial condition matching (identical scenario, seed, and step limits) with compatibility rating badges (`compatible`, `partially_compatible`, `incompatible`), metric delta cards ($\Delta$ Accuracy, $\Delta$ Return), step-by-step cumulative reward trajectory charts, and action frequency distributions.
4. **Held-Out Classification Evaluation (1,539 Samples):**
   Full test split report: 89.93% Accuracy, 0.7121 Macro F1, 0.9454 Weighted F1, 0.9364 MCC, 96.95% Top-2 Accuracy, confusion matrix heatmap, and 20-action per-class breakdown.
5. **Authentic Data Export:**
   Full chronological trajectory export to sanitized JSON and tabular CSV directly from the dashboard.

---

## Running Automated Tests

Run the full 51-test suite verifying API endpoints, simulation dynamics, architecture visualization, and policy comparison:

```bash
# Run all web tests
python -m unittest discover web/tests -v
```
All 51 tests execute in ~22 seconds with 100% pass rate.
