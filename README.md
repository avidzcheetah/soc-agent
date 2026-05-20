# Optimizing Cybersecurity Incident Response Actions Using a Transformer-Based Agent

This repository contains the implementation of a research-based autonomous Security Operations Center (SOC) incident response system. The project leverages **SecBERT** (a Transformer-based language model specialized for cybersecurity text) to embed security alerts, and a **Deep Reinforcement Learning (DRL) agent (PPO)** to automatically select and execute optimal mitigation actions in response to security incidents.

Developed at the **Department of Computer Engineering, University of Jaffna**.

## Research Team & Supervision
* **Authors:** 
  * Witharana A.D.S. (`2022/E/008`)
  * Perera W.P.C.K. (`2022/E/173`)
* **Supervisor:** Dr. J. Jananie
* **Co-Supervisor:** Mr. Y. Pirunthapan

---

## System Overview & Architecture

The system operates in two core phases:
1. **Training Phase:**
   * **Preprocessing:** Structure raw cybersecurity incident datasets (such as CISSM and CECILIA) into clean representations.
   * **Representation Learning:** Fine-tune SecBERT to encode textual incident descriptions/alerts into dense, 768-dimensional state vectors representing the security state.
   * **Reinforcement Learning:** Train a **Proximal Policy Optimization (PPO)** RL agent inside a custom **Gymnasium** environment. The agent learns to map SecBERT state embeddings to defensive response actions (e.g., isolate host, block IP) by maximizing a reward function based on mitigation efficacy and system uptime.
2. **Simulation & Evaluation Phase:**
   * Deploy the trained agent alongside a simulated enterprise network.
   * **Adversary Simulation:** Generate automated attacks via **MITRE Caldera**.
   * **Detection & Telemetry:** Collect event logs using **Wazuh** (SIEM), **OSQuery** (Host Monitoring), and **Zeek** (Network Security Monitoring).
   * **Response Execution:** The agent interacts with the Wazuh API and local firewalls/controllers to execute containment actions in real-time.

```
                  +-----------------------------------+
                  |      Enterprise Environment       |
                  |  (Wazuh, OSQuery, Zeek, Caldera)   |
                  +-----------------+-----------------+
                                    |
                               Alert Logs
                                    v
                  +-----------------+-----------------+
                  |      Alert Preprocessing          |
                  +-----------------+-----------------+
                                    |
                            Structured Text
                                    v
                  +-----------------+-----------------+
                  |     SecBERT Embeddings (State)    |
                  +-----------------+-----------------+
                                    |
                             State Embeddings
                                    v
                  +-----------------+-----------------+
                  |         PPO RL Agent              |
                  +-----------------+-----------------+
                                    |
                                 Action
                                    v
                  +-----------------+-----------------+
                  |         Mitigation Action         |
                  | (Monitor, Isolate, Block, Reset)  |
                  +-----------------------------------+
```

---

## Repository Structure

```directory
soc-agent/
├── data/
│   ├── raw/                  # Raw datasets (e.g., CISSM cyber events, CISA alerts)
│   └── processed/            # Cleaned data for model training
├── logs/                     # TensorBoard and execution log files
├── models/                   # Saved fine-tuned SecBERT and trained PPO models
├── results/                  # Evaluation statistics, charts, and metrics
├── preprocess_cissm.py       # Preprocesses raw CISSM data into tokenizable formats
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Environment & Requirements

### Hardware Requirements
* **CPU:** Minimum 4 cores (8+ cores recommended)
* **RAM:** Minimum 16 GB (32 GB recommended)
* **Storage:** 100 GB+ SSD space
* **GPU:** Optional, but highly recommended for SecBERT fine-tuning and DRL training.

### Software Requirements
* **OS:** Ubuntu Linux (tested on 22.04 LTS)
* **Python:** 3.10 or 3.12
* **Docker:** Required for running the Wazuh and Caldera simulation containers.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/avidzcheetah/soc-agent.git
   cd soc-agent
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Execution Guide

### 1. Data Preprocessing

Place your raw cyber event datasets in `data/raw/`. The repository includes support for preprocessing the CISSM Cyber Events dataset.

To run the CISSM dataset preprocessing script:
```bash
python preprocess_cissm.py
```
This script will:
* Map dataset columns to standard formats (e.g., `event_type` -> `attack_type`).
* Filter out entries missing critical fields.
* Formulate a single textual sentence representing the event context.
* Label the correct recommended containment action based on attack effects:
  * `0`: Monitor (Default)
  * `1`: Isolate Host (Lateral movement / Ransomware spreading)
  * `2`: Block IP (DDoS / Network scanning / Exploit attempts)
  * `3`: Reset Credentials (Brute-force / Credential theft)
* Output the preprocessed data to `data/processed/cissm_processed.csv`.

---

## Development Roadmap & Next Steps

- [x] Set up repository, environments, and basic workspace.
- [x] Implement preprocessing for the CISSM Dataset (`preprocess_cissm.py`).
- [ ] Preprocess the CECILIA alert dataset.
- [ ] Implement SecBERT fine-tuning script (`train_secbert.py`).
- [ ] Build custom Gymnasium RL environment (`soc_env.py`).
- [ ] Implement and train the PPO RL Agent (`train_ppo.py`).
- [ ] Configure Wazuh, Caldera, and OSQuery simulation setups.
- [ ] Run integrated evaluation and generate performance reports.
