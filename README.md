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
   * **Preprocessing:** Structure raw cybersecurity incident datasets (CISSM Cyber Events and rcATT threat intelligence reports) into clean representations.
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
                  |     Mitigation Action (1 of 20)   |
                  | (Isolate, Block IP, Sinkhole ...) |
                  +-----------------------------------+
```

---

## Repository Structure

```directory
soc-agent/
├── data/
│   ├── raw/                  # Raw datasets (CISSM, rcATT, CISA playbooks)
│   │   ├── cissm_cyber_events.csv
│   │   └── rcATT/            # rcATT threat intelligence reports
│   ├── processed/            # Cleaned data for model training
│   │   ├── cissm_processed.csv
│   │   └── rcatt_processed.csv
│   └── action_space.csv      # Canonical 20-action space definitions
├── logs/                     # TensorBoard and execution log files
├── models/                   # Saved fine-tuned SecBERT and trained PPO models
├── results/                  # Evaluation statistics, charts, and metrics
├── preprocess_cissm.py       # Preprocesses raw CISSM data into tokenizable formats
├── preprocess_rcatt.py       # Preprocesses rcATT threat reports with ATT&CK mapping
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

Place your raw cyber event datasets in `data/raw/`. The repository includes preprocessing scripts for two datasets.

#### CISSM Cyber Events Dataset
```bash
python scripts/secbert/preprocess_cissm.py
```
This script will:
* Map dataset columns to standard formats (e.g., `event_type` → `attack_type`).
* Filter out entries missing critical fields.
* Formulate a single textual sentence representing the event context.
* Label the correct recommended containment action using the **20-action discrete action space** (see [Action Space](#action-space--policy-design) below).
* Output **16,728 records** to `data/processed/cissm_processed.csv`.

#### rcATT Threat Intelligence Reports Dataset

The [rcATT dataset](https://github.com/vlegoy/rcATT) contains ~1,490 real-world cyber threat intelligence reports (scraped blog posts from Cisco Talos, Palo Alto Unit 42, Accenture iDefense, etc.) with multi-label MITRE ATT&CK tactic and technique annotations.

```bash
python scripts/secbert/preprocess_rcatt.py
```
This script will:
* Clean boilerplate/navigation text from scraped web pages.
* Truncate reports to ≤2,000 characters (to fit SecBERT's 512-token input limit).
* Map 215 MITRE ATT&CK technique columns to the 20-action space using a two-tier strategy:
  * **Tier 1 — Technique-level:** Direct mapping of specific technique IDs to actions (e.g., `T1486 Data Encrypted for Impact` → `isolate_host`).
  * **Tier 2 — Tactic-level fallback:** If no specific technique match exists, the primary tactic determines the action (e.g., `TA0006 Credential Access` → `reset_credentials`).
  * When multiple techniques/tactics are present, the **highest-severity action** is selected.
* Output **1,490 records** to `data/processed/rcatt_processed.csv`.

> **Combined training data: 18,218 labeled records** across both datasets.

### 2. SecBERT Fine-Tuning (Phase 1)

To fine-tune the SecBERT model on the preprocessed data, run:
```bash
python scripts/secbert/train_secbert.py
```
This will train the model to embed cyber event descriptions and classify them into the 20 mitigation actions. The trained model checkpoint will be saved in the `models/` directory.

### 3. Deep Reinforcement Learning (PPO) Training (Phase 2)

To train the PPO agent using the frozen fine-tuned SecBERT model as a state encoder, run:
```bash
python scripts/ppo/train.py
```
This trains the PPO policy inside the custom Gymnasium environment. Checkpoints will be saved during training.

### 4. Evaluation

You can evaluate both the baseline SecBERT classification performance and the PPO agent performance:
```bash
# Evaluate baseline SecBERT
python scripts/evaluate_baseline.py

# Evaluate PPO Agent
python scripts/ppo/evaluate_ppo.py
```

---

## Action Space & Policy Design

The agent's policy outputs one of **20 discrete actions** per alert, derived from:
* **[CISA Federal Cybersecurity Incident & Vulnerability Response Playbooks](https://www.cisa.gov/sites/default/files/publications/Federal_Government_Cybersecurity_Incident_and_Vulnerability_Response_Playbooks_508C.pdf)** — containment steps 7a–7n, eradication steps 8a–8m, and recovery procedures.
* **[MITRE Cyber Analytics Repository (CAR)](https://car.mitre.org)** — 70+ validated detection analytics mapped to ATT&CK techniques.

Actions are ordered by increasing intervention severity to support proportional-response reward shaping.

### Full Action Table

| Index | Action | CISA Phase | Threat Match | SOC Implementation |
|:---:|---|---|---|---|
| 0 | `monitor` | Detection | Baseline / Low Severity / FP | Log event, close alert as non-malicious. |
| 1 | `enable_deep_logging` | Detection | Insufficient Data / Suspicious | Flip Zeek/OSQuery to verbose; enable full PCAP on subnet. |
| 2 | `create_ioc_alert` | Detection | New Indicator / CTI Match | Extract IOCs, inject into Wazuh SIEM ruleset. |
| 3 | `block_source_ip` | Containment | Scanning / C2 Beaconing / DDoS | Perimeter firewall deny rule for source IP. |
| 4 | `block_dest_ip` | Containment | Outbound C2 / Exfiltration | Egress firewall deny rule to attacker infra. |
| 5 | `dns_sinkhole` | Containment | C2 Domain / Phishing Domain | Redirect domain to sinkhole in local DNS. |
| 6 | `block_port` | Containment | Exploit on Specific Port | Close port via firewall (RDP, SMB, etc). |
| 7 | `isolate_host` | Containment | Lateral Movement / Ransomware | Drop all traffic to/from host via EDR/VLAN. |
| 8 | `kill_process` | Containment | Malicious Execution / Miner / RAT | Kill PID via OSQuery/EDR, no full isolation. |
| 9 | `quarantine_file` | Containment | Trojan / Dropper / Malware Binary | Move binary to encrypted vault; block hash. |
| 10 | `quarantine_email` | Containment | Phishing / Malicious Attachment | Purge email from mailboxes; block sender. |
| 11 | `reset_credentials` | Containment | Brute Force / Credential Dump | Force password reset, expire tokens in AD/IAM. |
| 12 | `disable_account` | Containment | Compromised Account / Priv. Esc. | Disable account entirely in AD/LDAP. |
| 13 | `remove_persistence` | Eradication | Backdoor / Autorun / Webshell | Delete sched. task, registry key, or service. |
| 14 | `restore_registry` | Eradication | Registry Tampering / Hijack Flow | Revert registry keys to known-good baseline. |
| 15 | `restore_defense_config` | Eradication | AV/Defender Disabled / Log Tamper | Re-enable security tools; restore log config. |
| 16 | `patch_vulnerability` | Eradication | Known CVE / Unpatched Service | Trigger patch deployment for exploited CVE. |
| 17 | `snapshot_forensics` | Detection | High-Severity / Evidence Needed | Capture memory dump + disk image for forensics. |
| 18 | `sandbox_redirect` | Containment | Active Adversary — Deception | Redirect adversary to honeynet for TTP study. |
| 19 | `escalate_to_human` | All Phases | Critical / High-Uncertainty | Flag to Tier-3 analyst; freeze automation. |

> The full machine-readable definition is in [`data/action_space.csv`](data/action_space.csv).

### Action-to-ATT&CK Tactic Coverage

```
                    Init.  Exec.  Persist PrivEsc Def.Ev. Cred.  Discov. Lat.Mv. Exfil.  C2     Impact
Action              Access                                Access
────────────────────────────────────────────────────────────────────────────────────────────────────────
monitor              ·      ·      ·       ·       ·       ·      ●       ·       ·       ·      ·
enable_deep_log      ·      ·      ·       ·       ●       ·      ●       ·       ·       ·      ·
create_ioc_alert     ●      ●      ●       ·       ●       ●      ·       ●       ●       ●      ·
block_source_ip      ●      ·      ·       ·       ·       ●      ·       ·       ·       ●      ·
block_dest_ip        ·      ·      ·       ·       ·       ·      ·       ·       ●       ●      ·
dns_sinkhole         ●      ·      ·       ·       ·       ·      ·       ·       ·       ●      ·
block_port           ·      ·      ·       ·       ·       ·      ·       ●       ·       ●      ·
isolate_host         ·      ·      ·       ·       ·       ·      ·       ●       ●       ●      ●
kill_process         ·      ●      ●       ·       ·       ·      ·       ·       ·       ·      ·
quarantine_file      ●      ●      ●       ·       ●       ·      ·       ·       ·       ·      ·
quarantine_email     ●      ·      ·       ·       ·       ·      ·       ·       ·       ·      ·
reset_credentials    ·      ·      ·       ·       ·       ●      ·       ●       ·       ·      ·
disable_account      ·      ·      ●       ●       ·       ●      ·       ·       ·       ·      ·
remove_persistence   ·      ·      ●       ·       ·       ·      ·       ·       ·       ·      ·
restore_registry     ·      ·      ●       ·       ●       ·      ·       ·       ·       ·      ·
restore_defense      ·      ·      ·       ·       ●       ·      ·       ·       ·       ·      ·
patch_vulnerability  ●      ·      ·       ●       ·       ·      ·       ·       ·       ·      ·
snapshot_forensics   ●      ●      ●       ●       ●       ●      ●       ●       ●       ●      ●
sandbox_redirect     ·      ·      ·       ·       ·       ·      ·       ·       ·       ●      ·
escalate_to_human    ●      ●      ●       ●       ●       ●      ●       ●       ●       ●      ●
```
`●` = Primary effectiveness &nbsp;&nbsp; `·` = Not primary target

---

## Development Roadmap & Next Steps

### Phase 1: Representation Learning (Completed)
- [x] Set up repository, environments, and basic workspace.
- [x] Implement preprocessing for the CISSM Dataset (`scripts/secbert/preprocess_cissm.py`).
- [x] Define 20-action discrete action space from CISA Playbooks + MITRE CAR.
- [x] Preprocess the rcATT threat intelligence dataset (`scripts/secbert/preprocess_rcatt.py`).
- [x] Implement SecBERT fine-tuning script (`scripts/secbert/train_secbert.py`).

### Phase 2: Deep Reinforcement Learning (Completed)
- [x] Build custom Gymnasium RL environment (`src/environment.py`).
- [x] Implement SecBERT State Encoder (`src/encoder.py`).
- [x] Implement and train the PPO RL Agent (`scripts/ppo/train.py`).
- [x] Develop Evaluation Pipeline (`scripts/evaluate_baseline.py`, `scripts/ppo/evaluate_ppo.py`).

### Phase 3: Integration (Pending)
- [ ] Configure Wazuh, Caldera, and OSQuery simulation setups.
- [ ] Connect the agent to the Ubuntu SOC Lab for live telemetry and response automation.
- [ ] Run integrated evaluation and generate performance reports.

