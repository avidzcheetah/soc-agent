# SOC Simulation Environment (`soc_env`) — Baseline Performance Benchmark (v3)

## Executive Summary
This benchmark evaluates 6 candidate and baseline policies in the sequential SOC environment across 60 reproducible episodes (Seeds 1000–1059).
All policies were tested under identical scenario distributions, observation embeddings (frozen SecBERT), and transition dynamics.

---

## Comparative Benchmark Results

| Policy | Mean Return | Mean Length | Malicious Containment | Malicious Breach | Benign Clearance | False Disruption | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Uniform Random** | -0.63 ± 1.67 | 1.48 | 43.8% | 46.9% | 7.1% | 10.7% | 5.0% |
| **2. Always Monitor** | -0.81 ± 2.54 | 1.60 | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| **3. Always Kill Process** | -1.37 ± 1.74 | 1.37 | 31.2% | 68.8% | 0.0% | 100.0% | 0.0% |
| **4. Always Escalate** | -0.10 ± 0.00 | 1.00 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **5. Rule-Based SOC Policy** | +0.42 ± 1.92 | 1.13 | 75.0% | 25.0% | 42.9% | 57.1% | 0.0% |
| **6. Pretrained PPO (Exp 4D ATGP)** | -1.23 ± 0.59 | 1.60 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |

---

## Granular Policy Breakdown

### 1. Uniform Random
- **Overall Performance:** Mean Return = `-0.63`, Mean Episode Length = `1.48` steps
- **Malicious Incidents (N=32):** Containment Rate = `43.8%`, Breach Rate = `46.9%`, Mean Return = `-0.76`
- **Benign Incidents (N=28):** Clearance Rate = `7.1%`, False Disruption Rate = `10.7%`, Unresolved Rate = `82.1%`, Mean Return = `-0.47`
- **Escalation Rate:** `5.0%`
- **Outcome Distribution:** `{"escalated": 3, "benign_unresolved": 23, "contained": 13, "breach_compromised": 15, "correctly_cleared_benign": 2, "disruptive_false_positive": 3, "contained_by_prior_isolation": 1}`

### 2. Always Monitor
- **Overall Performance:** Mean Return = `-0.81`, Mean Episode Length = `1.60` steps
- **Malicious Incidents (N=32):** Containment Rate = `0.0%`, Breach Rate = `100.0%`, Mean Return = `-3.19`
- **Benign Incidents (N=28):** Clearance Rate = `100.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `0.0%`, Mean Return = `+1.90`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"breach_compromised": 32, "correctly_cleared_benign": 28}`

### 3. Always Kill Process
- **Overall Performance:** Mean Return = `-1.37`, Mean Episode Length = `1.37` steps
- **Malicious Incidents (N=32):** Containment Rate = `31.2%`, Breach Rate = `68.8%`, Mean Return = `-1.61`
- **Benign Incidents (N=28):** Clearance Rate = `0.0%`, False Disruption Rate = `100.0%`, Unresolved Rate = `0.0%`, Mean Return = `-1.10`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"contained": 10, "disruptive_false_positive": 28, "breach_compromised": 22}`

### 4. Always Escalate
- **Overall Performance:** Mean Return = `-0.10`, Mean Episode Length = `1.00` steps
- **Malicious Incidents (N=32):** Containment Rate = `0.0%`, Breach Rate = `0.0%`, Mean Return = `-0.10`
- **Benign Incidents (N=28):** Clearance Rate = `0.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `0.0%`, Mean Return = `-0.10`
- **Escalation Rate:** `100.0%`
- **Outcome Distribution:** `{"escalated": 60}`

### 5. Rule-Based SOC Policy
- **Overall Performance:** Mean Return = `+0.42`, Mean Episode Length = `1.13` steps
- **Malicious Incidents (N=32):** Containment Rate = `75.0%`, Breach Rate = `25.0%`, Mean Return = `+0.62`
- **Benign Incidents (N=28):** Clearance Rate = `42.9%`, False Disruption Rate = `57.1%`, Unresolved Rate = `0.0%`, Mean Return = `+0.19`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"contained": 24, "disruptive_false_positive": 16, "correctly_cleared_benign": 12, "breach_compromised": 8}`

### 6. Pretrained PPO (Exp 4D ATGP)
- **Overall Performance:** Mean Return = `-1.23`, Mean Episode Length = `1.60` steps
- **Malicious Incidents (N=32):** Containment Rate = `0.0%`, Breach Rate = `100.0%`, Mean Return = `-1.79`
- **Benign Incidents (N=28):** Clearance Rate = `0.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `100.0%`, Mean Return = `-0.60`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"breach_compromised": 32, "benign_unresolved": 28}`

---

## Methodological Notes & Limitations
1. **Separation of Metrics:** Malicious containment and benign clearance are reported using their respective incident subsets as denominators. Benign clearance does not inflate threat containment.
2. **Expanded 7-Scenario Curriculum:** The environment now contains 7 diverse scenario archetypes (4 malicious, 3 benign). It introduces credential theft (`scen_credential_lateral`) and auto-restarting web RCE tunnels (`scen_network_rce_tunnel`) where process termination alone is ineffective, as well as administrative false-positive challenges (`scen_benign_devops_tooling`, `scen_benign_security_audit`) where naive process killing or host isolation causes disruptive false positives.
3. **Breakdown of Simplistic Heuristics:** The expanded curriculum successfully eliminates the single-action bias: `Always Kill Process` plummets from 100.0% containment (+0.90 return) in v2 to 31.2% containment, 68.8% breach rate, and -1.37 return in v3.
4. **Realistic Rule-Based Baseline:** Operating strictly on observable telemetry, the Rule-Based SOC policy achieves 75.0% malicious containment and 42.9% benign clearance (+0.42 return), realistically struggling with administrative LOLBins (57.1% false disruption) and unmodeled SSH credential abuse (25.0% breach).
5. **Pretrained Policy Status:** The pretrained PPO policy from Phase 2 continues to collapse onto Action 17 (snapshot_forensics) across multi-stage alerts, confirming that zero-shot transfer fails and motivating sequential PPO fine-tuning.
