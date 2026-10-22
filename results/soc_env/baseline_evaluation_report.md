# SOC Simulation Environment (`soc_env`) — Baseline Performance Benchmark

## Executive Summary
This benchmark evaluates 6 candidate and baseline policies in the sequential SOC environment across 60 reproducible episodes (Seeds 1000–1059).
All policies were tested under identical scenario distributions, observation embeddings (frozen SecBERT), and transition dynamics.

---

## Comparative Benchmark Results

| Policy | Mean Return | Mean Length | Malicious Containment | Malicious Breach | Benign Clearance | False Disruption | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Uniform Random** | -0.73 ± 2.02 | 1.68 | 42.5% | 50.0% | 25.0% | 10.0% | 5.0% |
| **2. Always Monitor** | -1.47 ± 2.39 | 1.92 | 0.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| **3. Always Kill Process** | +0.90 ± 1.41 | 1.00 | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| **4. Always Escalate** | -0.10 ± 0.00 | 1.00 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **5. Rule-Based SOC Policy** | +1.90 ± 0.00 | 1.00 | 100.0% | 0.0% | 100.0% | 0.0% | 0.0% |
| **6. Pretrained PPO (Exp 4D ATGP)** | -1.37 ± 0.55 | 1.92 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |

---

## Granular Policy Breakdown

### 1. Uniform Random
- **Overall Performance:** Mean Return = `-0.73`, Mean Episode Length = `1.68` steps
- **Malicious Incidents (N=40):** Containment Rate = `42.5%`, Breach Rate = `50.0%`, Mean Return = `-1.08`
- **Benign Incidents (N=20):** Clearance Rate = `25.0%`, False Disruption Rate = `10.0%`, Unresolved Rate = `65.0%`, Mean Return = `-0.03`
- **Escalation Rate:** `5.0%`
- **Outcome Distribution:** `{"escalated": 3, "benign_unresolved": 13, "contained": 16, "breach_compromised": 20, "correctly_cleared_benign": 5, "disruptive_false_positive": 2, "contained_by_prior_isolation": 1}`

### 2. Always Monitor
- **Overall Performance:** Mean Return = `-1.47`, Mean Episode Length = `1.92` steps
- **Malicious Incidents (N=40):** Containment Rate = `0.0%`, Breach Rate = `100.0%`, Mean Return = `-3.16`
- **Benign Incidents (N=20):** Clearance Rate = `100.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `0.0%`, Mean Return = `+1.90`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"breach_compromised": 40, "correctly_cleared_benign": 20}`

### 3. Always Kill Process
- **Overall Performance:** Mean Return = `+0.90`, Mean Episode Length = `1.00` steps
- **Malicious Incidents (N=40):** Containment Rate = `100.0%`, Breach Rate = `0.0%`, Mean Return = `+1.90`
- **Benign Incidents (N=20):** Clearance Rate = `0.0%`, False Disruption Rate = `100.0%`, Unresolved Rate = `0.0%`, Mean Return = `-1.10`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"contained": 40, "disruptive_false_positive": 20}`

### 4. Always Escalate
- **Overall Performance:** Mean Return = `-0.10`, Mean Episode Length = `1.00` steps
- **Malicious Incidents (N=40):** Containment Rate = `0.0%`, Breach Rate = `0.0%`, Mean Return = `-0.10`
- **Benign Incidents (N=20):** Clearance Rate = `0.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `0.0%`, Mean Return = `-0.10`
- **Escalation Rate:** `100.0%`
- **Outcome Distribution:** `{"escalated": 60}`

### 5. Rule-Based SOC Policy
- **Overall Performance:** Mean Return = `+1.90`, Mean Episode Length = `1.00` steps
- **Malicious Incidents (N=40):** Containment Rate = `100.0%`, Breach Rate = `0.0%`, Mean Return = `+1.90`
- **Benign Incidents (N=20):** Clearance Rate = `100.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `0.0%`, Mean Return = `+1.90`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"contained": 40, "correctly_cleared_benign": 20}`

### 6. Pretrained PPO (Exp 4D ATGP)
- **Overall Performance:** Mean Return = `-1.37`, Mean Episode Length = `1.92` steps
- **Malicious Incidents (N=40):** Containment Rate = `0.0%`, Breach Rate = `100.0%`, Mean Return = `-1.76`
- **Benign Incidents (N=20):** Clearance Rate = `0.0%`, False Disruption Rate = `0.0%`, Unresolved Rate = `100.0%`, Mean Return = `-0.60`
- **Escalation Rate:** `0.0%`
- **Outcome Distribution:** `{"breach_compromised": 40, "benign_unresolved": 20}`

---

## Methodological Notes & Limitations
1. **Separation of Metrics:** Malicious containment and benign clearance are reported using their respective incident subsets as denominators. Benign clearance does not inflate threat containment.
2. **Curriculum Scope:** The environment currently contains 3 scenario templates (2 malicious, 1 benign), representing a controlled verification benchmark. A policy with heavy bias towards process termination (Action 8) scores artificially high on this small curriculum because process execution is a key vector in the default templates.
3. **Pretrained Policy Status:** The pretrained PPO policy was trained in Phase 2 as a contextual-bandit on static supervised text samples. Its sequential multi-step behavior represents zero-shot transfer and has not been fine-tuned on sequential trajectories.
