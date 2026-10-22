# Policy Diagnostics Report: Sequential PPO Experiment 002

**Experiment ID:** EXP_SEQ_002  
**Date:** 2026-10-12  
**Investigation Type:** Read-Only Diagnostic Audit  
**Artifact:** `results/soc_env/exp002/policy_diagnostics.md`  

---

## Executive Summary

Experiment 002 tested whether penalizing human escalation (`escalation_cost=1.0`) would eliminate the `always_escalate` degenerate policy observed in Experiment 001 and enable context-aware incident response. While escalation dropped to 0%, Exp002 converged to a different degenerate policy: **`always_isolate_host` (Action 7)**. In comparative evaluation across 60 benchmark episodes, Exp002 achieved:
- **100% Malicious Containment** (0% compromise/breach rate)
- **0% Benign Clearance** (100% disruptive false positive rate)
- **Mean Return:** $-0.09 \pm 1.02$

This diagnostic investigation evaluated the entire telemetry-to-action pipeline without modifying any source code, checkpoints, reward weights, or scenario definitions.

### Core Conclusion
The `always_isolate_host` collapse in Experiment 002 is driven by a **fatal synergy between representation anisotropy and asymmetric reward incentives**:
1. **Representation Anisotropy (The Cone Effect):** The frozen SecBERT encoder outputs 768-dimensional pooler embeddings that are compressed into an extremely narrow cone. The pairwise cosine similarities across all seven scenarios range between **0.9791 and 0.9954**. The inter-class cosine similarity between malicious and benign alerts (mean **0.9893**) is virtually indistinguishable from intra-malicious similarity (mean **0.9901**) and intra-benign similarity (mean **0.9855**).
2. **Reward Asymmetry:** Under the environment mechanics, network isolation persistently neutralizes all subsequent stages of malicious attacks (preventing a catastrophic $-3.00$ breach penalty and yielding $+0.79$ net malicious return), while incurring only a $-1.00$ false disruption penalty on benign incidents. Because the actor cannot easily distinguish malicious from benign embeddings, the expected return of unconditional isolation ($\approx -0.09$) overwhelmingly dominates the expected return of unconditional monitoring ($\approx -0.81$) or random guessing ($\approx -0.68$).
3. **Policy-Learning Trajectory:** PPO began initialized at Action 17 (`snapshot_forensics`) inherited from the pretrained checkpoint. As exploration increased entropy to a peak of $1.30$ at Iteration 21, PPO sampled Action 7 (`isolate_host`). By **Iteration 22**, Action 7 overtook Action 17, and by **Iteration 25**, the actor permanently collapsed onto Action 7 with over 75% probability across every scenario.

---

## 1. Observation Generation Pipeline Audit

The exact pipeline from simulated raw telemetry to action selection operates as follows:

```
[Incident Event / Wazuh & OSQuery Telemetry]
               │
               ▼
   IncidentRecord (Dataclass)
               │
               ▼
 IncidentFormatter.format_incident_text()
               │
               ▼
  Standardized Detection Text String
               │
               ▼
 SecBERTStateEncoder (BertModel, frozen 83.45M params)
   - BertTokenizer (vocab=52000, max_length=512)
   - 6 Transformer Layers (hidden_dim=768, heads=12)
   - BertPooler: pooler_output = tanh(dense(CLS))
               │
               ▼
 768-Dimensional Observation Vector s ∈ ℝ⁷⁶⁸
   - L2 normalization: DISABLED (norm ≈ 13.6)
               │
               ▼
   PPOAgent Policy Network (Actor MLP)
   - Linear(768 -> 256) + Tanh
   - Linear(256 -> 128) + Tanh
   - Linear(128 -> 20)
               │
               ▼
 Softmax Distribution over 20 Discrete Actions
```

### Numerical Validity of Embeddings
Across all seven scenarios at Stage 0:
- **Shape:** Exactly `(768,)` 1D float32 tensor
- **Numerical Validity:** 0 NaNs, 0 Infs across all dimensions
- **Vector Norms:** Consistent between $13.4355$ and $13.6659$
- **Dynamic Range:** Minimum values in $[-0.9535, -0.9326]$, maximum values in $[0.9210, 0.9447]$

---

## 2. Scenario-by-Scenario Policy & Actor Probability Analysis

We evaluated the initial state (Stage 0) of all seven benchmark scenarios using the trained Exp002 actor checkpoint (`models/ppo_sequential_exp002/best_sequential_policy.pt`), alongside the Exp001 actor, the Pretrained PPO actor, and the fine-tuned SecBERT classification head.

| Scenario ID | Ground-Truth Type | Exp002 Predicted Action | $P(\text{isolate\_host})$ | $P(\text{monitor})$ | $P(\text{escalate})$ | Critic $V(s)$ | Top 5 Action Probabilities (Exp002) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `scen_c2_exfil` | Malicious | **7 (`isolate_host`)** | **0.7687** | 0.00130 | 0.00275 | -0.0552 | `isolate_host`: 76.9%, `snapshot_forensics`: 13.5%, `reset_credentials`: 6.1%, `block_source_ip`: 0.4%, `dns_sinkhole`: 0.4% |
| `scen_privesc_persist` | Malicious | **7 (`isolate_host`)** | **0.7574** | 0.00137 | 0.00296 | -0.0617 | `isolate_host`: 75.7%, `snapshot_forensics`: 14.2%, `reset_credentials`: 6.3%, `block_source_ip`: 0.5%, `dns_sinkhole`: 0.4% |
| `scen_credential_lateral` | Malicious | **7 (`isolate_host`)** | **0.7634** | 0.00127 | 0.00274 | -0.0601 | `isolate_host`: 76.3%, `snapshot_forensics`: 13.9%, `reset_credentials`: 6.2%, `block_source_ip`: 0.5%, `dns_sinkhole`: 0.4% |
| `scen_network_rce_tunnel` | Malicious | **7 (`isolate_host`)** | **0.7886** | 0.00118 | 0.00243 | -0.0591 | `isolate_host`: 78.9%, `snapshot_forensics`: 12.2%, `reset_credentials`: 5.7%, `block_source_ip`: 0.4%, `dns_sinkhole`: 0.4% |
| `scen_benign_backup` | Benign | **7 (`isolate_host`)** | **0.7755** | 0.00124 | 0.00258 | -0.0558 | `isolate_host`: 77.6%, `snapshot_forensics`: 13.1%, `reset_credentials`: 6.0%, `block_source_ip`: 0.4%, `dns_sinkhole`: 0.4% |
| `scen_benign_devops_tooling` | Benign | **7 (`isolate_host`)** | **0.7747** | 0.00125 | 0.00264 | -0.0547 | `isolate_host`: 77.5%, `snapshot_forensics`: 13.1%, `reset_credentials`: 6.0%, `block_source_ip`: 0.4%, `dns_sinkhole`: 0.4% |
| `scen_benign_security_audit` | Benign | **7 (`isolate_host`)** | **0.7558** | 0.00152 | 0.00310 | -0.0542 | `isolate_host`: 75.6%, `snapshot_forensics`: 13.9%, `reset_credentials`: 6.5%, `block_source_ip`: 0.5%, `dns_sinkhole`: 0.4% |

### Key Observations on Actor Output:
1. **Total Policy Invariance:** The Exp002 actor produces **no meaningful differences in action distributions** across scenarios. The predicted action is unconditionally Action 7 (`isolate_host`) for all 7 scenarios.
2. **Probability Concentration:** The probability assigned to `isolate_host` is strictly bounded between **$0.7558$ and $0.7886$** (a spread of only $3.28\%$).
3. **Top 5 Ranking Invariance:** Across all seven scenarios, the top 5 action rankings are completely identical:
   - Rank 1: `isolate_host` ($\approx 75.6\% - 78.9\%$)
   - Rank 2: `snapshot_forensics` ($\approx 12.2\% - 14.2\%$)
   - Rank 3: `reset_credentials` ($\approx 5.7\% - 6.5\%$)
   - Rank 4: `block_source_ip` ($\approx 0.4\% - 0.5\%$)
   - Rank 5: `dns_sinkhole` ($\approx 0.35\% - 0.43\%$)
4. **Near-Zero Clearance Probability:** The probability assigned to `monitor` (the necessary action to clear benign incidents without false disruption) is depressed to **$\approx 0.0012 - 0.0015$** ($<0.15\%$).
5. **Suppressed Escalation:** Escalation probability is depressed to **$\approx 0.0024 - 0.0031$** ($<0.31\%$), confirming the penalty successfully suppressed escalation.
6. **Flat Critic Value Function:** The critic network predicts nearly identical state values across all seven scenarios, ranging between $-0.0542$ and $-0.0617$.

---

## 3. Representation Separability Analysis

To determine whether the representations provide a discriminatory signal before RL policy optimization, we analyzed the pairwise geometric relationships of the Stage 0 768-dimensional embeddings.

### Pairwise Cosine Similarity Matrix

| Scenario | `scen_c2_exfil` | `scen_privesc` | `scen_backup` | `scen_lateral` | `scen_rce_tunnel` | `scen_devops` | `scen_sec_audit` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`scen_c2_exfil`** (Mal) | 1.0000 | 0.9879 | 0.9928 | 0.9888 | 0.9933 | 0.9954 | 0.9831 |
| **`scen_privesc_persist`** (Mal) | 0.9879 | 1.0000 | 0.9850 | 0.9899 | 0.9886 | 0.9885 | 0.9857 |
| **`scen_benign_backup`** (Ben) | 0.9928 | 0.9850 | 1.0000 | 0.9882 | 0.9936 | 0.9934 | 0.9791 |
| **`scen_credential_lateral`** (Mal) | 0.9888 | 0.9899 | 0.9882 | 1.0000 | 0.9922 | 0.9897 | 0.9901 |
| **`scen_network_rce_tunnel`** (Mal) | 0.9933 | 0.9886 | 0.9936 | 0.9922 | 1.0000 | 0.9942 | 0.9849 |
| **`scen_benign_devops`** (Ben) | 0.9954 | 0.9885 | 0.9934 | 0.9897 | 0.9942 | 1.0000 | 0.9838 |
| **`scen_benign_sec_audit`** (Ben) | 0.9831 | 0.9857 | 0.9791 | 0.9901 | 0.9849 | 0.9838 | 1.0000 |

### Pairwise Euclidean Distance Matrix

| Scenario | `scen_c2_exfil` | `scen_privesc` | `scen_backup` | `scen_lateral` | `scen_rce_tunnel` | `scen_devops` | `scen_sec_audit` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`scen_c2_exfil`** (Mal) | 0.0000 | 2.1133 | 1.6299 | 2.0386 | 1.5709 | 1.3141 | 2.4881 |
| **`scen_privesc_persist`** (Mal) | 2.1133 | 0.0000 | 2.3598 | 1.9297 | 2.0546 | 2.0692 | 2.2900 |
| **`scen_benign_backup`** (Ben) | 1.6299 | 2.3598 | 0.0000 | 2.0919 | 1.5402 | 1.5665 | 2.7710 |
| **`scen_credential_lateral`** (Mal) | 2.0386 | 1.9297 | 2.0919 | 0.0000 | 1.7032 | 1.9587 | 1.9141 |
| **`scen_network_rce_tunnel`** (Mal) | 1.5709 | 2.0546 | 1.5402 | 1.7032 | 0.0000 | 1.4744 | 2.3583 |
| **`scen_benign_devops`** (Ben) | 1.3141 | 2.0692 | 1.5665 | 1.9587 | 1.4744 | 0.0000 | 2.4495 |
| **`scen_benign_sec_audit`** (Ben) | 2.4881 | 2.2900 | 2.7710 | 1.9141 | 2.3583 | 2.4495 | 0.0000 |

### Group Descriptive Statistics

| Relationship | Pair Count | Cosine Mean $\pm$ Std | Cosine [Min, Max] | Euclidean Mean $\pm$ Std | Euclidean [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intra-Malicious** | 6 pairs | $0.9901 \pm 0.0020$ | $[0.9879, 0.9933]$ | $1.9017 \pm 0.1985$ | $[1.5709, 2.1133]$ |
| **Intra-Benign** | 3 pairs | $0.9855 \pm 0.0059$ | $[0.9791, 0.9934]$ | $2.2623 \pm 0.5092$ | $[1.5665, 2.7710]$ |
| **Inter-Class (Mal vs Ben)** | 12 pairs | $0.9893 \pm 0.0039$ | $[0.9831, 0.9954]$ | $1.9574 \pm 0.3737$ | $[1.3141, 2.4881]$ |

### Representation Diagnostic Findings:
- **Severe Representation Anisotropy:** The embeddings occupy an extremely narrow angular cone. The mean cosine similarity across any two incident alert texts is $\approx 0.99$.
- **Lack of Natural Cluster Separability:** The inter-class cosine similarity ($0.9893$) falls directly between intra-malicious ($0.9901$) and intra-benign ($0.9855$). Notice that `scen_c2_exfil` (Malicious) has a higher cosine similarity to `scen_benign_devops_tooling` ($0.9954$) than it does to `scen_privesc_persist` ($0.9879$).
- **Euclidean Overlap:** The minimum inter-class distance ($1.3141$, between C2 exfil and benign DevOps tooling) is smaller than any intra-malicious distance (minimum $1.5709$).
- **Consequence for MLP:** An MLP receiving inputs with standard deviation $\approx 0.49$ but pairwise vector differences of only $1.3 - 2.0$ against an overall norm of $13.6$ faces an extremely low signal-to-noise ratio in separating these classes without feature centering, whitening, or encoder fine-tuning.

---

## 4. SecBERT Classification Head Inspection

We evaluated the fine-tuned sequence classification head (`AutoModelForSequenceClassification.from_pretrained('models/secbert_finetuned')`) directly on the Stage 0 texts of all seven scenarios.

| Scenario ID | Scenario Nature | Classifier Predicted Action | Probability | Top 3 Classifier Predictions |
| :--- | :---: | :---: | :---: | :--- |
| `scen_c2_exfil` | Malicious | **17 (`snapshot_forensics`)** | 0.9169 | `snapshot_forensics`: 91.7%, `enable_deep_logging`: 0.48%, `quarantine_file`: 0.47% |
| `scen_privesc_persist` | Malicious | **17 (`snapshot_forensics`)** | 0.9134 | `snapshot_forensics`: 91.3%, `enable_deep_logging`: 0.58%, `block_port`: 0.54% |
| `scen_credential_lateral` | Malicious | **17 (`snapshot_forensics`)** | 0.9177 | `snapshot_forensics`: 91.8%, `enable_deep_logging`: 0.53%, `quarantine_file`: 0.46% |
| `scen_network_rce_tunnel` | Malicious | **17 (`snapshot_forensics`)** | 0.9171 | `snapshot_forensics`: 91.7%, `quarantine_file`: 0.49%, `isolate_host`: 0.46% |
| `scen_benign_backup` | Benign | **17 (`snapshot_forensics`)** | 0.9181 | `snapshot_forensics`: 91.8%, `quarantine_file`: 0.47%, `restore_registry`: 0.45% |
| `scen_benign_devops_tooling` | Benign | **17 (`snapshot_forensics`)** | 0.9178 | `snapshot_forensics`: 91.8%, `restore_registry`: 0.49%, `enable_deep_logging`: 0.49% |
| `scen_benign_security_audit` | Benign | **17 (`snapshot_forensics`)** | 0.9111 | `snapshot_forensics`: 91.1%, `enable_deep_logging`: 0.67%, `block_port`: 0.53% |

### Critical Context on Label Discrepancy:
1. **Training Objective vs Simulation Objective:**
   - The SecBERT classification head was fine-tuned in Phase 1 and Phase 2 on static, single-event security logs mapped to 20 discrete remediation classes derived from MITRE ATT&CK techniques.
   - In that supervised dataset, any alert containing EDR memory detection, suspicious process execution, or credential access was heavily mapped to evidence preservation (`snapshot_forensics`) through targeted synthetic augmentation (Experiment 002D in Phase 2 added targeted samples to bolster rare classes).
2. **Zero-Shot Domain Mismatch:**
   - The supervised classifier was trained on single-step labels, whereas the sequential simulation models multi-stage progression where actions have downstream operational impacts.
   - In particular, the supervised dataset contained **no benign operational classes** or explicit "clear without action" labels; benign scenarios in the simulation are expected to be met with `monitor` (Action 0), but the classifier assigns $P(\text{monitor}) < 0.001$.
3. **Implication for PPO Initialization:**
   - The pretrained PPO policy was initialized from this exact classifier representation. Consequently, both Exp001 and Exp002 started training with a massive prior bias toward Action 17 (`snapshot_forensics`, $>92\%$).

---

## 5. Experiment 002 Training Dynamics Audit

Training logs (`results/soc_env/exp002/training_logs.json`) contain 50 iterations (64 rollout steps per iteration, total 3,200 training steps).

### Chronological Learning Progression

```
Iter  1 - 15: Pretrained Anchor (Action 17 Dominated)
              - Policy dominated by Action 17 (55-62 actions per 64 steps)
              - Rollout return poor (-1.43 to -1.01)
              - Value loss drops from 2.93 to 0.40
              - Entropy starts at 0.47, bottoms at 0.20 (Iter 8), then begins rising

Iter 16 - 21: Entropy Surge & Active Exploration
              - Entropy climbs rapidly: 0.89 (Iter 16) -> 1.05 (Iter 17) -> 1.30 (Iter 20-21)
              - Value loss spikes to 2.79 as diverse action rewards are encountered
              - Action 7 sampling increases: 6 (Iter 16) -> 12 (Iter 19) -> 21 (Iter 20)

Iter 22:      CRITICAL TIPPING POINT: Action 7 Takes the Lead
              - Action 7: 32 counts vs Action 17: 22 counts
              - Rollout mean return improves from -1.11 to -0.498

Iter 24 - 25: Dominance & Checkpoint Selection
              - Iter 24: Rollout return spikes to +0.037 (Action 7: 46 counts)
              - Iter 25: Evaluation return reaches -0.370 (Saved as best_sequential_policy.pt)

Iter 26 - 50: Lock-in & Consolidation
              - Action 7 maintains 82-86% of all rollout actions (53-55 per 64 steps)
              - Policy entropy gradually stabilizes between 0.45 and 0.93
              - All validation evaluations at Iter 30, 35, 40, 45, 50 yield identical -0.370 return
```

### Measured Iteration Metrics

| Iteration | Rollout Mean Return | Policy Loss | Value Loss | Policy Entropy | Dominant Rollout Actions | Eval Mean Return |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **1** | -1.100 | -0.0117 | 2.9299 | 0.4734 | Act 17: 57, Act 7: 1, Act 18: 1 | — |
| **5** | -1.429 | -0.0090 | 0.4005 | 0.3131 | Act 17: 61, Act 10: 1, Act 6: 1 | -1.075 |
| **8** | -1.129 | -0.0055 | 0.3733 | **0.2057** (Min) | Act 17: 59, Act 19: 2, Act 0: 1 | — |
| **15** | -1.015 | -0.0116 | 0.7869 | 0.6578 | Act 17: 56, Act 7: 3, Act 10: 1 | -1.075 |
| **18** | -1.115 | -0.0101 | 1.2353 | 1.1757 | Act 17: 48, Act 7: 4, Act 19: 3 | — |
| **20** | -0.331 | -0.0229 | 2.7940 | 1.2919 | Act 17: 35, Act 7: 21, Act 3: 2 | -1.075 |
| **21** | -0.432 | -0.0194 | 2.5821 | **1.2994** (Max) | Act 17: 37, Act 7: 15, Act 19: 4 | — |
| **22** | **-0.498** | -0.0181 | 2.0312 | 1.1507 | **Act 7: 32**, Act 17: 22, Act 19: 2 | — |
| **24** | **+0.037** | -0.0098 | 2.5345 | 0.8819 | **Act 7: 46**, Act 17: 11, Act 11: 4 | — |
| **25** | -0.278 | -0.0132 | 2.4928 | 0.8476 | **Act 7: 48**, Act 17: 10, Act 11: 4 | **-0.370** (Best) |
| **30** | -0.200 | -0.0069 | 2.1138 | 0.5361 | **Act 7: 55**, Act 11: 4, Act 17: 4 | -0.370 |
| **40** | -0.140 | -0.0148 | 1.0523 | 0.6080 | **Act 7: 54**, Act 17: 5, Act 11: 4 | -0.370 |
| **50** | -0.092 | -0.0073 | 0.7710 | 0.9322 | **Act 7: 53**, Act 11: 5, Act 17: 3 | -0.370 |

---

## 6. Evaluation Methodology & Reward Function Audit

A critical finding in auditing previous reports is the risk of **cross-reward evaluation distortion**.

### Reward Parameter Calibration Across Experiments

| Parameter | Exp001 Setting | Exp002 Setting | Operational Meaning |
| :--- | :---: | :---: | :--- |
| `step_penalty` | -0.10 | -0.10 | Incurred on every action taken |
| `effective_containment` | +1.00 | +1.00 | Awarded when an attack stage is neutralized |
| `benign_clearance` | +1.00 | +1.00 | Awarded when benign incident is cleared with `monitor` |
| `malicious_breach` | -3.00 | -3.00 | Incurred if attack reaches final stage uncontained |
| `false_disruption` | -1.00 | -1.00 | Incurred if disruptive action applied to benign incident |
| **`escalation_cost`** | **0.00** | **1.00** | Additional penalty when invoking human escalation (Act 19) |

### The Exp001 vs Exp002 Return Comparison Fallacy

In Exp001, human escalation incurred only the `-0.10` step penalty because `escalation_cost=0.0`. Immediate escalation terminated the episode instantly, yielding an episodic return of:
$$R_{\text{esc}}(\text{Exp001}) = -0.10 + 0.00 = -0.10 \pm 0.00$$

When the Exp001 checkpoint is re-evaluated under Exp002's reward settings (`escalation_cost=1.0`), its return is:
$$R_{\text{esc}}(\text{Exp002 settings}) = -0.10 + (-1.00) = -1.10 \pm 0.00$$

Comparing Exp001's published return ($-0.10$) to Exp002's return ($-0.09$) as if they used the same scale is **invalid**. Under identical reward conditions (`escalation_cost=1.0`):
- **Exp001 Policy (Always Escalate):** Mean Return = **$-1.10 \pm 0.00$**
- **Exp002 Policy (Always Isolate):** Mean Return = **$-0.09 \pm 1.02$**

### Why Always Isolate Host Outperforms Other Fixed Policies

Under the 7-scenario benchmark (4 malicious, 3 benign, with evaluation sampling 32 malicious and 28 benign episodes), the expected cumulative returns of fixed policies under `escalation_cost=1.0` are:

| Policy | Return on Malicious | Return on Benign | Overall Return (32 Mal / 28 Ben) | Failure Mode |
| :--- | :---: | :---: | :---: | :--- |
| **Always Monitor (Act 0)** | $-3.18$ | **+0.90** | $-0.81$ | Catastrophic breach on all attacks |
| **Always Kill Process (Act 8)** | $-1.55$ | $-1.10$ | $-1.37$ | Ineffective against SSH/tunneling; disrupts benign |
| **Always Escalate (Act 19)** | $-1.10$ | $-1.10$ | $-1.10$ | Incurs escalation penalty on every episode |
| **Uniform Random** | $-0.32$ | $-1.08$ | $-0.68$ | Unpredictable, high breach rate |
| **Rule-Based SOC Policy** | **+1.56** | $-1.10$ | **+0.42** | Distinguishes observable telemetry (host/PID/rule) |
| **Always Isolate Host (Act 7)** | **+0.79** | $-1.10$ | **-0.09** | **Neutralizes all attacks via network cut; disrupts benign** |

Because Action 7 persistently sets `isolated_hosts`, it neutralizes network communications for C2 beaconing, SSH lateral movement, and reverse TCP tunnels. The environment awards $+1.00$ containment ($-0.10$ step penalty $= +0.79$ avg return). On benign incidents, it incurs a $-1.00$ disruption penalty ($-0.10$ step penalty $= -1.10$ return).
Weighted average across 60 episodes:
$$\mathbb{E}[R] = \frac{32 \times (+0.79) + 28 \times (-1.10)}{60} = \frac{25.28 - 30.80}{60} = -0.092$$

To an optimizer unable to sharply differentiate benign from malicious embeddings, **`always_isolate_host` is the mathematical optimum among all single-action policies**.

---

## 7. Comparative Policy Analysis

| Evaluation Metric | Pretrained PPO | Exp001 Sequential PPO | Exp002 Sequential PPO | Rule-Based Heuristic |
| :--- | :---: | :---: | :---: | :---: |
| **Dominant Action** | 17 (`snapshot_forensics`) | 19 (`escalate_to_human`) | 7 (`isolate_host`) | Context-driven (0, 4, 7, 8, 11) |
| **Action Diversity** | Degenerate (100% Act 17) | Degenerate (100% Act 19) | Degenerate (100% Act 7) | Dynamic (5 distinct actions) |
| **Mean Return** (`cost=1.0`) | $-1.23 \pm 0.59$ | $-1.10 \pm 0.00$ | **$-0.09 \pm 1.02$** | **$+0.42 \pm 1.92$** |
| **Malicious Containment** | 0.0% | 0.0% | **100.0%** | **75.0%** |
| **Malicious Breach Rate** | 68.8% | 0.0% | **0.0%** | **25.0%** |
| **Benign Clearance** | 0.0% | 0.0% | **0.0%** | **42.9%** |
| **False Disruption Rate** | 0.0% | 0.0% | **100.0%** | **57.1%** |
| **Escalation Rate** | 0.0% | 100.0% | **0.0%** | 0.0% |
| **Mean Episode Length** | 1.60 | 1.00 | 1.23 | 1.13 |

---

## 8. Root Cause Synthesis: Why the Policy Collapsed

The diagnostic findings conclusively identify three interacting factors that created the collapse:

```
┌────────────────────────────────────────────────────────┐
│  Factor 1: Representation Anisotropy (The Bottleneck)  │
│  - Pairwise cosine similarity > 0.985 for all alerts   │
│  - Inter-class similarity indistinguishable from intra │
│  - Raw pooler output CLS vector clusters in a cone     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Factor 2: Asymmetric Reward Landscape                │
│  - Malicious Breach penalty: -3.00                     │
│  - Benign False Disruption penalty: -1.00              │
│  - Host Isolation neutralizes all multi-stage attacks  │
│  - Return(Isolate) = -0.09  >>  Return(Monitor) = -0.81│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│  Factor 3: PPO Optimization & Value Learning           │
│  - Frozen encoder prevents representation adaptation   │
│  - Entropy rise at Iter 16-21 exposes Action 7         │
│  - Value function collapses to scalar baseline (-0.06) │
│  - Tipping point at Iteration 22 locks into Action 7   │
└────────────────────────────────────────────────────────┘
```

1. **Representation Limitations (Primary Enabler):**
   The frozen SecBERT encoder pooler output does not provide a linear or easily separable boundary between malicious and benign alert texts in the 768-dimensional observation space. Because the encoder weights are frozen during sequential PPO training, the actor network cannot reshape the state representations to pull malicious and benign alerts apart.

2. **Reward Incentives (Objective Driver):**
   Given that the agent cannot reliably distinguish benign from malicious alerts, it is forced to choose between betting on "benign" (monitoring) or betting on "malicious" (disrupting/isolating). Because the penalty for missing a malicious incident ($-3.00$ breach penalty) is three times larger than the penalty for falsely disrupting a benign incident ($-1.00$), and because isolation provides 100% containment across diverse attack types, risk-averse expected utility maximization mathematically favors aggressive containment over passive monitoring.

3. **Policy-Learning Behavior (Convergence Mechanism):**
   The policy did not oscillate; it executed a classic exploration-exploitation shift. It broke free from the pretrained prior (`snapshot_forensics`) through policy entropy between Iterations 16 and 21, discovered the high positive returns of `isolate_host` on malicious episodes, and quickly learned that Action 7 yields the highest expected return under observation uncertainty. Once Action 7 dominated at Iteration 22, the policy distribution hardened, collapsing policy entropy back down and preventing subsequent discovery of benign clearance.

---

## 9. Limitations of This Diagnostic

1. **Static Stage 0 Evaluation:** The diagnostic evaluated initial observation states (Stage 0). While sequential progression introduces subsequent stages, 100% of benign episodes terminate at Stage 0 (either via clearance or false disruption), meaning Stage 0 separability is strictly required for benign clearance.
2. **Pretrained Checkpoint Lineage:** The base checkpoint inherited biases from Phase 2 contextual-bandit training where rare classes were synthetically over-sampled, explaining the initial lock-in on Action 17.
3. **No Dynamic Fine-Tuning in Scope:** Under the strict read-only constraints of this task, no encoder fine-tuning, embedding whitening, or reward adjustments were tested.

---

## 10. Recommended Next Steps (For Future Experiments)

To achieve true context-aware action selection without degenerate collapse, future sequential experiments should address the root causes identified above:

1. **Address Embedding Anisotropy:**
   - **Representation Whitening / Normalization:** Apply L2 normalization, mean centering, or PCA whitening to the 768-dim embeddings so that subtle textual variations produce larger angular and Euclidean differences.
   - **Observable Telemetry Feature Augmentation:** Supplement or replace the dense 768-dim text embedding with explicit observable telemetry feature vectors (e.g., host role, process name, rule level, user privilege, network destination), which the rule-based heuristic demonstrated is highly discriminative.
2. **End-to-End or LoRA Adapter Tuning:**
   - Allow the SecBERT encoder's top layers (or lightweight LoRA adapters) to adapt under sequential task gradients, enabling the model to learn a task-specific projection that cleanly separates benign from malicious alerts.
3. **Balanced Reward Structuring:**
   - Rebalance the ratio between `malicious_breach` and `false_disruption` or introduce progressive investigative rewards (e.g., granting reward for investigative actions that confirm benign status) to make gathering evidence competitive with immediate host isolation.

---
*Report generated autonomously by Antigravity Read-Only Diagnostic Inspector.*
