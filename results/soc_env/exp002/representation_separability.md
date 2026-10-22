# Representation Separability Diagnostic Report: SOC Simulation Environment

**Investigation Scope:** Representation Separability of Frozen SecBERT Embeddings  
**Experiment Context:** Sequential PPO Experiment 002 Policy Collapse Investigation  
**Date:** 2026-10-12  
**Target Artifact:** `results/soc_env/exp002/representation_separability.md`  

---

## Executive Summary

This diagnostic evaluates whether the frozen SecBERT 768-dimensional pooler embeddings used by `SequentialSOCEnvironment` contain sufficient discriminative signal to separate malicious incidents from benign incidents across all defined scenario stages.

In earlier experiments, sequential PPO collapsed to `always_escalate` (Exp001) and `always_isolate_host` (Exp002). An initial inspection of the 7 initial states showed pairwise cosine similarities exceeding $0.98$. To determine whether this was merely an artifact of evaluating only seven initial states or a systemic representation limitation across the simulation, this investigation extracted and analyzed the embeddings of **every distinct incident stage** across the entire scenario catalog.

### Key Empirical Findings

1. **Dataset Composition:** Across the 7 benchmark scenarios, there are **12 distinct incident stages** (9 malicious, 3 benign). Every stage possesses a unique formatted text string (0 duplicate strings).
2. **Embedding Anisotropy:** The 768-dimensional pooler embeddings exhibit severe anisotropy (the "cone effect"). Pairwise cosine similarity across all 66 stage pairs averages **$0.9914 \pm 0.0034$** (range: $0.9791$ to $0.9962$).
3. **Absence of Natural Class Boundary:** The mean inter-class cosine similarity between malicious and benign stages (**$0.9904 \pm 0.0032$**) is nestled directly between intra-malicious similarity (**$0.9926 \pm 0.0024$**) and intra-benign similarity (**$0.9855 \pm 0.0059$**). Certain malicious stages are geometrically closer to benign stages than to other malicious stages (e.g., C2 exfiltration Stage 0 and benign DevOps tooling have cosine similarity $0.9954$).
4. **Linear Separability Under Grouped Cross-Validation:**
   - In **Leave-One-Group-Out Cross-Validation (LOGO-CV)** grouped by scenario (7 folds), a logistic regression classifier achieves a **Balanced Accuracy of 0.5000**, with **0.0000 recall on benign incidents** (classifying 100% of held-out stages as Malicious, identical to a majority-class baseline).
   - Even with balanced class weights (`class_weight='balanced'`), LOGO-CV achieves a **Balanced Accuracy of 0.4444** (worse than majority-class baseline, misclassifying all 3 held-out benign scenarios as Malicious and misclassifying C2 initial execution as Benign).
5. **Formatter & Telemetry Audit:** The `IncidentFormatter` preserves all primary observed telemetry fields (rules, processes, users, PIDs, MITRE tags). However, the standardized clause template and the legitimate presence of MITRE ATT&CK tags on benign operational activity (e.g., T1059.006 on CI/CD tooling, T1046 on internal vulnerability scanning) create heavy token and semantic overlap.

---

## 1. Scenario Catalog & Incident Stage Inventory

Every scenario in `src/environment_sequential.py` was inspected to identify all distinct stages and their telemetry payloads.

| Scenario ID | Scenario Name | Class | Total Stages | Stage ID | Stage Name | Formatted Text Char Count | Token Count (SecBERT) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| `scen_c2_exfil` | Multi-Stage C2 & Data Exfiltration | Malicious | 3 | 0 | Discovery & Local Execution (`whoami`) | 124 | 41 |
| | | | | 1 | Outbound C2 Communication (Beacon) | 304 | 99 |
| | | | | 2 | Data Exfiltration & Impact (Exfil) | 296 | 96 |
| `scen_privesc_persist` | Local Privilege Escalation & Persistence | Malicious | 2 | 0 | Privilege Escalation Attempt (`sudo`) | 249 | 76 |
| | | | | 1 | Persistence Mechanism Installation (`cron`) | 230 | 67 |
| `scen_credential_lateral` | Stolen Credential & Lateral Movement | Malicious | 2 | 0 | Stolen Credential Auth (Bastion SSH) | 314 | 90 |
| | | | | 1 | Lateral DB Dump & Exfiltration (`mysqldump`) | 325 | 97 |
| `scen_network_rce_tunnel` | Web App RCE & Reverse TCP Tunnel | Malicious | 2 | 0 | Web Service Exploit (Reverse Socket) | 302 | 96 |
| | | | | 1 | Encrypted Stream Exfiltration | 276 | 88 |
| `scen_benign_backup` | Scheduled System Backup | Benign | 1 | 0 | Routine Admin Backup (`rsync`) | 233 | 71 |
| `scen_benign_devops_tooling` | CI/CD DevOps Pipeline Operations | Benign | 1 | 0 | Automated Health Verification (`pytest`) | 299 | 96 |
| `scen_benign_security_audit` | Scheduled Security Vulnerability Scan | Benign | 1 | 0 | Approved Vulnerability Scan (`nmap`) | 268 | 88 |

### Inventory Summary
- **Total Scenarios:** 7 (4 Malicious, 3 Benign)
- **Total Stages:** 12 (9 Malicious stages, 3 Benign stages)
- **Class Balance:** 75.0% Malicious, 25.0% Benign
- **Text Uniqueness:** 12 unique texts (0 duplicates, 0 near-duplicates with identical payload)
- **Token Lengths:** Range from 41 to 99 tokens. All well within the model's 512-token limit (0 truncation).

---

## 2. Embedding Extraction & Numerical Validity

Embeddings were extracted using the exact production pipeline utilized by `SequentialSOCEnvironment`:
$$\text{IncidentRecord} \xrightarrow{\text{IncidentFormatter}} \text{Standardized Text} \xrightarrow{\text{SecBERTStateEncoder}} \mathbf{s} \in \mathbb{R}^{768}$$

- **Model Architecture:** Frozen 6-layer `BertModel` (83,450,880 parameters) from `models/secbert_finetuned`
- **Pooling Mechanism:** `BertPooler` ($\text{pooler\_output} = \tanh(\mathbf{W}_{\text{pool}} \mathbf{h}_{\text{CLS}} + \mathbf{b}_{\text{pool}})$)
- **Shape:** `(12, 768)`
- **Numerical Health:**
  - NaN count: 0 (0.0%)
  - Inf count: 0 (0.0%)
  - Norm Distribution: Mean $13.6025 \pm 0.0557$ (Min: $13.4355$, Max: $13.6659$)
  - Coordinate Values: Mean $-0.0071 \pm 0.4908$, bounded strictly within $[-0.9535, 0.9447]$

---

## 3. Geometric Separability & Distributional Analysis

To determine how the representations cluster in $\mathbb{R}^{768}$, we computed the full $12 \times 12$ pairwise cosine similarity and Euclidean distance matrices.

### Complete $12 \times 12$ Pairwise Cosine Similarity Matrix

| Short Stage Identifier | `c2_s0` | `c2_s1` | `c2_s2` | `priv_s0` | `priv_s1` | `back_s0` | `lat_s0` | `lat_s1` | `rce_s0` | `rce_s1` | `dev_s0` | `aud_s0` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`c2_s0`** (Mal) | 1.0000 | 0.9935 | 0.9936 | 0.9879 | 0.9920 | 0.9928 | 0.9888 | 0.9900 | 0.9933 | 0.9921 | **0.9954** | 0.9831 |
| **`c2_s1`** (Mal) | 0.9935 | 1.0000 | 0.9948 | 0.9885 | 0.9933 | 0.9924 | 0.9923 | 0.9935 | 0.9951 | 0.9952 | 0.9929 | 0.9877 |
| **`c2_s2`** (Mal) | 0.9936 | 0.9948 | 1.0000 | 0.9910 | 0.9945 | 0.9919 | 0.9919 | 0.9959 | 0.9943 | 0.9955 | 0.9932 | 0.9897 |
| **`priv_s0`** (Mal) | 0.9879 | 0.9885 | 0.9910 | 1.0000 | 0.9907 | 0.9850 | 0.9899 | 0.9907 | 0.9886 | 0.9863 | 0.9885 | 0.9857 |
| **`priv_s1`** (Mal) | 0.9920 | 0.9933 | 0.9945 | 0.9907 | 1.0000 | 0.9926 | 0.9937 | 0.9951 | 0.9932 | 0.9940 | 0.9924 | 0.9892 |
| **`back_s0`** (Ben) | 0.9928 | 0.9924 | 0.9919 | 0.9850 | 0.9926 | 1.0000 | 0.9882 | 0.9900 | 0.9936 | 0.9942 | 0.9934 | 0.9791 |
| **`lat_s0`** (Mal) | 0.9888 | 0.9923 | 0.9919 | 0.9899 | 0.9937 | 0.9882 | 1.0000 | 0.9931 | 0.9922 | 0.9919 | 0.9897 | 0.9901 |
| **`lat_s1`** (Mal) | 0.9900 | 0.9935 | 0.9959 | 0.9907 | 0.9951 | 0.9900 | 0.9931 | 1.0000 | 0.9945 | 0.9948 | 0.9922 | 0.9916 |
| **`rce_s0`** (Mal) | 0.9933 | 0.9951 | 0.9943 | 0.9886 | 0.9932 | 0.9936 | 0.9922 | 0.9945 | 1.0000 | **0.9962** | 0.9942 | 0.9849 |
| **`rce_s1`** (Mal) | 0.9921 | 0.9952 | 0.9955 | 0.9863 | 0.9940 | 0.9942 | 0.9919 | 0.9948 | **0.9962** | 1.0000 | 0.9938 | 0.9865 |
| **`dev_s0`** (Ben) | **0.9954** | 0.9929 | 0.9932 | 0.9885 | 0.9924 | 0.9934 | 0.9897 | 0.9922 | 0.9942 | 0.9938 | 1.0000 | 0.9838 |
| **`aud_s0`** (Ben) | 0.9831 | 0.9877 | 0.9897 | 0.9857 | 0.9892 | 0.9791 | 0.9901 | 0.9916 | 0.9849 | 0.9865 | 0.9838 | 1.0000 |

### Within-Class vs Between-Class Distributions

| Distribution Subset | Pair Count | Cosine Mean $\pm$ Std | Cosine [Min, Max] | Euclidean Mean $\pm$ Std | Euclidean [Min, Max] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Unique Pairs** | 66 pairs | $0.9914 \pm 0.0034$ | $[0.9791, 0.9962]$ | $1.7820 \pm 0.3541$ | $[1.1908, 2.7710]$ |
| **Intra-Malicious** | 36 pairs | $0.9926 \pm 0.0024$ | $[0.9863, 0.9962]$ | $1.6399 \pm 0.2649$ | $[1.1908, 2.2576]$ |
| **Intra-Benign** | 3 pairs | $0.9855 \pm 0.0059$ | $[0.9791, 0.9934]$ | $2.2623 \pm 0.5092$ | $[1.5665, 2.7710]$ |
| **Inter-Class (Mal vs Ben)** | 27 pairs | **$0.9904 \pm 0.0032$** | $[0.9831, 0.9954]$ | **$1.8571 \pm 0.3089$** | $[1.3141, 2.4881]$ |

### Geometric Diagnostic Observations:
1. **Severe Anisotropy:** All 12 embeddings occupy a cone of less than $12^\circ$ half-angle. The lowest pairwise cosine similarity across the entire catalog is $0.9791$ (between routine backup and security audit).
2. **Inter-Class Sandwiching:** The inter-class cosine similarity ($0.9904$) is higher than the intra-benign similarity ($0.9855$) and only $0.0022$ below intra-malicious similarity ($0.9926$).
3. **Cross-Class Proximity:** The malicious Stage 0 of C2 exfiltration (`scen_c2_exfil_s0`) is closer in cosine similarity to benign DevOps tooling ($0.9954$) and routine backup ($0.9928$) than it is to malicious privilege escalation ($0.9879$).
4. **Euclidean Overlap:** The minimum inter-class distance ($1.3141$) is smaller than 28 of the 36 intra-malicious distances. There is no simple hyper-spherical or angular margin separating the two classes.

---

## 4. Empirical Separability Evaluation: Logistic Regression

To test whether a linear decision surface can separate the frozen embeddings without modifying SecBERT weights, we evaluated a regularized Logistic Regression classifier ($C=1.0$) across multiple validation strategies.

### Cross-Validation Strategy: Leakage Control via Grouping
Because multiple stages originate from the same scenario (e.g., C2 exfiltration has 3 stages sharing the host `workstation-01`), evaluating with standard random k-fold would leak hostnames, PIDs, and scenario context between train and test splits. Therefore, we group stages by `scenario_id`:
- **Leave-One-Group-Out CV (LOGO-CV):** 7 folds. In each fold, all stages of one scenario are held out for testing, while the classifier is trained on the remaining 6 scenarios.
- **StratifiedGroupKFold (3 Folds):** 3 folds, ensuring each fold contains 1 held-out benign scenario and 1–2 held-out malicious scenarios.

### Comparative Classification Results

| Model / Evaluation Scheme | Balanced Accuracy | Precision (Benign) | Recall (Benign) | F1 (Benign) | Macro F1 | Confusion Matrix $\begin{bmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{bmatrix}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-Class Baseline** (Always Malicious) | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4286 | $\begin{bmatrix} 9 & 0 \\ 3 & 0 \end{bmatrix}$ |
| **Whole Dataset Fit** (Train on all 12, $C=1.0$) | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.4286 | $\begin{bmatrix} 9 & 0 \\ 3 & 0 \end{bmatrix}$ |
| **Whole Dataset Fit** (Balanced class weights) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | $\begin{bmatrix} 9 & 0 \\ 0 & 3 \end{bmatrix}$ |
| **LOGO-CV** (7 Folds, $C=1.0$, default weights) | **0.5000** | **0.0000** | **0.0000** | **0.0000** | **0.4286** | $\begin{bmatrix} 9 & 0 \\ 3 & 0 \end{bmatrix}$ |
| **LOGO-CV** (7 Folds, $C=1.0$, balanced weights) | **0.4444** | **0.0000** | **0.0000** | **0.0000** | **0.4000** | $\begin{bmatrix} 8 & 1 \\ 3 & 0 \end{bmatrix}$ |
| **StratifiedGroupKFold** (3 Folds, $C=1.0$) | **0.5000** | **0.0000** | **0.0000** | **0.0000** | **0.4286** | $\begin{bmatrix} 9 & 0 \\ 3 & 0 \end{bmatrix}$ |

### Fold-by-Fold Inspection of LOGO-CV

#### Standard Logistic Regression ($C=1.0$):
- **Fold 1** (Held out: `scen_benign_backup`, True: Benign) $\rightarrow$ Pred: `Malicious` ($P(\text{Benign}) = 0.1180$)
- **Fold 2** (Held out: `scen_benign_devops_tooling`, True: Benign) $\rightarrow$ Pred: `Malicious` ($P(\text{Benign}) = 0.1616$)
- **Fold 3** (Held out: `scen_benign_security_audit`, True: Benign) $\rightarrow$ Pred: `Malicious` ($P(\text{Benign}) = 0.0589$)
- **Fold 4** (Held out: `scen_c2_exfil`, 3 stages, True: Malicious) $\rightarrow$ Pred: `[Mal, Mal, Mal]` ($P(\text{Benign}) = [0.453, 0.336, 0.324]$)
- **Fold 5** (Held out: `scen_credential_lateral`, 2 stages, True: Malicious) $\rightarrow$ Pred: `[Mal, Mal]` ($P(\text{Benign}) = [0.288, 0.287]$)
- **Fold 6** (Held out: `scen_network_rce_tunnel`, 2 stages, True: Malicious) $\rightarrow$ Pred: `[Mal, Mal]` ($P(\text{Benign}) = [0.280, 0.328]$)
- **Fold 7** (Held out: `scen_privesc_persist`, 2 stages, True: Malicious) $\rightarrow$ Pred: `[Mal, Mal]` ($P(\text{Benign}) = [0.292, 0.312]$)

#### Balanced Logistic Regression (`class_weight='balanced'`):
- **Held-out Benign Scenarios:** $P(\text{Benign})$ remained strictly below threshold: Backup ($0.2757$), DevOps ($0.3762$), Security Audit ($0.1125$). All 3 misclassified as Malicious.
- **Held-out Malicious Scenarios:** Fold 4 misclassified C2 exfiltration Stage 0 (`whoami`) as Benign ($P(\text{Benign}) = 0.5971$).
- **Net Outcome:** 0 true positives on Benign, 1 false positive on Malicious $\rightarrow$ Balanced Accuracy drops to $0.4444$.

### Separability Finding:
A linear probe on the frozen SecBERT embeddings **fails to outperform the majority-class baseline** when evaluated on unseen scenarios. In fact, under unweighted loss, the optimal regularized linear hyperplane collapses to predicting `Malicious` for every single input.

---

## 5. Inspection of Formatter Output & Telemetry Representation

We audited `IncidentFormatter.format_incident_text()` and the underlying `IncidentRecord` objects to assess whether formatting introduces artificial similarity or suppresses discriminative telemetry fields.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Observed Telemetry Record                       │
│  - host: 'backup-node-01'           - rule_id: 100010 (Level 2)        │
│  - process: '/usr/bin/rsync'        - username: 'backup_admin'         │
│  - cmdline: 'rsync -avz ...'        - metadata: {'benign': True}       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              IncidentFormatter.format_incident_text()                  │
│                                                                        │
│  Template Clauses:                                                     │
│  1. Prefix: '[Detection] '                                             │
│  2. Headline: 'Alert: Scheduled system backup...'                      │
│  3. Host: 'Host: backup-node-01'                                       │
│  4. Rule: 'Rule: 100010 (Level 2)'                                     │
│  5. Process/Command: 'Process: /usr/bin/rsync. Command: ...'           │
│  6. PID/User: 'PID: 12040. User: backup_admin.'                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                           SecBERT Tokenizer                            │
│  - Identical template syntax across all 12 stages                      │
│  - 35-45% of tokens are shared syntactic boilerplate                   │
│  - Legitimate MITRE tags present on benign events (T1059, T1046)       │
└────────────────────────────────────────────────────────────────────────┘
```

### Formatter Integrity Assessment

1. **Telemetry Preservation (Pass):**
   - No primary telemetry fields are discarded. Rule descriptions, hostnames, rule IDs, alert levels, process paths, command lines, PIDs, parent PIDs, usernames, and working directories are faithfully converted into text clauses.
   - Raw simulator flags (such as `metadata['benign'] = True` or `metadata['simulated'] = True`) are correctly excluded to prevent simulator information leakage.
2. **Syntactic Boilerplate Contribution:**
   - Every formatted string shares the exact prefix `[Detection] `, followed by repetitive clause keys: `Alert: ... Host: ... Rule: ... PID: ... User: ...`.
   - Across a 75-token alert, approximately $25 - 35$ tokens ($\approx 35 - 45\%$) consist of identical structural boilerplate.
3. **MITRE ATT&CK Attribution on Benign Incidents:**
   - Real-world SIEM rules often tag benign tools with MITRE ATT&CK identifiers. In our benchmark:
     - `scen_benign_devops_tooling` is tagged with **T1059.006 (Execution)** due to python script execution in `/tmp`.
     - `scen_benign_security_audit` is tagged with **T1046 (Discovery)** due to network SYN scanning with `nmap`.
   - While realistic, this means a language model fine-tuned on MITRE ATT&CK texts sees threat tactics in both malicious and benign alerts.
4. **The Pooling Bottleneck:**
   - The frozen `BertPooler` maps the sequence to a single vector via $\tanh(\mathbf{W} \mathbf{h}_{\text{CLS}} + \mathbf{b})$. Because the [CLS] embedding aggregates self-attention over the entire sequence—dominated by shared syntactic tokens, SIEM boilerplate, and MITRE keywords—the small distinguishing tokens (`backup_admin` vs `avidz`, level 2 vs level 8) are compressed into minor dimensional perturbations relative to the overall embedding norm ($13.6$).

---

## 6. Critical Limitations of This Diagnostic

To ensure scientific rigor, the conclusions must be interpreted within the boundaries of the available data:

1. **Small Scenario Sample Size ($N=7$):**
   - With only 4 malicious scenarios and 3 benign scenarios (12 stages total), cross-validation estimates have high variance and wide confidence intervals.
   - **This evaluation does NOT establish that SecBERT is fundamentally incapable of separating arbitrary cyber text.** A fine-tuned transformer with millions of parameters could easily separate these concepts if adapted.
2. **Scope of the Frozen Condition:**
   - SecBERT was evaluated in a **strictly frozen** state without adapter layers, contrastive fine-tuning, or embedding whitening.
   - The results demonstrate that the *existing, frozen pooler embeddings* as currently fed to the PPO agent do not provide a separable linear surface.
3. **Class Imbalance in Scenarios:**
   - The catalog features 9 malicious stages vs 3 benign stages ($3:1$ ratio). In small-sample regimes, regularized estimators naturally gravitate toward the majority class.

---

## 7. Conclusions: What the Evidence Does and Does Not Establish

### What the Evidence DOES Establish:
1. **The representations explain the PPO collapse:**
   When an MLP actor is trained via PPO on top of these frozen embeddings, it is presented with state vectors that have $>0.985$ cosine similarity and are linearly inseparable under grouped cross-validation. Combined with the environment's reward structure (where a missed attack incurs a $-3.00$ breach penalty, whereas false disruption costs only $-1.00$), the policy optimizer inevitably converges to the risk-averse majority action: **unconditional host isolation (`isolate_host`)**.
2. **The linear probe fails out-of-scenario generalization:**
   When evaluated on a held-out scenario, a logistic regression classifier trained on the frozen embeddings predicts `Malicious` for 100% of held-out benign scenarios.
3. **Severe embedding anisotropy exists:**
   The 768-dimensional pooler outputs are concentrated in an extremely narrow cone ($0.9914$ average cosine similarity), with inter-class distance indistinguishable from intra-class distance.

### What the Evidence DOES NOT Establish:
1. **It does NOT prove that incident text lacks information:**
   Human analysts and rule-based heuristics easily distinguish `rsync by backup_admin` from `unauthorized sudo by avidz`. The raw text clearly contains sufficient information.
2. **It does NOT prove that SecBERT cannot learn the distinction:**
   The failure occurs because SecBERT's 83.45M parameters are completely **frozen**. If the encoder's top layers were fine-tuned end-to-end or augmented with a parameter-efficient adapter (e.g., LoRA), the model could readily learn to project benign and malicious alerts into distinct clusters.
3. **It does NOT indicate a flaw in the telemetry formatter:**
   The formatter correctly preserves all observable fields without leakage. The lack of separability arises from the frozen pooling geometry, not from dropped telemetry.

---

## 8. Architectural Implications for Next Experiments

Based on these findings, future iterations of the sequential incident-response agent must resolve the representation bottleneck through one or more of the following mechanisms:

1. **Observation Augmentation (Hybrid Representation):**
   Combine the dense 768-dim text embedding with a structured, normalized telemetry vector containing explicit categorical/numerical features (e.g., host classification, rule alert level, user privilege level, network direction). As demonstrated by the rule-based policy (return $+0.42$), structured telemetry provides immediate separability.
2. **Embedding Centering & Whitening:**
   Apply standard BERT anisotropy corrections—such as mean subtraction, PCA whitening, or cosine-margin scaling—to expand the angular distribution of the embeddings prior to the actor MLP.
3. **End-to-End or Adapter Fine-Tuning:**
   Unfreeze the top transformer layer or introduce a low-rank adapter (LoRA) trained with a contrastive loss or sequential policy gradients, allowing the text encoder to align its representations with the sequential decision task.

---
*Report autonomously generated by Antigravity Representation Separability Inspector.*
