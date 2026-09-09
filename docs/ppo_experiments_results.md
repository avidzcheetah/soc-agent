# PPO Improvement Experiments: Validation Results

This document serves as the official tracking record for all PPO improvement experiments conducted during Phase 2 of the thesis methodology. All results below are evaluated purely on the isolated **1,539-sample Validation Set**. 

The test set remains completely frozen and untouched during this phase.

## Experiment Baselines

| Experiment | Description | Validation Macro F1 |
| :--- | :--- | :--- |
| **Baseline (SecBERT)** | Pure classifier baseline (No PPO). | **0.7375** |
| **Exp 0 (Original PPO)** | Original PPO with naive `+1/-1` reward structure. | **0.6114** |

---

## Experiment 1: Class-Aware Logarithmic Reward

**Hypothesis:** Applying an inverse-frequency logarithmic reward to minority classes will encourage the agent to explore and successfully learn them, improving the Macro F1 score without overcompensating gradients.
**Formulation:** `W_c = 1.0 + ln(N_max / N_c)`

### Overall Metrics
| Metric | Exp 0 (Original PPO) | Exp 1 (Class-Aware) | Diff |
| :--- | ---: | ---: | ---: |
| Accuracy | 93.44% | **93.83%** | +0.39% |
| Macro F1 | 0.6114 | **0.6281** | +0.0167 |
| Weighted F1 | 0.9300 | **0.9342** | +0.0042 |
| MCC | 0.9224 | **0.9268** | +0.0044 |

### Notable Class Movements
- `enable_deep_logging`: 0.0000 $\rightarrow$ **0.8000** (Recovered)
- `block_dest_ip`: 0.6250 $\rightarrow$ **0.0000** (Collapsed)
- `escalate_to_human`: 0.0000 $\rightarrow$ **0.0000** (Unchanged)

**Conclusion:** Class-aware logarithmic rewards improved PPO's overall validation performance and recovered some minority-class performance, but the improvement was inconsistent across rare response actions. This indicates that class-frequency weighting alone is insufficient to achieve balanced response-action selection.

---

## Experiment 2: Response-Aware Reward (Semantic Phase Penalty)

**Hypothesis:** Injecting the semantic CISA response framework into the reward landscape prevents the model from taking blind leaps. A phase-compatible mistake is penalized less (-0.5) than a completely inappropriate cross-phase response (-1.0).
**Formulation:** (Independent of Exp 1)
- Exact action correct: `+1.0`
- Wrong action, but correct response phase: `-0.5`
- Wrong phase: `-1.0`
- `escalate_to_human` (All Phases): `-0.5` if wrong action but phase-compatible.

### Comparative Validation Summary

| Metric | SecBERT (Supervised) | Exp 0 (Original PPO) | Exp 1 (Class-Aware PPO) | **Exp 2 (Response-Aware PPO)** | Diff vs Exp 0 | Diff vs Exp 1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Accuracy** | 94.48% | 93.44% | 93.83% | **93.83%** | +0.39% | +0.00% |
| **Macro F1** | **0.7375** | 0.6114 | **0.6281** | **0.6219** | +0.0105 | -0.0062 |
| **Weighted F1** | 0.9445 | 0.9300 | 0.9342 | **0.9332** | +0.0032 | -0.0010 |
| **MCC** | 0.9347 | 0.9224 | 0.9268 | **0.9269** | +0.0045 | +0.0001 |

### Notable Class Movements & Per-Class Analysis
- `dns_sinkhole`: 0.8889 $\rightarrow$ **0.9231** (+0.0342 vs SecBERT)
- `reset_credentials`: 0.8736 $\rightarrow$ **0.8876** (+0.0140 vs SecBERT)
- `disable_account`: 0.9448 $\rightarrow$ **0.9459** (+0.0011 vs SecBERT)
- `escalate_to_human`: 0.0000 $\rightarrow$ **0.3636** (Successfully recovered from 0 in Exp 0 & Exp 1)
- `enable_deep_logging`: 0.0000 (Regressed from 0.8000 in Exp 1; collapsed back to 0.0000)
- `block_dest_ip`: 0.0000 (Remained collapsed)

### Methodological Insights & Next Steps
1. **Response-Aware Value:** The phase penalty prevents catastrophic cross-phase penalties and successfully salvaged `escalate_to_human` (F1 = 0.3636), but it did not provide global balance across severely under-represented classes (`enable_deep_logging`, `block_dest_ip`).
2. **Current Ranking:** SecBERT (0.7375) > Class-Aware PPO (0.6281) > Response-Aware PPO (0.6219) > Original PPO (0.6114).
3. **Diagnostic Prerequisite for Experiment 3:** Rather than blindly combining rewards into a hybrid, we perform a diagnostic prediction-distribution and confusion analysis on validation to pinpoint exactly which dominant classes absorb the predictions of `enable_deep_logging`, `block_dest_ip`, and other minority actions.

---

## Diagnostic Analysis: Policy Collapse & Misclassification Destinations

To uncover why minority-class F1 drops under PPO, an independent diagnostic inspection of the 1,539 validation predictions was conducted on `models/ppo/best_ppo_policy.pt`.

### 1. Where do the Missing Minority Actions Go?

| Ground Truth Action | GT Count | SecBERT Correct | PPO Correct | Primary PPO Misclassification Destinations | Semantic Mechanism |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`enable_deep_logging`** | 8 | 6 (75%) | **0 (0%)** | `escalate_to_human` (6 / 8 = 75%)<br>`monitor` (2 / 8 = 25%) | `escalate_to_human` spans All Phases (-0.5 penalty fallback); `monitor` is Detection phase. |
| **`block_dest_ip`** | 10 | 6 (60%) | **0 (0%)** | `kill_process` (4 / 10 = 40%)<br>`isolate_host` (4 / 10 = 40%)<br>`disable_account` (1/10)<br>`quarantine_email` (1/10) | **100% of errors stayed within Containment phase.** The agent collapsed rare containment into dominant containment hubs. |
| **`escalate_to_human`** | 2 | 2 (100%) | **2 (100%)** | *100% Recall*, but **Precision = 22.2%** (PPO predicted it 9 times instead of 2). | Because `escalate_to_human` carries a mild penalty across all phases, the policy over-samples it as a hedge. |
| **`block_port`** | 1 | 0 (0%) | 0 (0%) | `kill_process` (1 / 1 = 100%) | Collapsed into dominant Containment hub. |
| **`restore_registry`** | 1 | 0 (0%) | 0 (0%) | `remove_persistence` (1 / 1 = 100%) | Eradication phase neighbor. |

### 2. Dominant Attractor Classes (Where do all 95 PPO Errors go?)

Across all 95 validation misclassifications made by PPO:
1. **`kill_process`**: Absorbed **29.5%** of all errors (28 / 95).
2. **`isolate_host`**: Absorbed **15.8%** of all errors (15 / 95).
3. **`reset_credentials`**: Absorbed **12.6%** of all errors (12 / 95).
4. **`escalate_to_human`**: Absorbed **7.4%** of all errors (7 / 95 false positives).

**Key Takeaway for Experiment 3 (Class-Aware + Response-Aware Hybrid):**
The phase penalty in Experiment 2 was so effective at constraining the agent to the correct phase that it created **intra-phase majority attractors**: rare Containment actions (`block_dest_ip`, `block_port`) collapsed into the massive Containment anchors (`kill_process`, `isolate_host`), while rare Detection actions (`enable_deep_logging`) escaped to the universal fallback `escalate_to_human`. 

Experiment 3 must combine **class-frequency inverse weighting** with **phase penalties** specifically to push the gradient out of these intra-phase attractor traps.


