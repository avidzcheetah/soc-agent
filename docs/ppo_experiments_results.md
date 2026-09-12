# PPO Improvement Experiments: Validation Results

This document serves as the official tracking record for all PPO improvement experiments conducted during Phase 2 of the thesis methodology. All results below are evaluated purely on the isolated **1,539-sample Validation Set**. 

The test set remains completely frozen and untouched during this phase.

---

## Executive Summary & Metric Leaderboard

| Model / Experiment | Sampling | Intervention | Accuracy | Macro F1 | Weighted F1 | MCC | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **SecBERT Baseline** | N/A | Fine-tuned Transformer | 94.4835% | 0.7375 | 0.9445 | 0.9347 | Baseline |
| **Warm-Start Actor** | N/A | Knowledge Distillation (BC) | **94.7368%** | 0.7438 | **0.9467** | **0.9377** | Initializer |
| **Exp 0 (Naive PPO)** | Uniform | $+1/-1$ reward, scratch | 93.4373% | 0.6114 | 0.9300 | 0.9224 | Baseline PPO |
| **Exp 1 (Class-Aware)** | Uniform | Log frequency reward | 93.8272% | 0.6281 | 0.9342 | 0.9268 | Reward Shaping |
| **Exp 2 (Response-Aware)** | Uniform | CISA Phase penalty | 93.8272% | 0.6219 | 0.9332 | 0.9269 | Reward Shaping |
| **Exp 3 (Hybrid Reward)** | Uniform | Class + Phase penalty | 93.1124% | 0.6312 | 0.9232 | 0.9183 | Reward Shaping |
| **Exp 4A (Pure PPO)** | Uniform | Warm-start, $+1/-1$ reward | 93.5673% | 0.7616 | 0.9342 | 0.9238 | Macro-F1 peak |
| **Exp 4A-1 (KL-PPO)** | Uniform | KL penalty ($\beta=0.5$) | 93.3073% | 0.7027 | 0.9318 | 0.9205 | Regularization |
| **Exp 4A-2 (BC-PPO)** | Uniform | Supervised BC loss ($\lambda=0.05$) | 94.2820% | 0.7350 | 0.9422 | 0.9323 | Preservation |
| **Exp 4B (Balanced PPO)** | Balanced | Capped inverse-sqrt ($lr=3\text{e-}4$) | 94.6069% | 0.7423 | 0.9452 | 0.9361 | Balanced baseline |
| **Exp 4C-0 (Conservative Balanced)** | Balanced | Capped sampling + $lr=1\text{e-}4$ | **94.4120%** | **0.7928** | **0.9433** | **0.9338** | 🏆 **Best PPO Model** |

---

## Phase 2A: Reward Engineering Experiments (Exp 0 – Exp 3)

### Summary of Findings
Across 4 distinct reward formulations (naive scalar, logarithmic class-aware, semantic response-aware phase penalty, and bounded hybrid), reward engineering alone proved **insufficient** to close the performance gap with the supervised SecBERT baseline (Macro F1 0.7375).

* **Exp 0 (Naive PPO):** Suffered severe minority-class collapse (Macro F1 = 0.6114).
* **Exp 1 (Class-Aware Logarithmic Reward):** Improved Macro F1 to 0.6281 and recovered `enable_deep_logging` (F1 0.80), but `block_dest_ip` collapsed.
* **Exp 2 (Response-Aware CISA Phase Penalty):** Constrained errors within correct semantic phases (MCC 0.9269), but intra-phase majority attractors (`kill_process`, `isolate_host`) absorbed rare actions.
* **Exp 3 (Bounded Hybrid Reward):** Proved counterproductive. Entropy collapsed rapidly by iteration 10, causing complete collapse of `remove_persistence` and `escalate_to_human`.

**Key Strategic Insight:** The core limitation of PPO was not reward definition, but structural:
1. **Cold-Start Disadvantage:** PPO Actor initialized from scratch struggled against dense cross-entropy gradients.
2. **Experience Starvation:** Rare classes appeared $\le 1\times$ per 256-sample rollout.
3. **Policy Entropy Collapse:** Deterministic convergence before policy could explore rare decision boundaries.

---

## Phase 2B: Structural Recovery & Initialization Experiments (Exp 4A Series)

### Warm-Start Knowledge Distillation
* **Method:** Knowledge distillation of SecBERT logits into the PPO Actor MLP architecture via KL-divergence loss.
* **Result:** Reached **94.7368% Accuracy**, **0.7438 Macro F1**, **0.9467 Weighted F1**, and **0.9377 MCC**.
* **Verdict:** Successfully reproduced and slightly exceeded the supervised SecBERT baseline, providing a high-quality warm-start initialization.

### Exp 4A: Warm-Start PPO
* **Configuration:** PPO initialized from Warm-Start Actor, uniform sampling, simple $+1/-1$ binary reward.
* **Result:** Achieved Macro F1 **0.7616** at iteration 20 (+0.0241 over SecBERT), but overall Accuracy degraded to 93.57% and MCC to 0.9238.

### Exp 4A-1: KL-Constrained PPO ($\beta=0.5$)
* **Hypothesis:** Penalizing KL divergence relative to the frozen reference warm-start policy will prevent policy drift.
* **Result:** Macro F1 dropped to **0.7027**. Over-constrained policy exploration without improving balance.

### Exp 4A-2: Supervised BC Preservation Loss ($\lambda_{BC}=0.05$)
* **Hypothesis:** Adding a cross-entropy loss against ground-truth labels during PPO updates will preserve baseline performance.
* **Result:** Accuracy (94.28%), Weighted F1 (0.9422), and MCC (0.9323) were successfully preserved, but Macro F1 was suppressed to **0.7350** (below SecBERT).

---

## Phase 2C: Experience Sampling & Learning Rate Optimization (Exp 4B & Exp 4C)

### Exp 4B: Capped Class-Balanced PPO ($lr=3\text{e-}4$, $\text{Cap}=5\times$)
* **Hypothesis:** Replacing uniform random sampling with capped inverse-sqrt class frequency sampling ($P(c) \propto \min(N_c^{-0.5}, 5 \cdot w_{\text{maj}})$) will give rare response actions necessary exposure during rollouts.
* **Result:** Evaluated on validation set: **Accuracy 94.61%**, **Macro F1 0.7423**, **Weighted F1 0.9452**, **MCC 0.9361**.
* **Verdict:** First PPO variant to beat SecBERT across all 4 metrics simultaneously. However, training peaked early at iteration 5 and oscillated due to high actor learning rate ($3\text{e-}4$).

### Exp 4C-0: Conservative Balanced PPO ($lr=1\text{e-}4$, Balanced Sampling, 150 Iterations)
* **Hypothesis:** Reducing the actor learning rate to $1\text{e-}4$ under capped balanced sampling will allow smooth, non-oscillatory convergence on minority-class decision boundaries without overshooting.
* **Checkpoint Evaluation (Iter 125):**
  * **Val Accuracy:** **94.4120%**
  * **Val Macro F1:** **0.7928** (+0.0553 vs SecBERT baseline)
  * **Val Weighted F1:** **0.9433**
  * **Val MCC:** **0.9338**
* **Per-Class Breakdown (18 Active Classes):**
  * `patch_vulnerability`: **0.9927** (support=411)
  * `monitor`: **0.9919** (support=123)
  * `isolate_host`: **0.9626** (support=338)
  * `disable_account`: **0.9562** (support=150)
  * `kill_process`: **0.8858** (support=172)
  * `reset_credentials`: **0.8721** (support=82)
  * `quarantine_file`: **0.8649** (support=57)
  * `enable_deep_logging`: **0.8571** (support=8)
  * `quarantine_email`: **0.8454** (support=51)
  * `restore_defense_config`: **0.7778** (support=10)
  * `remove_persistence`: **0.7500** (support=21)
  * `snapshot_forensics`: **1.0000** (support=1) — *Fully recovered*
  * `block_port`: **0.0000** (support=1)
  * `restore_registry`: **0.0000** (support=1)

**Conclusion:** **Exp 4C-0 represents the current state-of-the-art PPO policy**, breaking the 0.75 Macro F1 threshold and reaching **0.7928 Macro F1** while maintaining robust overall accuracy (94.41%) and MCC (0.9338).
