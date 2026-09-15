# PPO Improvement Experiments: Validation Results

This document serves as the official tracking record for all PPO improvement experiments conducted during Phase 2 of the thesis methodology. All results below are evaluated purely on the isolated **1,539-sample Validation Set**. 

The test set remains completely frozen and untouched during this phase.

---

## Executive Summary & Metric Leaderboard

| Model / Experiment | Sampling | Intervention | Accuracy | Macro F1 | Weighted F1 | MCC | Best Iter | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **SecBERT Baseline** | N/A | Fine-tuned Transformer | 94.4835% | 0.7375 | 0.9445 | 0.9347 | N/A | Baseline |
| **Warm-Start Actor** | N/A | Knowledge Distillation (BC) | **94.7368%** | 0.7438 | **0.9467** | **0.9377** | N/A | Initializer |
| **Exp 0 (Naive PPO)** | Uniform | $+1/-1$ reward, scratch | 93.4373% | 0.6114 | 0.9300 | 0.9224 | N/A | Baseline PPO |
| **Exp 1 (Class-Aware)** | Uniform | Log frequency reward | 93.8272% | 0.6281 | 0.9342 | 0.9268 | N/A | Reward Shaping |
| **Exp 2 (Response-Aware)** | Uniform | CISA Phase penalty | 93.8272% | 0.6219 | 0.9332 | 0.9269 | N/A | Reward Shaping |
| **Exp 3 (Hybrid Reward)** | Uniform | Class + Phase penalty | 93.1124% | 0.6312 | 0.9232 | 0.9183 | 70 | Reward Shaping |
| **Exp 4A (Pure PPO)** | Uniform | Warm-start, $+1/-1$ reward | 93.5673% | 0.7616 | 0.9342 | 0.9238 | 20 | Macro-F1 peak |
| **Exp 4A-1 (KL-PPO)** | Uniform | KL penalty ($\beta=0.5$) | 93.3073% | 0.7027 | 0.9318 | 0.9205 | N/A | Regularization |
| **Exp 4A-2 (BC-PPO)** | Uniform | Supervised BC loss ($\lambda=0.05$) | 94.2820% | 0.7350 | 0.9422 | 0.9323 | 70 | Preservation |
| **Exp 4B (Balanced PPO)** | Balanced | Capped inverse-sqrt ($lr=3\text{e-}4$) | 94.6069% | 0.7423 | 0.9452 | 0.9361 | 5 | Balanced baseline |
| **Exp 4C-0 (Conservative Balanced)** | Balanced | Capped sampling + $lr=1\text{e-}4$ | **94.4120%** | **0.7928** | **0.9433** | **0.9338** | 125 | 🏆 **Best PPO Model** |
| **Exp 4C-1 (Refined LR)** | Balanced | Capped sampling + $lr=7.5\text{e-}5$ | 94.3470% | 0.7900 | 0.9428 | 0.9331 | 135 | LR Sensitivity |
| **Exp 4C-2 (Balanced + Mild BC)** | Balanced | Capped sampling + $lr=1\text{e-}4$, $\lambda=0.01$ | 94.5419% | 0.7451 | 0.9446 | 0.9353 | 20 | BC Hybrid |
| **Exp 4C-3 (Higher Entropy)** | Balanced | Capped sampling + $lr=1\text{e-}4$, $c_2=0.02$ | 94.4769% | 0.7926 | 0.9439 | 0.9346 | 135 | Entropy Analysis |
| **Exp 4C-4 (Early Stopping)** | Balanced | 4C-3 config + early stopping (patience=4) | 94.2820% | 0.7398 | 0.9420 | 0.9323 | 30 | Stopped too early |
| **Exp 4C-5 (Periodic Checkpoints)**| Balanced | 4C-3 config + periodic checkpointing | **94.4769%**| **0.7926** | **0.9439** | **0.9346** | 135 | **Validated Peak** |

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
* **Verdict:** **State-of-the-Art PPO Policy**, breaking the 0.75 Macro F1 threshold and reaching **0.7928 Macro F1** while maintaining robust overall accuracy (94.41%) and MCC (0.9338).

### Exp 4C-1: Refined Learning Rate ($lr=7.5\text{e-}5$, Balanced Sampling, 150 Iterations)
* **Hypothesis:** Testing whether a slightly lower actor learning rate ($7.5\text{e-}5$) can improve accuracy preservation while retaining high Macro F1.
* **Checkpoint Evaluation (Iter 135):**
  * **Val Accuracy:** **94.3470%**
  * **Val Macro F1:** **0.7900**
  * **Val Weighted F1:** **0.9428**
  * **Val MCC:** **0.9331**
* **Verdict:** Confirms that the $1\text{e-}4$ region is near-optimal. $7.5\text{e-}5$ yields virtually identical performance (0.7900 Macro F1 vs 0.7928), confirming stable convergence around this hyperparameter setting.

### Exp 4C-2: Balanced Sampling + Mild BC Preservation ($lr=1\text{e-}4$, $\lambda_{BC}=0.01$)
* **Hypothesis:** Combining capped balanced sampling with a very small behavioral cloning loss ($\lambda=0.01$) will anchor majority-class boundaries (boosting Accuracy/MCC) while allowing minority classes to benefit from balanced sampling.
* **Checkpoint Evaluation (Iter 20):**
  * **Val Accuracy:** **94.5419%**
  * **Val Macro F1:** **0.7451**
  * **Val Weighted F1:** **0.9446**
  * **Val MCC:** **0.9353**
* **Verdict:** Even a tiny BC preservation loss ($\lambda=0.01$) strongly pulls the policy back toward the warm-start distribution, preserving Accuracy (94.54%) but capping Macro F1 at 0.7451. Pure capped balanced sampling without BC loss (Exp 4C-0) remains the superior configuration for maximizing Macro F1 gain.

### Exp 4C-3: Higher Entropy Coefficient ($lr=1\text{e-}4$, $c_2=0.02$)
* **Hypothesis:** Doubling the entropy coefficient from 0.01 to 0.02 will maintain exploration and prevent the late-training performance collapse (from Macro F1 ~0.79 down to ~0.70) observed in 4C-0.
* **Checkpoint Evaluation (Iter 135):**
  * **Val Accuracy:** **94.4769%**
  * **Val Macro F1:** **0.7926**
  * **Val Weighted F1:** **0.9439**
  * **Val MCC:** **0.9346**
* **Verdict:** Essentially matched 4C-0's peak (Macro F1 0.7926 vs 0.7928). Crucially, **the late-training collapse still occurred** (dropping to 0.7136 by iteration 150). This proves that simply increasing the entropy penalty does not prevent policy drift caused by over-optimization.

### Structural Improvement: Validation-Based Early Stopping (Exp 4C-4)
* **Finding:** The degradation seen in late iterations (140-150) of 4C-0 and 4C-3 demonstrates that PPO continues optimizing beyond the ideal generalized policy, progressively destroying minority-class performance in favor of common actions (e.g. `patch_vulnerability`, `kill_process`). 
* **Solution (Exp 4C-4):** A validation composite score `(Macro F1 + Accuracy + Weighted F1 + MCC) / 4` early stopping mechanism was implemented directly into `trainer.py` with a patience of 4 evaluations (20 iterations) to formalize the model selection process.
* **Verdict:** Early stopping with patience=4 was too aggressive. It triggered at iteration 50 (Macro F1: 0.7398), failing to reach the high-performance region that emerges around iteration 125. The training trajectory is non-monotonic, meaning temporary validation valleys must be tolerated.

### Exp 4C-5: Periodic Checkpointing & Trajectory Analysis ($lr=1\text{e-}4$, $c_2=0.02$)
* **Hypothesis:** By disabling early stopping (patience=150) and saving checkpoints every 5 iterations, we can definitively prove the reproducibility of the high-Macro-F1 region and map the exact trajectory of policy drift.
* **Checkpoint Evaluation (Iter 135 Peak):**
  * **Val Accuracy:** **94.4769%**
  * **Val Macro F1:** **0.7926**
  * **Val Weighted F1:** **0.9439**
  * **Val MCC:** **0.9346**
* **4C-5 Checkpoint Trajectory Comparison:**

| Iteration |     Accuracy |   Macro F1 | Weighted F1 |        MCC |
| --------: | -----------: | ---------: | ----------: | ---------: |
|       120 |     94.0221% | **0.7842** |      0.9394 |     0.9292 |
|       125 |     94.3470% | **0.7919** |      0.9424 |     0.9329 |
|       130 |     94.2170% | **0.7911** |      0.9414 |     0.9315 |
|   **135** | **94.4769%** | **0.7926** |  **0.9439** | **0.9346** |
|       140 |     94.2820% |     0.7335 |      0.9417 |     0.9323 |
|       145 |     94.0221% |     0.7163 |      0.9381 |     0.9291 |
|       150 |     93.9571% |     0.7136 |      0.9373 |     0.9283 |

* **Verdict:** The independent evaluation of periodic checkpoints confirmed that the policy reaches a stable, high-performance region between iterations 125–135 (consistently >0.78 Macro F1) before undergoing a sharp degradation (Macro F1 drops to ~0.71 by iteration 150). Iteration 135 is currently the **best PPO validation checkpoint**, achieving a +0.0551 Macro F1 improvement over the SecBERT baseline while maintaining robust overall accuracy and MCC.
* **Remaining Weakness:** The three validation classes with only one sample each (`block_port`, `restore_registry`, `snapshot_forensics`) remain extremely unstable (F1=0 or F1=1.0 depending on the checkpoint). However, because these one-sample classes lack statistically significant support, the overall PPO optimization strategy should not be constrained to cater exclusively to them.
