# Thesis Notes: EXP_002A (Negative Result)

## 1. The Double-Weighting Problem

**Hypothesis:** Adding a weighted CrossEntropyLoss function to the training pipeline will force the model to prioritize rare classes, thereby improving the Macro F1 score on imbalanced cybersecurity datasets.

**Result:** The experiment (**EXP_002A**) resulted in a significant degradation of performance across all metrics compared to the baseline (**EXP_001**):
- **Accuracy:** 0.899 → 0.801 (-9.8%)
- **Macro F1:** 0.625 → 0.574 (-0.051)

**Conclusion:** Replacing the default CrossEntropyLoss with an inverse-frequency weighted version is insufficient to improve Macro F1 and is actively harmful under the current sampling strategy. 

**Root Cause:** The training pipeline already utilized a `WeightedRandomSampler` to equalize class representation in each batch. By applying the same inverse-frequency weights to the loss function, the minority classes were doubly compensated. For example, the rarest class (Class 17) was oversampled by a factor of ~456x in the batch composition, and its loss gradient was amplified by another 95x. This compounding effect led to a catastrophic overemphasis on minority-class gradients, causing the model to rapidly overfit on rare samples and collapse on majority classes (e.g., Class 8 recall dropped from 0.81 to 0.54).

## 2. The Value of Top-2 Accuracy for Downstream PPO

In EXP_002A, a new metric was introduced: **Top-2 Accuracy**.

The test set evaluation reported a Top-2 Accuracy of **0.903** (90.3%). This means that even when the model misclassifies a security incident, the correct action is within its top two predictions 90% of the time. 

**Thesis Impact:** This is a highly encouraging signal for the downstream Reinforcement Learning (PPO) phase. The classification head's primary purpose is to pre-train the underlying SecBERT encoder so that the hidden state representations encode discriminative information about SOC actions. The high Top-2 accuracy demonstrates that the encoder successfully structures the embedding space such that the correct action is almost always nearby, even if the absolute highest logit is slightly miscalibrated. The PPO policy network will be able to extract this latent structure to choose optimal actions during simulation.

## 3. Revised Methodology: Strict Ablation

Moving forward, the research methodology follows a strict ablation study approach:
1. Every experiment branches from the proven baseline (**EXP_001**).
2. Only one variable is changed per experiment to isolate causality.
3. Only techniques proven to improve the baseline will be combined in the final model (**EXP_003**).

The next phase explores Label Smoothing (**EXP_002B**) and CISA Phase Tagging (**EXP_002C**). If these regularization and semantic techniques fail to elevate the rare classes, the study will have established strong empirical justification for introducing generative data augmentation.

## 4. Architectural Adjustment: SecBERT and PPO Separation

Rather than describing SecBERT as predicting the final response action, the architecture should be described as follows:

**SecBERT is fine-tuned on incident-response labels so that its encoder learns incident semantics. During the reinforcement learning phase, the classifier head is discarded. The frozen encoder generates contextual incident embeddings, which form part of the PPO agent's state representation. The PPO agent is responsible for selecting the optimal response action and sequence of actions.**

This aligns the implementation with standard reinforcement learning practice and clearly separates the roles of representation learning (SecBERT) and decision-making (PPO). It resolves the apparent contradiction: SecBERT is not replacing PPO, but rather providing the rich semantic understanding that allows PPO to make better decisions.

## 5. Final Model Selection: FP32 Precision Experiment (EXP_002G / EXP_20260727_001)

The FP32 experiment was conducted to verify whether mixed-precision training influenced the model. Although the Macro F1 decreased slightly from 0.7296 to 0.7121, the FP32 model achieved the highest Weighted F1 (0.9454), the highest MCC (0.9364), and the highest Top-2 Accuracy (0.9695). Since our dataset is highly imbalanced and most real SOC events belong to the common action classes, Weighted F1 and MCC provide a more representative measure of overall deployment performance. Therefore, we selected the FP32 model as the final model.

## 6. Formulation of the SOC Environment as a Contextual Bandit

**Formal Statement for Thesis & Methodology:**

> *"The environment is formulated as a contextual bandit, where each SOC alert is treated as an independent decision point. Consequently, the PPO agent optimizes per-incident action selection without modeling temporal dependencies between alerts. This formulation is appropriate because the objective of the proposed system is to recommend the most suitable response for each individual security incident rather than to learn long-horizon control policies."*

### Mathematical Justification:
In standard Markov Decision Processes (MDPs), transitions follow $P(s_{t+1} | s_t, a_t)$ where actions alter the environment state dynamics across sequential trajectories. 

In enterprise Security Operations Centers:
1. **Incident Granularity:** Each security alert represents a discrete, self-contained event signature (e.g., an individual C2 beacon, brute force spike, or privilege escalation attempt).
2. **Action Independence:** Selecting a mitigation action $a_t \in \{0, \dots, 19\}$ for incident $t$ yields immediate feedback (correct containment vs. operational penalty: $r_t \in \{+1.0, -1.0\}$) without dictating the arrival distribution or semantic content of subsequent un-correlated alerts.
3. **PPO Applicability:** Utilizing Proximal Policy Optimization under this contextual bandit formulation allows the agent to maintain stable policy improvement, leverage Generalized Advantage Estimation across rollout batches, and prevent policy collapse via entropy regularization while learning non-linear policy mappings over 768-dimensional contextual representations.


## 7. Evaluation Metrics and Checkpoint Selection for PPO

To establish a robust, research-grade evaluation pipeline for the PPO agent in the presence of an imbalanced 20-action space:

1. **Deterministic Evaluation**: During evaluation phases, the PPO agent uses greedy rgmax selection over policy logits instead of stochastic sampling. This provides a true measure of the learned policy's capability rather than its exploration noise.
2. **Macro F1 Metric**: Because the SOC dataset contains highly imbalanced response classes (e.g., monitor is overwhelmingly more common than quarantine_file), pure accuracy is an insufficient metric. A policy could achieve high accuracy by merely predicting the majority class. Therefore, the evaluation pipeline computes **Macro F1**, **Weighted F1**, and **MCC (Matthews Correlation Coefficient)**.
3. **Checkpoint Strategy**: The "best" PPO policy checkpoint (est_ppo_policy.pt) is saved exclusively when the validation **Macro F1** improves, ensuring the final research model is the one that generalized best across all response classes, not just the majority classes.
## 8. Step 10.4 PPO Training Methodology

The final PPO training experiment utilizes the following configuration:
- **Total steps**: 204,800 (100 iterations × 2,048 rollout steps)
- **Validation**: 1,539 samples (evaluated entirely every 5 iterations to capture a highly stable Macro F1 metric)

**Stochastic Sampling Strategy**: 
The 204,800 environment interactions do *not* represent 204,800 unique incidents. The environment continuously samples stochastically from the pool of 12,312 distinct training incidents. Because the environment is formulated as an independent contextual bandit (done=True at every step), this repeated random sampling is mathematically equivalent to independently drawing i.i.d. incident contexts from the dataset distribution. The PPO agent optimizes the policy by repeatedly experiencing permutations of these incidents over multiple epochs, analogous to standard supervised mini-batch training.
