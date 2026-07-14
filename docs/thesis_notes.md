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
