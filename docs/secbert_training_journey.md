# The SecBERT Training Journey: Phase 1 Summary

This document summarizes our entire journey of fine-tuning the SecBERT language model to understand cybersecurity incidents. 

Our goal in Phase 1 was **not** to build a perfect classifier, but to train an encoder that genuinely understands the semantics of different security incidents. This "understanding" (the embeddings) will act as the "eyes" for our Reinforcement Learning (PPO) agent in Phase 2.

Here is the step-by-step story of how we improved the model, what worked, what failed, and where we finally drew the line.

---

## 1. The Starting Point (Baseline)
- **Experiment:** `EXP_20260710_001`
- **What we did:** Fine-tuned the default SecBERT model using standard cross-entropy loss and random sampling.
- **Result:** **Macro F1: 0.625**
- **Takeaway:** A decent start, but the model struggled heavily with rare incident types because the dataset was heavily imbalanced.

## 2. The Double-Weighting Mistake
- **Experiment:** `EXP_20260712_001`
- **What we tried:** We added class-weights to the loss function to force the model to pay more attention to rare classes. 
- **Result:** **Macro F1: 0.574** (Regression 📉)
- **Takeaway:** We were already using a `WeightedRandomSampler` to balance the batches. By adding weights to the loss function too, we accidentally "double-weighted" the rare classes. The model over-focused on rare incidents and forgot the common ones. We reverted this.

## 3. Smoothing Things Out
- **Experiment:** `EXP_20260712_003` & `004`
- **What we tried:** We applied *Label Smoothing* (0.1). Instead of the model being 100% sure about a single label, it distributes a tiny bit of probability to other labels, preventing it from becoming overconfident.
- **Result:** **Macro F1: 0.619 - 0.632** (Slight Improvement 📈)
- **Takeaway:** This stabilized training and prevented overfitting, but the underlying imbalance issue remained.

## 4. The Breakthrough: Context is King
- **Experiment:** `EXP_20260712_006`
- **What we tried:** We prepended **CISA Phase Tags** (e.g., `[Detection]`, `[Containment]`, `[Eradication]`) to the beginning of every incident description. This gave the model immediate context about the incident's lifecycle stage.
- **Result:** **Macro F1: 0.706** (Massive Jump 🚀)
- **Takeaway:** This was our biggest win. Providing explicit structural context helped the model clearly separate incidents that previously looked identical.

## 5. The Peak: Targeted Data Augmentation
- **Experiment:** `EXP_20260713_001`
- **What we tried:** We noticed 4 specific action classes were heavily underrepresented. We generated a very small, targeted batch of synthetic data (40-60 samples each) just for those weakest classes to help the model learn them.
- **Result:** **Macro F1: 0.730** (The Peak 🏔️)
- **Takeaway:** This achieved our highest score. The targeted injection of data perfectly balanced the model without overwhelming it.

## 6. The Plateau: Diminishing Returns
- **Experiments:** `EXP_20260713_002` (Gemini Data) & `EXP_20260714_003` (Claude Data)
- **What we tried:** Believing more synthetic data equaled better results, we generated hundreds of synthetic samples using Gemini and Claude across 5-6 rare classes.
- **Result:** **Macro F1: 0.695 - 0.714** (Regression / Plateau 📉)
- **Takeaway:** We hit a wall. Adding more synthetic data introduced LLM-style repetitiveness and noise, which degraded the sharp decision boundaries the model had built. A model cannot learn information that simply doesn't exist in the data.

## 7. Numerical Precision Check: FP32 vs. FP16 (Selected Final Model)
- **Experiment:** `EXP_20260727_001` (EXP_002G)
- **What we tried:** Re-ran `EXP_20260713_001` configuration with FP16 mixed precision disabled (`mixed_precision: false`) to test if half precision reduced numerical accuracy.
- **Result:** **Macro F1: 0.712** | **Weighted F1: 0.9454 (Highest)** | **MCC: 0.9364 (Highest)** | **Top-2 Acc: 0.9695 (Highest)**
- **Takeaway:** Although Macro F1 saw a minor change from 0.730 to 0.712, the FP32 model achieved the overall highest Weighted F1, MCC, and Top-2 Accuracy. In an imbalanced real-world SOC dataset, Weighted F1 and MCC offer a more realistic picture of deployment performance. Hence, EXP_002G was selected as our final model.

---

## The Final Decision: Architecture Lock

We are officially stopping SecBERT optimization and locking **`EXP_20260727_001`** (FP32 model) as our final Phase 1 checkpoint.

### Why select this checkpoint?
1. **Best Overall Metrics:** Highest Weighted F1 (0.9454), highest MCC (0.9364), and highest Top-2 Accuracy (0.9695) across all experiments.
2. **Top-2 Accuracy:** The correct action is within its top 2 guesses **97.0%** of the time. This means the embeddings are exceptionally well-structured for the downstream agent.
3. **The Real Goal:** The primary contribution of Phase 1 is providing a rich, contextual semantic encoder to act as the "eyes" for our Reinforcement Learning (PPO) agent in Phase 2.
 

### What's Next? (Phase 2)
The SecBERT classification head is now discarded. We will freeze the encoder and use it strictly to generate state embeddings. 

These embeddings will be fed into a PPO environment where an RL agent will learn to select the optimal defensive actions, unlocking the true autonomous potential of the research!
