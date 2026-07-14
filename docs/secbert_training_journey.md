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

---

## The Final Decision: Architecture Lock

We are officially stopping SecBERT optimization and locking **`EXP_20260713_001`** as our final model.

### Why stop here?
1. **The Plateau:** We have squeezed every bit of performance out of this dataset (Smoothing, Context Tags, Synthetic Augmentation). Going from 0.73 to 0.74 is not worth weeks of compute time.
2. **Top-2 Accuracy:** Even when the model is wrong, the correct action is in its top 2 guesses **96.8%** of the time. This means the embeddings are incredibly rich and well-structured.
3. **The Real Goal:** The thesis contribution is **not** building the world's most accurate standalone SecBERT classifier. The novelty is combining a transformer with a **Reinforcement Learning Agent (PPO)**. 

### What's Next? (Phase 2)
The SecBERT classification head is now discarded. We will freeze the encoder and use it strictly to generate state embeddings. 

These embeddings will be fed into a PPO environment where an RL agent will learn to select the optimal defensive actions, unlocking the true autonomous potential of the research!
