# Experiment Index

This document maps our logical experiment names (e.g., `EXP_001`) to the system paths and tracks the primary changes and outcomes of each phase.

## Phase 1 & 2: Baseline and Strict Ablation

| Logical Name | System ID | Description | Result / Verdict |
|--------------|-----------|-------------|------------------|
| **EXP_001** | `EXP_20260710_001` | **Baseline.** Default CrossEntropyLoss, `WeightedRandomSampler`, 10 epochs. | Macro F1: 0.625. Top-2 Acc: N/A. (Baseline reference) |
| **EXP_002A** | `EXP_20260712_001` | **Weighted CE Loss.** Replaced default loss with inverse-frequency weighted CE. Kept sampler. | ❌ Macro F1: 0.574 (-0.05). Double-weighting caused majority-class collapse. *(Negative Result)* |
| **EXP_002B** | `EXP_20260712_003` | **Label Smoothing.** Reverted weighted CE. Added `label_smoothing=0.1`. | ✅ Macro F1: 0.632 (+0.007). Top-2 Acc: 0.950. Stabilized training, but stopped due to 10-epoch limit. |
| **EXP_002B_Extended** | `EXP_20260712_004` | **Label Smoothing (Extended).** Same as EXP_002B but extended to 20 epochs with patience 4 to allow full convergence. | ❌ Macro F1: 0.614. Model overfit validation split. Stopped hyperparam tuning. |
| **EXP_002C** | `EXP_20260712_006` | **CISA Phase Tags.** Base = EXP_002B (10 epochs, ls=0.1). Prepended CISA Phase (e.g. `[Detection]`) to all text inputs. | ✅ Macro F1: 0.706 (+0.074). Top-2 Acc: 0.968. Major Success. Resolved semantic ambiguity for majority classes. |
| **EXP_002D** | `EXP_20260713_001` | **Targeted Synthetic Augmentation.** Base = EXP_002C. Synthetically enrich the 4 weakest classes (`block_port`, `restore_registry`, `restore_defense_config`, `snapshot_forensics`) by 40-60 samples. | ✅ Macro F1: 0.730 (+0.024). Top-2 Acc: 0.968. Highest performance achieved. |
| **EXP_002E** | `EXP_20260713_002` | **Full Synthetic Augmentation.** Base = EXP_002D pipeline. Fully augmented 6 rare classes with 300 Gemini API generated samples. | ❌ Macro F1: 0.695 (-0.035). Top-2 Acc: 0.966. Added noise/LLM-style repetition degraded decision boundaries. Rejected. |
| **EXP_002F** | `EXP_20260714_003` | **Claude Targeted Augmentation.** Base = EXP_002C. Added highly templated, high-quality Claude synthetic data for 5 rarest classes. | ❌ Macro F1: 0.714 (-0.016 from 002D). Top-2 Acc: 0.968. Diminishing returns/plateau reached. Rejected. |
| **EXP_002G** | `EXP_20260727_001` | **FP32 Ablation.** Base = EXP_002D (EXP_20260713_001). Identical config with `mixed_precision: false` to measure FP16 precision impact. | ✅ Macro F1: 0.7375, Weighted F1: **0.9445**, MCC: **0.9347**, Acc: **94.48%**. Selected as final model checkpoint (`secbert_finetuned`). |

---

## Phase 2A: PPO Reward Formulation Experiments (Validation Split: 1,539 samples)

| Logical Name | System ID / Save Path | Description / Reward Formulation | Result / Verdict |
|--------------|-----------------------|----------------------------------|------------------|
| **EXP_PPO_000** | `models/ppo_sanity` | **Original PPO.** Naive scalar rewards (+1 correct, -1 incorrect). | Val Acc: 93.44%, Macro F1: 0.6114, Weighted F1: 0.9300, MCC: 0.9224. Baseline PPO. Severe minority collapse. |
| **EXP_PPO_001** | `models/ppo_exp1` | **Class-Aware Logarithmic Reward.** Inverse-frequency weighted: $W_c = 1.0 + \ln(N_{\max}/N_c)$. | Val Acc: 93.83%, Macro F1: **0.6281** (+0.0167), Weighted F1: 0.9342, MCC: 0.9268. Recovered `enable_deep_logging` (0.80). |
| **EXP_PPO_002** | `models/ppo_exp2` | **Response-Aware Reward (Semantic Phase Penalty).** Exact correct: +1.0; wrong action but correct CISA phase: -0.5; wrong phase: -1.0. | Val Acc: 93.83%, Macro F1: **0.6219**, Weighted F1: 0.9332, MCC: 0.9269. Recovered `escalate_to_human` (0.3636), `dns_sinkhole` (0.9231). |
| **EXP_PPO_003** | `models/ppo_exp3` | **Bounded Hybrid Reward (Class-Aware + Response-Aware).** Correct: $+W_c$; wrong action, same phase: $-0.5$; wrong phase: $-1.0$; false escalation: $-1.0$. | ❌ Val Acc: 93.11%, Macro F1: **0.6312**, Weighted F1: 0.9232, MCC: 0.9183. Rapid entropy collapse. **Reward shaping exhausted — pivot to structural fixes.** |

---

## Phase 2B & 2C: Warm-Start, Preservation & Balanced Sampling (Validation Split: 1,539 samples)

| Logical Name | Save Directory | Description | Result / Verdict |
|--------------|----------------|-------------|------------------|
| **EXP_WARMSTART** | `models/ppo/warmstart_actor.pt` | **Knowledge Distillation.** Distill SecBERT logits into PPO Actor MLP architecture. | ✅ Val Acc: **94.7368%**, Macro F1: **0.7438**, Weighted F1: **0.9467**, MCC: **0.9377**. Successfully reproduced SecBERT baseline. |
| **EXP_PPO_004A** | `models/ppo_exp4a` | **Warm-Start PPO.** PPO fine-tuning from warm-started Actor, uniform sampling, simple $+1/-1$ reward. | Val Acc: 93.57%, Macro F1: **0.7616** (Iter 20), Weighted F1: 0.9342, MCC: 0.9238. High Macro F1, but degraded overall accuracy/MCC. |
| **EXP_PPO_004A-1** | `models/ppo_exp4a1` | **KL-Constrained PPO.** Added KL penalty ($\beta=0.5$) relative to frozen warm-start reference policy. | ❌ Val Acc: 93.31%, Macro F1: **0.7027**. Over-constrained policy exploration. |
| **EXP_PPO_004A-2** | `models/ppo_exp4a2_bc` | **Supervised BC Preservation Loss.** Added supervised cross-entropy loss ($\lambda=0.05$) during PPO updates. | Val Acc: 94.28%, Macro F1: **0.7350**, Weighted F1: 0.9422, MCC: 0.9323. Retained baseline accuracy, but suppressed Macro F1. |
| **EXP_PPO_004B** | `models/ppo_exp4b_balanced` | **Capped Class-Balanced PPO.** Capped inverse-sqrt frequency sampling ($lr=3\text{e-}4$). | ✅ Val Acc: **94.61%**, Macro F1: **0.7423**, Weighted F1: **0.9452**, MCC: **0.9361**. First PPO variant to beat SecBERT across all 4 metrics. |
| **EXP_PPO_004C-0** | `models/ppo_exp4c0_balanced_lowlr` | **Conservative Balanced PPO.** Capped balanced sampling + reduced actor learning rate ($lr=1\text{e-}4$, 150 iterations). | 🏆 **Val Acc: 94.4120%, Macro F1: 0.7928, Weighted F1: 0.9433, MCC: 0.9338**. Peak at Iter 125. **Current State-of-the-Art PPO Model**. |
