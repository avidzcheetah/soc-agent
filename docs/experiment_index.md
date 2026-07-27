# Experiment Index

This document maps our logical experiment names (e.g., `EXP_001`) to the automatically generated system IDs (e.g., `EXP_20260710_001`) and tracks the primary changes and outcomes of each phase.

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
| **EXP_002G** | `EXP_20260727_001` | **FP32 Ablation.** Base = EXP_002D (EXP_20260713_001). Identical config with `mixed_precision: false` to measure FP16 precision impact. | ✅ Macro F1: 0.712, Weighted F1: **0.9454**, MCC: **0.9364**, Top-2 Acc: **0.970**. Selected as final model checkpoint. |

*Note: Following EXP_002G, we are officially locking **EXP_20260727_001** (FP32) as the final Phase 1 model checkpoint (Weighted F1: 0.9454, MCC: 0.9364, Top-2 Acc: 0.970). The SecBERT encoder is now frozen. Development moves entirely to Phase 2: Building the PPO environment and training the RL decision-making agent.*

