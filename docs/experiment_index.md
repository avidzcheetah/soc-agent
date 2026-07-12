# Experiment Index

This document maps our logical experiment names (e.g., `EXP_001`) to the automatically generated system IDs (e.g., `EXP_20260710_001`) and tracks the primary changes and outcomes of each phase.

## Phase 1 & 2: Baseline and Strict Ablation

| Logical Name | System ID | Description | Result / Verdict |
|--------------|-----------|-------------|------------------|
| **EXP_001** | `EXP_20260710_001` | **Baseline.** Default CrossEntropyLoss, `WeightedRandomSampler`, 10 epochs. | Macro F1: 0.625. Top-2 Acc: N/A. (Baseline reference) |
| **EXP_002A** | `EXP_20260712_001` | **Weighted CE Loss.** Replaced default loss with inverse-frequency weighted CE. Kept sampler. | ❌ Macro F1: 0.574 (-0.05). Double-weighting caused majority-class collapse. *(Negative Result)* |
| **EXP_002B** | `EXP_20260712_003` | **Label Smoothing.** Reverted weighted CE. Added `label_smoothing=0.1`. | ✅ Macro F1: 0.632 (+0.007). Top-2 Acc: 0.950. Stabilized training, but stopped due to 10-epoch limit. |
| **EXP_002B_Extended** | `EXP_20260712_004` | **Label Smoothing (Extended).** Same as EXP_002B but extended to 20 epochs with patience 4 to allow full convergence. | ❌ Macro F1: 0.614. Model overfit validation split. Stopped hyperparam tuning. |
| **EXP_002C** | `(Pending)` | **CISA Phase Tags.** Base = EXP_002B (10 epochs, ls=0.1). Prepended CISA Phase (e.g. `[Detection]`) to all text inputs. | *(Pending)* |
*Note: Future experiments (e.g. Synthetic Augmentation) will continue this ablation sequence depending on the outcome of EXP_002C.*
