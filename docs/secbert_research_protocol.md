# SecBERT Fine-Tuning Research Protocol

This document defines the rigorous research protocol and methodology for fine-tuning the `jackaduma/SecBERT` model on the combined CISSM and rcATT datasets for automated SOC incident response action selection.

## 1. Data Verification and Cleaning Rules
- **Schema Alignment:** Both datasets must be reduced to a unified schema consisting of two essential columns: `text` (the input feature) and `action_label` (the target variable, integer 0-19). All other auxiliary columns (e.g., source IP, tactic booleans) will be dropped prior to merging.
- **Missing Values:** Any row with a `NaN` or empty `text` or `action_label` will be strictly dropped.
- **Label Validation:** `action_label` must be an integer bounded between `0` and `19` inclusive.

## 2. Dataset Merging and Leakage Prevention
- **Concatenation:** The 16,728 records from CISSM and 1,490 records from rcATT will be vertically concatenated to form a single corpus of 18,218 records.
- **Shuffle & Random State:** The merged dataset will be globally shuffled using a fixed random seed (`42`) to ensure reproducibility.
- **Leakage Prevention:** Deduplication will be applied to the `text` column *before* splitting to ensure identical text strings do not appear in both the training and testing sets, which would artificially inflate evaluation metrics.

## 3. Label Consistency Checks
- **Class Imbalance Analysis:** Based on preliminary data analysis, the dataset exhibits significant class imbalance. For example, Action 16 (`patch_vulnerability`) and Action 7 (`isolate_host`) dominate the CISSM dataset, while Actions 13 (`remove_persistence`) and 11 (`reset_credentials`) are frequent in rcATT. Some actions (e.g., 2, 18) may have very few or zero samples.
- **Mitigation Strategy:** 
  - We will compute **Class Weights** based on inverse class frequencies using scikit-learn (`compute_class_weight`) and pass these weights to the CrossEntropyLoss function during PyTorch training.
  - Stratification will be strictly enforced during data splitting.

## 4. Token Length Analysis and Sequence Length Selection
- **Distribution Analysis:** 
  - **CISSM:** Mean word count is ~41 words; 95th percentile is ~77 words; Max is 260 words.
  - **rcATT:** Mean word count is ~261 words; 95th percentile is ~320 words; Max is 822 words.
- **Sequence Length Selection:** Since SecBERT uses a standard BERT architecture with a hard limit of **512 tokens**, and the 95th percentile of our longest dataset (rcATT) is ~320 words (which translates to roughly 400-450 subword tokens), we will set the `MAX_LEN` to **512**.
- **Truncation Policy:** `truncation=True` will be applied, truncating from the right (end of the text).

## 5. Split Strategy
- **Ratio:** The deduplicated dataset will be split into **80% Training**, **10% Validation**, and **10% Test**.
- **Stratified Splitting:** Due to the severe class imbalance, `StratifiedShuffleSplit` (or `train_test_split` with `stratify=y`) will be used to ensure the train, validation, and test sets maintain the exact same proportional distribution of the 20 action classes.
- **Edge Cases:** Any class with fewer than 3 samples total will be isolated and warned about, as stratified splitting requires at least a few samples per class.

## 6. Hyperparameter Justification
- **Learning Rate:** `2e-5` to `5e-5`. Fine-tuning Transformer models on domain-specific downstream tasks typically converges best with lower learning rates to avoid catastrophic forgetting of the pre-trained SecBERT representations. We will default to **3e-5**.
- **Batch Size:** **16** or **32** (depending on VRAM limits in the cloud). Batch size 16 with gradient accumulation steps = 2 provides an effective batch size of 32, ensuring stable gradient updates despite noisy labels.
- **Epochs:** **4 to 5**. Transformer fine-tuning usually peaks quickly. We will set `num_epochs=5` with early stopping.
- **Optimizer:** `AdamW` with linear learning rate decay and a warmup ratio of 0.1 (10% of total training steps).

## 7. Training Configuration
- **Model Checkpoint:** `jackaduma/SecBERT`
- **Framework:** PyTorch & HuggingFace `Trainer` API.
- **Mixed Precision (FP16):** Enabled to cut training time in half and reduce memory usage without losing precision in the gradients.
- **Loss Function:** Weighted Cross-Entropy Loss (to handle class imbalances).

## 8. Validation Strategy
- **Evaluation Strategy:** Evaluate at the end of each epoch (`evaluation_strategy="epoch"`).
- **Early Stopping:** Implemented with a patience of 2 epochs monitoring the validation loss (`val_loss`). If `val_loss` increases for two consecutive epochs, training halts to prevent overfitting.

## 9. Model Selection Criteria
- **Primary Metric:** **Macro F1-Score**. Because of the extreme class imbalance, raw Accuracy is a deceptive metric (a model could just guess Action 16 and get a high score). Macro F1 ensures that the model's performance is equally weighted across all 20 action classes, penalizing the model if it fails to predict minority classes.
- The model checkpoint with the highest validation Macro F1-Score will be saved as the definitive model to `models/secbert_finetuned/`.

## 10. Final Evaluation Protocol
- **Hold-out Test Set:** After the best model is selected, it will be evaluated exactly once on the 10% Hold-out Test Set.
- **Artifact Generation:** The script will output a detailed `classification_report` (Precision, Recall, F1 for each of the 20 classes) and plot a **Confusion Matrix** saved to the `results/` directory. 
- **Readiness for RL:** The final validated model will then be frozen (gradients disabled) and used exclusively as the state-embedding mechanism for the subsequent Proximal Policy Optimization (PPO) reinforcement learning phase.
