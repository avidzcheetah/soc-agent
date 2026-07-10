# SecBERT Fine-Tuning Research Protocol

This document defines the rigorous research protocol and methodology for fine-tuning the `jackaduma/SecBERT` model on the combined CISSM and rcATT datasets for automated SOC incident response action selection.

## 1. Data Verification and Cleaning Rules
- **Schema Alignment:** Both datasets must be reduced to a unified schema consisting of three columns:
  - `text` — the input feature (incident description / threat report).
  - `action_label` — the target variable (integer 0–19).
  - `source_dataset` — provenance tag (`"CISSM"` or `"rcATT"`). This column is **not** used as a model feature, but is preserved for post-hoc analysis (e.g., "Does SecBERT perform equally well on both datasets?"). Without this column, dataset-level performance breakdowns are lost permanently.
  
  All other auxiliary columns (e.g., source IP, tactic booleans) will be dropped prior to merging.
- **Missing Values and Garbage Filtering:** Any row containing `NaN`, `NULL`, empty strings, pure whitespace, or placeholders like `"N/A"` or `"unknown"` in the `text` or `action_label` columns will be strictly dropped. Additionally, to ensure semantic substance, records where the `text` field length is less than 15 characters will be removed as garbage data.
- **Label Validation:** `action_label` must be an integer bounded between `0` and `19` inclusive. Furthermore, we will explicitly verify that every unique label present in the datasets actually exists in the `Index` column of `data/action_space.csv`. Any label mismatch will trigger a critical failure, as we cannot trust preprocessing outputs implicitly.

## 2. Dataset Merging and Leakage Prevention
- **Concatenation:** The 16,728 records from CISSM and 1,490 records from rcATT will be vertically concatenated to form a single corpus of 18,218 records.
- **Shuffle & Random State:** The merged dataset will be globally shuffled using a fixed random seed (`42`) to ensure reproducibility.
- **Leakage Prevention:** 
  - **Stage 1: Exact Duplicate Removal:** Remove all exact string matches in the `text` column before splitting.
  - **Stage 2: Near-Duplicate Detection:** To prevent "Near-Duplicate Leakage" (e.g., templates differing only by IP addresses, hashes, or domain names), we will apply near-duplicate detection. We will compute pairwise similarity using TF-IDF with Cosine Similarity (threshold ≥ 0.90) or SentenceTransformer (`all-MiniLM-L6-v2`) embeddings. Records exceeding the similarity threshold within the same class or across splits will be pruned, ensuring the model generalizes rather than memorizing templates.
  - **Stage 3: Train/Test Split:** Split the remaining clean, deduplicated dataset.

## 3. Label Consistency Checks
- **Class Imbalance Analysis:** Based on preliminary data analysis, the dataset exhibits significant class imbalance. For example, Action 16 (`patch_vulnerability`) and Action 7 (`isolate_host`) dominate the CISSM dataset, while Actions 13 (`remove_persistence`) and 11 (`reset_credentials`) are frequent in rcATT. Actions 2 (`create_ioc_alert`) and 18 (`sandbox_redirect`) have exactly zero samples across both datasets.
- **Mitigation Strategy:** 
  - **Class Weights:** We will compute class weights based on inverse class frequencies using scikit-learn (`compute_class_weight`) and pass these to the `CrossEntropyLoss` function during PyTorch training to penalize minority misclassifications.
  - **Zero-Shot Classes:** Zero-shot classes (classes with zero labeled training instances, such as Actions 2 and 18) remain in the classification head (preserving `num_labels = 20`) to preserve compatibility with the downstream 20-action reinforcement learning policy. These classes receive zero loss weight (weight = 0.0) and are explicitly excluded from performance interpretation because no supervised learning signal exists for them. This maintains consistent interface sizes across the pipeline.
  - **WeightedRandomSampler:** We will apply a `WeightedRandomSampler` to the PyTorch `DataLoader` to adjust which batches are seen, ensuring minority classes are sufficiently sampled. Using both techniques together guarantees much better minority learning.
  - Stratification will be strictly enforced during data splitting.

## 4. Token Length Analysis and Sequence Length Selection
- **Distribution Analysis via SecBERT Tokenizer:** Counting words is insufficient for cybersecurity data (e.g., `CVE-2025-49122` becomes multiple BERT tokens like `CV ## E - 2025 - 49122`). Therefore, we will run the official `jackaduma/SecBERT` tokenizer over every report in the pristine dataset.
- **Sequence Length Selection:** We will calculate the average token length, the 95th percentile, and the maximum token output. Only based on these precise tokenizer statistics will we decide whether `MAX_LEN = 512` is appropriate, or if a smaller sequence length suffices.
- **Truncation Policy:** We will evaluate the structure of the incident reports to justify the truncation strategy. If crucial data (e.g., "Recommended response", "IOC") is typically found at the end of the text, we will investigate whether `head-only`, `head+tail`, or a `sliding window` is optimal before defaulting to standard right-side truncation.

## 5. Split Strategy
- **Ratio:** The deduplicated dataset will be split into **80% Training**, **10% Validation**, and **10% Test**.
- **Stratified Splitting:** Due to the severe class imbalance, `StratifiedShuffleSplit` (or `train_test_split` with `stratify=y`) will be used to ensure the train, validation, and test sets maintain the exact same proportional distribution of the 20 action classes.
- **Reproducibility:** A fixed global random seed (`42`) will be strictly enforced during data splitting to ensure identical splits across experiments.
- **Edge Cases:** Any class with fewer than 3 samples total will be isolated and warned about, as stratified splitting requires at least a few samples per class.

## 6. Hyperparameter Justification
- **Learning Rate:** We will default to **3e-5**. Fine-tuning Transformer models on domain-specific downstream tasks typically converges best with lower learning rates to avoid catastrophic forgetting of the pre-trained SecBERT representations.
- **Gradient Clipping:** Explicitly set to **1.0**. Gradient clipping is an almost universal best practice for BERT fine-tuning to prevent exploding gradients during the initial stages of training.
- **Weight Decay:** Explicitly set to **0.01** to penalize large weights and act as a strong regularizer.
- **Dropout:** We will leave the native SecBERT dropout probability (typically `0.1`) intact, as the pre-trained checkpoint expects this level of regularization. If we determine we need to change it (e.g. to combat severe overfitting), the decision will be explicitly documented and justified.
- **Batch Size:** **16** or **32** (depending on VRAM limits in the cloud). Batch size 16 with gradient accumulation steps = 2 provides an effective batch size of 32, ensuring stable gradient updates despite noisy labels.
- **Epochs:** **4 to 5**. Transformer fine-tuning usually peaks quickly. We will set `num_epochs=5` with early stopping.
- **Optimizer:** `AdamW` with linear learning rate decay and a warmup ratio of 0.1 (10% of total training steps).

## 6.5 Global Random Seed Enforcement
To guarantee identical training loops and weight initialization, and because GPU architectures can introduce non-deterministic operations, we will strictly enforce global random seeds (`42`) for `python`, `numpy`, `torch`, and `transformers` before training begins:
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

## 6.7 Pre-Training Hardware Verification
Before training begins, the script must explicitly log the environment to ensure reproducibility:
- CUDA Available (Boolean)
- GPU Name and VRAM capacity
- CUDA Version and PyTorch Version

## 7. Training Configuration & Logging
- **Model Checkpoint:** `jackaduma/SecBERT`
- **Framework:** PyTorch & HuggingFace `Trainer` API.
- **Mixed Precision (FP16):** Enabled to cut training time in half and reduce memory usage.
- **Loss Function:** Weighted Cross-Entropy Loss.
- **Logging & TensorBoard:** TensorBoard will be fully enabled to plot Training Loss, Validation Loss, Macro F1, and Learning Rate curves over time. At each epoch, we will log: Training Loss, Validation Loss, Learning Rate, Macro F1, Accuracy, Precision, and Recall.
- **Checkpointing:** We will save a checkpoint at the end of *every* epoch, in addition to the best model. Storage is cheap; lost checkpoints are expensive.

## 8. Validation Strategy & Model Selection
- **Early Stopping:** Early stopping will monitor **Validation Loss** (`val_loss`) with a patience of 2 epochs to halt training when overfitting begins.
- **Best Checkpoint Selection:** The "best" model will be strictly selected based on the highest **Macro F1-Score**, not the lowest validation loss. This decouples the stopping criteria from the selection criteria.

## 9. Final Evaluation Protocol
- **Hold-out Test Set:** After the best model is selected, it will be evaluated exactly once on the 10% Hold-out Test Set.
- **Extensive Metrics:** In addition to Macro F1, we will report: Weighted F1, Micro F1, Balanced Accuracy, and the **Matthews Correlation Coefficient (MCC)**. MCC is highly respected for imbalanced classification as it considers all elements of the confusion matrix.
- **Normalized Confusion Matrix:** A visually interpretable, normalized confusion matrix will be generated and saved to `results/`.
- **Experiment Tracking:** Every run must generate an `experiment.json` containing: date, learning rate, batch size, epochs, seed, git commit, dataset size, token length, best F1, and best epoch.

## 10. Final Encoder Verification (Readiness for RL)
- **Sanity Check:** Before passing the frozen SecBERT encoder to the PPO reinforcement learning agent, we will pass 20 random incident reports through it to verify the embeddings.
- **Assertions:** We will explicitly assert that the output dimension is exactly `768`, that there are no `NaN`s or `Inf`s, and that the vector norms are reasonable.
