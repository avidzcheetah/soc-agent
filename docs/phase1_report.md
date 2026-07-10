# Phase 1 Data Prep & Validation Report

This report documents the dataset preparation, validation, and profiling steps executed in Phase 1 before fine-tuning SecBERT.

## 1. Executive Summary & Datasets
- **Total Combined Dataset Size:** 15390 records (after strict deduplication).
- **Train Set Size (80%):** 12312 records
- **Validation Set Size (10%):** 1539 records
- **Test Set Size (10%):** 1539 records (Locked for final evaluation only)
- **Random Seed:** `42` (strictly enforced across Python, NumPy, and split functions)

### Dataset Provenance
- **CISSM (Cyber Events):** 14370 records
- **rcATT (MITRE ATT&CK Reports):** 1020 records

---

## 2. Text Length and SecBERT Token Statistics
Transformers ingest tokens, not words or characters. The following statistics reflect the output of the official `jackaduma/SecBERT` tokenizer run over the pristine dataset.

| Dataset / Split | Minimum Tokens | Mean Tokens | Median Tokens | 95th Percentile | Maximum Tokens | Truncated (>512 Tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CISSM Only** | 26 | 61.20 | 57 | 101.0 | 394 | 0.00% |
| **rcATT Only** | 73 | 438.30 | 382 | 857.6 | 1688 | 11.47% |
| **Combined (All)** | 26 | 86.20 | 58 | 350.0 | 1688 | 0.76% |

**Sequence Length Justification:**
The 95th percentile for the combined dataset is **350.0 tokens**, and only **0.76%** of the reports exceed the 512-token limit. Therefore, configuring `MAX_LEN = 512` is highly appropriate and guarantees minimal information loss.

---

## 3. Class Distribution & Loss Weights (20-Action Space)
The combined dataset exhibits severe class imbalance. To stabilize training, we compute inverse-frequency class weights for the 20 actions. Zero-shot actions are handled separately to preserve consistency.

| Class Index | Action Name | Sample Count | Percentage | Class Weight | Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `monitor` | 1228 | 7.98% | 0.6963 | Balanced Class |
| 1 | `enable_deep_logging` | 74 | 0.48% | 11.5541 | Minority Class |
| 2 | `create_ioc_alert` | 0 | 0.00% | 0.0000 | Zero-Shot (Excluded from Loss) |
| 3 | `block_source_ip` | 878 | 5.71% | 0.9738 | Balanced Class |
| 4 | `block_dest_ip` | 100 | 0.65% | 8.5500 | Minority Class |
| 5 | `dns_sinkhole` | 125 | 0.81% | 6.8400 | Minority Class |
| 6 | `block_port` | 12 | 0.08% | 71.2500 | Minority Class |
| 7 | `isolate_host` | 3378 | 21.95% | 0.2531 | Majority Class |
| 8 | `kill_process` | 1727 | 11.22% | 0.4951 | Majority Class |
| 9 | `quarantine_file` | 573 | 3.72% | 1.4921 | Balanced Class |
| 10 | `quarantine_email` | 515 | 3.35% | 1.6602 | Balanced Class |
| 11 | `reset_credentials` | 820 | 5.33% | 1.0427 | Balanced Class |
| 12 | `disable_account` | 1497 | 9.73% | 0.5711 | Balanced Class |
| 13 | `remove_persistence` | 212 | 1.38% | 4.0330 | Balanced Class |
| 14 | `restore_registry` | 17 | 0.11% | 50.2941 | Minority Class |
| 15 | `restore_defense_config` | 94 | 0.61% | 9.0957 | Minority Class |
| 16 | `patch_vulnerability` | 4105 | 26.67% | 0.2083 | Majority Class |
| 17 | `snapshot_forensics` | 9 | 0.06% | 95.0000 | Minority Class |
| 18 | `sandbox_redirect` | 0 | 0.00% | 0.0000 | Zero-Shot (Excluded from Loss) |
| 19 | `escalate_to_human` | 26 | 0.17% | 32.8846 | Minority Class |

### Zero-Shot Classes Justification
Actions **2 (`create_ioc_alert`)** and **18 (`sandbox_redirect`)** contain exactly zero training samples.
- **Why they remain:** To preserve interface compatibility with the downstream 20-action reinforcement learning (PPO) policy.
- **How they are handled:** They are assigned a loss weight of `0.0`. They remain in the classification head, but are explicitly excluded from classification metric evaluation as no learning signal exists for them.

---

## 4. Preprocessing & Cleanliness Decisions
1. **Schema Standardization:** Both inputs aligned to `['text', 'action_label', 'source_dataset']`.
2. **Strict Garbage Filtering:** Dropped all reports under 15 characters or consisting of generic placeholders (`null`, `nan`, `n/a`, `unknown`).
3. **No Target Alterations:** Kept critical indicators (CVEs, IPs, domain names, file hashes) intact to preserve SecBERT's domain-specific capability.
4. **Near-Duplicate Leakage Prevention:** Removed 2,604 near-duplicate entries using a TF-IDF character-level Cosine Similarity threshold of `≥ 0.90` (preventing train-test contamination).
5. **Stratified Split:** Split using `StratifiedShuffleSplit` (80/10/10) to guarantee identical label proportions across train, val, and test splits.
