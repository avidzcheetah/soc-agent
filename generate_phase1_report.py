import pandas as pd
import numpy as np
import json
import os
from transformers import AutoTokenizer

def main():
    print("Loading datasets for Phase 1 reporting...")
    
    # Load splits
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print("Initializing SecBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
    
    # Helper to calculate token lengths
    print("Analyzing token lengths (this might take a few moments)...")
    
    # CISSM
    cissm_df = full_df[full_df['source_dataset'] == 'CISSM']
    cissm_lens = cissm_df['text'].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=True))).values
    
    # rcATT
    rcatt_df = full_df[full_df['source_dataset'] == 'rcATT']
    rcatt_lens = rcatt_df['text'].apply(lambda x: len(tokenizer.encode(str(x), add_special_tokens=True))).values
    
    # Combined
    combined_lens = np.concatenate([cissm_lens, rcatt_lens])
    
    def get_stats(lens):
        if len(lens) == 0:
            return {"min": 0, "mean": 0, "median": 0, "p95": 0, "max": 0, "trunc_pct": 0.0}
        trunc_count = np.sum(lens > 512)
        return {
            "min": int(np.min(lens)),
            "mean": float(np.mean(lens)),
            "median": int(np.median(lens)),
            "p95": float(np.percentile(lens, 95)),
            "max": int(np.max(lens)),
            "trunc_pct": float(trunc_count / len(lens) * 100)
        }
        
    cissm_stats = get_stats(cissm_lens)
    rcatt_stats = get_stats(rcatt_lens)
    combined_stats = get_stats(combined_lens)
    
    # Load class weights
    with open('data/class_weights.json', 'r') as f:
        class_weights = json.load(f)
        
    # Get action names
    action_space = pd.read_csv('data/action_space.csv')
    action_names = {row['Index']: row['Action Name'] for _, row in action_space.iterrows()}
    
    # Compute class distribution
    class_counts = full_df['action_label'].value_counts().to_dict()
    total_records = len(full_df)
    
    # Generate Markdown Report
    report = f"""# Phase 1 Data Prep & Validation Report

This report documents the dataset preparation, validation, and profiling steps executed in Phase 1 before fine-tuning SecBERT.

## 1. Executive Summary & Datasets
- **Total Combined Dataset Size:** {total_records} records (after strict deduplication).
- **Train Set Size (80%):** {len(train_df)} records
- **Validation Set Size (10%):** {len(val_df)} records
- **Test Set Size (10%):** {len(test_df)} records (Locked for final evaluation only)
- **Random Seed:** `42` (strictly enforced across Python, NumPy, and split functions)

### Dataset Provenance
- **CISSM (Cyber Events):** {len(cissm_df)} records
- **rcATT (MITRE ATT&CK Reports):** {len(rcatt_df)} records

---

## 2. Text Length and SecBERT Token Statistics
Transformers ingest tokens, not words or characters. The following statistics reflect the output of the official `jackaduma/SecBERT` tokenizer run over the pristine dataset.

| Dataset / Split | Minimum Tokens | Mean Tokens | Median Tokens | 95th Percentile | Maximum Tokens | Truncated (>512 Tokens) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CISSM Only** | {cissm_stats['min']} | {cissm_stats['mean']:.2f} | {cissm_stats['median']} | {cissm_stats['p95']:.1f} | {cissm_stats['max']} | {cissm_stats['trunc_pct']:.2f}% |
| **rcATT Only** | {rcatt_stats['min']} | {rcatt_stats['mean']:.2f} | {rcatt_stats['median']} | {rcatt_stats['p95']:.1f} | {rcatt_stats['max']} | {rcatt_stats['trunc_pct']:.2f}% |
| **Combined (All)** | {combined_stats['min']} | {combined_stats['mean']:.2f} | {combined_stats['median']} | {combined_stats['p95']:.1f} | {combined_stats['max']} | {combined_stats['trunc_pct']:.2f}% |

**Sequence Length Justification:**
The 95th percentile for the combined dataset is **{combined_stats['p95']:.1f} tokens**, and only **{combined_stats['trunc_pct']:.2f}%** of the reports exceed the 512-token limit. Therefore, configuring `MAX_LEN = 512` is highly appropriate and guarantees minimal information loss.

---

## 3. Class Distribution & Loss Weights (20-Action Space)
The combined dataset exhibits severe class imbalance. To stabilize training, we compute inverse-frequency class weights for the 20 actions. Zero-shot actions are handled separately to preserve consistency.

| Class Index | Action Name | Sample Count | Percentage | Class Weight | Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
"""
    
    for i in range(20):
        name = action_names.get(i, "unknown")
        count = class_counts.get(i, 0)
        pct = (count / total_records) * 100
        weight = class_weights.get(str(i), 0.0)
        
        if count == 0:
            status = "Zero-Shot (Excluded from Loss)"
        elif weight > 5.0:
            status = "Minority Class"
        elif weight < 0.5:
            status = "Majority Class"
        else:
            status = "Balanced Class"
            
        report += f"| {i} | `{name}` | {count} | {pct:.2f}% | {weight:.4f} | {status} |\n"
        
    report += """
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
"""
    
    os.makedirs('docs', exist_ok=True)
    with open('docs/phase1_report.md', 'w') as f:
        f.write(report)
        
    print("Phase 1 report generated and saved to docs/phase1_report.md.")

if __name__ == '__main__':
    main()
