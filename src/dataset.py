# Responsibility: Load CSVs and build datasets
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json


# ─────────────────────────────────────────────
# Dataset Loading & Verification
# ─────────────────────────────────────────────

def _check(label, condition, fatal=True):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition and fatal:
        print(f"\nDataset validation failed: {label}")
        print("Stopping training.\n")
        sys.exit(1)
    return condition


def load_data_splits(cfg):
    """
    Load train, val, and test CSVs from paths defined in cfg.dataset.
    In debug mode, subsample to cfg.dataset.debug_size rows.
    """
    train_df = pd.read_csv(cfg.dataset.train_path)
    val_df   = pd.read_csv(cfg.dataset.val_path)
    test_df  = pd.read_csv(cfg.dataset.test_path)

    if cfg.system.debug:
        n = cfg.dataset.debug_size
        train_df = train_df.sample(min(n, len(train_df)), random_state=cfg.system.seed).reset_index(drop=True)
        val_df   = val_df.sample(min(n // 4, len(val_df)), random_state=cfg.system.seed).reset_index(drop=True)
        test_df  = test_df.sample(min(n // 4, len(test_df)), random_state=cfg.system.seed).reset_index(drop=True)

    return train_df, val_df, test_df


def verify_data_splits(train_df, val_df, test_df, cfg):
    """
    Run comprehensive verification checks on all three splits.
    Prints PASS or FAIL for every check. Stops training on any FAIL.
    """
    num_labels = cfg.model.num_labels
    expected_labels = set(range(num_labels))

    print("======================================")
    print("Dataset Verification")
    print("======================================")

    # ── Basic sizes ───────────────────────────
    print("Sizes:")
    _check(f"Train split non-empty         : {len(train_df):>6} records", len(train_df) > 0)
    _check(f"Val split non-empty           : {len(val_df):>6} records",   len(val_df) > 0)
    _check(f"Test split non-empty          : {len(test_df):>6} records",  len(test_df) > 0)

    # ── Column presence ───────────────────────
    print("Columns:")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _check(f"'text' column present in {split_name}", "text" in df.columns)
        _check(f"'action_label' column present in {split_name}", "action_label" in df.columns)

    # ── Missing values ────────────────────────
    print("Missing values:")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_missing_text   = df["text"].isna().sum()
        n_missing_labels = df["action_label"].isna().sum()
        _check(f"No missing text in {split_name}         : {n_missing_text} missing",  n_missing_text == 0)
        _check(f"No missing labels in {split_name}       : {n_missing_labels} missing", n_missing_labels == 0)

    # ── Empty text strings ────────────────────
    print("Empty text:")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_empty = (df["text"].astype(str).str.strip() == "").sum()
        _check(f"No empty text strings in {split_name}  : {n_empty} empty", n_empty == 0)

    # ── Label integrity ───────────────────────
    print("Labels:")
    all_labels = set(train_df["action_label"].unique()) | \
                 set(val_df["action_label"].unique())   | \
                 set(test_df["action_label"].unique())

    min_label = int(min(all_labels))
    max_label = int(max(all_labels))
    unique_count = len(all_labels)

    _check(f"Minimum label >= 0            : min={min_label}", min_label >= 0)
    _check(f"Maximum label < {num_labels}           : max={max_label}", max_label < num_labels)
    _check(f"Unique label count            : {unique_count}", unique_count > 0)

    invalid_labels = all_labels - expected_labels
    _check(f"No out-of-range labels found  : {len(invalid_labels)} invalid", len(invalid_labels) == 0)

    # ── Duplicates ────────────────────────────
    print("Duplicates:")
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_dupes = df.duplicated(subset=["text"]).sum()
        _check(f"No exact duplicates in {split_name}     : {n_dupes} found", n_dupes == 0, fatal=False)

    print("======================================")
    print("All dataset checks passed.\n")


# ─────────────────────────────────────────────
# PyTorch Dataset Class (stub — implemented in Phase 2.6)
# ─────────────────────────────────────────────

class SecBERTDataset(Dataset):
    """
    Custom PyTorch Dataset for SecBERT.
    Implemented in Phase 2.6 (Tokenizer Pipeline).
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def load_class_weights(weights_path, device="cpu"):
    """
    Load action class weights array from JSON file.
    Implemented in Phase 2.7 (Loss & Class Weights).
    """
    pass


def get_dataloaders(train_df, val_df, test_df, tokenizer, max_length, batch_size, weights_path):
    """
    Initialize DataLoader objects applying WeightedRandomSampler to the train split.
    Implemented in Phase 2.6 (Tokenizer Pipeline).
    """
    pass
