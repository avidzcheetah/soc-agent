#!/usr/bin/env python3
"""
Experiment 4A — Subtask 2.3: Warm-Start Actor Diagnostic Evaluation

Evaluates any Actor checkpoint on the full 1,539-sample validation set.

Usage (from project root):
    python scripts/ppo/eval_warmstart.py
    python scripts/ppo/eval_warmstart.py --checkpoint models/ppo/best_ppo_policy.pt
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

import argparse

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ppo.networks import ActorNetwork
from src.secbert.dataset import SecBERTDataset


# ── Config ────────────────────────────────────────────────────────────
SECBERT_PATH      = "models/secbert_finetuned"
VAL_DATA_PATH     = "data/processed/val.csv"
ACTION_SPACE_PATH = "data/action_space.csv"
BATCH_SIZE        = 64
MAX_LENGTH        = 512


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Actor checkpoint on validation set")
    parser.add_argument("--checkpoint", type=str, default="models/ppo/warmstart_actor.pt",
                        help="Path to Actor checkpoint (.pt file)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    device = torch.device("cpu")

    print("=" * 65)
    print("Actor Diagnostic Evaluation on Validation Set")
    print("=" * 65)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Val split  : {VAL_DATA_PATH}")
    print(f"  Device     : {device}")
    print()

    # ── 1. Load action space ──────────────────────────────────────────
    action_space_df = pd.read_csv(ACTION_SPACE_PATH)
    phase_map  = dict(zip(action_space_df["Index"], action_space_df["CISA Phase"]))
    label_map  = dict(zip(action_space_df["Index"], action_space_df["Action Name"]))
    action_names = [label_map[i] for i in range(len(label_map))]

    # ── 2. Load val data ──────────────────────────────────────────────
    val_df = pd.read_csv(VAL_DATA_PATH)
    val_df["text"] = val_df.apply(
        lambda row: f"[{phase_map[int(row['action_label'])]}] {row['text']}", axis=1
    )

    # ── 3. Load tokenizer + encoder (from SecBERT) ───────────────────
    print("[1/3] Loading SecBERT encoder (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(SECBERT_PATH)
    secbert   = AutoModelForSequenceClassification.from_pretrained(SECBERT_PATH)
    secbert.eval()
    encoder = getattr(secbert, secbert.base_model_prefix)
    encoder.eval()

    # ── 4. Build val dataloader ───────────────────────────────────────
    collator     = DataCollatorWithPadding(tokenizer=tokenizer)
    val_dataset  = SecBERTDataset(
        val_df["text"].tolist(), val_df["action_label"].tolist(),
        tokenizer, max_length=MAX_LENGTH
    )
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collator)

    # ── 5. Load Actor checkpoint ──────────────────────────────────────
    print("[2/3] Loading Actor checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor = ActorNetwork(state_dim=768, action_dim=20).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # Print any stored metadata
    for key in ["warmstart_val_acc", "warmstart_val_macro_f1", "best_eval_macro_f1", "method"]:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")
    print()

    # ── 6. Run inference ──────────────────────────────────────────────
    print("[3/3] Running inference on 1,539 validation samples...")
    y_true, y_pred = [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            print(f"\r  Batch {i+1}/{len(val_loader)}", end="", flush=True)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            outputs    = encoder(input_ids, attention_mask=attention_mask)
            embeddings = outputs.pooler_output          # aligned with classifier head

            logits = actor(embeddings)
            preds  = torch.argmax(logits, dim=-1).cpu()

            y_pred.extend(preds.numpy())
            y_true.extend(labels.numpy())

    print()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── 7. Overall metrics ────────────────────────────────────────────
    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc         = matthews_corrcoef(y_true, y_pred)
    active_labels = sorted(set(y_true.tolist()))

    print()
    print("=" * 65)
    print("Overall Metrics (Warm-Start Actor vs SecBERT Baseline)")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Warm-Start Actor':>20}  {'SecBERT Baseline':>18}")
    print(f"  {'-'*60}")
    print(f"  {'Val Accuracy':<22} {acc:>19.4%}  {'94.4835%':>18}")
    print(f"  {'Val Macro F1':<22} {macro_f1:>20.4f}  {'0.7375':>18}")
    print(f"  {'Val Weighted F1':<22} {weighted_f1:>20.4f}  {'0.9445':>18}")
    print(f"  {'Val MCC':<22} {mcc:>20.4f}  {'0.9347':>18}")

    # ── 8. Per-class report ───────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"Per-Class F1 — {len(active_labels)} active classes (0-support classes excluded)")
    print("=" * 65)
    active_names = [action_names[i] for i in active_labels]
    report = classification_report(
        y_true, y_pred,
        labels=active_labels,
        target_names=active_names,
        zero_division=0,
        digits=4,
    )
    print(report)

    # ── 9. Collapsed classes ─────────────────────────────────────────────────────────
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=active_labels)
    print("=" * 65)
    print("Struggling Active Classes (F1 < 0.50)")
    print("=" * 65)
    any_collapsed = False
    for idx, f1 in zip(active_labels, per_class_f1):
        support = int((y_true == idx).sum())
        if f1 < 0.50:
            print(f"  [{idx:02d}] {action_names[idx]:<30}  F1={f1:.4f}  support={support}")
            any_collapsed = True
    if not any_collapsed:
        print("  (none — all active classes F1 >= 0.50)")

    # ── 10. Confusion matrix summary (top misclassifications) ─────────
    print()
    print("=" * 65)
    print("Top Misclassifications (ground truth → predicted)")
    print("=" * 65)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(20)))
    errors = []
    for gt in range(20):
        for pred in range(20):
            if gt != pred and cm[gt, pred] > 0:
                errors.append((cm[gt, pred], gt, pred))
    errors.sort(reverse=True)
    for count, gt, pred in errors[:15]:
        print(f"  {action_names[gt]:<30} → {action_names[pred]:<30}  (n={count})")

    print()
    print("=" * 65)
    print("Subtask 2.3 complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
