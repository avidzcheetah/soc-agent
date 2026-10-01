#!/usr/bin/env python3
"""
Diagnostic: PPO Actor Logits Analysis for Classes 13 & 15

Analyzes the raw probabilities output by the PPO Actor (and SecBERT for comparison)
to understand why it deterministically selects `isolate_host` for classes 13 and 15.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ppo.networks import ActorNetwork
from src.secbert.dataset import SecBERTDataset

ACTION_NAMES = {
    0: "monitor", 1: "enable_deep_logging", 2: "create_ioc_alert",
    3: "block_source_ip", 4: "block_dest_ip", 5: "dns_sinkhole",
    6: "block_port", 7: "isolate_host", 8: "kill_process",
    9: "quarantine_file", 10: "quarantine_email", 11: "reset_credentials",
    12: "disable_account", 13: "remove_persistence", 14: "restore_registry",
    15: "restore_defense_config", 16: "patch_vulnerability",
    17: "snapshot_forensics", 18: "sandbox_redirect", 19: "escalate_to_human",
}

ISOLATE_HOST_ID = 7


def get_stats(probs, y_true):
    """
    probs: (N, 20) tensor of probabilities
    y_true: (N,) list of ground truths
    """
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, {}

    gt_probs = []
    iso_probs = []
    ranks = []
    iso_top1_count = 0
    gt_top1_count = 0
    top1_actions = []

    for i in range(n):
        gt = y_true[i]
        p = probs[i]

        gt_probs.append(p[gt].item())
        iso_probs.append(p[ISOLATE_HOST_ID].item())

        sorted_indices = torch.argsort(p, descending=True)
        rank = (sorted_indices == gt).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

        top1 = sorted_indices[0].item()
        top1_actions.append(top1)
        if top1 == ISOLATE_HOST_ID:
            iso_top1_count += 1
        if top1 == gt:
            gt_top1_count += 1

    return (
        np.mean(gt_probs),
        np.mean(iso_probs),
        np.mean(ranks),
        (gt_top1_count / n) * 100,
        (iso_top1_count / n) * 100,
        Counter(top1_actions),
    )


def run_analysis(df, split_name, secbert_model, encoder_only, actor, tokenizer, device):
    action_space_df = pd.read_csv("data/action_space.csv")
    phase_map = dict(zip(action_space_df["Index"], action_space_df["CISA Phase"]))

    # Filter df to only classes 13 and 15
    sub_df = df[df["action_label"].isin([13, 15])].copy()
    if len(sub_df) == 0:
        return

    sub_df["text"] = sub_df.apply(
        lambda row: f"[{phase_map[row['action_label']]}] {row['text']}", axis=1
    )

    texts = sub_df["text"].tolist()
    labels = sub_df["action_label"].tolist()
    original_indices = sub_df.index.tolist()

    dataset = SecBERTDataset(texts, labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    all_y_true = []
    all_secbert_probs = []
    all_ppo_probs = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            # SecBERT classifier
            outputs = secbert_model(input_ids, attention_mask=attention_mask)
            secbert_probs = F.softmax(outputs.logits, dim=-1)

            # PPO actor: must use pooler_output to match the trained representation
            enc_outputs = encoder_only(input_ids, attention_mask=attention_mask)
            embeddings = enc_outputs.pooler_output
            ppo_logits = actor(embeddings)
            ppo_probs = F.softmax(ppo_logits, dim=-1)

            all_y_true.extend(batch_labels.cpu().numpy().tolist())
            all_secbert_probs.append(secbert_probs.cpu())
            all_ppo_probs.append(ppo_probs.cpu())

    all_secbert_probs = torch.cat(all_secbert_probs, dim=0)
    all_ppo_probs = torch.cat(all_ppo_probs, dim=0)

    print(f"\n" + "=" * 90)
    print(f"Detailed Sample Analysis — {split_name} (Classes 13 & 15)")
    print("=" * 90)
    print(
        f"{'Idx':>5} {'GT':>4} | {'Sec GT%':>8} {'PPO GT%':>8} {'Sec Iso%':>9} {'PPO Iso%':>9} | {'PPO Top-1 Action':<24} {'Prob':>6} | {'Rank':>4}"
    )
    print("-" * 90)

    for i in range(len(all_y_true)):
        gt = all_y_true[i]
        orig_idx = original_indices[i]
        s_probs = all_secbert_probs[i]
        p_probs = all_ppo_probs[i]

        s_gt_prob = s_probs[gt].item() * 100
        p_gt_prob = p_probs[gt].item() * 100
        s_iso_prob = s_probs[ISOLATE_HOST_ID].item() * 100
        p_iso_prob = p_probs[ISOLATE_HOST_ID].item() * 100

        sorted_indices = torch.argsort(p_probs, descending=True)
        top1_action = sorted_indices[0].item()
        top1_prob = p_probs[top1_action].item() * 100
        rank = (sorted_indices == gt).nonzero(as_tuple=True)[0].item() + 1

        print(
            f"{orig_idx:>5} {gt:>4} | {s_gt_prob:>7.1f}% {p_gt_prob:>7.1f}% {s_iso_prob:>8.1f}% {p_iso_prob:>8.1f}% | "
            f"{ACTION_NAMES.get(top1_action, str(top1_action)):<24} {top1_prob:>5.1f}% | {rank:>4}"
        )

    print(f"\n" + "=" * 90)
    print(f"Summary Statistics — Classes 13 & 15 ({split_name})")
    print("=" * 90)

    for model_name, probs in [("SecBERT", all_secbert_probs), ("PPO (ATGP)", all_ppo_probs)]:
        print(f"\n--- {model_name} ---")
        for cls in [13, 15]:
            indices = [i for i, x in enumerate(all_y_true) if x == cls]
            cls_probs = probs[indices]
            cls_y = [cls] * len(indices)

            mean_gt, mean_iso, mean_rank, gt_pct, iso_pct, top1_counts = get_stats(cls_probs, cls_y)
            print(f"Class {cls} ({ACTION_NAMES[cls]}) - {len(indices)} samples:")
            print(f"  Mean prob of correct action : {mean_gt:.4f} ({mean_gt*100:.1f}%)")
            print(f"  Mean prob of isolate_host   : {mean_iso:.4f} ({mean_iso*100:.1f}%)")
            print(f"  Mean rank of correct action : {mean_rank:.2f}")
            print(f"  Correct action is top-1     : {gt_pct:.1f}%")
            print(f"  isolate_host is top-1       : {iso_pct:.1f}%")
            top1_str = ", ".join(f"{ACTION_NAMES.get(a, str(a))}: {c}" for a, c in sorted(top1_counts.items(), key=lambda x: -x[1]))
            print(f"  Top-1 action distribution  : {top1_str}")


def run_full_cm(df, split_name, encoder_only, actor, tokenizer, device):
    """Computes and prints full 20-class evaluation and confusion matrix on the given split."""
    action_space_df = pd.read_csv("data/action_space.csv")
    phase_map = dict(zip(action_space_df["Index"], action_space_df["CISA Phase"]))

    prep_df = df.copy()
    prep_df["text"] = prep_df.apply(
        lambda row: f"[{phase_map[row['action_label']]}] {row['text']}", axis=1
    )

    texts = prep_df["text"].tolist()
    labels = prep_df["action_label"].tolist()

    dataset = SecBERTDataset(texts, labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collator)

    y_true = []
    y_pred = []

    print(f"\nRunning full {split_name} inference for confusion matrix...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"\r  {split_name} batch {i+1}/{len(loader)}", end="")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            enc_outputs = encoder_only(input_ids, attention_mask=attention_mask)
            embeddings = enc_outputs.pooler_output
            logits = actor(embeddings)
            preds = torch.argmax(logits, dim=-1)

            y_true.extend(batch_labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    print(f"\n  {split_name} inference complete.")

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\n" + "=" * 90)
    print(f"ATGP FULL {split_name} EVALUATION METRICS")
    print("=" * 90)
    print(f"  {split_name} Samples: {len(y_true)}")
    print(f"  Accuracy:           {acc:.4%}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Weighted F1:        {weighted_f1:.4f}")
    print(f"  MCC:                {mcc:+.4f}")
    print("=" * 90)

    # Classification report
    print(f"\n[PER-CLASS CLASSIFICATION REPORT — {split_name}]")
    present_labels = sorted(set(y_true))
    target_names = [f"[{lbl}] {ACTION_NAMES.get(lbl, str(lbl))}" for lbl in present_labels]
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=target_names, zero_division=0, digits=4))

    # Confusion matrix
    print("\n" + "=" * 90)
    print("VALIDATION CONFUSION MATRIX")
    print("=" * 90)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(20)))

    # Header
    print(f"{'True \\ Pred':<28} " + " ".join(f"{i:>3}" for i in range(20)) + f" {'Total':>6}")
    print("-" * 115)
    for i in range(20):
        row_sum = cm[i].sum()
        if row_sum > 0 or cm[:, i].sum() > 0:
            name = f"[{i:>2}] {ACTION_NAMES.get(i, str(i)):<23}"
            row_vals = " ".join(f"{cm[i, j]:>3}" if cm[i, j] > 0 else "  ." for j in range(20))
            print(f"{name} {row_vals} {row_sum:>6}")

    # Top misclassifications
    print("\n" + "=" * 90)
    print("NOTABLE MISCLASSIFICATIONS (True != Pred)")
    print("=" * 90)
    misclass = []
    for i in range(20):
        for j in range(20):
            if i != j and cm[i, j] > 0:
                misclass.append((i, j, cm[i, j]))
    misclass.sort(key=lambda x: -x[2])
    for t, p, count in misclass:
        t_name = ACTION_NAMES.get(t, str(t))
        p_name = ACTION_NAMES.get(p, str(p))
        print(f"  [{t:>2}] {t_name:<26} -> [{p:>2}] {p_name:<26} : {count:>3} errors")


def main():
    parser = argparse.ArgumentParser(description="Diagnose PPO Logits and Performance")
    parser.add_argument("--checkpoint", type=str, default="models/ppo_exp4d_atgp/best_ppo_policy.pt",
                        help="Path to PPO checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "both"],
                        help="Split to diagnose (default: val)")
    parser.add_argument("--val_data", type=str, default="data/processed/val.csv")
    parser.add_argument("--test_data", type=str, default="data/processed/test.csv")
    parser.add_argument("--full_cm", action="store_true", default=True,
                        help="Run full 20-class validation evaluation and confusion matrix")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("=" * 90)
    print("PPO Actor Logits Diagnosis & Evaluation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Split      : {args.split}")
    print("=" * 90)

    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained("models/secbert_finetuned")
    secbert_model = AutoModelForSequenceClassification.from_pretrained("models/secbert_finetuned")
    secbert_model.to(device)
    secbert_model.eval()

    encoder_only = getattr(secbert_model, secbert_model.base_model_prefix)
    encoder_only.eval()

    actor = ActorNetwork(state_dim=768, action_dim=20).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    print(f"  Checkpoint loaded. Stored Macro F1: {checkpoint.get('best_eval_macro_f1', 'N/A')}")

    if args.split in ("val", "both"):
        val_df = pd.read_csv(args.val_data)
        run_analysis(val_df, "VALIDATION", secbert_model, encoder_only, actor, tokenizer, device)
        if args.full_cm:
            run_full_cm(val_df, "VALIDATION", encoder_only, actor, tokenizer, device)

    if args.split in ("test", "both"):
        test_df = pd.read_csv(args.test_data)
        run_analysis(test_df, "TEST", secbert_model, encoder_only, actor, tokenizer, device)
        if args.full_cm:
            run_full_cm(test_df, "TEST", encoder_only, actor, tokenizer, device)


if __name__ == "__main__":
    main()

