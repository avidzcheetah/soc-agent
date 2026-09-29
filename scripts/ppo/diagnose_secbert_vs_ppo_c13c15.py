#!/usr/bin/env python3
"""
Diagnostic: SecBERT vs PPO on Classes 13 & 15 (Val + Test)

Determines whether the class 15→13 confusion is inherent to SecBERT's
representations or specific to the PPO policy layer.

Uses the same SecBERT classifier and PPO Actor loading as evaluate_baseline.py.
Does not modify or retrain anything.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.metrics import f1_score, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from src.secbert.dataset import SecBERTDataset
from src.ppo.networks import ActorNetwork


ACTION_NAMES = {
    0: "monitor", 1: "enable_deep_logging", 2: "create_ioc_alert",
    3: "block_source_ip", 4: "block_dest_ip", 5: "dns_sinkhole",
    6: "block_port", 7: "isolate_host", 8: "kill_process",
    9: "quarantine_file", 10: "quarantine_email", 11: "reset_credentials",
    12: "disable_account", 13: "remove_persistence", 14: "restore_registry",
    15: "restore_defense_config", 16: "patch_vulnerability",
    17: "snapshot_forensics", 18: "sandbox_redirect", 19: "escalate_to_human",
}

SECBERT_PATH = "models/secbert_finetuned"
PPO_PATH = "models/ppo_final/best_ppo_policy.pt"
ACTION_SPACE_PATH = "data/action_space.csv"


def run_secbert_and_ppo(df, split_name, secbert_model, encoder_only, actor, tokenizer, device):
    """Run both SecBERT classifier and PPO actor on a dataset, return predictions."""
    # Prepend phase tags (same as evaluate_baseline.py)
    action_space_df = pd.read_csv(ACTION_SPACE_PATH)
    phase_map = dict(zip(action_space_df["Index"], action_space_df["CISA Phase"]))

    df_prep = df.copy()
    df_prep["text"] = df_prep.apply(
        lambda row: f"[{phase_map[row['action_label']]}] {row['text']}", axis=1
    )

    texts = df_prep["text"].tolist()
    labels = df_prep["action_label"].tolist()

    dataset = SecBERTDataset(texts, labels, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    y_true = []
    y_pred_secbert = []
    y_pred_ppo = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            print(f"\r  {split_name} batch {i+1}/{len(loader)}", end="")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            # SecBERT classifier
            outputs = secbert_model(input_ids, attention_mask=attention_mask)
            secbert_preds = torch.argmax(outputs.logits, dim=-1)

            # PPO actor (via encoder_only → actor)
            enc_outputs = encoder_only(input_ids, attention_mask=attention_mask)
            embeddings = enc_outputs.last_hidden_state[:, 0, :]
            ppo_logits = actor(embeddings)
            ppo_preds = torch.argmax(ppo_logits, dim=-1)

            y_true.extend(batch_labels.cpu().numpy().tolist())
            y_pred_secbert.extend(secbert_preds.cpu().numpy().tolist())
            y_pred_ppo.extend(ppo_preds.cpu().numpy().tolist())

    print()
    return y_true, y_pred_secbert, y_pred_ppo


def analyze_class(y_true, y_pred_secbert, y_pred_ppo, target_class, split_name):
    """Detailed analysis for a single target class."""
    indices = [i for i, gt in enumerate(y_true) if gt == target_class]
    n = len(indices)
    name = ACTION_NAMES[target_class]

    print(f"\n  Class {target_class} ({name}) — {n} samples in {split_name}")
    print("  " + "-" * 60)

    # SecBERT predictions for this class
    secbert_preds = [y_pred_secbert[i] for i in indices]
    secbert_correct = sum(1 for p in secbert_preds if p == target_class)
    secbert_counts = Counter(secbert_preds)

    # PPO predictions for this class
    ppo_preds = [y_pred_ppo[i] for i in indices]
    ppo_correct = sum(1 for p in ppo_preds if p == target_class)
    ppo_counts = Counter(ppo_preds)

    print(f"  SecBERT correct: {secbert_correct}/{n}")
    for pred, count in sorted(secbert_counts.items(), key=lambda x: -x[1]):
        mark = " ✓" if pred == target_class else ""
        print(f"    → {ACTION_NAMES.get(pred, str(pred)):<28} {count:>3} times{mark}")

    print(f"\n  PPO correct:    {ppo_correct}/{n}")
    for pred, count in sorted(ppo_counts.items(), key=lambda x: -x[1]):
        mark = " ✓" if pred == target_class else ""
        print(f"    → {ACTION_NAMES.get(pred, str(pred)):<28} {count:>3} times{mark}")

    # Per-sample comparison
    print(f"\n  Per-sample comparison (class {target_class}, {split_name}):")
    print(f"  {'Idx':>5}  {'SecBERT':>8}  {'PPO':>8}  SecBERT Action              PPO Action")
    print("  " + "-" * 75)
    for i in indices:
        sp = y_pred_secbert[i]
        pp = y_pred_ppo[i]
        s_mark = "✓" if sp == target_class else "✗"
        p_mark = "✓" if pp == target_class else "✗"
        print(f"  {i:>5}  {s_mark:>8}  {p_mark:>8}  {ACTION_NAMES.get(sp, str(sp)):<28}{ACTION_NAMES.get(pp, str(pp))}")


def main():
    device = torch.device("cpu")

    print("=" * 80)
    print("SecBERT vs PPO — Class 13 & 15 Focused Diagnosis (Val + Test)")
    print("=" * 80)

    # Load models
    print("\nLoading SecBERT classifier...")
    tokenizer = AutoTokenizer.from_pretrained(SECBERT_PATH)
    secbert_model = AutoModelForSequenceClassification.from_pretrained(SECBERT_PATH)
    secbert_model.to(device)
    secbert_model.eval()

    encoder_only = getattr(secbert_model, secbert_model.base_model_prefix)
    encoder_only.eval()

    print("Loading PPO Actor...")
    actor = ActorNetwork(state_dim=768, action_dim=20).to(device)
    checkpoint = torch.load(PPO_PATH, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    # Process both splits
    for split_name, path in [("VALIDATION", "data/processed/val.csv"),
                              ("TEST", "data/processed/test.csv")]:
        df = pd.read_csv(path)
        print(f"\n{'=' * 80}")
        print(f"{split_name} SET ({len(df)} samples)")
        print("=" * 80)

        y_true, y_pred_secbert, y_pred_ppo = run_secbert_and_ppo(
            df, split_name, secbert_model, encoder_only, actor, tokenizer, device
        )

        # Overall class 13 & 15 F1
        all_labels = sorted(set(y_true))
        secbert_f1 = f1_score(y_true, y_pred_secbert, labels=all_labels, average=None, zero_division=0)
        ppo_f1 = f1_score(y_true, y_pred_ppo, labels=all_labels, average=None, zero_division=0)

        secbert_f1_map = dict(zip(all_labels, secbert_f1))
        ppo_f1_map = dict(zip(all_labels, ppo_f1))

        print(f"\n  {'Class':>5} {'Action':<28} {'SecBERT F1':>11} {'PPO F1':>9} {'Diff':>9}")
        print("  " + "-" * 65)
        for c in [13, 15]:
            sf = secbert_f1_map.get(c, 0.0)
            pf = ppo_f1_map.get(c, 0.0)
            print(f"  {c:>5} {ACTION_NAMES[c]:<28} {sf:>11.4f} {pf:>9.4f} {pf - sf:>+9.4f}")

        # Detailed per-class analysis
        for c in [13, 15]:
            analyze_class(y_true, y_pred_secbert, y_pred_ppo, c, split_name)

        # 2x2 confusion submatrix for classes 13 & 15
        print(f"\n  2×2 Confusion Submatrix (classes 13 & 15 only) — {split_name}")
        mask = [i for i, gt in enumerate(y_true) if gt in (13, 15)]
        sub_true = [y_true[i] for i in mask]

        print("\n  SecBERT:")
        sub_secbert = [y_pred_secbert[i] for i in mask]
        cm_s = confusion_matrix(sub_true, sub_secbert, labels=[13, 15])
        print(f"                    Pred 13   Pred 15   Other")
        other_s_13 = sum(1 for i in mask if y_true[i] == 13 and y_pred_secbert[i] not in (13, 15))
        other_s_15 = sum(1 for i in mask if y_true[i] == 15 and y_pred_secbert[i] not in (13, 15))
        print(f"    True 13 (n={sum(1 for t in sub_true if t==13):>3})   {cm_s[0,0]:>5}     {cm_s[0,1]:>5}     {other_s_13:>5}")
        print(f"    True 15 (n={sum(1 for t in sub_true if t==15):>3})   {cm_s[1,0]:>5}     {cm_s[1,1]:>5}     {other_s_15:>5}")

        print("\n  PPO:")
        sub_ppo = [y_pred_ppo[i] for i in mask]
        cm_p = confusion_matrix(sub_true, sub_ppo, labels=[13, 15])
        other_p_13 = sum(1 for i in mask if y_true[i] == 13 and y_pred_ppo[i] not in (13, 15))
        other_p_15 = sum(1 for i in mask if y_true[i] == 15 and y_pred_ppo[i] not in (13, 15))
        print(f"                    Pred 13   Pred 15   Other")
        print(f"    True 13 (n={sum(1 for t in sub_true if t==13):>3})   {cm_p[0,0]:>5}     {cm_p[0,1]:>5}     {other_p_13:>5}")
        print(f"    True 15 (n={sum(1 for t in sub_true if t==15):>3})   {cm_p[1,0]:>5}     {cm_p[1,1]:>5}     {other_p_15:>5}")


if __name__ == "__main__":
    main()
