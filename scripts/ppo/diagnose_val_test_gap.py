#!/usr/bin/env python3
"""
Diagnostic Script: Validation vs Test Per-Class F1 Gap Analysis

Evaluates PPO-135 checkpoint on both val.csv and test.csv using the same
frozen SecBERT encoder and deterministic action selection, then outputs
a per-class comparison table to identify which actions cause the
Macro F1 drop from validation (0.7926) to test (0.7057).
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from src.encoder import SecBERTStateEncoder
from src.environment import SOCEnvironment
from src.ppo.agent import PPOAgent


ACTION_NAMES = {
    0: "monitor", 1: "enable_deep_logging", 2: "create_ioc_alert",
    3: "block_source_ip", 4: "block_dest_ip", 5: "dns_sinkhole",
    6: "block_port", 7: "isolate_host", 8: "kill_process",
    9: "quarantine_file", 10: "quarantine_email", 11: "reset_credentials",
    12: "disable_account", 13: "remove_persistence", 14: "restore_registry",
    15: "restore_defense_config", 16: "patch_vulnerability",
    17: "snapshot_forensics", 18: "sandbox_redirect", 19: "escalate_to_human",
}


def evaluate_on_split(agent, encoder, df, split_name):
    """Run deterministic evaluation on a dataframe, return y_true and y_pred."""
    env = SOCEnvironment(df=df, encoder=encoder, mode="eval")
    env.current_index = 0
    obs, _ = env.reset()

    y_true = []
    y_pred = []

    for i in range(len(df)):
        action, _, _ = agent.select_action(obs, deterministic=True)
        next_obs, _, _, _, info = env.step(action)

        y_pred.append(action)
        y_true.append(info["ground_truth"])
        obs = next_obs

        if (i + 1) % 200 == 0 or (i + 1) == len(df):
            print(f"  {split_name}: {i+1}/{len(df)}")

    return y_true, y_pred


def per_class_f1(y_true, y_pred, labels):
    """Compute per-class F1 scores as a dict {label: f1}."""
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {label: (float(f[i]), int(s[i])) for i, label in enumerate(labels)}


def main():
    checkpoint_path = "models/ppo_final/best_ppo_policy.pt"
    model_path = "models/secbert_finetuned"
    device = "cpu"

    print("=" * 80)
    print("PPO-135 Validation vs Test Per-Class F1 Gap Diagnosis")
    print("=" * 80)

    # 1. Load encoder
    print("\n[1/4] Loading SecBERT encoder...")
    encoder = SecBERTStateEncoder(checkpoint_path=model_path)

    # 2. Load agent (same pattern as evaluate_ppo.py)
    print(f"[2/4] Loading PPO agent from {checkpoint_path}...")
    agent = PPOAgent(state_dim=768, action_dim=20, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()

    # 3. Evaluate on both splits
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    print(f"\n[3/4] Evaluating on validation ({len(val_df)} samples)...")
    val_true, val_pred = evaluate_on_split(agent, encoder, val_df, "VAL")

    print(f"\n[4/4] Evaluating on test ({len(test_df)} samples)...")
    test_true, test_pred = evaluate_on_split(agent, encoder, test_df, "TEST")

    # 4. Compute overall metrics
    val_macro = f1_score(val_true, val_pred, average="macro", zero_division=0)
    test_macro = f1_score(test_true, test_pred, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print(f"  Validation Macro F1:  {val_macro:.4f}")
    print(f"  Test Macro F1:        {test_macro:.4f}")
    print(f"  Gap (Test - Val):     {test_macro - val_macro:+.4f}")
    print("=" * 80)

    # 5. Per-class comparison table
    # Union of all labels present in either split
    all_labels = sorted(set(val_true) | set(test_true))

    val_f1s = per_class_f1(val_true, val_pred, all_labels)
    test_f1s = per_class_f1(test_true, test_pred, all_labels)

    print(f"\n{'Idx':<5}{'Action':<28}{'Val F1':>8}{'Val N':>7}{'Test F1':>9}{'Test N':>8}{'Gap':>9}")
    print("-" * 74)

    gaps = []
    for label in all_labels:
        name = ACTION_NAMES.get(label, f"action_{label}")
        vf1, vs = val_f1s.get(label, (0.0, 0))
        tf1, ts = test_f1s.get(label, (0.0, 0))
        gap = tf1 - vf1
        gaps.append((label, name, vf1, vs, tf1, ts, gap))
        print(f"{label:<5}{name:<28}{vf1:>8.4f}{vs:>7}{tf1:>9.4f}{ts:>8}{gap:>+9.4f}")

    # 6. Sort by gap to highlight worst regressions
    print("\n" + "=" * 80)
    print("WORST REGRESSIONS (sorted by gap, ascending)")
    print("=" * 80)
    gaps_sorted = sorted(gaps, key=lambda x: x[6])
    for label, name, vf1, vs, tf1, ts, gap in gaps_sorted:
        if gap < -0.01:
            print(f"  [{label:>2}] {name:<28} Val={vf1:.4f} (n={vs})  Test={tf1:.4f} (n={ts})  Gap={gap:+.4f}")

    # 7. Contribution analysis
    print("\n" + "=" * 80)
    print("MACRO F1 CONTRIBUTION ANALYSIS")
    print("=" * 80)
    n_classes = len(all_labels)
    print(f"  Number of active classes: {n_classes}")
    print(f"  Per-class weight in Macro F1: 1/{n_classes} = {1/n_classes:.4f}")
    print()
    for label, name, vf1, vs, tf1, ts, gap in gaps_sorted:
        contribution = gap / n_classes
        if abs(contribution) > 0.001:
            print(f"  [{label:>2}] {name:<28} contributes {contribution:+.4f} to Macro F1 gap")

    total_gap = sum(g[6] for g in gaps) / n_classes
    print(f"\n  Sum of contributions = {total_gap:+.4f} (should ≈ {test_macro - val_macro:+.4f})")


if __name__ == "__main__":
    main()
