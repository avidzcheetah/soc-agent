#!/usr/bin/env python3
"""
Diagnostic Script: Class 15 (restore_defense_config) Confusion Analysis

For both val.csv and test.csv, finds every sample with ground-truth label 15,
runs PPO-135 deterministic inference, and reports what action was predicted.

Does not modify or retrain anything.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
from collections import Counter

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


def evaluate_class15(agent, encoder, df, split_name):
    """Run deterministic eval on full dataset, return details for class-15 samples."""
    env = SOCEnvironment(df=df, encoder=encoder, mode="eval")
    env.current_index = 0
    obs, _ = env.reset()

    class15_results = []

    for i in range(len(df)):
        action, _, _ = agent.select_action(obs, deterministic=True)
        next_obs, _, _, _, info = env.step(action)
        gt = info["ground_truth"]

        if gt == 15:
            class15_results.append({
                "dataset": split_name,
                "sample_index": i,
                "ground_truth": gt,
                "ground_truth_name": ACTION_NAMES[gt],
                "predicted": action,
                "predicted_name": ACTION_NAMES.get(action, f"action_{action}"),
                "correct": action == gt,
            })

        obs = next_obs

    return class15_results


def main():
    checkpoint_path = "models/ppo_final/best_ppo_policy.pt"
    model_path = "models/secbert_finetuned"
    device = "cpu"

    print("=" * 70)
    print("Class 15 (restore_defense_config) Confusion Diagnosis")
    print("=" * 70)

    # Load encoder
    print("\nLoading SecBERT encoder...")
    encoder = SecBERTStateEncoder(checkpoint_path=model_path)

    # Load agent
    print(f"Loading PPO agent from {checkpoint_path}...")
    agent = PPOAgent(state_dim=768, action_dim=20, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()

    # Evaluate both splits
    for split_name, path in [("VALIDATION", "data/processed/val.csv"), ("TEST", "data/processed/test.csv")]:
        df = pd.read_csv(path)
        total_class15 = (df["action_label"] == 15).sum()
        print(f"\n{'=' * 70}")
        print(f"{split_name} — {total_class15} class-15 samples in {len(df)} total")
        print("=" * 70)

        results = evaluate_class15(agent, encoder, df, split_name)

        # Per-sample detail
        print(f"\n{'Idx':>5}  {'GT':>3}  {'Pred':>5}  {'Correct':>8}  Predicted Action")
        print("-" * 60)
        for r in results:
            mark = "✓" if r["correct"] else "✗"
            print(f"{r['sample_index']:>5}  {r['ground_truth']:>3}  {r['predicted']:>5}  {mark:>8}  {r['predicted_name']}")

        # Summary
        pred_counts = Counter(r["predicted"] for r in results)
        correct = sum(1 for r in results if r["correct"])
        print(f"\n{split_name} class-15 prediction summary:")
        print(f"  Correct: {correct}/{len(results)}")
        for action_id, count in sorted(pred_counts.items(), key=lambda x: -x[1]):
            print(f"  predicted {ACTION_NAMES.get(action_id, str(action_id)):<28} → {count} times")


if __name__ == "__main__":
    main()
