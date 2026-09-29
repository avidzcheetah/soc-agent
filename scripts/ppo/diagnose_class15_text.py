#!/usr/bin/env python3
"""
Diagnostic Script: Class 15 (restore_defense_config) — Raw Text Inspection

Shows the actual incident text for every class-15 sample in val and test,
alongside PPO-135's prediction. Helps determine whether test samples are
semantically different from validation or if PPO has a systematic boundary issue.

Does not modify or retrain anything.
"""

import os
import sys
import textwrap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd

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


def evaluate_and_collect(agent, encoder, df, split_name, target_class=15):
    """Run deterministic eval, return details for target class samples."""
    env = SOCEnvironment(df=df, encoder=encoder, mode="eval")
    env.current_index = 0
    obs, _ = env.reset()

    results = []

    for i in range(len(df)):
        action, _, _ = agent.select_action(obs, deterministic=True)
        next_obs, _, _, _, info = env.step(action)
        gt = info["ground_truth"]

        if gt == target_class:
            # Get the raw text from the dataframe
            raw_text = str(df.iloc[i].get("text", ""))
            results.append({
                "dataset": split_name,
                "index": i,
                "text": raw_text,
                "ground_truth": gt,
                "predicted": action,
                "predicted_name": ACTION_NAMES.get(action, f"action_{action}"),
                "correct": action == gt,
            })

        obs = next_obs

    return results


def print_sample(r, width=100):
    """Pretty-print one sample."""
    mark = "✓ CORRECT" if r["correct"] else f"✗ WRONG → {r['predicted_name']} ({r['predicted']})"
    print(f"\n  [{r['dataset']}] Index {r['index']}  |  {mark}")
    print("  " + "-" * (width - 2))
    wrapped = textwrap.fill(r["text"], width=width - 4)
    for line in wrapped.split("\n"):
        print(f"    {line}")


def main():
    checkpoint_path = "models/ppo_final/best_ppo_policy.pt"
    model_path = "models/secbert_finetuned"
    device = "cpu"

    print("=" * 100)
    print("Class 15 (restore_defense_config) — Raw Incident Text Inspection")
    print("=" * 100)

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

    # Process both splits
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    print("\nEvaluating validation set...")
    val_results = evaluate_and_collect(agent, encoder, val_df, "VAL")

    print("Evaluating test set...")
    test_results = evaluate_and_collect(agent, encoder, test_df, "TEST")

    # ─── Validation Samples ───
    print("\n" + "=" * 100)
    print(f"VALIDATION — {len(val_results)} class-15 samples")
    print("=" * 100)

    print("\n── Correctly Classified ──")
    correct_val = [r for r in val_results if r["correct"]]
    for r in correct_val:
        print_sample(r)

    print("\n── Misclassified ──")
    wrong_val = [r for r in val_results if not r["correct"]]
    for r in wrong_val:
        print_sample(r)

    # ─── Test Samples ───
    print("\n" + "=" * 100)
    print(f"TEST — {len(test_results)} class-15 samples")
    print("=" * 100)

    print("\n── Correctly Classified ──")
    correct_test = [r for r in test_results if r["correct"]]
    for r in correct_test:
        print_sample(r)

    print("\n── Misclassified (predicted as remove_persistence) ──")
    wrong_13 = [r for r in test_results if not r["correct"] and r["predicted"] == 13]
    for r in wrong_13:
        print_sample(r)

    print("\n── Misclassified (other) ──")
    wrong_other = [r for r in test_results if not r["correct"] and r["predicted"] != 13]
    for r in wrong_other:
        print_sample(r)

    # ─── Summary ───
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\n  Validation: {len(correct_val)}/{len(val_results)} correct")
    print(f"  Test:       {len(correct_test)}/{len(test_results)} correct")
    print(f"  Test misclassified as remove_persistence: {len(wrong_13)}/{len(test_results)}")


if __name__ == "__main__":
    main()
