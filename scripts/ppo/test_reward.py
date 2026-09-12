#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.environment import SOCEnvironment

def test_reward():
    try:
        train_df = pd.read_csv("data/processed/train.csv")
    except FileNotFoundError:
        print("Error: train.csv not found")
        sys.exit(1)
        
    print("Testing Bounded Hybrid Reward Logic...\n")
    
    class MockEncoder:
        def encode_incident(self, text):
            return np.zeros(768)
            
    env = SOCEnvironment(df=train_df, encoder=MockEncoder(), mode="eval")
    
    action_names = pd.read_csv("data/action_space.csv").set_index("Index")["Action Name"].to_dict()
    name_to_idx = {v: k for k, v in action_names.items()}
    
    test_cases = [
        {"gt": "block_dest_ip", "pred": "block_dest_ip", "desc": "+Wc (Rare Containment)"},
        {"gt": "block_dest_ip", "pred": "kill_process", "desc": "-0.5 (Same phase)"},
        {"gt": "block_dest_ip", "pred": "patch_vulnerability", "desc": "-1.0 (Diff phase)"},
        {"gt": "enable_deep_logging", "pred": "monitor", "desc": "-0.5 (Same phase)"},
        {"gt": "enable_deep_logging", "pred": "escalate_to_human", "desc": "-1.0 (False escalation)"},
        {"gt": "restore_registry", "pred": "remove_persistence", "desc": "-0.5 (Same phase)"},
        {"gt": "snapshot_forensics", "pred": "patch_vulnerability", "desc": "-1.0 (Diff phase)"},
        {"gt": "escalate_to_human", "pred": "escalate_to_human", "desc": "+W19 (Correct escalation)"},
    ]
    
    print(f"{'Ground Truth':<22} | {'Prediction':<22} | {'Reward':<8} | {'Description'}")
    print("-" * 80)
    
    for case in test_cases:
        gt_name = case["gt"]
        pred_name = case["pred"]
        gt_idx = name_to_idx[gt_name]
        pred_idx = name_to_idx[pred_name]
        
        env.current_sample = pd.Series({"action_label": gt_idx})
        
        _, reward, _, _, _ = env.step(pred_idx)
        
        print(f"{gt_name:<22} | {pred_name:<22} | {reward:>8.4f} | {case['desc']}")

    print("-" * 80)
    print("\nWeight verification for key classes:")
    key_classes = ["patch_vulnerability", "quarantine_file", "block_dest_ip", "enable_deep_logging", "restore_registry", "snapshot_forensics", "escalate_to_human", "sandbox_redirect"]
    
    for cls in key_classes:
        idx = name_to_idx[cls]
        w = env.class_weights.get(idx, 1.0)
        c = train_df["action_label"].value_counts().get(idx, 0)
        print(f"{cls:<22} (N_c={c:<4}): W_c = {w:.4f}")

if __name__ == "__main__":
    test_reward()
