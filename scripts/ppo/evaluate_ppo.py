#!/usr/bin/env python3
"""
Step 10.5: Final PPO Evaluation Script
Evaluates the best trained PPO policy checkpoint on the untouched test set.

Calculates:
- Accuracy, Macro F1, Weighted F1, MCC
- Per-class metrics (Precision, Recall, F1)
- Confusion Matrix
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix

from src.environment import SOCEnvironment
from src.encoder import SecBERTStateEncoder
from src.ppo.agent import PPOAgent

def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Agent")
    parser.add_argument("--test_data", type=str, default="data/processed/test.csv", help="Path to test CSV")
    parser.add_argument("--model_path", type=str, default="models/secbert_finetuned", help="Path to SecBERT")
    parser.add_argument("--checkpoint_path", type=str, default="models/ppo_final/best_ppo_policy.pt", help="Path to PPO checkpoint")
    args = parser.parse_argument()

    print("=" * 70)
    print("SOC PPO Final Evaluation (Step 10.5)")
    print(f"  Test Data:  {args.test_data}")
    print(f"  SecBERT:    {args.model_path}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print("=" * 70)

    # 1. Load test data
    test_df = pd.read_csv(args.test_data)
    test_size = len(test_df)
    print(f"[*] Loaded {test_size} untouched test incidents.")

    # 2. Initialize Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing SecBERT State Encoder on {device}...")
    encoder = SecBERTStateEncoder(checkpoint_path=args.model_path)

    # 3. Create Environment
    print("[*] Creating Test Environment...")
    test_env = SOCEnvironment(df=test_df, encoder=encoder, mode="eval")

    # 4. Initialize and Load PPO Agent
    print(f"[*] Loading PPO Agent from {args.checkpoint_path}...")
    agent = PPOAgent(state_dim=768, action_dim=20, device=device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()
    
    best_val_f1 = checkpoint.get("best_eval_macro_f1", "N/A")
    print(f"[*] Agent loaded. Best Validation Macro F1 recorded at save: {best_val_f1}")

    # 5. Run Evaluation Loop
    print("[*] Running Deterministic Evaluation on Test Set...")
    obs, _ = test_env.reset()
    
    y_true = []
    y_pred = []
    
    for _ in range(test_size):
        action, _, _ = agent.select_action(obs, deterministic=True)
        next_obs, _, _, _, info = test_env.step(action)
        
        y_pred.append(action)
        y_true.append(info["ground_truth"])
        
        obs = next_obs
        
    print("[*] Evaluation complete. Computing metrics...")
    
    # 6. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("FINAL TEST SET METRICS")
    print("=" * 70)
    print(f"  Test Samples: {test_size}")
    print(f"  Accuracy:     {acc:.4%}")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  Weighted F1:  {weighted_f1:.4f}")
    print(f"  MCC:          {mcc:+.4f}")
    print("=" * 70)
    
    # 7. Classification Report
    print("\n[PER-CLASS CLASSIFICATION REPORT]")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # 8. Confusion Matrix
    print("\n[CONFUSION MATRIX]")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Save results to file
    os.makedirs("results", exist_ok=True)
    with open("results/final_evaluation.txt", "w") as f:
        f.write(f"Test Samples: {test_size}\n")
        f.write(f"Accuracy:     {acc:.4f}\n")
        f.write(f"Macro F1:     {macro_f1:.4f}\n")
        f.write(f"Weighted F1:  {weighted_f1:.4f}\n")
        f.write(f"MCC:          {mcc:+.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        
    print("\n[*] Full results saved to results/final_evaluation.txt")
    print("======================================================================")

if __name__ == "__main__":
    main()
