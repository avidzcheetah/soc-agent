#!/usr/bin/env python3
"""
PPO Training CLI Script
Usage:
    python scripts/ppo/train.py --train_data data/processed/train.csv --eval_data data/processed/val.csv
"""

import os
import sys
import argparse
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.encoder import SecBERTStateEncoder
from src.environment import SOCEnvironment
from src.ppo import PPOAgent, PPOMemory, PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO Agent for SOC Incident Response")

    # Data arguments
    parser.add_argument("--train_data", type=str, default="data/processed/train.csv", help="Path to training CSV")
    parser.add_argument("--eval_data", type=str, default="data/processed/val.csv", help="Path to evaluation CSV")
    parser.add_argument("--model_path", type=str, default="models/secbert_finetuned", help="Path to fine-tuned SecBERT")

    # PPO Hyperparameters
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--c2_entropy", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--k_epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size for training")

    # Training loop arguments
    parser.add_argument("--total_iterations", type=int, default=100, help="Total rollout-update iterations")
    parser.add_argument("--rollout_steps", type=int, default=256, help="Rollout steps per iteration")
    parser.add_argument("--eval_interval", type=int, default=10, help="Validation interval (in iterations)")
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of eval steps per validation")
    parser.add_argument("--log_interval", type=int, default=5, help="Logging interval (in iterations)")
    parser.add_argument("--save_dir", type=str, default="models/ppo", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("Initializing SOC PPO Training Pipeline")
    print(f"  Training Data:   {args.train_data}")
    print(f"  Evaluation Data: {args.eval_data}")
    print(f"  SecBERT Model:   {args.model_path}")
    print(f"  Save Directory:  {args.save_dir}")
    print("=" * 70)

    # 1. Load Data
    train_df = pd.read_csv(args.train_data)
    eval_df = pd.read_csv(args.eval_data) if os.path.exists(args.eval_data) else None
    print(f"[*] Loaded {len(train_df)} training incidents, {len(eval_df) if eval_df is not None else 0} eval incidents.")

    # 2. Initialize SecBERT State Encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing SecBERT State Encoder on {device}...")
    encoder = SecBERTStateEncoder(model_path=args.model_path, device=device)

    # 3. Create SOC Environments
    print("[*] Creating Gymnasium SOC Environments...")
    train_env = SOCEnvironment(df=train_df, encoder=encoder, mode="train")
    eval_env = SOCEnvironment(df=eval_df, encoder=encoder, mode="eval") if eval_df is not None else None

    # 4. Initialize PPO Agent
    print("[*] Initializing PPO Agent (Actor-Critic)...")
    agent = PPOAgent(
        state_dim=768,
        action_dim=20,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        c2_entropy=args.c2_entropy,
        k_epochs=args.k_epochs,
        batch_size=args.batch_size,
        device=device,
    )

    # 5. Initialize Trainer
    trainer = PPOTrainer(
        env=train_env,
        agent=agent,
        eval_env=eval_env,
        rollout_steps=args.rollout_steps,
        total_iterations=args.total_iterations,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        save_dir=args.save_dir,
    )

    # 6. Execute Training
    history = trainer.train(verbose=True)

    print("\n" + "=" * 70)
    print(f"[SUCCESS] PPO Training completed across {len(history)} iterations.")
    if trainer.best_eval_accuracy >= 0:
        print(f"          Best Evaluation Accuracy: {trainer.best_eval_accuracy:.4%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
