#!/usr/bin/env python3
"""
PPO Training CLI Script
Usage:
    python scripts/ppo/train.py --train_data data/processed/train.csv --eval_data data/processed/val.csv
"""

import os
import sys
import random
import argparse
import numpy as np
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
    parser.add_argument("--early_stopping_patience", type=int, default=4, help="Patience (eval intervals) for early stopping")
    parser.add_argument("--log_interval", type=int, default=5, help="Logging interval (in iterations)")
    parser.add_argument("--save_dir", type=str, default="models/ppo", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--warmstart_path", type=str, default=None,
                        help="Path to warm-started Actor weights (from warmstart_actor.py)")
    parser.add_argument("--kl_beta", type=float, default=0.0,
                        help="KL penalty coefficient against warm-start reference policy (0.0 = disabled)")
    parser.add_argument("--bc_lambda", type=float, default=0.0,
                        help="BC cross-entropy preservation coefficient (0.0 = disabled). "
                             "Adds supervised cross-entropy loss to the actor during PPO updates.")
    parser.add_argument("--sampling", type=str, default="uniform",
                        choices=["uniform", "balanced"],
                        help="Training experience sampling strategy: "
                             "'uniform' = random (default), "
                             "'balanced' = capped class-frequency-weighted sampling to improve minority-class coverage.")
    parser.add_argument("--atgp_lambda", type=float, default=0.0,
                        help="ATGP teacher-preservation coefficient (0.0 = disabled). "
                             "Adaptive Teacher-Guided Preservation: per-sample, confidence-gated KL "
                             "against frozen teacher to prevent policy collapse on minority classes.")
    parser.add_argument("--atgp_tau", type=float, default=0.3,
                        help="ATGP teacher confidence threshold. Only samples where the teacher's "
                             "max probability exceeds this threshold receive the preservation loss.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds for full reproducibility across all random sources
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    encoder = SecBERTStateEncoder(checkpoint_path=args.model_path)

    # 3. Create SOC Environments
    print("[*] Creating Gymnasium SOC Environments...")
    print(f"    Sampling strategy: {args.sampling}")
    train_env = SOCEnvironment(df=train_df, encoder=encoder, mode="train", sampling=args.sampling)
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
    agent.bc_lambda = args.bc_lambda  # BC preservation coefficient
    agent.atgp_lambda = args.atgp_lambda  # ATGP preservation coefficient
    agent.atgp_tau = args.atgp_tau        # ATGP confidence threshold

    # 4.5 Load warm-started Actor weights if provided (Experiment 4A)
    ref_actor = None
    if args.warmstart_path is not None:
        print(f"[*] Loading warm-started Actor from {args.warmstart_path}...")
        warmstart_ckpt = torch.load(args.warmstart_path, map_location=device, weights_only=False)
        agent.actor.load_state_dict(warmstart_ckpt["actor_state_dict"])
        ws_acc = warmstart_ckpt.get("warmstart_val_acc", "N/A")
        ws_f1  = warmstart_ckpt.get("warmstart_val_macro_f1", "N/A")
        print(f"    Warm-start Val Acc    : {ws_acc}")
        print(f"    Warm-start Val Macro F1 (stored): {ws_f1}")
        print(f"    Authoritative Macro F1 (eval_warmstart.py): 0.7438")
        print(f"    Method: {warmstart_ckpt.get('method', 'unknown')}")

        # Build a frozen reference Actor for KL / ATGP constraints
        if args.kl_beta > 0.0 or args.atgp_lambda > 0.0:
            from src.ppo.networks import ActorNetwork
            ref_actor = ActorNetwork(state_dim=768, action_dim=20).to(device)
            ref_actor.load_state_dict(warmstart_ckpt["actor_state_dict"])
            ref_actor.eval()
            for p in ref_actor.parameters():
                p.requires_grad = False
            if args.kl_beta > 0.0:
                print(f"    KL reference policy loaded (beta={args.kl_beta})")
            if args.atgp_lambda > 0.0:
                print(f"    ATGP reference policy loaded (lambda={args.atgp_lambda}, tau={args.atgp_tau})")

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
        ref_actor=ref_actor,
        kl_beta=args.kl_beta,
        early_stopping_patience=args.early_stopping_patience,
    )

    # 6. Execute Training
    history = trainer.train(verbose=True)

    # ── Final Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[SUCCESS] PPO Training completed across {len(history)} iterations.")
    if trainer.best_composite_info:
        info = trainer.best_composite_info
        print(f"\n  Best checkpoint saved at iteration {int(info['iteration'])}/{args.total_iterations}:")
        print(f"    Composite Score : {info['composite']:.4f}")
        print(f"    Macro F1        : {info['macro_f1']:.4f}")
        print(f"    Accuracy        : {info['accuracy']:.4%}")
        print(f"    Weighted F1     : {info['weighted_f1']:.4f}")
        print(f"    MCC             : {info['mcc']:.4f}")
        print(f"    Saved to        : {os.path.join(args.save_dir, 'best_ppo_policy.pt')}")
    else:
        print("  No evaluation checkpoint was saved (no eval_interval hit).")
    print("=" * 70)


if __name__ == "__main__":
    main()
