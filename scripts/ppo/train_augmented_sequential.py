#!/usr/bin/env python3
# Responsibility: Controlled Sequential PPO Training Pipeline for Experiment 003
#
# Executes:
#   1. Training configuration capture and logging.
#   2. Dimension-safe warm-start initialization from Exp002 checkpoint (768 -> 802).
#   3. Controlled sequential PPO training on TelemetryAugmentedSOCEnvironment (802-dim).
#   4. Validation-selected checkpoint tracking on held-out seeds (2000..2019).
#   5. Multi-seed training loop for seeds 42, 43, 44 with separate artifacts.
#   6. Complete logging of training dynamics and validation history.

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment_augmented import TelemetryAugmentedSOCEnvironment
from src.environment_sequential import RewardConfig
from src.ppo.agent_augmented import AugmentedPPOAgent
from src.ppo.trainer_sequential import SequentialPPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Execute Controlled Sequential PPO Experiment 003")
    parser.add_argument("--exp_id", type=str, default="exp003", help="Experiment identifier")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to train")
    parser.add_argument("--total_iterations", type=int, default=50, help="Total PPO update iterations")
    parser.add_argument("--rollout_steps", type=int, default=64, help="Environment steps collected per rollout")
    parser.add_argument("--eval_interval", type=int, default=5, help="Iterations between periodic evaluations")
    parser.add_argument("--eval_episodes", type=int, default=20, help="Episodes per periodic evaluation")
    parser.add_argument("--eval_seed_start", type=int, default=2000, help="Start seed for periodic evaluation")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="Learning rate for Actor network")
    parser.add_argument("--lr_critic", type=float, default=5e-4, help="Learning rate for Critic network")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--c2_entropy", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--k_epochs", type=int, default=4, help="PPO update epochs per iteration")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size for PPO updates")
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="models/ppo_sequential_exp002/best_sequential_policy.pt",
        help="Path to pre-existing 768-dim Exp002 checkpoint for dimension-safe warm-start",
    )
    parser.add_argument(
        "--escalation_cost",
        type=float,
        default=1.0,
        help="Operational escalation penalty cost (default: 1.0)",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_seed(seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    print("=" * 80)
    print(f"Executing Sequential PPO Training Run | Experiment: {args.exp_id} | Seed: {seed}")
    print("=" * 80)

    # 1. Output directories for this seed
    seed_tag = f"seed_{seed}"
    models_dir = os.path.join(PROJECT_ROOT, f"models/ppo_sequential_{args.exp_id}/{seed_tag}")
    results_dir = os.path.join(PROJECT_ROOT, f"results/soc_env/{args.exp_id}/{seed_tag}")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Set seed
    set_seed(seed)

    # 3. Save training configuration
    init_ckpt_abs = (
        os.path.join(PROJECT_ROOT, args.init_checkpoint)
        if not os.path.isabs(args.init_checkpoint)
        else args.init_checkpoint
    )
    if not os.path.exists(init_ckpt_abs):
        raise FileNotFoundError(f"Required warm-start checkpoint not found: {init_ckpt_abs}")

    config = {
        "experiment_id": args.exp_id,
        "seed": seed,
        "init_checkpoint": args.init_checkpoint,
        "total_iterations": args.total_iterations,
        "rollout_steps": args.rollout_steps,
        "total_env_steps": args.total_iterations * args.rollout_steps,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "eval_seed_start": args.eval_seed_start,
        "hyperparameters": {
            "state_dim": 802,
            "action_dim": 20,
            "lr_actor": args.lr_actor,
            "lr_critic": args.lr_critic,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_eps": args.clip_eps,
            "c2_entropy": args.c2_entropy,
            "k_epochs": args.k_epochs,
            "batch_size": args.batch_size,
        },
        "environment": {
            "name": "TelemetryAugmentedSOCEnvironment",
            "observation_dim": 802,
            "secbert_dim": 768,
            "telemetry_dim": 34,
            "action_dim": 20,
            "num_scenarios": 7,
            "max_steps": 10,
            "escalation_cost": args.escalation_cost,
        },
    }

    config_path = os.path.join(results_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # 4. Initialize Environments
    reward_config = RewardConfig(escalation_cost=args.escalation_cost)
    train_env = TelemetryAugmentedSOCEnvironment(max_steps=10, reward_config=reward_config)
    eval_env = TelemetryAugmentedSOCEnvironment(
        encoder=train_env.encoder, max_steps=10, reward_config=reward_config
    )

    # 5. Initialize Augmented PPO Agent and warmstart
    agent = AugmentedPPOAgent(
        state_dim=802,
        action_dim=20,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        c2_entropy=args.c2_entropy,
        k_epochs=args.k_epochs,
        batch_size=args.batch_size,
        device="cpu",
    )

    print(f"Dimension-safe warm-starting weights from: {init_ckpt_abs}")
    agent.warmstart_from_768_checkpoint(init_ckpt_abs, init_telemetry_weight="zero")

    # Save initial policy
    initial_ckpt_path = os.path.join(models_dir, "initial_policy.pt")
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "hyperparameters": config["hyperparameters"],
            "source": args.init_checkpoint,
        },
        initial_ckpt_path,
    )

    # 6. Initialize SequentialPPOTrainer
    trainer = SequentialPPOTrainer(
        env=train_env,
        agent=agent,
        eval_env=eval_env,
        rollout_steps=args.rollout_steps,
        total_iterations=args.total_iterations,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_seed_start=args.eval_seed_start,
        save_dir=models_dir,
    )

    # Initial pre-training evaluation (Iteration 0)
    print("Evaluating pre-training policy on validation seeds (Iteration 0)...")
    init_eval = trainer.evaluate(
        num_episodes=args.eval_episodes,
        deterministic=True,
        seed_start=args.eval_seed_start,
    )
    trainer.best_eval_return = init_eval["eval_mean_return"]
    print(
        f"  Iter 000 Eval Return: {init_eval['eval_mean_return']:+.3f} | "
        f"Mal Contain: {init_eval['eval_malicious']['containment_rate']:.1%} | "
        f"Ben Clear: {init_eval['eval_benign']['clearance_rate']:.1%}"
    )

    # Save initial checkpoint as default best
    best_ckpt_path = os.path.join(models_dir, "best_sequential_policy.pt")
    trainer.save_checkpoint(best_ckpt_path)

    # 7. Training loop
    print("Executing training loop (50 iterations)...")
    start_time = time.time()
    history = trainer.train(verbose=True)
    train_duration = time.time() - start_time
    print(f"[OK] Training complete for seed {seed} in {train_duration:.2f}s.")

    # Save final checkpoint
    final_ckpt_path = os.path.join(models_dir, "final_sequential_policy.pt")
    trainer.save_checkpoint(final_ckpt_path)

    # 8. Save training logs
    logs_path = os.path.join(results_dir, "training_logs.json")
    training_summary = {
        "experiment_id": args.exp_id,
        "seed": seed,
        "duration_seconds": train_duration,
        "iterations_completed": len(history),
        "best_eval_return": trainer.best_eval_return,
        "initial_eval": init_eval,
        "history": history,
    }
    with open(logs_path, "w") as f:
        json.dump(training_summary, f, indent=2)

    print(f"[OK] Seed {seed} artifacts saved successfully to:\n  Models : {models_dir}\n  Results: {results_dir}\n")

    return {
        "seed": seed,
        "duration_seconds": train_duration,
        "best_eval_return": trainer.best_eval_return,
        "best_checkpoint_path": best_ckpt_path,
        "final_checkpoint_path": final_ckpt_path,
        "training_logs_path": logs_path,
        "training_config_path": config_path,
    }


def main():
    args = parse_args()
    print("=" * 80)
    print(f"Starting Multi-Seed Experiment 003 Sequential Training: Seeds {args.seeds}")
    print("=" * 80)

    all_seed_results = []
    total_start = time.time()
    for s in args.seeds:
        res = train_single_seed(s, args)
        all_seed_results.append(res)

    # Save multi-seed summary
    exp_summary_dir = os.path.join(PROJECT_ROOT, f"results/soc_env/{args.exp_id}")
    os.makedirs(exp_summary_dir, exist_ok=True)
    summary_path = os.path.join(exp_summary_dir, "multi_seed_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "experiment_id": args.exp_id,
                "seeds": args.seeds,
                "total_duration_seconds": time.time() - total_start,
                "seed_results": all_seed_results,
            },
            f,
            indent=2,
        )
    print(f"[ALL SEEDS COMPLETE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
