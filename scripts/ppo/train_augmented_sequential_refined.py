#!/usr/bin/env python3
# Responsibility: Targeted Refinement Training & Final Benchmark Comparison for Experiment 003
#
# Addresses the two identified failure causes from Exp003 diagnosis:
#   1. Disables premature early-stopping truncation (early_stopping_patience = 0)
#   2. Restores exploratory entropy (c2_entropy = 0.03) to enable discovery of non-disruptive actions.
#
# Preserves all existing Exp001, Exp002, and Exp003 artifacts without modification.

import os
import sys
import json
import time
import random
import argparse
from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment_augmented import TelemetryAugmentedSOCEnvironment
from src.environment_sequential import SequentialSOCEnvironment, RewardConfig
from src.ppo.agent import PPOAgent
from src.ppo.agent_augmented import AugmentedPPOAgent
from src.ppo.trainer_sequential import (
    SequentialPPOTrainer,
    make_random_policy,
    make_fixed_action_policy,
    rule_based_policy,
)
from scripts.evaluate_exp003_comparative import evaluate_policy_detailed, compute_paired_comparison


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_refined_training(seeds: List[int] = [42, 43, 44]):
    reward_config = RewardConfig(escalation_cost=1.0)
    train_env = TelemetryAugmentedSOCEnvironment(max_steps=10, reward_config=reward_config)
    eval_env = TelemetryAugmentedSOCEnvironment(encoder=train_env.encoder, max_steps=10, reward_config=reward_config)

    models_base = os.path.join(PROJECT_ROOT, "models/ppo_sequential_exp003_refined")
    results_base = os.path.join(PROJECT_ROOT, "results/soc_env/exp003/refined")
    os.makedirs(models_base, exist_ok=True)
    os.makedirs(results_base, exist_ok=True)

    trained_agents = {}
    seed_train_summaries = []

    for s in seeds:
        print(f"\n--- Training Refined Exp003 Agent | Seed {s} ---")
        set_seed(s)

        seed_models_dir = os.path.join(models_base, f"seed_{s}")
        seed_results_dir = os.path.join(results_base, f"seed_{s}")
        os.makedirs(seed_models_dir, exist_ok=True)
        os.makedirs(seed_results_dir, exist_ok=True)

        agent = AugmentedPPOAgent(
            state_dim=802,
            action_dim=20,
            lr_actor=1e-4,
            lr_critic=5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_eps=0.2,
            c2_entropy=0.03,  # Principled exploration parameter
            k_epochs=4,
            batch_size=32,
            device="cpu",
        )
        agent.warmstart_from_768_checkpoint(
            "models/ppo_sequential_exp002/best_sequential_policy.pt",
            init_telemetry_weight="zero",
        )

        trainer = SequentialPPOTrainer(
            env=train_env,
            agent=agent,
            eval_env=eval_env,
            rollout_steps=64,
            total_iterations=50,
            eval_interval=5,
            eval_episodes=20,
            eval_seed_start=2000,
            early_stopping_patience=0,  # Eliminates premature cutoff at iteration 25
            save_dir=seed_models_dir,
        )

        # Pre-training eval
        init_eval = trainer.evaluate(num_episodes=20, deterministic=True, seed_start=2000)
        trainer.best_eval_return = init_eval["eval_mean_return"]
        trainer.save_checkpoint(os.path.join(seed_models_dir, "best_sequential_policy.pt"))

        t_start = time.time()
        history = trainer.train(verbose=False)
        duration = time.time() - t_start

        final_ckpt = os.path.join(seed_models_dir, "final_sequential_policy.pt")
        trainer.save_checkpoint(final_ckpt)

        train_log = {
            "experiment_id": "exp003_refined",
            "seed": s,
            "duration_seconds": duration,
            "best_eval_return": trainer.best_eval_return,
            "iterations_completed": len(history),
            "history": history,
        }
        with open(os.path.join(seed_results_dir, "training_logs.json"), "w") as f:
            json.dump(train_log, f, indent=2)

        seed_train_summaries.append({
            "seed": s,
            "duration": duration,
            "best_validation_return": trainer.best_eval_return,
            "checkpoint_dir": seed_models_dir,
        })

        # Load best validation policy for evaluation
        best_agent = AugmentedPPOAgent(state_dim=802, action_dim=20, device="cpu")
        best_agent.load_checkpoint(os.path.join(seed_models_dir, "best_sequential_policy.pt"))
        best_agent.actor.eval()
        trained_agents[s] = best_agent

        print(f"  Seed {s} Complete: Duration {duration:.1f}s | Best Val Return: {trainer.best_eval_return:+.3f}")

    return trained_agents, seed_train_summaries


def evaluate_final_comparison(refined_agents: Dict[int, AugmentedPPOAgent]):
    benchmark_seeds = list(range(1000, 1060))
    cfg = RewardConfig(escalation_cost=1.0)
    std_env = SequentialSOCEnvironment(max_steps=10, reward_config=cfg)
    aug_env = TelemetryAugmentedSOCEnvironment(encoder=std_env.encoder, max_steps=10, reward_config=cfg)

    policies = {}
    policies["1. Uniform Random"] = (std_env, make_random_policy(seed=42))
    policies["2. Always Monitor"] = (std_env, make_fixed_action_policy(0))
    policies["3. Always Kill Process"] = (std_env, make_fixed_action_policy(8))
    policies["4. Always Escalate"] = (std_env, make_fixed_action_policy(19))
    policies["5. Observable Rule-Based Policy"] = (std_env, rule_based_policy)

    # Pretrained Contextual PPO
    ag_4d = PPOAgent(state_dim=768, action_dim=20, device="cpu")
    ag_4d.actor.load_state_dict(torch.load("models/ppo_exp4d_atgp/best_ppo_policy.pt", map_location="cpu")["actor_state_dict"])
    ag_4d.actor.eval()
    policies["6. Contextual PPO (Exp 4D ATGP)"] = (std_env, lambda o, i, a=ag_4d: a.select_action(o, deterministic=True)[0])

    # Exp002
    ag_002 = PPOAgent(state_dim=768, action_dim=20, device="cpu")
    ag_002.actor.load_state_dict(torch.load("models/ppo_sequential_exp002/best_sequential_policy.pt", map_location="cpu")["actor_state_dict"])
    ag_002.actor.eval()
    policies["7. Sequential PPO Exp002 (Isolate Host Trap)"] = (std_env, lambda o, i, a=ag_002: a.select_action(o, deterministic=True)[0])

    # Initial Exp003 Best (Seed 44)
    ag_003_orig = AugmentedPPOAgent(state_dim=802, action_dim=20, device="cpu")
    ag_003_orig.actor.load_state_dict(torch.load("models/ppo_sequential_exp003/seed_44/best_sequential_policy.pt", map_location="cpu")["actor_state_dict"])
    ag_003_orig.actor.eval()
    policies["8. Exp003 Initial (Seed 44)"] = (aug_env, lambda o, i, a=ag_003_orig: a.select_action(o, deterministic=True)[0])

    # Refined Exp003 Agents
    for s, ag in refined_agents.items():
        policies[f"9. Exp003 Refined (Seed {s})"] = (aug_env, lambda o, i, a=ag: a.select_action(o, deterministic=True)[0])

    raw_results = {}
    clean_results = {}
    print("\n--- Evaluating Final Benchmark Policies on 60 Seeds (1000..1059) ---")
    for name, (env, p_fn) in policies.items():
        res = evaluate_policy_detailed(env, name, p_fn, benchmark_seeds)
        raw_results[name] = res
        clean = dict(res)
        clean.pop("raw_episodes", None)
        clean_results[name] = clean
        print(f"  {name:<42} | Ret: {res['mean_return']:+.2f} | MalCont: {res['malicious']['containment_rate']:.1%} | BenClr: {res['benign']['clearance_rate']:.1%} | BenDisrupt: {res['benign']['disruption_rate']:.1%}")

    # Statistical comparisons
    stats_list = []
    rule_res = raw_results["5. Observable Rule-Based Policy"]
    exp002_res = raw_results["7. Sequential PPO Exp002 (Isolate Host Trap)"]
    exp003_orig_res = raw_results["8. Exp003 Initial (Seed 44)"]

    for s in [42, 43, 44]:
        ref_res = raw_results[f"9. Exp003 Refined (Seed {s})"]
        stats_list.append(compute_paired_comparison(ref_res, exp002_res, f"Exp003_Refined_s{s}", "Exp002"))
        stats_list.append(compute_paired_comparison(ref_res, rule_res, f"Exp003_Refined_s{s}", "RuleBased"))
        stats_list.append(compute_paired_comparison(ref_res, exp003_orig_res, f"Exp003_Refined_s{s}", "Exp003_Initial_s44"))

    # Multi-seed aggregate for Refined Exp003
    ref_names = [f"9. Exp003 Refined (Seed {s})" for s in [42, 43, 44]]
    returns = [clean_results[n]["mean_return"] for n in ref_names]
    conts = [clean_results[n]["malicious"]["containment_rate"] for n in ref_names]
    clrs = [clean_results[n]["benign"]["clearance_rate"] for n in ref_names]
    disrupts = [clean_results[n]["benign"]["disruption_rate"] for n in ref_names]

    aggregate = {
        "mean_return_across_seeds": float(np.mean(returns)),
        "std_return_across_seeds": float(np.std(returns)),
        "mean_containment": float(np.mean(conts)),
        "mean_clearance": float(np.mean(clrs)),
        "mean_disruption": float(np.mean(disrupts)),
    }

    final_payload = {
        "benchmark_seeds": benchmark_seeds,
        "escalation_cost": 1.0,
        "policies": clean_results,
        "exp003_refined_aggregate": aggregate,
        "statistical_comparisons": stats_list,
    }

    out_file = os.path.join(PROJECT_ROOT, "results/soc_env/exp003/final_comparison_results.json")
    with open(out_file, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[OK] Final comparison results saved to: {out_file}")

    return final_payload


def main():
    print("=" * 80)
    print("Executing Targeted Improvement: Refined Experiment 003")
    print("=" * 80)
    agents, summaries = run_refined_training(seeds=[42, 43, 44])
    evaluate_final_comparison(agents)
    print("\n[COMPLETE] Refinement run and final evaluation finished successfully.")


if __name__ == "__main__":
    main()
