#!/usr/bin/env python3
# Responsibility: Comprehensive Benchmark Comparative Evaluation for Experiment 003
#
# Evaluates:
#   1. Random policy
#   2. Always monitor (Action 0)
#   3. Always kill_process (Action 8)
#   4. Always escalate_to_human (Action 19)
#   5. Observable rule-based policy
#   6. Pretrained contextual PPO (Exp 4D ATGP)
#   7. Sequential PPO Exp001
#   8. Sequential PPO Exp002
#   9. Sequential PPO Exp003 (Seed 42)
#   10. Sequential PPO Exp003 (Seed 43)
#   11. Sequential PPO Exp003 (Seed 44)
#
# Features:
#   - Evaluates across 60 benchmark episodes (seeds 1000..1059)
#   - Primary evaluation at escalation_cost = 1.0
#   - Sensitivity evaluation for Exp003 at escalation_cost = 0.0
#   - Computes statistical comparisons (paired t-test, Wilcoxon, 95% CIs)
#   - Computes simulation-step proxy for MTTR (mean/median steps to containment)
#   - Computes full 20-action distributions and scenario-level breakdowns
#   - Outputs machine-readable JSON results.

import os
import sys
import json
import time
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import Counter
import numpy as np
import scipy.stats as stats
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment_sequential import SequentialSOCEnvironment, RewardConfig
from src.environment_augmented import TelemetryAugmentedSOCEnvironment
from src.ppo.agent import PPOAgent
from src.ppo.agent_augmented import AugmentedPPOAgent
from src.ppo.trainer_sequential import (
    make_random_policy,
    make_fixed_action_policy,
    rule_based_policy,
)


def evaluate_policy_detailed(
    env: Any,
    policy_name: str,
    policy_fn: Callable[[torch.Tensor, Dict[str, Any]], int],
    seeds: List[int],
) -> Dict[str, Any]:
    """
    Run policy over given seeds, capturing full episode trajectories, returns,
    actions, containment steps, and outcomes.
    """
    episodes: List[Dict[str, Any]] = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        ep_reward = 0.0
        ep_len = 0
        actions: List[int] = []
        term = False
        trunc = False

        scenario_id = str(info.get("scenario_id", "unknown"))
        is_benign = bool(info.get("is_benign", False))

        while not (term or trunc):
            action = int(policy_fn(obs, info))
            actions.append(action)
            obs, r, term, trunc, info = env.step(action)
            ep_reward += r
            ep_len += 1

        episodes.append({
            "seed": seed,
            "scenario_id": scenario_id,
            "is_benign": is_benign,
            "return": float(ep_reward),
            "length": ep_len,
            "actions": actions,
            "contained": bool(info.get("contained", False)),
            "cleared": bool(info.get("cleared", False)),
            "compromised": bool(info.get("compromised", False)),
            "escalated": bool(info.get("escalated", False)),
            "outcome": str(info.get("outcome", "unknown")),
        })

    total_n = len(episodes)
    mal_eps = [ep for ep in episodes if not ep["is_benign"]]
    ben_eps = [ep for ep in episodes if ep["is_benign"]]
    mal_n = len(mal_eps)
    ben_n = len(ben_eps)

    returns = [ep["return"] for ep in episodes]
    lengths = [ep["length"] for ep in episodes]
    escalation_count = sum(1 for ep in episodes if ep["escalated"])

    mal_contained_eps = [ep for ep in mal_eps if ep["contained"]]
    mal_contained_count = len(mal_contained_eps)
    mal_breached_count = sum(1 for ep in mal_eps if ep["compromised"])
    mal_returns = [ep["return"] for ep in mal_eps] if mal_eps else []
    mal_lengths = [ep["length"] for ep in mal_eps] if mal_eps else []

    # Simulation-step proxy for containment time
    containment_steps = [ep["length"] for ep in mal_contained_eps]
    mean_steps_to_contain = float(np.mean(containment_steps)) if containment_steps else None
    median_steps_to_contain = float(np.median(containment_steps)) if containment_steps else None
    step_distribution = dict(Counter(containment_steps))

    ben_cleared_count = sum(1 for ep in ben_eps if ep["cleared"])
    ben_disrupted_count = sum(1 for ep in ben_eps if ep["outcome"] == "disruptive_false_positive")
    ben_returns = [ep["return"] for ep in ben_eps] if ben_eps else []
    ben_lengths = [ep["length"] for ep in ben_eps] if ben_eps else []

    # 20-Action frequency
    all_actions = [a for ep in episodes for a in ep["actions"]]
    action_counts_20 = {a: 0 for a in range(20)}
    for a in all_actions:
        action_counts_20[a] = action_counts_20.get(a, 0) + 1

    outcome_counts: Dict[str, int] = {}
    for ep in episodes:
        outc = ep["outcome"]
        outcome_counts[outc] = outcome_counts.get(outc, 0) + 1

    # Scenario-level breakdown
    scenario_groups: Dict[str, List[Dict[str, Any]]] = {}
    for ep in episodes:
        sc_id = ep["scenario_id"]
        scenario_groups.setdefault(sc_id, []).append(ep)

    scenario_metrics: Dict[str, Any] = {}
    for sc_id, sc_eps in scenario_groups.items():
        sc_returns = [e["return"] for e in sc_eps]
        sc_contained = sum(1 for e in sc_eps if e["contained"])
        sc_cleared = sum(1 for e in sc_eps if e["cleared"])
        sc_is_benign = sc_eps[0]["is_benign"]
        scenario_metrics[sc_id] = {
            "num_episodes": len(sc_eps),
            "is_benign": sc_is_benign,
            "mean_return": float(np.mean(sc_returns)),
            "containment_rate": float(sc_contained / len(sc_eps)) if not sc_is_benign else None,
            "clearance_rate": float(sc_cleared / len(sc_eps)) if sc_is_benign else None,
            "action_distribution": dict(Counter([a for e in sc_eps for a in e["actions"]])),
        }

    return {
        "policy_name": policy_name,
        "total_episodes": total_n,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "sem_return": float(stats.sem(returns)) if total_n > 1 else 0.0,
        "mean_length": float(np.mean(lengths)),
        "escalation_rate": float(escalation_count / total_n) if total_n > 0 else 0.0,
        "outcomes": outcome_counts,
        "action_counts": action_counts_20,
        "total_action_steps": len(all_actions),
        "malicious": {
            "num_episodes": mal_n,
            "containment_rate": float(mal_contained_count / mal_n) if mal_n > 0 else 0.0,
            "breach_rate": float(mal_breached_count / mal_n) if mal_n > 0 else 0.0,
            "mean_return": float(np.mean(mal_returns)) if mal_returns else 0.0,
            "mean_length": float(np.mean(mal_lengths)) if mal_lengths else 0.0,
            "mean_steps_to_containment": mean_steps_to_contain,
            "median_steps_to_containment": median_steps_to_contain,
            "containment_steps_distribution": step_distribution,
        },
        "benign": {
            "num_episodes": ben_n,
            "clearance_rate": float(ben_cleared_count / ben_n) if ben_n > 0 else 0.0,
            "disruption_rate": float(ben_disrupted_count / ben_n) if ben_n > 0 else 0.0,
            "mean_return": float(np.mean(ben_returns)) if ben_returns else 0.0,
            "mean_length": float(np.mean(ben_lengths)) if ben_lengths else 0.0,
        },
        "scenario_breakdown": scenario_metrics,
        "raw_episodes": episodes,
    }


def compute_paired_comparison(
    pol_a_results: Dict[str, Any],
    pol_b_results: Dict[str, Any],
    name_a: str,
    name_b: str,
) -> Dict[str, Any]:
    """
    Compute paired statistical comparisons between Policy A and Policy B across identical seeds.
    """
    eps_a = pol_a_results["raw_episodes"]
    eps_b = pol_b_results["raw_episodes"]

    assert len(eps_a) == len(eps_b), "Episode count mismatch for paired test"

    returns_a = np.array([ep["return"] for ep in eps_a])
    returns_b = np.array([ep["return"] for ep in eps_b])
    diffs = returns_a - returns_b

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))
    sem_diff = float(stats.sem(diffs))
    ci95_low = mean_diff - 1.96 * sem_diff
    ci95_high = mean_diff + 1.96 * sem_diff

    # Paired t-test
    if np.all(diffs == 0):
        t_stat, p_val_t = 0.0, 1.0
        w_stat, p_val_w = 0.0, 1.0
    else:
        t_stat, p_val_t = stats.ttest_rel(returns_a, returns_b)
        try:
            w_stat, p_val_w = stats.wilcoxon(returns_a, returns_b)
        except Exception:
            w_stat, p_val_w = float("nan"), float("nan")

    # Containment diff
    mal_a = [ep["contained"] for ep in eps_a if not ep["is_benign"]]
    mal_b = [ep["contained"] for ep in eps_b if not ep["is_benign"]]
    cont_rate_a = float(np.mean(mal_a)) if mal_a else 0.0
    cont_rate_b = float(np.mean(mal_b)) if mal_b else 0.0
    cont_diff_pp = (cont_rate_a - cont_rate_b) * 100.0

    # Clearance diff
    ben_a = [ep["cleared"] for ep in eps_a if ep["is_benign"]]
    ben_b = [ep["cleared"] for ep in eps_b if ep["is_benign"]]
    clr_rate_a = float(np.mean(ben_a)) if ben_a else 0.0
    clr_rate_b = float(np.mean(ben_b)) if ben_b else 0.0
    clr_diff_pp = (clr_rate_a - clr_rate_b) * 100.0

    return {
        "comparison": f"{name_a} vs {name_b}",
        "policy_a": name_a,
        "policy_b": name_b,
        "mean_diff_return": mean_diff,
        "std_diff_return": std_diff,
        "sem_diff_return": sem_diff,
        "ci95_diff_return": [float(ci95_low), float(ci95_high)],
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_val_t),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(p_val_w),
        "containment_rate_a": cont_rate_a,
        "containment_rate_b": cont_rate_b,
        "containment_diff_percentage_points": cont_diff_pp,
        "clearance_rate_a": clr_rate_a,
        "clearance_rate_b": clr_rate_b,
        "clearance_diff_percentage_points": clr_diff_pp,
    }


def run_comparative_benchmark():
    benchmark_seeds = list(range(1000, 1060))
    print(f"Executing Comparative Benchmark over {len(benchmark_seeds)} seeds ({benchmark_seeds[0]}..{benchmark_seeds[-1]})...")

    # 1. Initialize Environments
    cfg_cost1 = RewardConfig(escalation_cost=1.0)
    cfg_cost0 = RewardConfig(escalation_cost=0.0)

    standard_env_cost1 = SequentialSOCEnvironment(max_steps=10, reward_config=cfg_cost1)
    aug_env_cost1 = TelemetryAugmentedSOCEnvironment(
        encoder=standard_env_cost1.encoder, max_steps=10, reward_config=cfg_cost1
    )
    aug_env_cost0 = TelemetryAugmentedSOCEnvironment(
        encoder=standard_env_cost1.encoder, max_steps=10, reward_config=cfg_cost0
    )

    # 2. Define Policies
    # Baselines 1-4
    baseline_policies = {
        "1. Uniform Random": (standard_env_cost1, make_random_policy(seed=42)),
        "2. Always Monitor": (standard_env_cost1, make_fixed_action_policy(0)),
        "3. Always Kill Process": (standard_env_cost1, make_fixed_action_policy(8)),
        "4. Always Escalate": (standard_env_cost1, make_fixed_action_policy(19)),
        "5. Observable Rule-Based Policy": (standard_env_cost1, rule_based_policy),
    }

    # Baseline 6: Pretrained Contextual PPO (Exp 4D ATGP)
    exp4d_path = os.path.join(PROJECT_ROOT, "models/ppo_exp4d_atgp/best_ppo_policy.pt")
    if os.path.exists(exp4d_path):
        ag_4d = PPOAgent(state_dim=768, action_dim=20, device="cpu")
        ag_4d_ckpt = torch.load(exp4d_path, map_location="cpu")
        ag_4d.actor.load_state_dict(ag_4d_ckpt["actor_state_dict"])
        ag_4d.actor.eval()
        baseline_policies["6. Contextual PPO (Exp 4D ATGP)"] = (
            standard_env_cost1,
            lambda obs, info, a=ag_4d: a.select_action(obs, deterministic=True)[0],
        )

    # Baseline 7: Sequential PPO Exp001
    exp001_path = os.path.join(PROJECT_ROOT, "models/ppo_sequential_exp001/best_sequential_policy.pt")
    if os.path.exists(exp001_path):
        ag_001 = PPOAgent(state_dim=768, action_dim=20, device="cpu")
        ag_001_ckpt = torch.load(exp001_path, map_location="cpu")
        ag_001.actor.load_state_dict(ag_001_ckpt["actor_state_dict"])
        ag_001.actor.eval()
        baseline_policies["7. Sequential PPO Exp001 (Escalate Trap)"] = (
            standard_env_cost1,
            lambda obs, info, a=ag_001: a.select_action(obs, deterministic=True)[0],
        )

    # Baseline 8: Sequential PPO Exp002
    exp002_path = os.path.join(PROJECT_ROOT, "models/ppo_sequential_exp002/best_sequential_policy.pt")
    if os.path.exists(exp002_path):
        ag_002 = PPOAgent(state_dim=768, action_dim=20, device="cpu")
        ag_002_ckpt = torch.load(exp002_path, map_location="cpu")
        ag_002.actor.load_state_dict(ag_002_ckpt["actor_state_dict"])
        ag_002.actor.eval()
        baseline_policies["8. Sequential PPO Exp002 (Isolate Host Trap)"] = (
            standard_env_cost1,
            lambda obs, info, a=ag_002: a.select_action(obs, deterministic=True)[0],
        )

    # Exp003 Policies (Seeds 42, 43, 44)
    exp003_agents: Dict[int, AugmentedPPOAgent] = {}
    for seed in [42, 43, 44]:
        ckpt_path = os.path.join(PROJECT_ROOT, f"models/ppo_sequential_exp003/seed_{seed}/best_sequential_policy.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Exp003 checkpoint for seed {seed} not found at {ckpt_path}. Train it first.")
        ag_003 = AugmentedPPOAgent(state_dim=802, action_dim=20, device="cpu")
        ag_003_ckpt = torch.load(ckpt_path, map_location="cpu")
        ag_003.actor.load_state_dict(ag_003_ckpt["actor_state_dict"])
        ag_003.actor.eval()
        exp003_agents[seed] = ag_003

    all_eval_configs = dict(baseline_policies)
    for seed, ag in exp003_agents.items():
        all_eval_configs[f"9. Exp003 Sequential PPO (Seed {seed})"] = (
            aug_env_cost1,
            lambda obs, info, a=ag: a.select_action(obs, deterministic=True)[0],
        )

    # Execute primary evaluations (cost = 1.0)
    print("\n--- Running Primary Benchmark Evaluations (escalation_cost = 1.0) ---")
    results: Dict[str, Any] = {}
    for name, (env, pol_fn) in all_eval_configs.items():
        print(f"  Evaluating: {name} ...", end=" ", flush=True)
        summary = evaluate_policy_detailed(env, name, pol_fn, benchmark_seeds)
        results[name] = summary
        print(
            f"Mean Ret: {summary['mean_return']:+.2f} | "
            f"Contain: {summary['malicious']['containment_rate']:.1%} | "
            f"Clear: {summary['benign']['clearance_rate']:.1%} | "
            f"Escalate: {summary['escalation_rate']:.1%}"
        )

    # Execute sensitivity analysis for Exp003 at escalation_cost = 0.0
    print("\n--- Running Sensitivity Analysis for Exp003 (escalation_cost = 0.0) ---")
    sensitivity_results: Dict[str, Any] = {}
    for seed, ag in exp003_agents.items():
        pol_name = f"Exp003 (Seed {seed}, escalation_cost=0.0)"
        print(f"  Evaluating: {pol_name} ...", end=" ", flush=True)
        pol_fn = lambda obs, info, a=ag: a.select_action(obs, deterministic=True)[0]
        sens_summary = evaluate_policy_detailed(aug_env_cost0, pol_name, pol_fn, benchmark_seeds)
        sensitivity_results[pol_name] = sens_summary
        print(
            f"Mean Ret: {sens_summary['mean_return']:+.2f} | "
            f"Contain: {sens_summary['malicious']['containment_rate']:.1%} | "
            f"Clear: {sens_summary['benign']['clearance_rate']:.1%}"
        )

    # 3. Statistical comparisons
    print("\n--- Computing Statistical Comparisons ---")
    statistical_comparisons: List[Dict[str, Any]] = []

    exp003_seed42 = results["9. Exp003 Sequential PPO (Seed 42)"]
    exp003_seed44 = results["9. Exp003 Sequential PPO (Seed 44)"]
    exp002 = results["8. Sequential PPO Exp002 (Isolate Host Trap)"]
    rule_soc = results["5. Observable Rule-Based Policy"]

    # Exp003 (seed 42) vs Exp002
    stat_s42_vs_exp002 = compute_paired_comparison(
        exp003_seed42, exp002, "Exp003 (Seed 42)", "Exp002 (Isolate Host)"
    )
    statistical_comparisons.append(stat_s42_vs_exp002)

    # Exp003 (seed 42) vs Rule-Based
    stat_s42_vs_rule = compute_paired_comparison(
        exp003_seed42, rule_soc, "Exp003 (Seed 42)", "Observable Rule-Based"
    )
    statistical_comparisons.append(stat_s42_vs_rule)

    # Exp003 (seed 44) vs Exp002
    stat_s44_vs_exp002 = compute_paired_comparison(
        exp003_seed44, exp002, "Exp003 (Seed 44)", "Exp002 (Isolate Host)"
    )
    statistical_comparisons.append(stat_s44_vs_exp002)

    # Exp003 (seed 44) vs Rule-Based
    stat_s44_vs_rule = compute_paired_comparison(
        exp003_seed44, rule_soc, "Exp003 (Seed 44)", "Observable Rule-Based"
    )
    statistical_comparisons.append(stat_s44_vs_rule)

    # Exp003 Multi-Seed Aggregation
    exp003_names = [f"9. Exp003 Sequential PPO (Seed {s})" for s in [42, 43, 44]]
    seed_returns = [results[n]["mean_return"] for n in exp003_names]
    seed_containments = [results[n]["malicious"]["containment_rate"] for n in exp003_names]
    seed_clearances = [results[n]["benign"]["clearance_rate"] for n in exp003_names]
    seed_escalations = [results[n]["escalation_rate"] for n in exp003_names]

    exp003_aggregate = {
        "mean_return": float(np.mean(seed_returns)),
        "std_across_seeds_return": float(np.std(seed_returns)),
        "mean_containment": float(np.mean(seed_containments)),
        "std_across_seeds_containment": float(np.std(seed_containments)),
        "mean_clearance": float(np.mean(seed_clearances)),
        "std_across_seeds_clearance": float(np.std(seed_clearances)),
        "mean_escalation": float(np.mean(seed_escalations)),
        "std_across_seeds_escalation": float(np.std(seed_escalations)),
    }

    # Save comparative results JSON (clean version without huge raw arrays)
    output_dir = os.path.join(PROJECT_ROOT, "results/soc_env/exp003")
    os.makedirs(output_dir, exist_ok=True)

    clean_results = {}
    for k, v in results.items():
        v_copy = dict(v)
        v_copy.pop("raw_episodes", None)
        clean_results[k] = v_copy

    clean_sensitivity = {}
    for k, v in sensitivity_results.items():
        v_copy = dict(v)
        v_copy.pop("raw_episodes", None)
        clean_sensitivity[k] = v_copy

    comp_data = {
        "benchmark_seeds": benchmark_seeds,
        "escalation_cost": 1.0,
        "results": clean_results,
        "exp003_aggregate_across_seeds": exp003_aggregate,
        "statistical_comparisons": statistical_comparisons,
    }

    comp_json_path = os.path.join(output_dir, "comparative_evaluation_results.json")
    with open(comp_json_path, "w") as f:
        json.dump(comp_data, f, indent=2)
    print(f"[OK] Benchmark comparative results saved to: {comp_json_path}")

    sens_json_path = os.path.join(output_dir, "sensitivity_evaluation_results.json")
    with open(sens_json_path, "w") as f:
        json.dump(clean_sensitivity, f, indent=2)
    print(f"[OK] Sensitivity evaluation results saved to: {sens_json_path}")

    return comp_data, sensitivity_results


if __name__ == "__main__":
    run_comparative_benchmark()
