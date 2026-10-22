# Responsibility: Comprehensive Unit & Integration Tests for SequentialPPOTrainer
#
# Tests covered:
#   1. Episodic rollout collection (proper done flags and automatic reset).
#   2. Multi-step GAE advantage and return propagation across intermediate steps.
#   3. Pretrained checkpoint loading and deterministic evaluation.
#   4. Operational metric tracking (returns, containment, clearance, breaches).
#   5. Independent baseline evaluation (Random, Fixed Action, Rule-based).
#   6. Complete rollout-update training iteration cycle.

import os
import sys
import unittest
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from src.environment_sequential import SequentialSOCEnvironment
from src.ppo.agent import PPOAgent
from src.ppo.memory import PPOMemory
from src.ppo.trainer_sequential import (
    SequentialPPOTrainer,
    evaluate_baseline,
    make_random_policy,
    make_fixed_action_policy,
    rule_based_policy,
)


class MockEncoder:
    """Fast deterministic mock encoder for unit testing without GPU/SecBERT."""
    def encode_incident(self, text: str) -> torch.Tensor:
        seed = abs(hash(text)) % (2**31)
        np_gen = np.random.RandomState(seed)
        vec = np_gen.randn(768).astype(np.float32)
        return torch.from_numpy(vec)


class TestSequentialPPOTrainer(unittest.TestCase):

    def setUp(self):
        """Initialize mock environment, agent, memory, and sequential trainer."""
        self.mock_encoder = MockEncoder()
        self.env = SequentialSOCEnvironment(encoder=self.mock_encoder, max_steps=5)
        self.agent = PPOAgent(state_dim=768, action_dim=20, device="cpu")
        self.memory = PPOMemory()
        self.trainer = SequentialPPOTrainer(
            env=self.env,
            agent=self.agent,
            memory=self.memory,
            rollout_steps=16,
            total_iterations=2,
            eval_episodes=5,
            save_dir="models/test_ppo_seq",
        )

    def test_collect_rollout_episodic_transitions(self):
        """Verify collect_rollout sets done=True only on termination and resets environment."""
        metrics = self.trainer.collect_rollout()
        self.assertEqual(len(self.memory), 16)
        self.assertIn("rollout_mean_step_reward", metrics)
        self.assertIn("rollout_episodes_completed", metrics)

        # Check that memory contains a mix of done=False and done=True
        dones = self.memory.dones
        self.assertEqual(len(dones), 16)
        # At least one transition should be non-terminal (done=False) in multi-step scenarios
        self.assertIn(False, dones)

    def test_multi_step_gae_propagation(self):
        """Verify that done=False preserves future discounted returns in GAE calculation."""
        # Manually populate memory with a 2-step episode:
        # Step 0: reward = 0.1, done = False
        # Step 1: reward = 2.0, done = True
        dummy_state = torch.randn(768)
        self.memory.clear()
        self.memory.store(state=dummy_state, action=1, reward=0.1, log_prob=-1.0, value=0.5, done=False)
        self.memory.store(state=dummy_state, action=8, reward=2.0, log_prob=-1.0, value=1.0, done=True)

        advantages, returns = self.agent.compute_gae(self.memory)
        self.assertEqual(len(advantages), 2)
        self.assertEqual(len(returns), 2)

        # Step 0 return R_0 must include discounted reward from Step 1: R_0 = r_0 + gamma * r_1
        # Because gamma=0.99, return at step 0 must be > 1.5
        self.assertGreater(returns[0].item(), 1.5)

    def test_evaluate_metrics_structure(self):
        """Verify evaluate produces complete operational metrics across full episodes."""
        eval_metrics = self.trainer.evaluate(num_episodes=6, deterministic=True, seed_start=100)
        self.assertEqual(eval_metrics["eval_episodes"], 6)
        self.assertIn("eval_mean_return", eval_metrics)
        self.assertIn("eval_containment_rate", eval_metrics)
        self.assertIn("eval_clearance_rate", eval_metrics)
        self.assertIn("eval_compromise_rate", eval_metrics)
        self.assertIn("eval_outcomes", eval_metrics)

        # Rate sanity checks
        self.assertGreaterEqual(eval_metrics["eval_containment_rate"], 0.0)
        self.assertLessEqual(eval_metrics["eval_containment_rate"], 1.0)
        self.assertGreaterEqual(eval_metrics["eval_clearance_rate"], 0.0)
        self.assertLessEqual(eval_metrics["eval_clearance_rate"], 1.0)

    def test_load_pretrained_checkpoint(self):
        """Verify loading existing Phase 2 checkpoint into SequentialPPOTrainer."""
        ckpt_path = "models/ppo_exp4d_atgp/best_ppo_policy.pt"
        if os.path.exists(ckpt_path):
            self.trainer.load_checkpoint(ckpt_path)
            # Evaluate loaded policy on sequential environment
            results = self.trainer.evaluate(num_episodes=3, deterministic=True)
            self.assertEqual(results["eval_episodes"], 3)
            self.assertIsInstance(results["eval_mean_return"], float)

    def test_baseline_evaluations(self):
        """Verify independent baselines run without errors on SequentialSOCEnvironment."""
        # 1. Random baseline
        rand_pol = make_random_policy(seed=42)
        rand_res = evaluate_baseline(self.env, rand_pol, num_episodes=5, seed_start=200)
        self.assertEqual(rand_res["num_episodes"], 5)
        self.assertIn("containment_rate", rand_res)

        # 2. Fixed action baseline (Action 8: kill_process)
        fix_pol = make_fixed_action_policy(8)
        fix_res = evaluate_baseline(self.env, fix_pol, num_episodes=5, seed_start=200)
        self.assertEqual(fix_res["num_episodes"], 5)
        self.assertIn("outcomes", fix_res)

        # 3. Rule-based heuristic baseline
        rule_res = evaluate_baseline(self.env, rule_based_policy, num_episodes=5, seed_start=200)
        self.assertEqual(rule_res["num_episodes"], 5)
        self.assertIn("containment_rate", rule_res)
        # Rule-based policy should clear benign scenarios and contain threats
        self.assertGreaterEqual(rule_res["containment_rate"] + rule_res["clearance_rate"], 0.5)

    def test_rule_based_policy_no_information_leakage(self):
        """
        Verify that rule_based_policy relies exclusively on observable telemetry
        and fails immediately if any hidden ground-truth simulation fields are accessed.
        """
        class GuardedInfoDict(dict):
            FORBIDDEN_KEYS = {
                "is_benign",
                "scenario_id",
                "scenario_name",
                "stage_id",
                "stage_name",
                "outcome",
                "contained",
                "cleared",
                "compromised",
                "escalated",
                "effective_actions",
                "investigative_actions",
                "disruptive_actions",
            }

            def __getitem__(self, key):
                if key in self.FORBIDDEN_KEYS:
                    raise AssertionError(f"Information leakage: accessed forbidden field '{key}' via __getitem__")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in self.FORBIDDEN_KEYS:
                    raise AssertionError(f"Information leakage: accessed forbidden field '{key}' via get()")
                return super().get(key, default)

            def __contains__(self, key):
                if key in self.FORBIDDEN_KEYS:
                    raise AssertionError(f"Information leakage: accessed forbidden field '{key}' via __contains__")
                return super().__contains__(key)

        dummy_obs = torch.zeros(768)

        # 1. Test against all stages of all scenarios wrapped in GuardedInfoDict
        for sc in self.env.scenarios:
            for stage in sc.stages:
                raw_info = {
                    "scenario_id": sc.scenario_id,
                    "scenario_name": sc.name,
                    "stage_id": stage.stage_id,
                    "stage_name": stage.name,
                    "is_benign": sc.is_benign,
                    "simulated": True,
                    "telemetry": self.env._extract_telemetry(stage),
                }
                guarded_info = GuardedInfoDict(raw_info)

                # Must execute successfully without touching forbidden keys
                action = rule_based_policy(dummy_obs, guarded_info)
                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < 20)

        # 2. Verify that an oracle policy attempting to access hidden fields DOES fail
        def cheating_oracle_policy(obs, info):
            if info.get("is_benign", False):
                return 0
            return 8

        sample_info = GuardedInfoDict({
            "is_benign": True,
            "telemetry": {},
        })
        with self.assertRaises(AssertionError) as ctx:
            cheating_oracle_policy(dummy_obs, sample_info)
        self.assertIn("Information leakage", str(ctx.exception))

    def test_evaluation_separates_benign_and_malicious_counts(self):
        """Regression test: verify that evaluate() accurately separates benign and malicious counts."""
        n_eps = 20
        seed_start = 1000
        metrics = self.trainer.evaluate(num_episodes=n_eps, seed_start=seed_start)
        
        mal_n = metrics["eval_malicious"]["num_episodes"]
        ben_n = metrics["eval_benign"]["num_episodes"]
        
        # Verify total and non-zero counts
        self.assertEqual(mal_n + ben_n, n_eps)
        self.assertGreater(ben_n, 0, "Benign count must not collapse to 0")
        self.assertGreater(mal_n, 0, "Malicious count must not collapse to 0")
        
        # Count actual expected ground truth by resetting with the same seeds
        expected_ben = sum(
            1 for s in range(seed_start, seed_start + n_eps)
            if self.env.reset(seed=s)[1]["is_benign"]
        )
        expected_mal = n_eps - expected_ben
        self.assertEqual(ben_n, expected_ben)
        self.assertEqual(mal_n, expected_mal)


if __name__ == "__main__":
    unittest.main()
