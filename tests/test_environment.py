# Responsibility: Comprehensive Verification Test Suite for SOCEnvironment
#
# Tests covered:
#   1. Initialization & Attribute Verification
#   2. Reset Execution & 768-dim Embedding Verification
#   3. Single Step Execution & Gymnasium 5-tuple Verification
#   4. Exact Reward Mechanics (+1.0 for Correct, -1.0 for Incorrect)
#   5. Evaluation Mode Deterministic Sequential Traversal
#   6. Multi-Step Stress Run (100+ steps, NaN/Inf checks, memory safety)
#   7. CISA Phase Text Preparation & Formatting Integrity

import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

from src.environment import SOCEnvironment
from src.encoder import SecBERTStateEncoder


class MockEncoder:
    """
    Fast mock encoder for rapid isolated testing of environment logic
    without requiring heavy GPU/transformer model loads.
    Produces deterministic 768-dimensional float32 tensors.
    """
    def __init__(self, normalize: bool = False):
        self.normalize = normalize

    def encode_incident(self, text: str) -> torch.Tensor:
        # Generate deterministic vector based on text hash
        seed = abs(hash(text)) % (2**31)
        np_gen = np.random.RandomState(seed)
        vec = np_gen.randn(768).astype(np.float32)
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return torch.from_numpy(vec)


class TestSOCEnvironment(unittest.TestCase):

    def setUp(self):
        """Create a synthetic test dataset for isolated unit tests."""
        self.sample_data = pd.DataFrame({
            "text": [
                "Attack: Unauthorized port scan detected on 192.168.1.10",
                "Phishing email with malicious attachment received by CFO",
                "Ransomware encrypting files in C:/Users/Shared",
                "Outbound C2 communication to external suspicious IP",
                "Compromised service account attempting privilege escalation",
            ],
            "action_label": [0, 10, 8, 4, 12],
            "source_dataset": ["Test", "Test", "Test", "Test", "Test"]
        })
        self.mock_encoder = MockEncoder()
        self.env_train = SOCEnvironment(df=self.sample_data, encoder=self.mock_encoder, mode="train")
        self.env_eval = SOCEnvironment(df=self.sample_data, encoder=self.mock_encoder, mode="eval")

    # ── Test 1: Initialization ──────────────────────────────────────────
    def test_initialization(self):
        """Verify that all 6 core attributes are cleanly initialized."""
        env = SOCEnvironment(df=self.sample_data, encoder=self.mock_encoder, mode="train")
        self.assertEqual(len(env.df), 5)
        self.assertIs(env.encoder, self.mock_encoder)
        self.assertIsNone(env.current_sample)
        self.assertIsNone(env.current_embedding)
        self.assertEqual(env.current_index, 0)
        self.assertEqual(env.mode, "train")

        # Invalid mode should raise ValueError
        with self.assertRaises(ValueError):
            SOCEnvironment(df=self.sample_data, encoder=self.mock_encoder, mode="invalid_mode")

    # ── Test 2: Reset Method ────────────────────────────────────────────
    def test_reset_behavior(self):
        """Verify env.reset() returns valid 768-dim tensor and info dict."""
        obs, info = self.env_train.reset(seed=42)

        # Check observation tensor properties
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape, (768,))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(torch.isnan(obs).sum().item(), 0, "Observation contains NaN values!")
        self.assertEqual(torch.isinf(obs).sum().item(), 0, "Observation contains Inf values!")

        # Check metadata
        self.assertIsInstance(info, dict)

        # Check environment internal state
        self.assertIsNotNone(self.env_train.current_sample)
        self.assertIn("text", self.env_train.current_sample)
        self.assertIn("action_label", self.env_train.current_sample)
        self.assertTrue(torch.equal(obs, self.env_train.current_embedding))

    # ── Test 3: Step Method & Gymnasium 5-tuple ─────────────────────────
    def test_step_gymnasium_5_tuple(self):
        """Verify env.step() returns (obs, reward, terminated, truncated, info)."""
        self.env_train.reset(seed=123)
        obs, reward, terminated, truncated, info = self.env_train.step(action=0)

        # Check observation
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape, (768,))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(torch.isnan(obs).sum().item(), 0)

        # Check reward and flags
        self.assertIn(reward, [1.0, -1.0])
        self.assertIsInstance(terminated, bool)
        self.assertFalse(terminated)
        self.assertIsInstance(truncated, bool)
        self.assertFalse(truncated)

        # Check info dictionary
        self.assertIn("correct", info)
        self.assertIn("ground_truth", info)
        self.assertIsInstance(info["correct"], bool)

    # ── Test 4: Reward Mechanics ────────────────────────────────────────
    def test_reward_mechanics(self):
        """Verify exact reward calculation (+1.0 for match, -1.0 for mismatch)."""
        self.env_train.reset(seed=99)
        
        # Determine ground truth for the current sample
        gt = int(self.env_train.current_sample["action_label"])
        wrong_action = (gt + 1) % 20

        # Step with CORRECT action
        _, reward_correct, _, _, info_correct = self.env_train.step(action=gt)
        self.assertEqual(reward_correct, 1.0, "Correct action must yield reward +1.0")
        self.assertTrue(info_correct["correct"])

        # Determine ground truth for the new sample
        new_gt = int(self.env_train.current_sample["action_label"])
        new_wrong = (new_gt + 1) % 20

        # Step with INCORRECT action
        _, reward_wrong, _, _, info_wrong = self.env_train.step(action=new_wrong)
        self.assertEqual(reward_wrong, -1.0, "Incorrect action must yield reward -1.0")
        self.assertFalse(info_wrong["correct"])

    # ── Test 5: Evaluation Mode Determinism ──────────────────────────────
    def test_evaluation_mode_sequential_traversal(self):
        """Verify eval mode steps through the dataset sequentially."""
        obs, info = self.env_eval.reset()

        # Step 0: Should have evaluated sample 0
        expected_first_sample_text = self.sample_data.iloc[0]["text"]
        self.assertIn(expected_first_sample_text, self.env_eval.current_sample["text"])

        # Execute steps sequentially across all rows
        for expected_idx in range(len(self.sample_data)):
            current_gt = int(self.sample_data.iloc[expected_idx]["action_label"])
            next_obs, reward, _, _, step_info = self.env_eval.step(action=current_gt)
            
            # Evaluated action should match ground truth of expected_idx
            self.assertEqual(step_info["ground_truth"], current_gt)
            self.assertEqual(reward, 1.0)
            self.assertEqual(self.env_eval.current_index, expected_idx + 1)

    # ── Test 6: Multi-Step Stress Run ────────────────────────────────────
    def test_multi_step_stress_run(self):
        """Stress test: 200 consecutive random steps without crash or NaN."""
        obs, info = self.env_train.reset(seed=42)
        rewards = []

        for step_i in range(200):
            random_action = np.random.randint(0, 20)
            obs, reward, term, trunc, step_info = self.env_train.step(action=random_action)

            # Robustness checks
            self.assertEqual(obs.shape, (768,))
            self.assertEqual(torch.isnan(obs).sum().item(), 0, f"NaN at step {step_i}")
            self.assertEqual(torch.isinf(obs).sum().item(), 0, f"Inf at step {step_i}")
            self.assertIn(reward, [1.0, -1.0])
            rewards.append(reward)

        self.assertEqual(len(rewards), 200)
        print(f"\n[Stress Test] Completed 200 random steps successfully. Mean reward: {np.mean(rewards):.2f}")

    # ── Test 7: CISA Phase Text Preparation ──────────────────────────────
    def test_prepare_text_formatting(self):
        """Verify that _prepare_text correctly injects CISA phase tags."""
        # Row without tag
        row_unformatted = pd.Series({"text": "Unauthorized login", "action_label": 0})
        formatted = self.env_train._prepare_text(row_unformatted)
        self.assertTrue(formatted.startswith("[Detection]"))

        # Row already formatted
        row_preformatted = pd.Series({"text": "[Containment] Blocked port 445", "action_label": 6})
        formatted2 = self.env_train._prepare_text(row_preformatted)
        self.assertEqual(formatted2, "[Containment] Blocked port 445", "Must not double-tag!")


def run_all_tests():
    """Run test suite and report summary."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE SOC ENVIRONMENT VERIFICATION SUITE")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSOCEnvironment)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        print("\n[FAIL] SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("\n[PASS] ALL SOC ENVIRONMENT TESTS PASSED CLEANLY!")



if __name__ == "__main__":
    run_all_tests()
