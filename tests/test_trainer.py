import os
import sys
import shutil
import unittest
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment import SOCEnvironment
from src.ppo import PPOAgent, PPOMemory, PPOTrainer
from tests.test_environment import MockEncoder


class TestPPOTrainer(unittest.TestCase):
    def setUp(self):
        train_df = pd.DataFrame({
            "text": [f"Incident alert description {i}" for i in range(80)],
            "action_label": [i % 20 for i in range(80)],
        })
        eval_df = pd.DataFrame({
            "text": [f"Evaluation incident alert {i}" for i in range(30)],
            "action_label": [i % 20 for i in range(30)],
        })
        encoder = MockEncoder()
        self.train_env = SOCEnvironment(train_df, encoder, mode="train")
        self.eval_env = SOCEnvironment(eval_df, encoder, mode="eval")

        self.test_save_dir = "models/ppo/test_trainer_pytest"
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)

        self.agent = PPOAgent(batch_size=8, k_epochs=2)
        self.trainer = PPOTrainer(
            env=self.train_env,
            agent=self.agent,
            eval_env=self.eval_env,
            rollout_steps=16,
            total_iterations=3,
            log_interval=1,
            eval_interval=1,
            eval_steps=10,
            save_dir=self.test_save_dir,
        )

    def tearDown(self):
        if os.path.exists(self.test_save_dir):
            shutil.rmtree(self.test_save_dir)

    def test_collect_rollout(self):
        metrics = self.trainer.collect_rollout()
        self.assertEqual(len(self.trainer.memory), 16)
        self.assertIn("rollout_mean_reward", metrics)
        self.assertIn("rollout_accuracy", metrics)
        self.assertEqual(metrics["rollout_steps"], 16.0)
        self.trainer.memory.clear()

    def test_evaluate(self):
        eval_metrics = self.trainer.evaluate(num_steps=15)
        self.assertIn("eval_accuracy", eval_metrics)
        self.assertIn("eval_mean_reward", eval_metrics)
        self.assertEqual(eval_metrics["eval_steps"], 30.0)

    def test_train_loop_and_checkpointing(self):
        history = self.trainer.train(verbose=False)
        self.assertEqual(len(history), 3)
        self.assertEqual(len(self.trainer.memory), 0)

        for record in history:
            self.assertIn("policy_loss", record)
            self.assertIn("value_loss", record)
            self.assertIn("entropy", record)
            self.assertIn("rollout_mean_reward", record)

        final_path = os.path.join(self.test_save_dir, "final_ppo_policy.pt")
        self.assertTrue(os.path.exists(final_path))

        new_agent = PPOAgent()
        trainer_new = PPOTrainer(env=self.trainer.env, agent=new_agent)
        trainer_new.load_checkpoint(final_path)

        for p1, p2 in zip(self.trainer.agent.actor.parameters(), trainer_new.agent.actor.parameters()):
            self.assertTrue(torch.equal(p1, p2))


if __name__ == "__main__":
    unittest.main()
