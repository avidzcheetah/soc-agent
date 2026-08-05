import os
import sys
import shutil
import pytest
import pandas as pd
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.environment import SOCEnvironment
from src.ppo import PPOAgent, PPOMemory, PPOTrainer
from tests.test_environment import MockEncoder


@pytest.fixture
def trainer_setup():
    train_df = pd.DataFrame({
        "text": [f"Incident alert description {i}" for i in range(80)],
        "action_label": [i % 20 for i in range(80)],
    })
    eval_df = pd.DataFrame({
        "text": [f"Evaluation incident alert {i}" for i in range(30)],
        "action_label": [i % 20 for i in range(30)],
    })
    encoder = MockEncoder()
    train_env = SOCEnvironment(train_df, encoder, mode="train")
    eval_env = SOCEnvironment(eval_df, encoder, mode="eval")

    test_save_dir = "models/ppo/test_trainer_pytest"
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)

    agent = PPOAgent(batch_size=8, k_epochs=2)
    trainer = PPOTrainer(
        env=train_env,
        agent=agent,
        eval_env=eval_env,
        rollout_steps=16,
        total_iterations=3,
        log_interval=1,
        eval_interval=1,
        eval_steps=10,
        save_dir=test_save_dir,
    )

    yield trainer, test_save_dir

    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)


def test_collect_rollout(trainer_setup):
    trainer, _ = trainer_setup
    metrics = trainer.collect_rollout()

    assert len(trainer.memory) == 16
    assert "rollout_mean_reward" in metrics
    assert "rollout_accuracy" in metrics
    assert metrics["rollout_steps"] == 16.0
    trainer.memory.clear()


def test_evaluate(trainer_setup):
    trainer, _ = trainer_setup
    eval_metrics = trainer.evaluate(num_steps=15)

    assert "eval_accuracy" in eval_metrics
    assert "eval_mean_reward" in eval_metrics
    assert eval_metrics["eval_steps"] == 15.0


def test_train_loop_and_checkpointing(trainer_setup):
    trainer, test_save_dir = trainer_setup
    history = trainer.train(verbose=False)

    assert len(history) == 3
    assert len(trainer.memory) == 0  # Memory cleared after updates

    for record in history:
        assert "policy_loss" in record
        assert "value_loss" in record
        assert "entropy" in record
        assert "rollout_mean_reward" in record

    # Verify saved checkpoint
    final_path = os.path.join(test_save_dir, "final_ppo_policy.pt")
    assert os.path.exists(final_path)

    # Verify loading checkpoint
    new_agent = PPOAgent()
    trainer_new = PPOTrainer(env=trainer.env, agent=new_agent)
    trainer_new.load_checkpoint(final_path)

    for p1, p2 in zip(trainer.agent.actor.parameters(), trainer_new.agent.actor.parameters()):
        assert torch.equal(p1, p2)
