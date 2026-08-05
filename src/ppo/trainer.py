# Responsibility: PPO Training Pipeline (PPOTrainer) Orchestrator
#
# This module coordinates the training lifecycle:
#   1. Collecting rollout trajectories from SOCEnvironment into PPOMemory.
#   2. Triggering PPOAgent policy and value updates.
#   3. Tracking training metrics (mean reward, accuracy, loss, entropy).
#   4. Periodic evaluation on validation datasets.
#   5. Model checkpointing for production deployment.

import os
import time
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import torch

from src.environment import SOCEnvironment
from src.ppo.agent import PPOAgent
from src.ppo.memory import PPOMemory


class PPOTrainer:
    """
    Orchestrator for training and evaluating the PPO Agent within the SOC Environment.

    Coordinates:
        - Environment: Generates contextual incident experiences.
        - Memory: Stores (s, a, r, log_prob, v, done) rollout trajectories.
        - Agent: Learns optimal mitigation policy via clipped PPO updates.

    Attributes:
        env (SOCEnvironment): Training incident environment.
        agent (PPOAgent): PPO Actor-Critic agent.
        memory (PPOMemory): Experience replay rollout buffer.
        eval_env (Optional[SOCEnvironment]): Evaluation environment for validation.
        rollout_steps (int): Number of environment interaction steps per update iteration.
        total_iterations (int): Total number of rollout-update cycles.
        log_interval (int): Frequency (in iterations) of progress logging.
        eval_interval (int): Frequency (in iterations) of validation evaluation.
        eval_steps (int): Number of incidents to evaluate during validation.
        save_dir (Optional[str]): Directory path to store model checkpoints.
        history (List[Dict[str, float]]): Accumulated training metric history.
    """

    def __init__(
        self,
        env: SOCEnvironment,
        agent: Optional[PPOAgent] = None,
        memory: Optional[PPOMemory] = None,
        eval_env: Optional[SOCEnvironment] = None,
        rollout_steps: int = 64,
        total_iterations: int = 100,
        log_interval: int = 10,
        eval_interval: int = 20,
        eval_steps: int = 50,
        save_dir: Optional[str] = "models/ppo",
    ):
        # 1. Core RL components
        self.env = env
        self.agent = agent if agent is not None else PPOAgent()
        self.memory = memory if memory is not None else PPOMemory()
        self.eval_env = eval_env

        # 2. Orchestration hyperparameters
        self.rollout_steps = rollout_steps
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_steps = eval_steps
        self.save_dir = save_dir

        # 3. State and metric tracking
        self.history: List[Dict[str, float]] = []
        self.best_eval_accuracy: float = -1.0
        self._current_obs: Optional[torch.Tensor] = None

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def collect_rollout(self) -> Dict[str, float]:
        """
        Collect a rollout trajectory of size `self.rollout_steps` from the environment.

        Steps:
            1. Ensure environment is initialized (reset if necessary).
            2. For each step:
               a. Agent samples action, log_prob, state_value from observation.
               b. Environment steps with chosen action.
               c. Store transition in PPOMemory buffer.
               d. Accumulate reward and accuracy statistics.
            3. Return rollout summary metrics.

        Returns:
            Dict[str, float]: Rollout statistics (mean_reward, accuracy, steps).
        """
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()

        total_reward = 0.0
        correct_count = 0

        for _ in range(self.rollout_steps):
            # 1. Agent decision
            action, log_prob, state_value = self.agent.select_action(self._current_obs)

            # 2. Environment transition
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            # 3. Store in memory buffer
            self.memory.store(
                state=self._current_obs,
                action=action,
                reward=reward,
                log_prob=log_prob,
                value=state_value,
                done=done,
            )

            # 4. Track statistics
            total_reward += reward
            if info.get("correct", False):
                correct_count += 1

            # Advance state (for contextual bandit, next_obs is the next incident)
            self._current_obs = next_obs

        mean_reward = total_reward / self.rollout_steps
        accuracy = correct_count / self.rollout_steps

        return {
            "rollout_mean_reward": float(mean_reward),
            "rollout_accuracy": float(accuracy),
            "rollout_steps": float(self.rollout_steps),
        }

    def evaluate(self, num_steps: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the current policy on the evaluation environment without training updates.

        Args:
            num_steps: Number of evaluation incidents to process. Defaults to `self.eval_steps`.

        Returns:
            Dict[str, float]: Evaluation metrics (eval_mean_reward, eval_accuracy).
        """
        target_env = self.eval_env if self.eval_env is not None else self.env
        steps = num_steps if num_steps is not None else self.eval_steps

        obs, _ = target_env.reset()
        total_reward = 0.0
        correct_count = 0

        for _ in range(steps):
            action, _, _ = self.agent.select_action(obs)
            next_obs, reward, _, _, info = target_env.step(action)

            total_reward += reward
            if info.get("correct", False):
                correct_count += 1

            obs = next_obs

        eval_accuracy = correct_count / steps if steps > 0 else 0.0
        eval_reward = total_reward / steps if steps > 0 else 0.0

        return {
            "eval_accuracy": float(eval_accuracy),
            "eval_mean_reward": float(eval_reward),
            "eval_steps": float(steps),
        }

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save Actor, Critic, and optimizer states to disk.

        Args:
            filepath: Destination .pt file path.
        """
        checkpoint = {
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "actor_optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
            "best_eval_accuracy": self.best_eval_accuracy,
            "hyperparameters": {
                "state_dim": self.agent.state_dim,
                "action_dim": self.agent.action_dim,
                "lr_actor": self.agent.lr_actor,
                "lr_critic": self.agent.lr_critic,
                "gamma": self.agent.gamma,
                "gae_lambda": self.agent.gae_lambda,
                "clip_eps": self.agent.clip_eps,
                "c2_entropy": self.agent.c2_entropy,
                "max_grad_norm": self.agent.max_grad_norm,
                "k_epochs": self.agent.k_epochs,
                "batch_size": self.agent.batch_size,
            },
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load Actor, Critic, and optimizer states from a saved checkpoint.

        Args:
            filepath: Path to the .pt checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "actor_optimizer_state_dict" in checkpoint and self.agent.actor_optimizer is not None:
            self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint and self.agent.critic_optimizer is not None:
            self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self.best_eval_accuracy = checkpoint.get("best_eval_accuracy", -1.0)

    def train(self, verbose: bool = True) -> List[Dict[str, float]]:
        """
        Execute the complete PPO training loop across all iterations.

        Cycle:
            1. Collect rollout experience trajectory.
            2. Run PPO clipped update via agent.update(memory).
            3. Track metrics (reward, accuracy, policy loss, value loss, entropy).
            4. Periodically log progress and run evaluation.
            5. Save best policy checkpoint if validation accuracy improves.

        Args:
            verbose: If True, prints formatted progress logs to stdout.

        Returns:
            List[Dict[str, float]]: History of per-iteration training and evaluation metrics.
        """
        start_time = time.time()
        if verbose:
            print("=" * 70)
            print("Starting PPO Training Pipeline")
            print(f"  Total Iterations: {self.total_iterations}")
            print(f"  Rollout Steps / Iteration: {self.rollout_steps}")
            print(f"  Mini-Batch Size: {self.agent.batch_size}")
            print(f"  Epochs / Update: {self.agent.k_epochs}")
            print(f"  Device: {self.agent.device}")
            print("=" * 70)

        # Reset environment before training starts
        self._current_obs, _ = self.env.reset()

        for iteration in range(1, self.total_iterations + 1):
            iter_start = time.time()

            # 1. Collect rollout
            rollout_metrics = self.collect_rollout()

            # 2. Update agent
            update_metrics = self.agent.update(self.memory)

            # 3. Combine iteration metrics
            iter_metrics = {
                "iteration": float(iteration),
                **rollout_metrics,
                **update_metrics,
                "elapsed_time": float(time.time() - start_time),
            }

            # 4. Periodic Evaluation
            if self.eval_env is not None and (iteration % self.eval_interval == 0 or iteration == self.total_iterations):
                eval_metrics = self.evaluate()
                iter_metrics.update(eval_metrics)

                # Checkpoint best policy
                if eval_metrics["eval_accuracy"] > self.best_eval_accuracy:
                    self.best_eval_accuracy = eval_metrics["eval_accuracy"]
                    if self.save_dir is not None:
                        best_path = os.path.join(self.save_dir, "best_ppo_policy.pt")
                        self.save_checkpoint(best_path)
                        if verbose:
                            print(f"  [*] New best eval accuracy: {self.best_eval_accuracy:.4f} -> Saved to {best_path}")

            self.history.append(iter_metrics)

            # 5. Logging
            if verbose and (iteration % self.log_interval == 0 or iteration == 1 or iteration == self.total_iterations):
                iter_time = time.time() - iter_start
                acc = iter_metrics.get("rollout_accuracy", 0.0)
                rew = iter_metrics.get("rollout_mean_reward", 0.0)
                p_loss = iter_metrics.get("policy_loss", 0.0)
                v_loss = iter_metrics.get("value_loss", 0.0)
                ent = iter_metrics.get("entropy", 0.0)

                log_str = (
                    f"Iter {iteration:4d}/{self.total_iterations:4d} | "
                    f"Acc: {acc:6.2%} | Rew: {rew:+6.2f} | "
                    f"PLoss: {p_loss:8.4f} | VLoss: {v_loss:8.4f} | "
                    f"Ent: {ent:6.3f} | {iter_time:.2f}s"
                )

                if "eval_accuracy" in iter_metrics:
                    log_str += f" | EvalAcc: {iter_metrics['eval_accuracy']:6.2%}"

                print(log_str)

        # Save final checkpoint
        if self.save_dir is not None:
            final_path = os.path.join(self.save_dir, "final_ppo_policy.pt")
            self.save_checkpoint(final_path)
            if verbose:
                print(f"[*] Training finished. Final policy saved to {final_path}")

        return self.history
