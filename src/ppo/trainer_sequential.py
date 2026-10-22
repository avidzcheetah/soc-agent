# Responsibility: Sequential PPO Training Pipeline (SequentialPPOTrainer) Orchestrator
#
# Coordinates multi-step episodic reinforcement learning within SequentialSOCEnvironment:
#   1. Episodic rollout collection tracking true termination and truncation (done = term or trunc).
#   2. Environment reset upon episode completion.
#   3. Multi-step Generalized Advantage Estimation (GAE) across sequential transitions.
#   4. Episode-level metric tracking (returns, episode length, containment, clearance, breaches).
#   5. Pretrained checkpoint loading and saving for fine-tuning and evaluation.
#   6. Standardized baseline evaluation functions (Random, Fixed Action, Rule-based).

import os
import time
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
import torch

from src.environment_sequential import SequentialSOCEnvironment
from src.ppo.agent import PPOAgent
from src.ppo.memory import PPOMemory


class SequentialPPOTrainer:
    """
    Orchestrator for multi-step episodic PPO training and evaluation in SequentialSOCEnvironment.

    Distinctions from contextual-bandit PPOTrainer:
      - Tracks full multi-step episodes; does NOT force done=True on every step.
      - Resets environment when an episode terminates or truncates.
      - Preserves temporal credit assignment via GAE discount (gamma) across stage transitions.
      - Tracks episodic operational metrics: containment rate, benign clearance rate, breach rate.
    """

    def __init__(
        self,
        env: SequentialSOCEnvironment,
        agent: Optional[PPOAgent] = None,
        memory: Optional[PPOMemory] = None,
        eval_env: Optional[SequentialSOCEnvironment] = None,
        rollout_steps: int = 64,
        total_iterations: int = 100,
        log_interval: int = 10,
        eval_interval: int = 20,
        eval_episodes: int = 20,
        save_dir: Optional[str] = "models/ppo_sequential",
        ref_actor: Optional[torch.nn.Module] = None,
        kl_beta: float = 0.0,
        early_stopping_patience: int = 5,
        eval_seed_start: Optional[int] = 5000,
    ):
        # 1. Core RL components
        self.env = env
        self.agent = agent if agent is not None else PPOAgent()
        self.memory = memory if memory is not None else PPOMemory()
        self.eval_env = eval_env

        # 2. Hyperparameters
        self.rollout_steps = rollout_steps
        self.total_iterations = total_iterations
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.save_dir = save_dir

        # KL constraint against reference warm-start policy (optional)
        self.ref_actor = ref_actor
        self.kl_beta = kl_beta

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.eval_seed_start = eval_seed_start

        # 3. Episodic state tracking
        self.history: List[Dict[str, Any]] = []
        self.best_eval_return: float = -np.inf
        self._current_obs: Optional[torch.Tensor] = None
        self._current_ep_reward: float = 0.0
        self._current_ep_len: int = 0

        # Completed episode buffer for current rollout iteration
        self._completed_episodes: List[Dict[str, Any]] = []

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def collect_rollout(self) -> Dict[str, float]:
        """
        Collect a rollout of `self.rollout_steps` transitions across multi-step episodes.

        Unlike contextual bandit training, `done` is set to True ONLY when an episode
        actually terminates or truncates, and the environment is automatically reset
        for the subsequent episode.

        Returns:
            Dict[str, float]: Rollout statistics including step and episode returns.
        """
        if self._current_obs is None:
            self._current_obs, _ = self.env.reset()
            self._current_ep_reward = 0.0
            self._current_ep_len = 0

        total_step_reward = 0.0
        self._completed_episodes.clear()

        for _ in range(self.rollout_steps):
            # 1. Agent decision
            action, log_prob, state_value = self.agent.select_action(
                self._current_obs, deterministic=False
            )

            # 2. Environment transition
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            # 3. Store transition in memory buffer
            self.memory.store(
                state=self._current_obs,
                action=action,
                reward=reward,
                log_prob=log_prob,
                value=state_value,
                done=done,
                ground_truth=-1,
            )

            total_step_reward += reward
            self._current_ep_reward += reward
            self._current_ep_len += 1

            # 4. Handle episode completion
            if done:
                self._completed_episodes.append({
                    "return": float(self._current_ep_reward),
                    "length": int(self._current_ep_len),
                    "contained": bool(info.get("contained", False)),
                    "cleared": bool(info.get("cleared", False)),
                    "compromised": bool(info.get("compromised", False)),
                    "escalated": bool(info.get("escalated", False)),
                    "outcome": str(info.get("outcome", "unknown")),
                })

                # Reset environment for next episode
                self._current_obs, _ = self.env.reset()
                self._current_ep_reward = 0.0
                self._current_ep_len = 0
            else:
                self._current_obs = next_obs

        # Compute summary metrics for this rollout
        mean_step_reward = total_step_reward / self.rollout_steps
        num_episodes = len(self._completed_episodes)

        metrics: Dict[str, float] = {
            "rollout_mean_step_reward": float(mean_step_reward),
            "rollout_episodes_completed": float(num_episodes),
            "rollout_steps": float(self.rollout_steps),
        }

        if num_episodes > 0:
            returns = [ep["return"] for ep in self._completed_episodes]
            lengths = [ep["length"] for ep in self._completed_episodes]
            metrics["rollout_mean_ep_return"] = float(np.mean(returns))
            metrics["rollout_mean_ep_length"] = float(np.mean(lengths))
            metrics["rollout_containment_rate"] = float(
                sum(1 for ep in self._completed_episodes if ep["contained"]) / num_episodes
            )
            metrics["rollout_clearance_rate"] = float(
                sum(1 for ep in self._completed_episodes if ep["cleared"]) / num_episodes
            )
            metrics["rollout_breach_rate"] = float(
                sum(1 for ep in self._completed_episodes if ep["compromised"]) / num_episodes
            )
            metrics["rollout_escalation_rate"] = float(
                sum(1 for ep in self._completed_episodes if ep["escalated"]) / num_episodes
            )
        else:
            metrics["rollout_mean_ep_return"] = 0.0
            metrics["rollout_mean_ep_length"] = 0.0
            metrics["rollout_containment_rate"] = 0.0
            metrics["rollout_clearance_rate"] = 0.0
            metrics["rollout_breach_rate"] = 0.0
        action_counts: Dict[int, int] = {}
        for a in self.memory.actions:
            act_int = int(a.item() if isinstance(a, torch.Tensor) else a)
            action_counts[act_int] = action_counts.get(act_int, 0) + 1
        metrics["rollout_action_counts"] = action_counts

        return metrics

    def update(self) -> Dict[str, float]:
        """
        Trigger PPO policy and value network updates on collected rollout transitions.
        """
        if len(self.memory) == 0:
            return {}

        # GAE bootstrapping for non-terminal rollout boundaries:
        # If the last transition was non-terminal, bootstrap next_value with Critic value of current observation
        next_value = 0.0
        if len(self.memory.dones) > 0 and not self.memory.dones[-1]:
            if self._current_obs is not None:
                with torch.no_grad():
                    obs_t = self._current_obs.float().to(self.agent.device)
                    if obs_t.dim() == 1:
                        obs_t = obs_t.unsqueeze(0)
                    next_val_t = self.agent.critic(obs_t)
                    next_value = float(next_val_t.squeeze().item())

        return self.agent.update(
            self.memory,
            ref_actor=self.ref_actor,
            kl_beta=self.kl_beta,
            next_value=next_value,
        )

    def evaluate(
        self,
        num_episodes: Optional[int] = None,
        deterministic: bool = True,
        seed_start: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the policy over a fixed number of complete episodes.

        Args:
            num_episodes: Number of episodes to evaluate (defaults to self.eval_episodes).
            deterministic: If True, uses greedy argmax actions.
            seed_start: Optional starting seed for repeatable episode sequence.

        Returns:
            Dict[str, Any]: Comprehensive operational evaluation metrics.
        """
        target_env = self.eval_env if self.eval_env is not None else self.env
        n_eps = num_episodes if num_episodes is not None else self.eval_episodes

        ep_returns: List[float] = []
        ep_lengths: List[int] = []
        contained_count = 0
        cleared_count = 0
        compromised_count = 0
        escalated_count = 0
        outcomes: Dict[str, int] = {}

        mal_total = 0
        mal_contained = 0
        mal_breached = 0
        ben_total = 0
        ben_cleared = 0
        ben_disrupted = 0

        action_counts: Dict[int, int] = {}
        for ep_idx in range(n_eps):
            ep_seed = (seed_start + ep_idx) if seed_start is not None else None
            obs, info = target_env.reset(seed=ep_seed)
            is_benign_ep = bool(info.get("is_benign", False))

            ep_reward = 0.0
            ep_len = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _, _ = self.agent.select_action(obs, deterministic=deterministic)
                action_counts[int(action)] = action_counts.get(int(action), 0) + 1
                obs, reward, terminated, truncated, info = target_env.step(action)
                ep_reward += reward
                ep_len += 1

            ep_returns.append(ep_reward)
            ep_lengths.append(ep_len)

            if is_benign_ep:
                ben_total += 1
                if info.get("cleared", False):
                    ben_cleared += 1
                if info.get("outcome") == "disruptive_false_positive":
                    ben_disrupted += 1
            else:
                mal_total += 1
                if info.get("contained", False):
                    mal_contained += 1
                if info.get("compromised", False):
                    mal_breached += 1

            if info.get("contained", False):
                contained_count += 1
            if info.get("cleared", False):
                cleared_count += 1
            if info.get("compromised", False):
                compromised_count += 1
            if info.get("escalated", False):
                escalated_count += 1

            outc = str(info.get("outcome", "unknown"))
            outcomes[outc] = outcomes.get(outc, 0) + 1

        return {
            "eval_episodes": n_eps,
            "eval_mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "eval_std_return": float(np.std(ep_returns)) if ep_returns else 0.0,
            "eval_mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "eval_containment_rate": float(contained_count / n_eps) if n_eps > 0 else 0.0,
            "eval_clearance_rate": float(cleared_count / n_eps) if n_eps > 0 else 0.0,
            "eval_compromise_rate": float(compromised_count / n_eps) if n_eps > 0 else 0.0,
            "eval_escalation_rate": float(escalated_count / n_eps) if n_eps > 0 else 0.0,
            "eval_malicious": {
                "num_episodes": mal_total,
                "containment_rate": float(mal_contained / mal_total) if mal_total > 0 else 0.0,
                "breach_rate": float(mal_breached / mal_total) if mal_total > 0 else 0.0,
            },
            "eval_benign": {
                "num_episodes": ben_total,
                "clearance_rate": float(ben_cleared / ben_total) if ben_total > 0 else 0.0,
                "disruption_rate": float(ben_disrupted / ben_total) if ben_total > 0 else 0.0,
            },
            "eval_outcomes": outcomes,
            "eval_action_counts": action_counts,
        }

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save Actor, Critic, and optimizer states to disk.
        """
        checkpoint = {
            "actor_state_dict": self.agent.actor.state_dict(),
            "critic_state_dict": self.agent.critic.state_dict(),
            "actor_optimizer_state_dict": self.agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.agent.critic_optimizer.state_dict(),
            "best_eval_return": self.best_eval_return,
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

    def load_checkpoint(self, filepath: str, load_optimizers: bool = True) -> None:
        """
        Load Actor, Critic, and optionally optimizer states from a saved checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.agent.device)
        self.agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.agent.critic.load_state_dict(checkpoint["critic_state_dict"])
        if load_optimizers:
            if "actor_optimizer_state_dict" in checkpoint and self.agent.actor_optimizer is not None:
                self.agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in checkpoint and self.agent.critic_optimizer is not None:
                self.agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        val = checkpoint.get("best_eval_return", -np.inf)
        self.best_eval_return = float(val) if val is not None else -np.inf

    def train(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run the complete episodic PPO training cycle.
        """
        start_time = time.time()
        if verbose:
            print("=" * 70)
            print("Starting Sequential PPO Training Pipeline")
            print(f"  Total Iterations: {self.total_iterations}")
            print(f"  Rollout Steps / Iteration: {self.rollout_steps}")
            print(f"  Mini-Batch Size: {self.agent.batch_size}")
            print(f"  Epochs / Update: {self.agent.k_epochs}")
            print(f"  Device: {self.agent.device}")
            print("=" * 70)

        for iteration in range(1, self.total_iterations + 1):
            # 1. Rollout collection
            rollout_metrics = self.collect_rollout()

            # 2. PPO Update
            update_metrics = self.update()

            # 3. Aggregate iteration record
            iter_record: Dict[str, Any] = {
                "iteration": iteration,
                **rollout_metrics,
                **update_metrics,
            }

            # 4. Periodic evaluation
            if iteration % self.eval_interval == 0:
                eval_metrics = self.evaluate(seed_start=self.eval_seed_start)
                iter_record.update(eval_metrics)

                mean_ret = eval_metrics["eval_mean_return"]
                if mean_ret > self.best_eval_return:
                    self.best_eval_return = mean_ret
                    self.early_stopping_counter = 0
                    if self.save_dir:
                        ckpt_path = os.path.join(self.save_dir, "best_sequential_policy.pt")
                        self.save_checkpoint(ckpt_path)
                else:
                    self.early_stopping_counter += 1

            self.history.append(iter_record)

            if verbose and (iteration % self.log_interval == 0 or iteration == 1):
                elapsed = time.time() - start_time
                print(
                    f"[Iter {iteration:03d}/{self.total_iterations:03d}] "
                    f"StepR: {rollout_metrics['rollout_mean_step_reward']:+.3f} | "
                    f"EpRet: {rollout_metrics['rollout_mean_ep_return']:+.3f} | "
                    f"Contain: {rollout_metrics['rollout_containment_rate']:.1%} | "
                    f"Clear: {rollout_metrics['rollout_clearance_rate']:.1%} | "
                    f"Breach: {rollout_metrics['rollout_breach_rate']:.1%} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # Early stopping check
            if self.early_stopping_patience > 0 and self.early_stopping_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at iteration {iteration} (patience={self.early_stopping_patience}).")
                break

        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# Baseline Evaluation Interfaces
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(
    env: SequentialSOCEnvironment,
    policy_fn: Callable[[torch.Tensor, Dict[str, Any]], int],
    num_episodes: int = 20,
    seed_start: Optional[int] = 1000,
) -> Dict[str, Any]:
    """
    Evaluate an arbitrary baseline policy on SequentialSOCEnvironment.

    Args:
        env: The SequentialSOCEnvironment instance.
        policy_fn: Function mapping (observation, info) -> discrete action int (0-19).
        num_episodes: Number of episodes to evaluate.
        seed_start: Starting integer seed for reproducible evaluation.

    Returns:
        Dict[str, Any]: Metrics comparing performance against baseline expectations.
    """
    returns: List[float] = []
    lengths: List[int] = []
    contained = 0
    cleared = 0
    compromised = 0
    escalated = 0
    outcomes: Dict[str, int] = {}

    for ep_idx in range(num_episodes):
        seed = (seed_start + ep_idx) if seed_start is not None else None
        obs, info = env.reset(seed=seed)
        ep_ret = 0.0
        ep_len = 0
        term = False
        trunc = False

        while not (term or trunc):
            action = int(policy_fn(obs, info))
            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            ep_len += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        if info.get("contained", False):
            contained += 1
        if info.get("cleared", False):
            cleared += 1
        if info.get("compromised", False):
            compromised += 1
        if info.get("escalated", False):
            escalated += 1

        outc = str(info.get("outcome", "unknown"))
        outcomes[outc] = outcomes.get(outc, 0) + 1

    return {
        "num_episodes": num_episodes,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "containment_rate": float(contained / num_episodes) if num_episodes > 0 else 0.0,
        "clearance_rate": float(cleared / num_episodes) if num_episodes > 0 else 0.0,
        "compromise_rate": float(compromised / num_episodes) if num_episodes > 0 else 0.0,
        "escalation_rate": float(escalated / num_episodes) if num_episodes > 0 else 0.0,
        "outcomes": outcomes,
    }


def make_random_policy(seed: Optional[int] = None) -> Callable[[torch.Tensor, Dict[str, Any]], int]:
    """Uniform random baseline policy over Discrete(20)."""
    rng = np.random.RandomState(seed)
    return lambda obs, info: int(rng.randint(0, 20))


def make_fixed_action_policy(action_idx: int) -> Callable[[torch.Tensor, Dict[str, Any]], int]:
    """Fixed-action baseline policy always selecting a constant action index."""
    if not (0 <= action_idx < 20):
        raise ValueError(f"Action index {action_idx} out of range [0, 19].")
    return lambda obs, info: action_idx


def rule_based_policy(obs: torch.Tensor, info: Dict[str, Any]) -> int:
    """
    Realistic SOC rule-based heuristic baseline operating strictly on observable telemetry.

    Guarantees:
      - Does NOT inspect hidden ground-truth simulation fields (is_benign, scenario_id, stage_id, outcome, etc.).
      - Inspects only observable SIEM alerts and endpoint process properties in info["telemetry"]
        (alert level, rule description, process name, command line, MITRE tactics/techniques, cwd, username).

    Rules:
      1. Routine maintenance / low-severity benign backup (alert level <= 3, rsync, backup service account):
         -> Action 0 (monitor)
      2. Outbound C2 communication or active data exfiltration (tactics, C2 destination, curl POST exfil):
         -> Action 4 (block_dest_ip)
      3. Privilege escalation attempt (sudo root exploitation, privilege escalation tactics):
         -> Action 12 (disable_account)
      4. Persistence installation (crontab modification, scheduled persistence):
         -> Action 13 (remove_persistence)
      5. Execution of suspicious process / reconnaissance in temporary directory (/tmp staging, whoami):
         -> Action 8 (kill_process)
      6. Fallback default:
         -> Action 8 (kill_process)
    """
    telemetry = info.get("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}

    alert_level = telemetry.get("alert_level")
    rule_desc = str(telemetry.get("rule_description") or "").lower()
    cmdline = str(telemetry.get("cmdline") or "").lower()
    proc_name = str(telemetry.get("process_name") or "").lower()
    proc_path = str(telemetry.get("process_path") or "").lower()
    cwd = str(telemetry.get("cwd") or "").lower()
    tactics = [str(t).lower() for t in telemetry.get("mitre_tactics") or []]
    techniques = [str(t).lower() for t in telemetry.get("mitre_techniques") or []]
    username = str(telemetry.get("username") or "").lower()

    # Rule 1: Routine administrative maintenance / low-severity benign backup
    if (
        (alert_level is not None and alert_level <= 3)
        or "backup" in rule_desc
        or "rsync" in proc_name
        or "rsync" in cmdline
        or "backup_admin" in username
    ):
        return 0  # monitor

    # Rule 2: Outbound C2 communication or active data exfiltration
    if (
        "command and control" in tactics
        or "exfiltration" in tactics
        or "t1071.001" in techniques
        or "t1048.002" in techniques
        or "outbound network connection" in rule_desc
        or "data transfer" in rule_desc
        or "--c2" in cmdline
        or "curl" in proc_name
        or "curl" in cmdline
    ):
        return 4  # block_dest_ip

    # Rule 3: Unauthorized privilege escalation / sudo abuse
    if (
        "privilege escalation" in tactics
        or "t1548.003" in techniques
        or "sudo" in rule_desc
        or "sudo" in proc_name
        or "sudo" in cmdline
    ):
        return 12  # disable_account

    # Rule 4: Scheduled task persistence / crontab modification
    if (
        "persistence" in tactics
        or "t1053.003" in techniques
        or "cron" in rule_desc
        or "crontab" in proc_name
        or "crontab" in cmdline
    ):
        return 13  # remove_persistence

    # Rule 5: Execution of suspicious reconnaissance / temporary staging process
    if (
        "whoami" in proc_name
        or "whoami" in cmdline
        or "/tmp" in cwd
        or "/tmp" in proc_path
    ):
        return 8  # kill_process

    # Fallback: kill_process
    return 8
