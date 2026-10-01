# Responsibility: Proximal Policy Optimization (PPO) Agent Skeleton
#
# This module defines the PPOAgent class, managing the Actor (Policy) network,
# Critic (Value) network, trajectory buffers, and PPO clipping optimization loop.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Dict, Any, Tuple, Union, List
from src.ppo.networks import ActorNetwork, CriticNetwork
from src.ppo.memory import PPOMemory


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for Autonomous SOC Incident Response.

    Owns:
        1. state_dim (int): Dimensionality of input state embeddings (768 from SecBERT).
        2. action_dim (int): Number of discrete mitigation actions (20 actions).
        3. device (torch.device): Computing device ('cuda' or 'cpu').
        4. actor (Optional[nn.Module]): Policy network mapping state (768) -> action logits (20).
        5. critic (Optional[nn.Module]): Value network mapping state (768) -> state value V(s).
        6. actor_optimizer (Optional[torch.optim.Optimizer]): Optimizer for policy network.
        7. critic_optimizer (Optional[torch.optim.Optimizer]): Optimizer for value network.
        8. Hyperparameters:
           - lr_actor: Learning rate for Actor.
           - lr_critic: Learning rate for Critic.
           - gamma: Discount factor for future rewards.
           - gae_lambda: Generalized Advantage Estimation (GAE) smoothing parameter.
           - clip_eps: PPO policy ratio clipping threshold (epsilon).
           - c1_value_loss: Value function loss weight.
           - c2_entropy: Entropy bonus weight for exploration.
           - k_epochs: Number of optimization passes over collected rollouts.
           - batch_size: Minibatch size for gradient updates.
    """

    def __init__(
        self,
        state_dim: int = 768,
        action_dim: int = 20,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        c1_value_loss: float = 0.5,  # Unused with separate optimizers; kept for API compatibility
        c2_entropy: float = 0.01,
        max_grad_norm: float = 1.0,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """
        Step 1 of PPO Agent: Initialize attributes and hyperparameters owned by the agent.
        """
        # 1. State and Action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 2. Device management
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 3. Neural Networks
        self.actor = ActorNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim=self.state_dim).to(self.device)

        # 4. Optimizers & Gradient Clipping (Step 7.5)
        self.max_grad_norm = max_grad_norm
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 5. PPO Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c1_value_loss = c1_value_loss
        self.c2_entropy = c2_entropy
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.bc_lambda = 0.0  # set via agent.bc_lambda after init
        self.atgp_lambda = 0.0  # Adaptive Teacher-Guided Preservation coefficient
        self.atgp_tau = 0.3     # Teacher confidence threshold for ATGP gate

    def select_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """
        Select an action given a state observation.

        Two modes of operation:
            - Stochastic (deterministic=False): Samples from the Categorical policy
              distribution for exploration during training rollouts.
            - Deterministic (deterministic=True): Selects the highest-probability action
              via argmax for consistent evaluation of the learned policy.

        Steps:
            1. Move state to the agent's computing device.
            2. Pass state through the Actor network to obtain logits.
            3. Create a Categorical distribution from the logits.
            4. Select action: sample (stochastic) or argmax (deterministic).
            5. Compute log probability log π(a|s) of the chosen action.
            6. Pass state through the Critic network to obtain expected state value V(s).
            7. Return (action, log_prob, state_value).

        Args:
            state: 768-dim state embedding (torch.Tensor or np.ndarray).
            deterministic: If True, use greedy argmax action selection instead of
                stochastic sampling. Use True for evaluation, False for training.

        Returns:
            Tuple[int, float, float]:
                - action (int): Discrete action ID in [0, action_dim - 1].
                - log_prob (float): Log probability of the chosen action.
                - state_value (float): Critic estimated state value V(s).
        """
        # 1. Convert to tensor and transfer to agent device
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            raise TypeError(f"Expected state to be torch.Tensor or np.ndarray, got {type(state)}")

        # Ensure correct 1D tensor shape [768]
        if state_tensor.dim() == 2 and state_tensor.shape[0] == 1:
            state_tensor = state_tensor.squeeze(0)

        # 2. Forward passes without gradient tracking during environment interaction
        with torch.no_grad():
            logits = self.actor(state_tensor)
            state_value = self.critic(state_tensor)

            # 3. Categorical distribution
            dist = Categorical(logits=logits)

            # 4. Select action: greedy argmax for evaluation, stochastic for training
            if deterministic:
                action = torch.argmax(logits)
            else:
                action = dist.sample()

            # 5. Compute log probability
            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item()), float(state_value.squeeze().item())

    def compute_gae(
        self,
        memory: PPOMemory,
        next_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) and discounted returns
        by iterating backwards through the collected rollout trajectory in memory.

        Formulation:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)

        Args:
            memory: PPOMemory buffer containing stored trajectory lists.
            next_value: Value of state following the last step (0.0 for terminal/cutoff).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - advantages: 1D Tensor of shape [T] representing GAE advantages.
                - returns: 1D Tensor of shape [T] representing target returns R(t).
        """
        rewards = memory.rewards
        values = memory.values
        dones = memory.dones
        t_len = len(rewards)

        if t_len == 0:
            return torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        advantages: List[float] = []
        returns: List[float] = []
        gae = 0.0

        for t in reversed(range(t_len)):
            # Determine next state value and non-terminal mask
            if t == t_len - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - float(dones[t])

            # 1. TD error delta_t
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]

            # 2. GAE advantage A_t
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.append(gae)

            # 3. Return R_t = A_t + V(s_t)
            ret = gae + values[t]
            returns.append(ret)

        # Reverse backwards lists to match original chronological trajectory order
        advantages.reverse()
        returns.reverse()

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        return advantages_tensor, returns_tensor

    def update(self, memory: PPOMemory, ref_actor=None, kl_beta: float = 0.0) -> Dict[str, float]:
        """
        Execute one PPO optimization cycle over the collected trajectory in memory.

        Args:
            memory: PPOMemory containing collected trajectory rollouts.
            ref_actor: Optional reference Actor (frozen warm-start). When provided, a
                       KL(ref || current) penalty is added to the actor loss to prevent
                       catastrophic forgetting of the supervised warm-start knowledge.
            kl_beta: Coefficient for the KL penalty term. 0.0 disables it.

        Returns:
            Dict[str, float]: Training metrics/losses dictionary.
        """
        if len(memory) == 0:
            return {}

        # 1. Compute GAE advantages and target returns
        advantages, returns = self.compute_gae(memory)

        # 2. Convert trajectory lists into tensors on target device
        states_list = [
            torch.as_tensor(s, dtype=torch.float32) if isinstance(s, np.ndarray) else s.float()
            for s in memory.states
        ]
        states_tensor = torch.stack(states_list).to(self.device)
        actions_tensor = torch.tensor(memory.actions, dtype=torch.int64, device=self.device)
        old_log_probs_tensor = torch.tensor(memory.log_probs, dtype=torch.float32, device=self.device)

        # 3. Normalize advantages across the entire rollout trajectory
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. K-Epochs training loop with Shuffled Mini-Batches (Step 8)
        num_samples = states_tensor.shape[0]
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl_penalty = 0.0
        total_bc_loss = 0.0
        total_atgp_loss = 0.0
        num_updates = 0

        # Ground truth labels for BC loss (stored alongside states in memory)
        has_gt = len(memory.ground_truths) == len(memory.states) and any(g >= 0 for g in memory.ground_truths)
        if has_gt:
            gt_tensor = torch.tensor(memory.ground_truths, dtype=torch.int64, device=self.device)

        for epoch in range(self.k_epochs):
            # Generate randomized index permutation for each epoch
            permutation = torch.randperm(num_samples, device=self.device)

            # Iterate through mini-batches of size self.batch_size
            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = permutation[start_idx : start_idx + self.batch_size]

                # Mini-batch slices
                b_states = states_tensor[batch_indices]
                b_actions = actions_tensor[batch_indices]
                b_old_log_probs = old_log_probs_tensor[batch_indices]
                b_advantages = advantages[batch_indices]
                b_returns = returns[batch_indices]

                # Step 7.2: Clipped Policy (Actor) Loss
                # A. Forward pass through the current Actor to obtain new logits
                new_logits = self.actor(b_states)

                # B. Create categorical distribution from new logits
                dist = Categorical(logits=new_logits)

                # Step 7.4: Policy Entropy (Exploration Bonus)
                dist_entropy = dist.entropy().mean()

                # C. Compute new log probabilities for the SAME actions taken during rollout
                new_log_probs = dist.log_prob(b_actions)

                # D. Compute probability ratio r_t = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_log_probs - b_old_log_probs)

                # E. Two surrogate objectives
                surr1 = ratio * b_advantages                                           # Unclipped
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages  # Clipped

                # F. Take the minimum (pessimistic bound)
                # G. Negate because PyTorch minimizes, but we want to maximize expected reward
                policy_loss = -torch.min(surr1, surr2).mean()

                # Step 7.5: Actor Optimization (PPO loss - entropy bonus + optional KL + optional BC)
                actor_loss = policy_loss - self.c2_entropy * dist_entropy

                # KL constraint against reference warm-start policy
                if ref_actor is not None and kl_beta > 0.0:
                    with torch.no_grad():
                        ref_logits = ref_actor(b_states)
                    ref_probs = F.softmax(ref_logits, dim=-1)
                    current_log_probs_all = F.log_softmax(new_logits, dim=-1)
                    kl_penalty = F.kl_div(
                        current_log_probs_all, ref_probs,
                        reduction="batchmean", log_target=False
                    )
                    actor_loss = actor_loss + kl_beta * kl_penalty
                    total_kl_penalty += kl_penalty.item()

                # BC (Behavioral Cloning) preservation loss:
                # Cross-entropy between actor logits and ground-truth labels.
                # Keeps the policy from drifting away from the supervised solution.
                if has_gt and self.bc_lambda > 0.0:
                    b_gt = gt_tensor[batch_indices]
                    valid_mask = b_gt >= 0
                    if valid_mask.any():
                        bc_loss = F.cross_entropy(new_logits[valid_mask], b_gt[valid_mask])
                        actor_loss = actor_loss + self.bc_lambda * bc_loss
                        total_bc_loss += bc_loss.item()

                # ATGP (Adaptive Teacher-Guided Preservation) loss:
                # Per-sample, confidence-gated KL divergence against the frozen
                # teacher policy. Selectively constrains the actor only where the
                # teacher is confident AND the actor has diverged, preventing
                # policy collapse on minority classes without globally freezing
                # the policy.
                if ref_actor is not None and self.atgp_lambda > 0.0:
                    with torch.no_grad():
                        ref_logits_atgp = ref_actor(b_states)
                        ref_probs_atgp = F.softmax(ref_logits_atgp, dim=-1)
                        # Teacher confidence: max probability per sample [batch_size]
                        teacher_confidence, _ = ref_probs_atgp.max(dim=-1)
                        # Soft linear gate: 0 below tau, ramps to 1 at confidence=1
                        gate = ((teacher_confidence - self.atgp_tau) / (1.0 - self.atgp_tau)).clamp(0, 1)

                    current_log_probs_atgp = F.log_softmax(new_logits, dim=-1)  # [batch_size, 20]
                    # Per-sample forward KL: sum over actions → [batch_size]
                    per_sample_kl = (ref_probs_atgp * (ref_probs_atgp.log() - current_log_probs_atgp)).sum(dim=-1)
                    # Gated, batch-averaged ATGP loss
                    atgp_loss = (gate * per_sample_kl).mean()
                    actor_loss = actor_loss + self.atgp_lambda * atgp_loss
                    total_atgp_loss += atgp_loss.item()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Step 7.3: Critic (Value) Loss
                # A. Forward pass through the Critic to obtain predicted state values
                state_values = self.critic(b_states)

                # B. Reshape [batch_size, 1] -> [batch_size] to match returns shape
                state_values = state_values.squeeze(-1)

                # C. Mean Squared Error (MSE) between predictions V(s) and target returns R(t)
                value_loss = F.mse_loss(state_values, b_returns.detach())

                # Step 7.5: Critic Optimization
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Accumulate step metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += dist_entropy.item()
                num_updates += 1

        # 5. Clear trajectory memory buffer after update
        memory.clear()

        return {
            "num_samples": float(num_samples),
            "mean_advantage": float(advantages.mean().item()),
            "std_advantage": float(advantages.std().item()),
            "policy_loss": total_policy_loss / num_updates if num_updates > 0 else 0.0,
            "value_loss": total_value_loss / num_updates if num_updates > 0 else 0.0,
            "entropy": total_entropy / num_updates if num_updates > 0 else 0.0,
            "kl_penalty": total_kl_penalty / num_updates if num_updates > 0 else 0.0,
            "bc_loss": total_bc_loss / num_updates if num_updates > 0 else 0.0,
            "atgp_loss": total_atgp_loss / num_updates if num_updates > 0 else 0.0,
        }



