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
        c1_value_loss: float = 0.5,
        c2_entropy: float = 0.01,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """
        Step 1 of PPO Agent: Initialize attributes and hyperparameters owned by the agent.
        No neural networks, learning logic, or buffers are initialized yet.
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

        # 4. Optimizers (Placeholders for upcoming steps)
        self.actor_optimizer = None
        self.critic_optimizer = None

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

    def select_action(
        self,
        state: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[int, float, float]:
        """
        Select an action given a state using stochastic sampling from the Actor policy,
        and evaluate the expected state value using the Critic.

        Steps:
            1. Move state to the agent's computing device.
            2. Pass state through the Actor network to obtain logits.
            3. Create a Categorical distribution from the logits.
            4. Sample one action from the distribution (stochastic exploration).
            5. Compute log probability log π(a|s) of the chosen action.
            6. Pass state through the Critic network to obtain expected state value V(s).
            7. Return (action, log_prob, state_value).

        Args:
            state: 768-dim state embedding (torch.Tensor or np.ndarray).

        Returns:
            Tuple[int, float, float]:
                - action (int): Sampled discrete action ID in [0, action_dim - 1].
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

            # 4. Sample action
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

    def update(self, memory: PPOMemory) -> Dict[str, float]:
        """
        Execute one PPO optimization cycle over the collected trajectory in memory.

        Step 7.1 (Preparation Stage):
            1. Compute advantages and returns via GAE.
            2. Convert memory lists (states, actions, log_probs) to PyTorch tensors on device.
            3. Normalize advantages for numerical stability across batches.
            4. Execute k_epochs optimization loop (loss computation to be implemented in 7.2-7.5).
            5. Clear memory after update.

        Args:
            memory: PPOMemory containing collected trajectory rollouts.

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

        # 4. K-Epochs training loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for epoch in range(self.k_epochs):
            # Step 7.2: Clipped Policy (Actor) Loss

            # A. Forward pass through the current Actor to obtain new logits
            new_logits = self.actor(states_tensor)

            # B. Create categorical distribution from new logits
            dist = Categorical(logits=new_logits)

            # Step 7.4: Policy Entropy (Exploration Bonus)
            dist_entropy = dist.entropy().mean()
            total_entropy += dist_entropy.item()

            # C. Compute new log probabilities for the SAME actions taken during rollout
            new_log_probs = dist.log_prob(actions_tensor)

            # D. Compute probability ratio r_t = π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            # E. Two surrogate objectives
            surr1 = ratio * advantages                                           # Unclipped
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages  # Clipped

            # F. Take the minimum (pessimistic bound)
            # G. Negate because PyTorch minimizes, but we want to maximize expected reward
            policy_loss = -torch.min(surr1, surr2).mean()

            total_policy_loss += policy_loss.item()

            # Step 7.3: Critic (Value) Loss

            # A. Forward pass through the Critic to obtain predicted state values
            state_values = self.critic(states_tensor)

            # B. Reshape [num_samples, 1] -> [num_samples] to match returns shape
            state_values = state_values.squeeze(-1)

            # C. Mean Squared Error (MSE) between predictions V(s) and target returns R(t)
            value_loss = F.mse_loss(state_values, returns)

            total_value_loss += value_loss.item()

            # Placeholders for Step 7.5 (Combined Loss & Optimizers)

        # 5. Clear trajectory memory buffer after update
        memory.clear()

        return {
            "num_samples": float(states_tensor.shape[0]),
            "mean_advantage": float(advantages.mean().item()),
            "std_advantage": float(advantages.std().item()),
            "policy_loss": total_policy_loss / self.k_epochs,
            "value_loss": total_value_loss / self.k_epochs,
            "entropy": total_entropy / self.k_epochs,
        }



