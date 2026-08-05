# Responsibility: Proximal Policy Optimization (PPO) Agent Skeleton
#
# This module defines the PPOAgent class, managing the Actor (Policy) network,
# Critic (Value) network, trajectory buffers, and PPO clipping optimization loop.

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Optional, Dict, Any, Tuple, Union
from src.ppo.networks import ActorNetwork, CriticNetwork


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

