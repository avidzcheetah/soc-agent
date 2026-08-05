# Responsibility: Neural Network Architectures for Actor (Policy) and Critic (Value) in PPO
#
# ActorNetwork: Maps 768-dim SecBERT state embedding -> 20 raw action logits.
# CriticNetwork: (Will be implemented in Step 3) Maps 768-dim state embedding -> scalar state value V(s).

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Actor (Policy) Network for Discrete SOC Incident Response Actions.

    Maps a 768-dimensional contextual state embedding (from frozen SecBERT)
    to unnormalized action logits across the 20 discrete response actions.

    Architecture:
        Input:  [Batch_Size, 768] (or unbatched [768])
        Layer 1: Linear(768 -> 512) + ReLU
        Layer 2: Linear(512 -> 256) + ReLU
        Output:  Linear(256 -> 20) -> Raw Action Logits [Batch_Size, 20]
    """

    def __init__(self, state_dim: int = 768, action_dim: int = 20):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw action logits.

        Args:
            state: Tensor of shape [768] or [Batch_Size, 768].

        Returns:
            torch.Tensor: Unnormalized logits of shape [20] or [Batch_Size, 20].
                          Softmax is NOT applied here; it will be handled by
                          torch.distributions.Categorical during action sampling.
        """
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Critic (Value) Network for Estimating Expected State Value V(s).

    Maps a 768-dimensional contextual state embedding (from frozen SecBERT)
    to a scalar state value V(s) estimating expected future rewards.

    Architecture:
        Input:  [Batch_Size, 768] (or unbatched [768])
        Layer 1: Linear(768 -> 512) + ReLU
        Layer 2: Linear(512 -> 256) + ReLU
        Output:  Linear(256 -> 1) -> Scalar State Value [Batch_Size, 1] (or [1])
    """

    def __init__(self, state_dim: int = 768):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim

        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing scalar state value V(s).

        Args:
            state: Tensor of shape [768] or [Batch_Size, 768].

        Returns:
            torch.Tensor: State value of shape [1] (for single state) or [Batch_Size, 1].
        """
        return self.network(state)

