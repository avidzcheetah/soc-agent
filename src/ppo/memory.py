# Responsibility: Trajectory Rollout Buffer (PPOMemory) for PPO Reinforcement Learning
#
# This module records transitions during environment rollouts (states, actions,
# rewards, log_probs, values, dones) to be processed for Advantage estimation (GAE)
# and policy gradient optimization epochs.

from typing import List, Any, Union
import torch
import numpy as np


class PPOMemory:
    """
    Trajectory Memory Buffer for storing rollout transitions in PPO.

    Stores the 6 essential components of an RL interaction:
        1. states:    The 768-dim environment observation at step t.
        2. actions:   The discrete action integer selected by the actor.
        3. rewards:   The scalar environment reward received at step t.
        4. log_probs: The log probability log π(a_t | s_t) under the behavior policy.
        5. values:    The Critic state value estimate V(s_t) at step t.
        6. dones:     Boolean flag indicating if the transition terminated the episode.
    """

    def __init__(self):
        self.states: List[Any] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def store(
        self,
        state: Union[torch.Tensor, np.ndarray],
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
    ) -> None:
        """
        Append one interaction step to the memory buffer.

        Args:
            state: The state embedding (Tensor or ndarray).
            action: Selected action integer.
            reward: Received reward float.
            log_prob: Log probability of the selected action.
            value: Critic value estimate of the state.
            done: Termination flag boolean.
        """
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self) -> None:
        """
        Clear all stored trajectory lists after policy update is completed.
        """
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        """Return the current number of stored interactions in the buffer."""
        return len(self.states)
