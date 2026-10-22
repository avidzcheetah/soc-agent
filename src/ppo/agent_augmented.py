# Responsibility: Policy Architecture for Telemetry-Augmented Experiment 003
#
# Provides an isolated PPOAgent variant designed specifically for 802-dimensional
# observations (768 SecBERT text embedding + 34 structured telemetry features).
#
# Safety & Integrity:
#   - Does NOT modify the original PPOAgent or original checkpoint files.
#   - Strictly forbids loading an existing 768-input actor/critic directly without adaptation.
#   - Supports fresh initialization or controlled warmstarting from 768-input checkpoints.

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import logging

from src.ppo.agent import PPOAgent
from src.environment_augmented import COMBINED_OBSERVATION_DIM

logger = logging.getLogger(__name__)


class AugmentedPPOAgent(PPOAgent):
    """
    PPO Agent architecture configured for combined SecBERT + telemetry states (dim 802).

    Guarantees:
      - Actor network input dimension is 802.
      - Critic network input dimension is 802.
      - Existing 768-input checkpoints cannot be loaded unchanged.
    """

    def __init__(
        self,
        state_dim: int = COMBINED_OBSERVATION_DIM,  # 802
        action_dim: int = 20,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        c1_value_loss: float = 0.5,
        c2_entropy: float = 0.01,
        max_grad_norm: float = 1.0,
        k_epochs: int = 4,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        if state_dim != COMBINED_OBSERVATION_DIM:
            logger.warning(
                f"AugmentedPPOAgent initialized with state_dim={state_dim}, "
                f"expected COMBINED_OBSERVATION_DIM={COMBINED_OBSERVATION_DIM}."
            )
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            c1_value_loss=c1_value_loss,
            c2_entropy=c2_entropy,
            max_grad_norm=max_grad_norm,
            k_epochs=k_epochs,
            batch_size=batch_size,
            device=device,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint saved specifically for the 802-dimensional architecture.

        Raises:
            ValueError: If an existing 768-input checkpoint is attempted to be loaded unchanged.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        actor_state = checkpoint.get("actor_state_dict", {})
        first_layer_weight = actor_state.get("network.0.weight")

        if first_layer_weight is not None and first_layer_weight.shape[1] == 768:
            raise ValueError(
                "Cannot load existing 768-input checkpoint unchanged into 802-input architecture. "
                "Use warmstart_from_768_checkpoint() if transferring weights."
            )

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        if "actor_optimizer_state_dict" in checkpoint and self.actor_optimizer:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in checkpoint and self.critic_optimizer:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

    def warmstart_from_768_checkpoint(
        self,
        checkpoint_path: str,
        init_telemetry_weight: str = "zero",
    ) -> None:
        """
        Transfer weights from a 768-input checkpoint to the 802-input architecture.

        Preserves all hidden layers and copies weight[:, :768] for the first layer.
        The new 34 telemetry input weights weight[:, 768:] are initialized:
          - 'zero': Zero-initialized so policy initially behaves identically to the 768 model.
          - 'normal': Small Gaussian initialization (std=0.01).

        Args:
            checkpoint_path: Path to existing 768-input checkpoint.
            init_telemetry_weight: Initialization strategy for new 34 features ('zero' or 'normal').
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Source 768 checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        src_actor = ckpt["actor_state_dict"]
        src_critic = ckpt["critic_state_dict"]

        with torch.no_grad():
            # 1. Warmstart Actor
            curr_actor_dict = self.actor.state_dict()
            for key, val in src_actor.items():
                if key == "network.0.weight":
                    curr_actor_dict[key][:, :768] = val
                    if init_telemetry_weight == "zero":
                        curr_actor_dict[key][:, 768:] = 0.0
                    else:
                        curr_actor_dict[key][:, 768:] = torch.randn_like(curr_actor_dict[key][:, 768:]) * 0.01
                else:
                    curr_actor_dict[key] = val
            self.actor.load_state_dict(curr_actor_dict)

            # 2. Warmstart Critic
            curr_critic_dict = self.critic.state_dict()
            for key, val in src_critic.items():
                if key == "network.0.weight":
                    curr_critic_dict[key][:, :768] = val
                    if init_telemetry_weight == "zero":
                        curr_critic_dict[key][:, 768:] = 0.0
                    else:
                        curr_critic_dict[key][:, 768:] = torch.randn_like(curr_critic_dict[key][:, 768:]) * 0.01
                else:
                    curr_critic_dict[key] = val
            self.critic.load_state_dict(curr_critic_dict)

        logger.info(
            f"Successfully warm-started 802-input actor and critic from 768 checkpoint: {checkpoint_path}"
        )
