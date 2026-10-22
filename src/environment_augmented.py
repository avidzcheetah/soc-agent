# Responsibility: Telemetry-Augmented Sequential SOC Environment for Experiment 003
#
# Provides an isolated environment subclassing SequentialSOCEnvironment that
# combines the frozen SecBERT 768-dimensional language embedding with a structured,
# deterministic 34-dimensional telemetry feature vector.
#
# Total observation dimension: 768 + 34 = 802.
#
# Safety & Integrity:
#   - Does NOT modify SequentialSOCEnvironment or existing 768-dim models.
#   - Does NOT use is_benign, scenario_id, stage_id, hidden simulator state,
#     or ground-truth labels in observations.
#   - Preserves identical reward functions, scenarios, and step dynamics.

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from src.environment_sequential import (
    SequentialSOCEnvironment,
    IncidentScenario,
    RewardConfig,
    SecBERTStateEncoder,
    get_default_scenarios,
)
from src.telemetry.incident_formatter import IncidentRecord
from src.telemetry_feature_extractor import (
    TelemetryFeatureExtractor,
    TELEMETRY_FEATURE_DIM,
    TELEMETRY_FEATURE_SCHEMA,
)

COMBINED_OBSERVATION_DIM: int = 768 + TELEMETRY_FEATURE_DIM  # 802


class TelemetryAugmentedSOCEnvironment(SequentialSOCEnvironment):
    """
    Experiment 003 Environment: Augments SecBERT contextual text embeddings
    with structured observable telemetry features.

    Observation Space: Box(802,)
      - Dimensions 0..767: Frozen SecBERT CLS pooler text embedding (768)
      - Dimensions 768..801: Structured observable telemetry feature vector (34)
    """

    def __init__(
        self,
        scenarios: Optional[List[IncidentScenario]] = None,
        encoder: Optional[SecBERTStateEncoder] = None,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 10,
        action_space_path: str = "data/action_space.csv",
    ):
        super().__init__(
            scenarios=scenarios,
            encoder=encoder,
            reward_config=reward_config,
            max_steps=max_steps,
            action_space_path=action_space_path,
        )

        # 1. Update observation space to 802 dimensions
        self.telemetry_extractor = TelemetryFeatureExtractor()
        self.telemetry_dim = self.telemetry_extractor.feature_dim
        self.total_obs_dim = 768 + self.telemetry_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.total_obs_dim,), dtype=np.float32
        )

    def _create_state(self, record: IncidentRecord) -> torch.Tensor:
        """
        Produce combined 802-dimensional state representation:
        [768-dim SecBERT embedding, 34-dim observable telemetry features].

        Derived strictly from observable IncidentRecord fields.
        """
        # 1. Base SecBERT embedding (dim 768)
        secbert_emb = super()._create_state(record)
        if not isinstance(secbert_emb, torch.Tensor):
            secbert_emb = torch.from_numpy(secbert_emb).float()
        else:
            secbert_emb = secbert_emb.float()

        # Ensure 1D [768] shape
        if secbert_emb.dim() > 1:
            secbert_emb = secbert_emb.squeeze(0)

        # 2. Structured observable telemetry feature vector (dim 34)
        telemetry_vec = self.telemetry_extractor.extract(record)
        telemetry_tensor = torch.from_numpy(telemetry_vec).float()

        # 3. Concatenate along last dimension -> [802]
        combined = torch.cat([secbert_emb, telemetry_tensor], dim=-1)
        return combined

    def get_feature_schema(self) -> List[str]:
        """Return the documentation schema for all observation dimensions."""
        secbert_dims = [f"secbert_dim_{i}" for i in range(768)]
        telemetry_dims = list(TELEMETRY_FEATURE_SCHEMA)
        return secbert_dims + telemetry_dims
