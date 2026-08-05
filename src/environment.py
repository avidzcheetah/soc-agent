# Responsibility: SOC Environment for PPO Agent (Contextual Bandit)
#
# This environment models cybersecurity incident response as a contextual bandit:
# It presents an incident's 768-dim SecBERT embedding to the agent, receives an
# action (0-19), evaluates the decision, and assigns a reward.

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, Union
import torch
from src.encoder import SecBERTStateEncoder


class SOCEnvironment:
    """
    Custom SOC Environment for training and evaluating the PPO agent.

    Owns:
        1. df (pd.DataFrame): The incident dataset (e.g., train.csv or test.csv).
        2. encoder (SecBERTStateEncoder): Frozen SecBERT encoder to produce 768-dim embeddings.
        3. current_sample (Optional[pd.Series]): The full row of the incident currently being handled.
        4. current_embedding (Optional[torch.Tensor]): The 768-dim vector representation of the current incident.
        5. current_index (int): Pointer to the next incident index (used in 'eval' mode).
        6. mode (str): 'train' for random incident sampling, 'eval' for sequential replay.
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str],
        encoder: Optional[SecBERTStateEncoder] = None,
        mode: str = "train",
    ):
        """
        Step 1: Initialize the 6 core attributes owned by the environment.

        Args:
            df: A pandas DataFrame containing incidents, or a path to the CSV file.
            encoder: An instance of SecBERTStateEncoder. If None, instantiates a default one.
            mode: 'train' (random sampling) or 'eval' (sequential evaluation).
        """
        # 1. The dataset
        if isinstance(df, str):
            self.df = pd.read_csv(df)
        else:
            self.df = df.reset_index(drop=True)

        # 2. The SecBERT encoder
        self.encoder = encoder if encoder is not None else SecBERTStateEncoder()

        # 3. The current incident (entire row)
        self.current_sample = None

        # 4. The current embedding (768-dim vector for PPO)
        self.current_embedding = None

        # 5. Current index (used for sequential evaluation)
        self.current_index = 0

        # 6. Mode ('train' or 'eval')
        if mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode '{mode}'. Must be either 'train' or 'eval'.")
        self.mode = mode

    # ── Step 2: Incident Selection ───────────────────────────
    def _get_next_sample(self) -> pd.Series:
        """
        Select and return the next incident row from the dataset.

        In 'train' mode:
            Randomly samples a row from self.df (prevents memorization).

        In 'eval' mode:
            Selects the row sequentially at self.current_index (ensures reproducibility).

        Returns:
            pd.Series: The complete row of the selected incident.
                       Does not modify current_sample or current_embedding.
        """
        if self.mode == "train":
            # Random selection for training
            return self.df.sample(1).iloc[0]
        else:
            # Sequential selection for evaluation (wraps around if needed)
            idx = self.current_index % len(self.df)
            return self.df.iloc[idx]

    # ── Step 3: State Construction ───────────────────────────
    # Phase mapping used during Phase 1 training (EXP_005)
    PHASE_MAP = {
        0: "Detection", 1: "Detection", 2: "Detection",
        3: "Containment", 4: "Containment", 5: "Containment", 6: "Containment",
        7: "Containment", 8: "Containment", 9: "Containment", 10: "Containment",
        11: "Containment", 12: "Containment", 13: "Eradication", 14: "Eradication",
        15: "Eradication", 16: "Eradication", 17: "Detection", 18: "Containment",
        19: "All Phases",
    }

    def _prepare_text(self, sample: pd.Series) -> str:
        """
        Reconstruct incident alert text exactly as during Phase 1 SecBERT training,
        including the [CISA Phase] prefix.

        Args:
            sample: The incident row (pd.Series).

        Returns:
            str: Preprocessed text string (e.g., "[Detection] Attack: ...").
        """
        raw_text = str(sample.get("text", "")).strip()

        # If already tagged with a phase (e.g. "[Detection] ..."), keep as is
        if raw_text.startswith("[") and "]" in raw_text[:25]:
            return raw_text

        # If 'phase' or 'CISA Phase' column is explicitly available
        if "phase" in sample and pd.notna(sample["phase"]):
            return f"[{sample['phase']}] {raw_text}"
        if "CISA Phase" in sample and pd.notna(sample["CISA Phase"]):
            return f"[{sample['CISA Phase']}] {raw_text}"

        # Otherwise look up phase by action_label index
        if "action_label" in sample and pd.notna(sample["action_label"]):
            action_idx = int(sample["action_label"])
            phase = self.PHASE_MAP.get(action_idx, "Detection")
            return f"[{phase}] {raw_text}"

        return raw_text

    def _create_state(self, sample: pd.Series):
        """
        Convert a single DataFrame incident row into a 768-dimensional state embedding.

        Args:
            sample: A single incident row (pd.Series).

        Returns:
            torch.Tensor of shape [768]: The verified, detached CPU embedding for PPO.
                                         Does not modify any internal environment state.
        """
        prepared_text = self._prepare_text(sample)
        embedding = self.encoder.encode_incident(prepared_text)
        return embedding

    # ── Step 4: Episode Initialization ──────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset the environment for a new episode and prepare the first incident state.

        Steps:
            1. Select next incident row via _get_next_sample().
            2. Store row in self.current_sample.
            3. Generate 768-dim embedding via _create_state().
            4. Store embedding in self.current_embedding.
            5. Return (observation, info) following standard Gymnasium conventions.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional additional settings.

        Returns:
            Tuple[torch.Tensor, dict]:
                - observation: 768-dim Tensor for PPO.
                - info: Metadata dict (empty for now).
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 1. Pick one incident
        sample = self._get_next_sample()

        # 2. Save current incident row
        self.current_sample = sample

        # 3. Create embedding
        embedding = self._create_state(sample)

        # 4. Save current embedding
        self.current_embedding = embedding

        # 5. Return (observation, info)
        info: Dict[str, Any] = {}
        return self.current_embedding, info

    # ── Step 5: Environment Step ─────────────────────────────
    def step(
        self,
        action: int,
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action, compute reward, and advance to the next incident.

        Steps:
            1. Retrieve ground-truth action from self.current_sample["action_label"].
            2. Compute simple binary reward: +1 if correct, -1 if wrong.
            3. Advance current_index if in 'eval' mode.
            4. Select next incident row via _get_next_sample().
            5. Update self.current_sample with the new row.
            6. Generate and update self.current_embedding via _create_state().
            7. Set terminated=False, truncated=False.
            8. Return (observation, reward, terminated, truncated, info).

        Args:
            action: The discrete action index chosen by the agent (0-19).

        Returns:
            Tuple[torch.Tensor, float, bool, bool, dict]:
                - observation: 768-dim Tensor of the next state.
                - reward: +1.0 for correct action, -1.0 otherwise.
                - terminated: bool indicating natural episode termination.
                - truncated: bool indicating early episode cutoff.
                - info: Metadata dict.
        """
        if self.current_sample is None:
            raise RuntimeError("Environment must be reset with env.reset() before calling env.step().")

        # 1. Look at ground truth
        ground_truth = int(self.current_sample["action_label"])

        # 2. Compare and calculate reward
        is_correct = (int(action) == ground_truth)
        reward = 1.0 if is_correct else -1.0

        # 3. Advance evaluation pointer if in eval mode
        if self.mode == "eval":
            self.current_index += 1

        # 4. Get next incident
        next_sample = self._get_next_sample()

        # 5. Save next incident
        self.current_sample = next_sample

        # 6. Create and save next embedding
        next_embedding = self._create_state(next_sample)
        self.current_embedding = next_embedding

        # 7. Episode finish flags
        terminated = False
        truncated = False

        # 8. Return Gymnasium 5-tuple
        info: Dict[str, Any] = {
            "correct": is_correct,
            "ground_truth": ground_truth,
        }

        return self.current_embedding, reward, terminated, truncated, info


