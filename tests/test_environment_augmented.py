# Unit and Integration Tests for Experiment 003: Telemetry-Augmented SOC Environment
#
# Tests cover:
#   1. Feature extractor schema, dimensions (34), and missing-field handling.
#   2. Telemetry-augmented environment dimensions (802) and finite values.
#   3. Strict absence of hidden label / ground-truth leakage.
#   4. Actor and Critic network compatibility with 802-dimensional inputs.
#   5. Rejection of unadapted 768-dim checkpoints and successful warmstarting.
#   6. Backward compatibility of SequentialSOCEnvironment (remains 768-dim).
#   7. Evaluation metric separation of benign vs. malicious episodes.

import os
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.telemetry.incident_formatter import IncidentRecord
from src.telemetry_feature_extractor import (
    TelemetryFeatureExtractor,
    TELEMETRY_FEATURE_SCHEMA,
    TELEMETRY_FEATURE_DIM,
)
from src.environment_sequential import (
    SequentialSOCEnvironment,
    IncidentScenario,
    ScenarioStage,
    RewardConfig,
    get_default_scenarios,
)
from src.environment_augmented import (
    TelemetryAugmentedSOCEnvironment,
    COMBINED_OBSERVATION_DIM,
)
from src.ppo.agent import PPOAgent
from src.ppo.agent_augmented import AugmentedPPOAgent
from src.ppo.trainer_sequential import SequentialPPOTrainer
from tests.test_environment_sequential import MockEncoder


class TestTelemetryAugmentedEnvironment(unittest.TestCase):
    def setUp(self):
        self.mock_encoder = MockEncoder()
        self.extractor = TelemetryFeatureExtractor()
        self.aug_env = TelemetryAugmentedSOCEnvironment(
            encoder=self.mock_encoder,
            reward_config=RewardConfig(escalation_cost=1.0),
        )
        self.standard_env = SequentialSOCEnvironment(
            encoder=self.mock_encoder,
            reward_config=RewardConfig(escalation_cost=1.0),
        )

    def test_feature_extractor_schema_and_dimensions(self):
        """Verify feature extractor produces exactly 34 dimensions matching schema."""
        self.assertEqual(len(TELEMETRY_FEATURE_SCHEMA), 34)
        self.assertEqual(self.extractor.feature_dim, 34)
        self.assertEqual(COMBINED_OBSERVATION_DIM, 802)
        # Verify all feature names are unique
        self.assertEqual(len(TELEMETRY_FEATURE_SCHEMA), len(set(TELEMETRY_FEATURE_SCHEMA)))

    def test_feature_extractor_missing_fields_and_finite_values(self):
        """Verify graceful missing-field handling without inventing facts or producing NaNs."""
        # 1. None record
        features_none = self.extractor.extract(None)
        self.assertEqual(features_none.shape, (34,))
        self.assertTrue(np.all(features_none == 0.0))
        self.assertTrue(np.isfinite(features_none).all())

        # 2. Empty IncidentRecord
        empty_rec = IncidentRecord()
        features_empty = self.extractor.extract(empty_rec)
        self.assertEqual(features_empty.shape, (34,))
        self.assertTrue(np.all(features_empty == 0.0))
        self.assertTrue(np.isfinite(features_empty).all())

        # 3. Partial record (e.g. only alert_level)
        partial_rec = IncidentRecord(alert_level=6)
        features_partial = self.extractor.extract(partial_rec)
        self.assertEqual(features_partial.shape, (34,))
        self.assertAlmostEqual(features_partial[0], 6.0 / 15.0)  # alert_level_norm
        self.assertEqual(features_partial[1], 1.0)               # has_alert_level
        self.assertEqual(features_partial[4], 0.0)               # has_uid is False
        self.assertEqual(features_partial[6], 0.0)               # has_cwd is False

    def test_feature_extractor_observable_grounding(self):
        """Verify specific observable telemetry features encode accurately."""
        # Record with root, sudo, tmp cmdline, and privilege escalation tactic
        rec = IncidentRecord(
            process_name="sudo",
            process_path="/usr/bin/sudo",
            cmdline="sudo -u root /tmp/exploit.sh",
            uid=0,
            username="root",
            alert_level=12,
            mitre_tactics=["Privilege Escalation", "Defense Evasion"],
        )
        vec = self.extractor.extract(rec)
        self.assertEqual(vec.shape, (34,))
        self.assertTrue(np.isfinite(vec).all())
        self.assertTrue(np.all((vec >= 0.0) & (vec <= 1.0)))

        # Indices verification
        self.assertAlmostEqual(vec[0], 12.0 / 15.0)  # alert_level_norm
        self.assertEqual(vec[1], 1.0)                # has_alert_level
        self.assertEqual(vec[3], 1.0)                # is_uid_zero
        self.assertEqual(vec[4], 1.0)                # has_uid
        self.assertEqual(vec[8], 1.0)                # is_cmdline_tmp
        self.assertEqual(vec[10], 1.0)               # proc_sudo
        self.assertEqual(vec[20], 1.0)               # has_mitre
        self.assertEqual(vec[23], 1.0)               # tactic_privilege_escalation
        self.assertEqual(vec[24], 1.0)               # tactic_defense_evasion
        self.assertEqual(vec[31], 1.0)               # has_username
        self.assertEqual(vec[32], 1.0)               # user_is_root

    def test_no_hidden_label_leakage(self):
        """Verify observation does NOT change when hidden labels/metadata vary."""
        # Two identical observable records with inverted scenario/stage/benign status
        rec1 = IncidentRecord(
            process_name="rsync",
            cmdline="rsync -a /var/log/ /backup/",
            uid=0,
            username="root",
            alert_level=3,
        )
        rec2 = IncidentRecord(
            process_name="rsync",
            cmdline="rsync -a /var/log/ /backup/",
            uid=0,
            username="root",
            alert_level=3,
        )

        stage1 = ScenarioStage(stage_id=0, name="StageA", description="Test Stage A", incident_record=rec1, effective_actions=[0])
        scen1 = IncidentScenario(scenario_id="scen_fake_mal", name="Mal", description="", stages=[stage1], is_benign=False)

        stage2 = ScenarioStage(stage_id=99, name="StageB", description="Test Stage B", incident_record=rec2, effective_actions=[15])
        scen2 = IncidentScenario(scenario_id="scen_fake_ben", name="Ben", description="", stages=[stage2], is_benign=True)

        env1 = TelemetryAugmentedSOCEnvironment(scenarios=[scen1], encoder=self.mock_encoder)
        env2 = TelemetryAugmentedSOCEnvironment(scenarios=[scen2], encoder=self.mock_encoder)

        obs1, info1 = env1.reset()
        obs2, info2 = env2.reset()

        # Observations must be identical: neither is_benign nor scenario_id/stage_id leak into state
        self.assertTrue(torch.equal(obs1, obs2))

    def test_augmented_environment_observation_space_and_finite_values(self):
        """Verify environment observation space is Box(802,) and returns finite tensors."""
        self.assertEqual(self.aug_env.observation_space.shape, (802,))
        self.assertEqual(self.aug_env.action_space.n, 20)

        for scen in get_default_scenarios():
            obs, info = self.aug_env.reset(options={"scenario_id": scen.scenario_id})
            self.assertEqual(obs.shape, torch.Size([802]))
            self.assertTrue(torch.isfinite(obs).all())

            # Step 1
            next_obs, reward, term, trunc, step_info = self.aug_env.step(0)
            self.assertEqual(next_obs.shape, torch.Size([802]))
            self.assertTrue(torch.isfinite(next_obs).all())

    def test_actor_critic_compatibility_with_802_dims(self):
        """Verify actor and critic networks accept 802 dimensions and reject unadapted 768 checkpoints."""
        agent = AugmentedPPOAgent(state_dim=802, action_dim=20)
        self.assertEqual(agent.state_dim, 802)
        self.assertEqual(agent.actor.network[0].in_features, 802)
        self.assertEqual(agent.critic.network[0].in_features, 802)

        # Single sample forward pass
        single_state = torch.randn(802)
        action, log_prob, val = agent.select_action(single_state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < 20)
        self.assertTrue(np.isfinite(log_prob))
        self.assertTrue(np.isfinite(val))

        # Batched forward pass
        batch_state = torch.randn(16, 802)
        actor_logits = agent.actor(batch_state)
        critic_values = agent.critic(batch_state)
        self.assertEqual(actor_logits.shape, torch.Size([16, 20]))
        self.assertEqual(critic_values.shape, torch.Size([16, 1]))

        # Checkpoint rejection test: create a mock 768-dim checkpoint
        old_agent = PPOAgent(state_dim=768, action_dim=20)
        mock_ckpt_path = "models/test_mock_768_ckpt.pt"
        os.makedirs("models", exist_ok=True)
        torch.save(
            {
                "actor_state_dict": old_agent.actor.state_dict(),
                "critic_state_dict": old_agent.critic.state_dict(),
            },
            mock_ckpt_path,
        )

        try:
            # Must raise ValueError when loading unadapted 768 checkpoint
            with self.assertRaises(ValueError):
                agent.load_checkpoint(mock_ckpt_path)

            # Warmstart must succeed
            agent.warmstart_from_768_checkpoint(mock_ckpt_path, init_telemetry_weight="zero")
            # Verify first 768 weights match old agent
            self.assertTrue(
                torch.equal(
                    agent.actor.network[0].weight[:, :768],
                    old_agent.actor.network[0].weight,
                )
            )
            # Verify telemetry slice is zero-initialized
            self.assertTrue(
                torch.equal(
                    agent.actor.network[0].weight[:, 768:],
                    torch.zeros(512, 34),
                )
            )
        finally:
            if os.path.exists(mock_ckpt_path):
                os.remove(mock_ckpt_path)

    def test_existing_environment_and_checkpoints_remain_unchanged(self):
        """Verify SequentialSOCEnvironment and standard PPOAgent remain strictly 768-dimensional."""
        self.assertEqual(self.standard_env.observation_space.shape, (768,))
        obs, _ = self.standard_env.reset()
        self.assertEqual(obs.shape, torch.Size([768]))

        default_agent = PPOAgent()
        self.assertEqual(default_agent.state_dim, 768)
        self.assertEqual(default_agent.actor.network[0].in_features, 768)
        self.assertEqual(default_agent.critic.network[0].in_features, 768)

    def test_evaluation_separates_benign_and_malicious_in_augmented_env(self):
        """Verify SequentialPPOTrainer reliably tracks benign/malicious counts in augmented environment."""
        trainer = SequentialPPOTrainer(
            env=self.aug_env,
            agent=AugmentedPPOAgent(state_dim=802),
            eval_env=self.aug_env,
            rollout_steps=16,
            total_iterations=1,
            eval_episodes=14,  # Exactly 2 cycles over 7 scenarios (4 malicious, 3 benign each)
        )
        metrics = trainer.evaluate(num_episodes=14)
        self.assertEqual(metrics["eval_episodes"], 14.0)
        mal_eps = metrics["eval_malicious"]["num_episodes"]
        ben_eps = metrics["eval_benign"]["num_episodes"]
        self.assertGreater(mal_eps, 0)
        self.assertGreater(ben_eps, 0)
        self.assertEqual(mal_eps + ben_eps, 14.0)


if __name__ == "__main__":
    unittest.main()
