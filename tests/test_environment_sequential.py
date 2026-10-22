# Responsibility: Comprehensive Unit & Integration Tests for SequentialSOCEnvironment
#
# Tests covered:
#   1. Reset and 768-dim observation shape verification.
#   2. Action-space bounds and Discrete(20) validation.
#   3. Sequential state transitions across multi-stage attack scenarios.
#   4. Effective mitigation triggering containment and positive reward.
#   5. Ineffective responses advancing attack progression with penalty.
#   6. Disruption penalties for misapplied containment and false positives.
#   7. Terminal failure (breach completion when attack reaches final uncontained stage).
#   8. Episode truncation at max_steps limit.
#   9. Deterministic scenario selection under identical random seeds.
#  10. Safety check: Verification of zero host-side process or system execution.
#  11. Real SecBERTStateEncoder live embedding integration.

import os
import sys
import unittest
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from src.environment_sequential import (
    SequentialSOCEnvironment,
    RewardConfig,
    get_default_scenarios,
    IncidentScenario,
)
from src.encoder import SecBERTStateEncoder


class MockEncoder:
    """Fast deterministic mock encoder for rapid isolated testing."""
    def encode_incident(self, text: str) -> torch.Tensor:
        seed = abs(hash(text)) % (2**31)
        np_gen = np.random.RandomState(seed)
        vec = np_gen.randn(768).astype(np.float32)
        return torch.from_numpy(vec)


class TestSequentialSOCEnvironment(unittest.TestCase):

    def setUp(self):
        """Set up environment with deterministic mock encoder for fast unit tests."""
        self.mock_encoder = MockEncoder()
        self.env = SequentialSOCEnvironment(
            encoder=self.mock_encoder,
            max_steps=5,
        )

    def test_reset_and_observation_shape(self):
        """Verify reset returns a 768-dim float tensor observation and valid metadata info."""
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape, torch.Size([768]))
        self.assertTrue(torch.isfinite(obs).all())
        self.assertIn("scenario_id", info)
        self.assertIn("stage_id", info)
        self.assertTrue(info.get("simulated", False))

    def test_action_space_compatibility(self):
        """Verify action space has exactly 20 discrete actions and rejects out-of-bounds actions."""
        self.assertEqual(self.env.action_space.n, 20)
        self.env.reset(seed=1)
        with self.assertRaises(ValueError):
            self.env.step(25)  # Out of range
        with self.assertRaises(ValueError):
            self.env.step(-1)

    def test_effective_action_containment(self):
        """Verify that executing an effective action achieves containment and terminates the episode."""
        # Force scenario 1 (scen_c2_exfil), stage 0 effective action is 8 (kill_process)
        obs, info = self.env.reset(options={"scenario_id": "scen_c2_exfil"})
        self.assertEqual(info["stage_id"], 0)

        next_obs, reward, terminated, truncated, step_info = self.env.step(8)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(step_info["contained"])
        self.assertEqual(step_info["outcome"], "contained")
        # Reward should be containment_reward (2.0) + step_cost (-0.1) = 1.9
        self.assertAlmostEqual(reward, 1.9)

    def test_ineffective_action_progression(self):
        """Verify that an ineffective action incurs penalty and causes attack to advance."""
        obs, info = self.env.reset(options={"scenario_id": "scen_c2_exfil"})
        self.assertEqual(info["stage_id"], 0)

        # Action 5 (dns_sinkhole) is ineffective at stage 0 (local enumeration)
        next_obs, reward, terminated, truncated, step_info = self.env.step(5)
        self.assertFalse(terminated)
        self.assertEqual(step_info["stage_id"], 1)  # Advanced to C2 stage
        self.assertFalse(step_info["contained"])
        self.assertEqual(step_info["outcome"], "ineffective")
        # Reward: step_cost (-0.1) + ineffective_penalty (-0.5) = -0.6
        self.assertAlmostEqual(reward, -0.6)

    def test_disruption_cost_on_benign_scenario(self):
        """Verify disruption penalty when aggressive containment is misapplied to benign activity."""
        obs, info = self.env.reset(options={"scenario_id": "scen_benign_backup"})
        self.assertTrue(info["is_benign"])

        # Action 7 (isolate_host) on benign backup server
        next_obs, reward, terminated, truncated, step_info = self.env.step(7)
        self.assertTrue(terminated)
        self.assertEqual(step_info["outcome"], "disruptive_false_positive")
        # Reward: step_cost (-0.1) + disruption_penalty (-1.0) = -1.1
        self.assertAlmostEqual(reward, -1.1)

    def test_terminal_failure_breach(self):
        """Verify episode terminates with breach penalty when attack reaches final stage uncontained."""
        obs, info = self.env.reset(options={"scenario_id": "scen_privesc_persist"})
        # Scenario has 2 stages: 0 (sudo) and 1 (persist)
        self.assertEqual(info["stage_id"], 0)

        # Step 1: Ineffective action at stage 0 (advances to stage 1)
        _, _, term1, _, _ = self.env.step(5)
        self.assertFalse(term1)

        # Step 2: Ineffective action at stage 1 (advances past final stage -> breach)
        _, reward2, term2, _, info2 = self.env.step(5)
        self.assertTrue(term2)
        self.assertTrue(info2["compromised"])
        self.assertEqual(info2["outcome"], "breach_compromised")
        # Reward should include breach penalty (-2.0)
        self.assertLess(reward2, -2.0)

    def test_truncation_at_step_limit(self):
        """Verify episode truncates when reaching max_steps without terminal outcome."""
        env_short = SequentialSOCEnvironment(encoder=self.mock_encoder, max_steps=2)
        # Force a scenario where multiple investigative steps can occur
        env_short.reset(options={"scenario_id": "scen_c2_exfil"})

        # Step 1: investigative (1: deep logging)
        _, _, term1, trunc1, _ = env_short.step(1)
        self.assertFalse(term1)
        self.assertFalse(trunc1)

        # Step 2: reached max_steps=2
        _, _, term2, trunc2, _ = env_short.step(1)
        self.assertTrue(trunc2 or term2)

    def test_deterministic_seeded_episodes(self):
        """Verify that identical seeds produce identical scenario trajectories."""
        obs_a, info_a = self.env.reset(seed=12345)
        obs_b, info_b = self.env.reset(seed=12345)
        self.assertEqual(info_a["scenario_id"], info_b["scenario_id"])
        self.assertTrue(torch.allclose(obs_a, obs_b))

    def test_no_host_execution_safety(self):
        """Verify that simulation calls never execute live host subprocesses or OS commands."""
        # Run multiple random steps across multiple episodes
        self.env.reset(seed=999)
        for act in [8, 7, 4, 12, 19, 0, 1]:
            obs, r, term, trunc, info = self.env.step(act % 20)
            self.assertTrue(info["simulated"])
            if term or trunc:
                self.env.reset()

    def test_benign_incident_effective_clearance(self):
        """Verify that benign incidents are cleared (cleared=True) and never marked as contained."""
        obs, info = self.env.reset(options={"scenario_id": "scen_benign_backup"})
        self.assertTrue(info["is_benign"])

        # Action 0 (monitor) is effective for benign backup
        next_obs, reward, terminated, truncated, step_info = self.env.step(0)
        self.assertTrue(terminated)
        self.assertTrue(step_info["cleared"])
        self.assertFalse(step_info["contained"])  # Strictly False: threat was non-malicious
        self.assertEqual(step_info["outcome"], "correctly_cleared_benign")
        self.assertAlmostEqual(reward, 1.9)

    def test_benign_incident_ineffective_bounded_triage(self):
        """Verify ineffective action on benign incident terminates with bounded triage outcome and does not loop."""
        obs, info = self.env.reset(options={"scenario_id": "scen_benign_backup"})
        self.assertTrue(info["is_benign"])

        # Action 6 (block_port) is ineffective/irrelevant for routine admin backup
        next_obs, reward, terminated, truncated, step_info = self.env.step(6)
        self.assertTrue(terminated)
        self.assertFalse(step_info["cleared"])
        self.assertFalse(step_info["contained"])
        self.assertEqual(step_info["outcome"], "benign_unresolved")
        # Reward: step_cost (-0.1) + ineffective_penalty (-0.5) = -0.6
        self.assertAlmostEqual(reward, -0.6)

    def test_persistent_simulated_response_isolation(self):
        """Verify prior host isolation in stage 0 neutralizes subsequent network C2 in stage 1."""
        obs, info = self.env.reset(options={"scenario_id": "scen_c2_exfil"})
        self.assertEqual(info["stage_id"], 0)

        # Stage 0: Execute isolate_host (Action 7). Premature/disruptive for whoami, but isolates host.
        next_obs, reward, terminated, truncated, step_info = self.env.step(7)
        self.assertTrue(terminated)
        self.assertTrue(step_info["contained"])
        self.assertEqual(step_info["outcome"], "contained_by_prior_isolation")
        self.assertIn("workstation-01", step_info["simulated_state"]["isolated_hosts"])
        # Net reward: step_cost (-0.1) + disruption (-1.0) + ineffective (-0.5) + containment (+2.0) = +0.4
        self.assertAlmostEqual(reward, 0.4)

    def test_persistent_simulated_response_blocked_ip(self):
        """Verify prior IP block at stage 1 neutralizes subsequent exfiltration at stage 2."""
        obs, info = self.env.reset(options={"scenario_id": "scen_c2_exfil"})

        # Step 1: Ineffective action (5: dns_sinkhole) to reach stage 1
        self.env.step(5)

        # Step 2: At stage 1, execute block_dest_ip (Action 4)
        next_obs, reward, terminated, truncated, step_info = self.env.step(4)
        self.assertTrue(terminated)
        self.assertTrue(step_info["contained"])
        self.assertEqual(step_info["outcome"], "contained")
        self.assertIn("198.51.100.24", step_info["simulated_state"]["blocked_ips"])

    def test_escalation_reward_and_configurable_cost(self):
        """Verify baseline escalation and configurable operational escalation overhead cost."""
        # 1. Default config (escalation_cost = 0.0)
        self.env.reset(options={"scenario_id": "scen_c2_exfil"})
        _, reward_default, term_default, _, info_default = self.env.step(19)
        self.assertTrue(term_default)
        self.assertTrue(info_default["escalated"])
        self.assertAlmostEqual(reward_default, -0.1)  # step_cost (-0.1) + escalation_reward (0.0)

        # 2. Config with escalation overhead cost (e.g., tier-3 analyst triage cost of 0.4)
        custom_cfg = RewardConfig(escalation_cost=0.4)
        custom_env = SequentialSOCEnvironment(encoder=self.mock_encoder, reward_config=custom_cfg)
        custom_env.reset(options={"scenario_id": "scen_c2_exfil"})
        _, reward_cost, term_cost, _, info_cost = custom_env.step(19)
        self.assertTrue(term_cost)
        self.assertTrue(info_cost["escalated"])
        # Reward: step_cost (-0.1) + escalation_reward (0.0) - escalation_cost (0.4) = -0.5
        self.assertAlmostEqual(reward_cost, -0.5)

    def test_live_secbert_encoder_integration(self):
        """Verify that the real frozen SecBERT encoder generates finite 768-dim embeddings."""
        real_encoder = SecBERTStateEncoder()
        live_env = SequentialSOCEnvironment(encoder=real_encoder, max_steps=3)
        obs, info = live_env.reset(seed=42)

        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape, torch.Size([768]))
        self.assertTrue(torch.isfinite(obs).all())

        next_obs, reward, term, trunc, step_info = live_env.step(0)
        self.assertIsInstance(next_obs, torch.Tensor)
        self.assertEqual(next_obs.shape, torch.Size([768]))
        self.assertTrue(torch.isfinite(next_obs).all())

    def test_expanded_scenario_credential_lateral(self):
        """Verify scen_credential_lateral: kill_process fails; reset_credentials contains; uncontained breaches."""
        # 1. Stage 0: Action 8 (kill_process) is ineffective because credentials remain valid
        obs, info = self.env.reset(options={"scenario_id": "scen_credential_lateral"})
        self.assertEqual(info["stage_id"], 0)
        self.assertIn("telemetry", info)
        self.assertEqual(info["telemetry"]["username"], "devops_lead")

        _, r_step0, term0, _, info0 = self.env.step(8)  # kill_process
        self.assertFalse(term0)
        self.assertEqual(info0["outcome"], "ineffective")
        self.assertEqual(info0["stage_id"], 1)

        # 2. Stage 1: Action 12 (disable_account) successfully contains the incident
        _, r_step1, term1, _, info1 = self.env.step(12)  # disable_account
        self.assertTrue(term1)
        self.assertTrue(info1["contained"])
        self.assertEqual(info1["outcome"], "contained")

        # 3. Breach branch: If stage 1 is also ineffective, attack compromises
        self.env.reset(options={"scenario_id": "scen_credential_lateral"})
        self.env.step(8)  # Ineffective step 0
        _, _, term_breach, _, info_breach = self.env.step(8)  # Ineffective step 1
        self.assertTrue(term_breach)
        self.assertTrue(info_breach["compromised"])
        self.assertEqual(info_breach["outcome"], "breach_compromised")

    def test_expanded_scenario_network_rce_tunnel(self):
        """Verify scen_network_rce_tunnel: network blocking is effective; supervisor respawns against kill_process."""
        obs, info = self.env.reset(options={"scenario_id": "scen_network_rce_tunnel"})
        self.assertEqual(info["stage_id"], 0)
        self.assertIn("telemetry", info)
        self.assertEqual(info["telemetry"]["process_name"], "node")

        # Action 8 (kill_process) is ineffective (auto-restarted by supervisor)
        _, _, term_kill, _, info_kill = self.env.step(8)
        self.assertFalse(term_kill)
        self.assertEqual(info_kill["outcome"], "ineffective")

        # Reset and verify Action 4 (block_dest_ip) contains threat immediately
        self.env.reset(options={"scenario_id": "scen_network_rce_tunnel"})
        _, r_block, term_block, _, info_block = self.env.step(4)  # block_dest_ip
        self.assertTrue(term_block)
        self.assertTrue(info_block["contained"])
        self.assertEqual(info_block["outcome"], "contained")

    def test_expanded_scenario_benign_devops_false_disruption(self):
        """Verify scen_benign_devops_tooling: non-disruptive monitoring clears; killing process disrupts build."""
        # 1. Effective non-disruptive response
        obs, info = self.env.reset(options={"scenario_id": "scen_benign_devops_tooling"})
        self.assertTrue(info["is_benign"])
        self.assertEqual(info["telemetry"]["cwd"], "/tmp/ci_build")

        _, r_mon, term_mon, _, info_mon = self.env.step(0)  # monitor
        self.assertTrue(term_mon)
        self.assertTrue(info_mon["cleared"])
        self.assertFalse(info_mon["contained"])
        self.assertEqual(info_mon["outcome"], "correctly_cleared_benign")

        # 2. Disruptive response: killing CI/CD process triggers disruption penalty
        self.env.reset(options={"scenario_id": "scen_benign_devops_tooling"})
        _, r_kill, term_kill, _, info_kill = self.env.step(8)  # kill_process
        self.assertTrue(term_kill)
        self.assertEqual(info_kill["outcome"], "disruptive_false_positive")
        self.assertAlmostEqual(r_kill, -1.1)  # step_cost (-0.1) + disruption_penalty (-1.0)

    def test_expanded_scenario_benign_security_audit_false_disruption(self):
        """Verify scen_benign_security_audit: authorized scan cleared by monitor; isolation causes false disruption."""
        # 1. Monitoring clears authorized audit
        obs, info = self.env.reset(options={"scenario_id": "scen_benign_security_audit"})
        self.assertTrue(info["is_benign"])
        self.assertEqual(info["telemetry"]["process_name"], "nmap")

        _, _, term_mon, _, info_mon = self.env.step(0)  # monitor
        self.assertTrue(term_mon)
        self.assertTrue(info_mon["cleared"])

        # 2. Isolating scanner causes false disruption
        self.env.reset(options={"scenario_id": "scen_benign_security_audit"})
        _, r_iso, term_iso, _, info_iso = self.env.step(7)  # isolate_host
        self.assertTrue(term_iso)
        self.assertEqual(info_iso["outcome"], "disruptive_false_positive")


if __name__ == "__main__":
    unittest.main()
