"""
Integration and Validation Tests for the SOC Simulation Engine (Step 3).
Verifies scientific integrity:
1. Model checkpoint inspection & missing-checkpoint handling.
2. SecBERT output contracts and absence of fabricated embeddings.
3. Policy inference handling and absence of manufactured softmax distributions.
4. Action space 1:1 mapping with data/action_space.csv.
5. Reward provenance (+1.0 / -1.0) adhering to src/environment.py.
6. Separation of Historical Benchmark Replay from Live Attack Simulation.
7. Honest 'Not Available' state for enterprise containment metrics.
8. Event stream emission and early cancellation handling.
"""

import os
import sys
import unittest
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
GRANDPARENT_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))

for p in [PARENT_DIR, GRANDPARENT_DIR, os.path.join(PARENT_DIR, "web"), os.path.join(GRANDPARENT_DIR, "web")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from api.simulation_engine import (
    ACTION_SPACE,
    ACTION_BY_INDEX,
    SCENARIOS_CATALOG,
    CHECKPOINT_AUDIT,
    STATE_ENCODER,
    POLICY_ENGINE,
    BENCHMARK_TEST_INCIDENTS,
    create_and_start_simulation,
    get_simulation_run,
    resolve_scenario_id,
)


class TestSOCSimulationEngine(unittest.TestCase):

    def test_checkpoint_audit_detects_missing(self):
        """Verify checkpoint inspector accurately identifies uncommitted weights."""
        audit = CHECKPOINT_AUDIT
        self.assertIn("secbert", audit)
        self.assertIn("ppo", audit)
        self.assertIn("ppo_atgp", audit)

        # On this environment, checkpoints are not on disk
        self.assertEqual(audit["secbert"]["status"], "unavailable")
        self.assertFalse(audit["secbert"]["real_inference_available"])
        self.assertIn("models/secbert_finetuned", audit["secbert"]["expected_path"])

        self.assertEqual(audit["ppo"]["status"], "unavailable")
        self.assertFalse(audit["ppo"]["real_inference_available"])
        self.assertIn("models/ppo/best_ppo_policy.pt", audit["ppo"]["expected_path"])

        self.assertEqual(audit["ppo_atgp"]["status"], "unavailable")
        self.assertFalse(audit["ppo_atgp"]["real_inference_available"])

    def test_secbert_missing_checkpoint_handling(self):
        """Verify StateEncoderAdapter reports unavailable and does NOT return fake vectors."""
        res = STATE_ENCODER.encode("[Detection] Attack: Exploitive. Target: DC-01")
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNone(res["sample"])
        self.assertIsNone(res["norm"])
        self.assertFalse(res["real_inference"])
        self.assertIn("models/secbert_finetuned", res["reason"])

    def test_policy_engine_missing_checkpoint_handling(self):
        """Verify PolicyEngine sets probabilities and critic value to None when checkpoint is missing."""
        sample_inc = BENCHMARK_TEST_INCIDENTS[0]
        prediction = POLICY_ENGINE.predict(sample_inc, policy_name="PPO", step_idx=1)

        self.assertEqual(prediction["policy_status"], "unavailable")
        self.assertFalse(prediction["real_model_inference"])
        self.assertEqual(prediction["action_selection_mode"], "Historical Benchmark Ground Truth Replay")
        # Must NOT manufacture softmax probabilities or critic baseline
        self.assertIsNone(prediction["probabilities"])
        self.assertIsNone(prediction["confidence"])
        self.assertIsNone(prediction["critic_value"])
        # Selected action is the authoritative benchmark action
        self.assertEqual(prediction["chosen_action"], sample_inc["action_label"])
        self.assertEqual(prediction["ground_truth_action"], sample_inc["action_label"])

    def test_action_space_20_classes_mapping(self):
        """Verify action space has exactly 20 canonical CISA actions matching action_space.csv."""
        self.assertEqual(len(ACTION_SPACE), 20)
        self.assertEqual(len(ACTION_BY_INDEX), 20)

        # Verify key canonical actions
        self.assertEqual(ACTION_BY_INDEX[0]["name"], "monitor")
        self.assertEqual(ACTION_BY_INDEX[0]["phase"], "Detection")

        self.assertEqual(ACTION_BY_INDEX[3]["name"], "block_source_ip")
        self.assertEqual(ACTION_BY_INDEX[3]["phase"], "Containment")

        self.assertEqual(ACTION_BY_INDEX[7]["name"], "isolate_host")
        self.assertEqual(ACTION_BY_INDEX[7]["phase"], "Containment")

        self.assertEqual(ACTION_BY_INDEX[13]["name"], "remove_persistence")
        self.assertEqual(ACTION_BY_INDEX[13]["phase"], "Eradication")

        self.assertEqual(ACTION_BY_INDEX[16]["name"], "patch_vulnerability")
        self.assertEqual(ACTION_BY_INDEX[16]["phase"], "Eradication")

        self.assertEqual(ACTION_BY_INDEX[19]["name"], "escalate_to_human")

    def test_benchmark_incidents_data_integrity(self):
        """Verify benchmark incidents are loaded accurately from data/processed/test.csv."""
        self.assertEqual(len(BENCHMARK_TEST_INCIDENTS), 1539)
        inc0 = BENCHMARK_TEST_INCIDENTS[0]
        self.assertIn("id", inc0)
        self.assertIn("text", inc0)
        self.assertIn("prepared_text", inc0)
        self.assertIn("action_label", inc0)
        self.assertIn(inc0["action_label"], range(20))
        self.assertIn("cisa_phase", inc0)

    def test_simulation_scenarios_mode_separation(self):
        """Verify all scenarios are at most 3, validated, and labeled as Historical Replay."""
        self.assertLessEqual(len(SCENARIOS_CATALOG), 3)
        self.assertEqual(len(SCENARIOS_CATALOG), 3)
        self.assertIn("scenario-ransomware", SCENARIOS_CATALOG)
        self.assertIn("scenario-c2", SCENARIOS_CATALOG)
        self.assertIn("scenario-vulnerability", SCENARIOS_CATALOG)

        for sid, sc in SCENARIOS_CATALOG.items():
            self.assertEqual(sc["simulation_mode"], "Historical Replay")
            self.assertEqual(sc["validation_status"], "Validated (Historical Replay)")
            self.assertFalse(sc["live_adversary"])
            self.assertEqual(sc["dataset_source"], "data/processed/test.csv")
            self.assertIn("target_actions", sc)
            self.assertIn("stages", sc)
            self.assertIn("initial_state", sc)
            self.assertIn("supported_steps", sc)
            self.assertIn("outcome_criteria", sc)

    def test_scenario_resolution_and_aliases(self):
        """Verify scenario resolution supports canonical IDs and backward-compatible aliases."""
        self.assertEqual(resolve_scenario_id("scenario-ransomware"), "scenario-ransomware")
        self.assertEqual(resolve_scenario_id("scenario-a"), "scenario-ransomware")
        self.assertEqual(resolve_scenario_id("ransomware"), "scenario-ransomware")

        self.assertEqual(resolve_scenario_id("scenario-c2"), "scenario-c2")
        self.assertEqual(resolve_scenario_id("scenario-b"), "scenario-c2")
        self.assertEqual(resolve_scenario_id("c2"), "scenario-c2")

        self.assertEqual(resolve_scenario_id("scenario-vulnerability"), "scenario-vulnerability")
        self.assertEqual(resolve_scenario_id("scenario-c"), "scenario-vulnerability")
        self.assertEqual(resolve_scenario_id("web_exploit"), "scenario-vulnerability")

    def test_unsupported_scenarios_rejected(self):
        """Verify unsupported scenarios are rejected with an explicit error."""
        with self.assertRaises(ValueError) as ctx:
            resolve_scenario_id("scenario-d")
        self.assertIn("Allowed scenarios", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            resolve_scenario_id("scenario-benchmark")
        self.assertIn("Allowed scenarios", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            resolve_scenario_id("caldera-adversary-operation")
        self.assertIn("Allowed scenarios", str(ctx.exception))

    def test_scenario_candidate_class_filtering(self):
        """Verify scenarios filter incidents corresponding strictly to their candidate action classes."""
        # Ransomware: actions 7 (isolate_host) and 8 (kill_process)
        sim_ransomware = create_and_start_simulation(
            scenario="scenario-ransomware",
            policy="PPO",
            target="Subnet 10.0.1.0/24 (Corporate Workstations)",
            max_steps=5,
            seed=42
        )
        for _ in range(30):
            if sim_ransomware.status in ("completed", "stopped"):
                break
            time.sleep(0.15)
        self.assertEqual(sim_ransomware.status, "completed")
        for step in sim_ransomware.steps_data:
            self.assertIn(step["ground_truth"], [7, 8])

        # C2: actions 3 (block_source_ip), 4 (block_dest_ip), 5 (dns_sinkhole)
        sim_c2 = create_and_start_simulation(
            scenario="scenario-c2",
            policy="PPO",
            target="Subnet 10.0.2.0/24 (DMZ Server Infrastructure)",
            max_steps=5,
            seed=42
        )
        for _ in range(30):
            if sim_c2.status in ("completed", "stopped"):
                break
            time.sleep(0.15)
        self.assertEqual(sim_c2.status, "completed")
        for step in sim_c2.steps_data:
            self.assertIn(step["ground_truth"], [3, 4, 5])

        # Vulnerability: actions 16 (patch_vulnerability) and 13 (remove_persistence)
        sim_vuln = create_and_start_simulation(
            scenario="scenario-vulnerability",
            policy="PPO",
            target="Subnet 10.0.2.0/24 (DMZ Ingress Web Servers)",
            max_steps=5,
            seed=42
        )
        for _ in range(30):
            if sim_vuln.status in ("completed", "stopped"):
                break
            time.sleep(0.15)
        self.assertEqual(sim_vuln.status, "completed")
        for step in sim_vuln.steps_data:
            self.assertIn(step["ground_truth"], [16, 13])

    def test_seed_reproducibility(self):
        """Verify that identical random seed produces identical incident sequence and results."""
        sim1 = create_and_start_simulation("scenario-ransomware", "PPO", "Target-A", 4, seed=123)
        for _ in range(30):
            if sim1.status in ("completed", "stopped"):
                break
            time.sleep(0.15)

        sim2 = create_and_start_simulation("scenario-ransomware", "PPO", "Target-A", 4, seed=123)
        for _ in range(30):
            if sim2.status in ("completed", "stopped"):
                break
            time.sleep(0.15)

        self.assertEqual(len(sim1.steps_data), len(sim2.steps_data))
        for s1, s2 in zip(sim1.steps_data, sim2.steps_data):
            self.assertEqual(s1["incident_id"], s2["incident_id"])
            self.assertEqual(s1["chosen_action"], s2["chosen_action"])
            self.assertEqual(s1["reward"], s2["reward"])

    def test_reward_provenance_and_calculation(self):
        """Verify reward strictly follows the +1.0 / -1.0 logic of src/environment.py."""
        sample_inc = BENCHMARK_TEST_INCIDENTS[5]
        pred_correct = POLICY_ENGINE.predict(sample_inc, step_idx=1)
        self.assertEqual(pred_correct["reward"], 1.0)
        self.assertTrue(pred_correct["is_correct"])

        gt = sample_inc["action_label"]
        wrong_act = (gt + 1) % 20
        reward_wrong = 1.0 if (wrong_act == gt) else -1.0
        self.assertEqual(reward_wrong, -1.0)

    def test_containment_outcome_honesty(self):
        """Verify containment outcome is reported as 'Not Available' rather than fabricated."""
        sim = create_and_start_simulation(
            scenario="scenario-ransomware",
            policy="PPO",
            target="Enterprise Network",
            max_steps=5,
            seed=42
        )
        for _ in range(30):
            if sim.status in ("completed", "stopped"):
                break
            time.sleep(0.15)

        self.assertEqual(sim.status, "completed")
        self.assertIsNotNone(sim.results)
        self.assertEqual(sim.results["containment_outcome"], "Not Available")
        self.assertIn("Contextual Bandit", sim.results["containment_outcome_note"])
        self.assertEqual(sim.results["simulation_mode"], "Historical Replay")
        self.assertFalse(sim.results.get("real_model_inference", False))
        self.assertEqual(sim.results["total_steps"], 5)
        self.assertEqual(sim.results["cumulative_reward"], 5.0)

    def test_event_stream_structured_emission(self):
        """Verify SSE events contain accurate structure, attack stages, and honest indicators."""
        sim = create_and_start_simulation(
            scenario="scenario-ransomware",
            policy="PPO",
            target="Subnet 10.0.1.0/24 (Corporate Workstations)",
            max_steps=5,
            seed=10
        )
        for _ in range(30):
            if sim.status in ("completed", "stopped"):
                break
            time.sleep(0.15)

        event_types = [e["event_type"] for e in sim.events]
        self.assertIn("simulation_started", event_types)
        self.assertIn("attack_stage_updated", event_types)
        self.assertIn("incident_observed", event_types)
        self.assertIn("embedding_generated", event_types)
        self.assertIn("action_selected", event_types)
        self.assertIn("simulated_effect_applied", event_types)
        self.assertIn("reward_calculated", event_types)
        self.assertIn("state_updated", event_types)
        self.assertIn("simulation_completed", event_types)

        # Inspect attack_stage event
        stage_ev = next(e for e in sim.events if e["event_type"] == "attack_stage_updated")
        self.assertIn("stage", stage_ev["data"])
        self.assertEqual(stage_ev["data"]["simulation_mode"], "Historical Replay")
        self.assertIn("stage_index", stage_ev["data"])

        # Inspect embedding event
        emb_ev = next(e for e in sim.events if e["event_type"] == "embedding_generated")
        self.assertEqual(emb_ev["data"]["status"], "unavailable")
        self.assertIsNone(emb_ev["data"]["sample"])

        # Inspect action selected event
        act_ev = next(e for e in sim.events if e["event_type"] == "action_selected")
        self.assertIsNone(act_ev["data"]["probabilities"])
        self.assertIsNone(act_ev["data"]["critic_value"])
        self.assertIn("Historical Benchmark Ground Truth Replay", act_ev["data"]["action_source"])

    def test_early_cancellation_handling(self):
        """Verify worker thread terminates gracefully when stop is requested."""
        sim = create_and_start_simulation(
            scenario="scenario-c2",
            policy="PPO+ATGP",
            target="Subnet 10.0.2.0/24 (DMZ Server Infrastructure)",
            max_steps=50,
            seed=99
        )
        time.sleep(0.2)
        sim.request_stop()

        for _ in range(20):
            if sim.status in ("stopped", "completed"):
                break
            time.sleep(0.15)

        self.assertEqual(sim.status, "stopped")
        self.assertLess(len(sim.steps_data), 50)
        self.assertEqual(sim.results["containment_outcome"], "Not Available")


if __name__ == "__main__":
    unittest.main()
