"""
Unit tests for Step 6: Results, Evaluation, Persistence, and Policy Comparison.
Verifies:
1. Retrieval and filtering of simulation runs.
2. Retrieval of individual simulation trajectories and results.
3. JSON and CSV export endpoints with download attachment headers.
4. PPO vs. PPO+ATGP comparison engine, compatibility checking, and delta metrics.
5. Strict separation between supervised classification evaluation and sequential simulation metrics.
"""

import os
import sys
import json
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
GRANDPARENT_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))

for p in [PARENT_DIR, GRANDPARENT_DIR, os.path.join(PARENT_DIR, "web"), os.path.join(GRANDPARENT_DIR, "web")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from api.server import app


class TestResultsAndComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_simulations_list_retrieval(self):
        """Verify GET /api/simulations returns persisted benchmarks."""
        response = self.client.get("/api/simulations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("simulations", data)
        self.assertIn("total", data)
        self.assertGreaterEqual(data["total"], 3)

        ids = [s["id"] for s in data["simulations"]]
        self.assertIn("SIM-BENCH-PPO-01", ids)
        self.assertIn("SIM-BENCH-ATGP-01", ids)
        self.assertIn("SIM-BENCH-STOPPED-01", ids)

    def test_simulations_filtering(self):
        """Verify query filters for scenario, policy, status, and mode."""
        # 1. Filter by scenario
        resp_sc = self.client.get("/api/simulations?scenario=ransomware")
        self.assertEqual(resp_sc.status_code, 200)
        data_sc = resp_sc.json()
        for r in data_sc["simulations"]:
            self.assertIn("ransomware", r["scenario"].lower())

        # 2. Filter by policy
        resp_pol = self.client.get("/api/simulations?policy=atgp")
        self.assertEqual(resp_pol.status_code, 200)
        data_pol = resp_pol.json()
        for r in data_pol["simulations"]:
            self.assertIn("atgp", r["policy"].lower())

        # 3. Filter by status
        resp_st = self.client.get("/api/simulations?status=stopped")
        self.assertEqual(resp_st.status_code, 200)
        data_st = resp_st.json()
        for r in data_st["simulations"]:
            self.assertEqual(r["status"].lower(), "stopped")

        # 4. Filter by mode
        resp_m = self.client.get("/api/simulations?mode=replay")
        self.assertEqual(resp_m.status_code, 200)
        data_m = resp_m.json()
        for r in data_m["simulations"]:
            self.assertIn("replay", r.get("simulation_mode", "").lower())

    def test_simulation_detail_retrieval(self):
        """Verify GET /api/simulations/{run_id} returns complete trajectory and metadata."""
        response = self.client.get("/api/simulations/SIM-BENCH-PPO-01")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["id"], "SIM-BENCH-PPO-01")
        self.assertEqual(data["policy"], "PPO")
        self.assertEqual(data["checkpoint"], "v1.0-ppo-core (uncommitted)")
        self.assertEqual(data["encoder"], "SecBERT FP32 (EXP_20260727_001)")
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["simulation_mode"], "Historical Replay")
        self.assertEqual(data["steps_count"], 20)
        self.assertEqual(len(data["steps"]), 20)

        # Inspect first step schema
        step = data["steps"][0]
        self.assertIn("step", step)
        self.assertIn("incident_id", step)
        self.assertIn("action_name", step)
        self.assertIn("chosen_action", step)
        self.assertIn("ground_truth", step)
        self.assertIn("ground_truth_name", step)
        self.assertIn("is_correct", step)
        self.assertIn("reward", step)
        self.assertIn("cumulative_reward", step)
        self.assertIn("critic_value", step)
        self.assertIn("simulated_state", step)

    def test_simulation_detail_404(self):
        """Verify GET /api/simulations/{run_id} returns 404 for nonexistent run."""
        response = self.client.get("/api/simulations/NONEXISTENT-RUN-999")
        self.assertEqual(response.status_code, 404)

    def test_simulation_results_honesty(self):
        """Verify GET /api/simulations/{run_id}/results explicitly marks containment as Not Available."""
        response = self.client.get("/api/simulations/SIM-BENCH-PPO-01/results")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["run_id"], "SIM-BENCH-PPO-01")
        self.assertEqual(data["cumulative_reward"], 14.0)
        self.assertEqual(data["mean_reward"], 0.7)
        self.assertEqual(data["accuracy"], 0.85)
        # Scientific honesty check
        self.assertEqual(data["containment_outcome"], "Not Available")
        self.assertEqual(data["operational_disruption"], "Not Available")
        self.assertEqual(data["containment_latency_seconds"], "Not Available")

    def test_simulation_export_json(self):
        """Verify GET /api/simulations/{run_id}/export?format=json returns download attachment."""
        response = self.client.get("/api/simulations/SIM-BENCH-PPO-01/export?format=json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/json", response.headers.get("content-type", ""))
        self.assertIn("attachment", response.headers.get("content-disposition", ""))
        self.assertIn("SIM-BENCH-PPO-01.json", response.headers.get("content-disposition", ""))

        parsed = json.loads(response.text)
        self.assertEqual(parsed["id"], "SIM-BENCH-PPO-01")
        self.assertEqual(len(parsed["steps"]), 20)

    def test_simulation_export_csv(self):
        """Verify GET /api/simulations/{run_id}/export?format=csv returns CSV attachment."""
        response = self.client.get("/api/simulations/SIM-BENCH-PPO-01/export?format=csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers.get("content-type", ""))
        self.assertIn("attachment", response.headers.get("content-disposition", ""))
        self.assertIn("SIM-BENCH-PPO-01_trajectory.csv", response.headers.get("content-disposition", ""))

        lines = response.text.strip().split("\n")
        header = lines[0].strip()
        self.assertIn("step", header)
        self.assertIn("incident_id", header)
        self.assertIn("action_name", header)
        self.assertIn("reward", header)
        self.assertIn("cumulative_reward", header)
        self.assertEqual(len(lines), 21)  # 1 header + 20 step rows

    def test_simulation_compare_compatible(self):
        """Verify comparison between matched benchmark runs (PPO vs PPO+ATGP on same scenario/seed)."""
        response = self.client.get("/api/simulations/compare?run_a=SIM-BENCH-PPO-01&run_b=SIM-BENCH-ATGP-01")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["is_compatible"])
        self.assertEqual(data["compatibility_level"], "compatible")
        self.assertEqual(data["scenario_match"], True)
        self.assertEqual(data["seed_match"], True)
        self.assertEqual(data["step_limit_match"], True)

        # Delta metrics: Run B (ATGP: cum 16, acc 0.90) - Run A (PPO: cum 14, acc 0.85)
        self.assertAlmostEqual(data["delta"]["cum_reward_delta"], 2.0, places=2)
        self.assertAlmostEqual(data["delta"]["acc_delta"], 0.05, places=3)
        self.assertAlmostEqual(data["delta"]["mean_reward_delta"], 0.1, places=3)

        # Trajectory comparison
        self.assertEqual(len(data["trajectory_comparison"]), 20)
        step1 = data["trajectory_comparison"][0]
        self.assertEqual(step1["step"], 1)
        self.assertIn("reward_a", step1)
        self.assertIn("reward_b", step1)

        # Action distributions
        self.assertIn("action_frequencies", data)
        self.assertIn("run_a", data["action_frequencies"])
        self.assertIn("run_b", data["action_frequencies"])

    def test_simulation_compare_incompatible_scenario(self):
        """Verify comparison flags incompatible when scenarios differ."""
        response = self.client.get("/api/simulations/compare?run_a=SIM-BENCH-PPO-01&run_b=SIM-BENCH-STOPPED-01")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["is_compatible"])
        self.assertEqual(data["compatibility_level"], "incompatible")
        self.assertFalse(data["scenario_match"])
        self.assertIn("Incompatible Scenarios", data["scientific_notes"].get("notes", ""))

    def test_evaluations_metric_separation(self):
        """Verify clear separation between supervised classification and sequential simulation metrics."""
        response = self.client.get("/api/evaluations")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 1. Supervised Classification Evaluation on held-out test split
        self.assertIn("classification_metrics", data)
        cls_metrics = data["classification_metrics"]
        self.assertEqual(cls_metrics["held_out_samples"], 1539)
        self.assertAlmostEqual(cls_metrics["overall_accuracy"], 0.8993, places=3)
        self.assertAlmostEqual(cls_metrics["macro_f1"], 0.7121, places=3)
        self.assertAlmostEqual(cls_metrics["weighted_f1"], 0.9454, places=3)
        self.assertAlmostEqual(cls_metrics["mcc"], 0.9364, places=3)
        self.assertAlmostEqual(cls_metrics["top2_accuracy"], 0.9695, places=3)
        self.assertIn("confusion_matrix_sample", data)
        self.assertEqual(len(data["confusion_matrix_sample"]["labels"]), 8)

        # 2. Sequential Simulation Evaluation
        self.assertIn("sequential_simulation_metrics", data)
        seq_metrics = data["sequential_simulation_metrics"]
        self.assertGreaterEqual(seq_metrics["total_runs_evaluated"], 3)
        self.assertEqual(seq_metrics["containment_success_rate"], "Not Available")
        self.assertIn("mean_cumulative_return", seq_metrics)
        self.assertIn("median_cumulative_return", seq_metrics)


if __name__ == "__main__":
    unittest.main()
