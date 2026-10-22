"""
Unit tests for the SOC Simulation Environment FastAPI Backend.
Verifies response codes, schemas, data integrity, and error states.
"""

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
GRANDPARENT_DIR = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))

for p in [PARENT_DIR, GRANDPARENT_DIR, os.path.join(PARENT_DIR, "web"), os.path.join(GRANDPARENT_DIR, "web")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from api.server import app


class TestSOCApiServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["action_space_size"], 20)
        self.assertEqual(data["total_benchmark_records"], 15390)

    def test_system_status_endpoint(self):
        response = self.client.get("/api/system/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("infrastructure", data)
        self.assertIn("data_pipeline", data)
        self.assertIn("ai_components", data)
        self.assertEqual(data["infrastructure"]["wazuh_manager"]["status"], "planned")
        self.assertEqual(data["ai_components"]["secbert"]["embedding_dim"], 768)

    def test_telemetry_sources_endpoint(self):
        response = self.client.get("/api/telemetry/sources")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        sources = data["sources"]
        self.assertEqual(len(sources), 5)
        names = [s["name"] for s in sources]
        self.assertIn("Wazuh SIEM", names)
        self.assertIn("OSQuery", names)
        self.assertIn("MITRE Caldera", names)
        self.assertIn("CISSM Cyber Events", names)

    def test_telemetry_events_pagination(self):
        response = self.client.get("/api/telemetry/events?page=1&limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["events"]), 10)
        self.assertGreater(data["total"], 50)
        event = data["events"][0]
        self.assertIn("id", event)
        self.assertIn("action_label", event)
        self.assertIn("cisa_phase", event)

    def test_incidents_detail(self):
        # Fetch first incident ID
        list_resp = self.client.get("/api/incidents?page=1&limit=1")
        self.assertEqual(list_resp.status_code, 200)
        inc_id = list_resp.json()["incidents"][0]["id"]

        detail_resp = self.client.get(f"/api/incidents/{inc_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()
        self.assertEqual(detail["id"], inc_id)
        self.assertIn("prepared_text", detail)

    def test_incident_not_found(self):
        response = self.client.get("/api/incidents/NON_EXISTENT_ID")
        self.assertEqual(response.status_code, 404)

    def test_decisions_list_and_detail(self):
        list_resp = self.client.get("/api/decisions?page=1&limit=5")
        self.assertEqual(list_resp.status_code, 200)
        data = list_resp.json()
        self.assertEqual(len(data["decisions"]), 5)
        dec_id = data["decisions"][0]["id"]

        detail_resp = self.client.get(f"/api/decisions/{dec_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()
        self.assertEqual(detail["id"], dec_id)
        self.assertEqual(detail["embedding_dim"], 768)
        self.assertEqual(len(detail["action_probabilities"]), 20)
        self.assertIn("critic_value", detail)
        self.assertIn("reward", detail)

    def test_simulations_list_and_detail(self):
        list_resp = self.client.get("/api/simulations")
        self.assertEqual(list_resp.status_code, 200)
        sims = list_resp.json()["simulations"]
        self.assertGreaterEqual(len(sims), 2)
        hist_run = next((s for s in sims if s["id"].startswith("SIM-RUN-")), sims[-1])
        run_id = hist_run["id"]

        detail_resp = self.client.get(f"/api/simulations/{run_id}")
        self.assertEqual(detail_resp.status_code, 200)
        detail = detail_resp.json()
        self.assertIn("steps", detail)
        self.assertGreater(len(detail["steps"]), 0)
        step1 = detail["steps"][0]
        self.assertIn("reward", step1)
        self.assertIn("cumulative_reward", step1)

    def test_run_simulation_post(self):
        payload = {"scenario": "Safe-Unit-Test-Run", "steps": 10, "seed": 42}
        response = self.client.post("/api/simulations/run", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["steps_count"], 10)
        self.assertIn("final_cumulative_reward", data)

    def test_evaluations_endpoint(self):
        response = self.client.get("/api/evaluations")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("architectures", data)
        self.assertIn("operational_metrics", data)
        self.assertIn("per_class_metrics", data)
        self.assertEqual(len(data["per_class_metrics"]), 20)
        
        # Verify locked checkpoint metrics match EXP_20260727_001
        secbert = next(a for a in data["architectures"] if a["id"] == "secbert_standalone")
        self.assertEqual(secbert["macro_f1"], 0.7121)
        self.assertEqual(secbert["weighted_f1"], 0.9454)
        self.assertEqual(secbert["mcc"], 0.9364)
        self.assertEqual(secbert["top2_accuracy"], 0.9695)

    def test_research_experiments_endpoint(self):
        response = self.client.get("/api/research/experiments")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("phase1_secbert", data)
        self.assertEqual(len(data["phase1_secbert"]), 9)
        # Check negative results preservation
        exp_002a = next(e for e in data["phase1_secbert"] if e["logical_name"] == "EXP_002A")
        self.assertIn("Negative Result", exp_002a["verdict"])

    # ---------------------------------------------------------
    # Step 2: Simulation Execution & API Lifecycle Tests
    # ---------------------------------------------------------

    def test_scenarios_catalog_endpoint(self):
        response = self.client.get("/api/scenarios")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("scenarios", data)
        self.assertEqual(len(data["scenarios"]), 3)
        ids = [s["id"] for s in data["scenarios"]]
        self.assertIn("scenario-ransomware", ids)
        self.assertIn("scenario-c2", ids)
        self.assertIn("scenario-vulnerability", ids)

    def test_create_simulation_success(self):
        payload = {
            "scenario": "scenario-ransomware",
            "policy": "PPO",
            "target": "Subnet 10.0.1.0/24 (Corporate Workstations)",
            "step_limit": 10,
            "seed": 42
        }
        response = self.client.post("/api/simulations", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("run_id", data)
        self.assertEqual(data["scenario_id"], "scenario-ransomware")
        self.assertEqual(data["policy"], "PPO")
        self.assertEqual(data["step_limit"], 10)
        self.assertEqual(data["status"], "running")

    def test_create_simulation_invalid_scenario_reject(self):
        payload = {
            "scenario": "non-existent-scenario-xyz",
            "policy": "PPO"
        }
        response = self.client.post("/api/simulations", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid scenario", response.json()["detail"])

    def test_create_simulation_invalid_policy_reject(self):
        payload = {
            "scenario": "scenario-a",
            "policy": "DQN_UNSUPPORTED"
        }
        response = self.client.post("/api/simulations", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid policy", response.json()["detail"])

    def test_create_simulation_invalid_steps_reject(self):
        payload = {
            "scenario": "scenario-a",
            "policy": "PPO",
            "step_limit": 999
        }
        response = self.client.post("/api/simulations", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("step_limit", response.json()["detail"])

    def test_stop_simulation_endpoint(self):
        # Create a run with 40 steps
        create_resp = self.client.post("/api/simulations", json={
            "scenario": "scenario-c",
            "policy": "PPO+ATGP",
            "step_limit": 40,
            "seed": 99
        })
        self.assertEqual(create_resp.status_code, 200)
        run_id = create_resp.json()["run_id"]

        # Signal stop
        stop_resp = self.client.post(f"/api/simulations/{run_id}/stop")
        self.assertEqual(stop_resp.status_code, 200)
        stop_data = stop_resp.json()
        self.assertEqual(stop_data["run_id"], run_id)
        self.assertEqual(stop_data["status"], "stopping")

    def test_simulation_results_lifecycle(self):
        create_resp = self.client.post("/api/simulations", json={
            "scenario": "scenario-b",
            "policy": "PPO",
            "step_limit": 5,
            "seed": 77
        })
        self.assertEqual(create_resp.status_code, 200)
        run_id = create_resp.json()["run_id"]

        # Poll results endpoint
        res_resp = self.client.get(f"/api/simulations/{run_id}/results")
        self.assertEqual(res_resp.status_code, 200)
        res_data = res_resp.json()
        self.assertIn("status", res_data)

    def test_reset_simulations_endpoint(self):
        response = self.client.post("/api/simulations/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
