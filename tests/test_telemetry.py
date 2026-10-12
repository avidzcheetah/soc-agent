# Responsibility: Comprehensive Unit & Integration Tests for Telemetry Processing
#
# Covers:
#   1. Parsing real observed OSQuery bpf_process_events records.
#   2. Parsing real observed Wazuh SIEM alerts.
#   3. Multi-source telemetry correlation.
#   4. Robust handling of missing fields, malformed JSON, and empty inputs.
#   5. Canonical SecBERT text formatting and CISA phase conventions.
#   6. Absence of fabricated evidence (no invented MITRE, severity, or phases).
#   7. Live SecBERTStateEncoder embedding generation (shape (768,), finite values).

import json
import os
import sys
import unittest
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.telemetry.incident_formatter import IncidentRecord, IncidentFormatter
from src.telemetry.aggregator import TelemetryAggregator
from src.encoder import SecBERTStateEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Real Observed Fixtures (captured from live testbed logs)
# ─────────────────────────────────────────────────────────────────────────────

REAL_OSQUERY_WHOAMI_EVENT: Dict[str, Any] = {
    "name": "bpf_process_events",
    "hostIdentifier": "meow",
    "calendarTime": "Mon Oct  5 08:07:30 2026 UTC",
    "unixTime": 1791187650,
    "epoch": 0,
    "counter": 394,
    "numerics": False,
    "columns": {
        "cid": "12099",
        "cmdline": "whoami",
        "cwd": "/home/avidz/Research/soc-agent",
        "duration": "316406",
        "exit_code": "0",
        "gid": "1000",
        "ntime": "13924080890792",
        "parent": "99385",
        "path": "/usr/bin/whoami",
        "pid": "99386",
        "probe_error": "0",
        "syscall": "exec",
        "tid": "99386",
        "uid": "1000"
    },
    "action": "added"
}

REAL_WAZUH_SUDO_ALERT: Dict[str, Any] = {
    "timestamp": "2026-10-12T07:17:40.464+0000",
    "rule": {
        "level": 3,
        "description": "Successful sudo to ROOT executed.",
        "id": "5402",
        "mitre": {
            "id": ["T1548.003"],
            "tactic": ["Privilege Escalation", "Defense Evasion"],
            "technique": ["Sudo and Sudo Caching"]
        },
        "firedtimes": 2,
        "mail": False,
        "groups": ["syslog", "sudo"]
    },
    "agent": {
        "id": "001",
        "name": "meow",
        "ip": "127.0.0.1"
    },
    "manager": {"name": "wazuh.manager"},
    "id": "1791789460.389648",
    "full_log": "Oct 12 07:17:38 meow sudo[63214]:    avidz : a password is required ; TTY=pts/3 ; PWD=/home/avidz/Research/soc-agent ; USER=root ; COMMAND=/usr/bin/true",
    "data": {
        "srcuser": "avidz",
        "dstuser": "root",
        "tty": "pts/3",
        "pwd": "/home/avidz/Research/soc-agent",
        "command": "/usr/bin/true"
    },
    "location": "journald"
}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Test Fixtures (clearly labeled for unit test boundary cases)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_MINIMAL_OSQUERY_EVENT: Dict[str, Any] = {
    # Synthetic: contains only bare minimum process execution fields
    "columns": {
        "path": "/opt/custom_agent",
        "pid": 5512
    }
}

SYNTHETIC_MINIMAL_WAZUH_ALERT: Dict[str, Any] = {
    # Synthetic: alert without MITRE mappings, host info, or data blocks
    "id": "synthetic_alert_001",
    "timestamp": "2026-10-12T12:00:00Z",
    "rule": {
        "id": "100001",
        "level": 7,
        "description": "Suspicious script invocation detected"
    }
}


class TestTelemetryAggregator(unittest.TestCase):

    def test_parse_real_osquery_process_event(self):
        """Verify parsing of real observed OSQuery bpf_process_events record."""
        rec = TelemetryAggregator.parse_osquery_process_event(REAL_OSQUERY_WHOAMI_EVENT)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.source, "osquery")
        self.assertEqual(rec.host, "meow")
        self.assertEqual(rec.pid, 99386)
        self.assertEqual(rec.parent_pid, 99385)
        self.assertEqual(rec.uid, 1000)
        self.assertEqual(rec.gid, 1000)
        self.assertEqual(rec.process_path, "/usr/bin/whoami")
        self.assertEqual(rec.process_name, "whoami")
        self.assertEqual(rec.cmdline, "whoami")
        self.assertEqual(rec.cwd, "/home/avidz/Research/soc-agent")
        self.assertEqual(rec.syscall, "exec")
        self.assertEqual(rec.phase, "Detection")
        self.assertEqual(len(rec.raw_events), 1)

    def test_parse_real_osquery_json_string(self):
        """Verify parsing of real OSQuery record when passed as JSON string."""
        raw_str = json.dumps(REAL_OSQUERY_WHOAMI_EVENT)
        rec = TelemetryAggregator.parse_osquery_process_event(raw_str)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.pid, 99386)
        self.assertEqual(rec.host, "meow")

    def test_parse_real_wazuh_alert(self):
        """Verify parsing of real observed Wazuh alerts.json entry."""
        rec = TelemetryAggregator.parse_wazuh_alert(REAL_WAZUH_SUDO_ALERT)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.source, "wazuh")
        self.assertEqual(rec.alert_id, "1791789460.389648")
        self.assertEqual(rec.host, "meow")
        self.assertEqual(rec.rule_id, "5402")
        self.assertEqual(rec.alert_level, 3)
        self.assertEqual(rec.rule_description, "Successful sudo to ROOT executed.")
        self.assertIn("T1548.003", rec.mitre_techniques)
        self.assertIn("Privilege Escalation", rec.mitre_tactics)
        self.assertEqual(rec.cmdline, "/usr/bin/true")
        self.assertEqual(rec.username, "root")
        self.assertEqual(rec.cwd, "/home/avidz/Research/soc-agent")
        self.assertEqual(rec.pid, 63214)  # Extracted from sudo[63214] syslog header
        self.assertEqual(rec.phase, "Detection")

    def test_correlation_wazuh_and_osquery(self):
        """Verify multi-source correlation produces a unified, enriched IncidentRecord."""
        w_rec = TelemetryAggregator.parse_wazuh_alert(REAL_WAZUH_SUDO_ALERT)
        o_rec = TelemetryAggregator.parse_osquery_process_event(REAL_OSQUERY_WHOAMI_EVENT)

        correlated = TelemetryAggregator.correlate(wazuh_alert=w_rec, osquery_event=o_rec)
        self.assertIsNotNone(correlated)
        self.assertEqual(correlated.source, "correlated")
        self.assertEqual(correlated.host, "meow")
        # Low-level process attributes prioritized from OSQuery
        self.assertEqual(correlated.pid, 99386)
        self.assertEqual(correlated.parent_pid, 99385)
        self.assertEqual(correlated.process_path, "/usr/bin/whoami")
        # SIEM metadata preserved from Wazuh
        self.assertEqual(correlated.rule_id, "5402")
        self.assertEqual(correlated.alert_level, 3)
        self.assertIn("T1548.003", correlated.mitre_techniques)
        self.assertEqual(len(correlated.raw_events), 2)

    def test_missing_optional_fields_synthetic(self):
        """Verify graceful fallback for synthetic records with minimal fields."""
        rec_o = TelemetryAggregator.parse_osquery_process_event(SYNTHETIC_MINIMAL_OSQUERY_EVENT)
        self.assertIsNotNone(rec_o)
        self.assertEqual(rec_o.pid, 5512)
        self.assertIsNone(rec_o.parent_pid)
        self.assertIsNone(rec_o.uid)
        self.assertIsNone(rec_o.host)

        rec_w = TelemetryAggregator.parse_wazuh_alert(SYNTHETIC_MINIMAL_WAZUH_ALERT)
        self.assertIsNotNone(rec_w)
        self.assertEqual(rec_w.rule_id, "100001")
        self.assertEqual(rec_w.alert_level, 7)
        self.assertEqual(rec_w.mitre_techniques, [])
        self.assertIsNone(rec_w.host)

    def test_malformed_and_empty_inputs(self):
        """Verify robust non-crashing behavior for malformed and empty inputs."""
        # Malformed JSON string
        self.assertIsNone(TelemetryAggregator.parse_osquery_process_event("NOT_VALID_JSON{{{"))
        self.assertIsNone(TelemetryAggregator.parse_wazuh_alert("{{bad:json"))

        # Empty string / None / empty dict
        self.assertIsNone(TelemetryAggregator.parse_osquery_process_event(""))
        self.assertIsNone(TelemetryAggregator.parse_osquery_process_event({}))
        self.assertIsNone(TelemetryAggregator.parse_osquery_process_event(None))
        self.assertIsNone(TelemetryAggregator.parse_wazuh_alert(""))
        self.assertIsNone(TelemetryAggregator.parse_wazuh_alert({}))
        self.assertIsNone(TelemetryAggregator.parse_wazuh_alert(None))

        # Correlate with empty inputs
        self.assertIsNone(TelemetryAggregator.correlate(None, None))


class TestIncidentFormatter(unittest.TestCase):

    def test_format_real_osquery_record(self):
        """Verify canonical incident text for real OSQuery process execution."""
        rec = TelemetryAggregator.parse_osquery_process_event(REAL_OSQUERY_WHOAMI_EVENT)
        text = IncidentFormatter.format_incident_text(rec)

        self.assertTrue(text.startswith("[Detection]"))
        self.assertIn("Process execution: /usr/bin/whoami", text)
        self.assertIn("Host: meow", text)
        self.assertIn("PID: 99386, PPID: 99385", text)
        self.assertIn("UID: 1000", text)
        self.assertIn("CWD: /home/avidz/Research/soc-agent", text)
        # Verify no unobserved MITRE or rule text is fabricated
        self.assertNotIn("MITRE:", text)
        self.assertNotIn("Rule:", text)

    def test_format_real_wazuh_record(self):
        """Verify canonical incident text for real Wazuh alert."""
        rec = TelemetryAggregator.parse_wazuh_alert(REAL_WAZUH_SUDO_ALERT)
        text = IncidentFormatter.format_incident_text(rec)

        self.assertTrue(text.startswith("[Detection]"))
        self.assertIn("Alert: Successful sudo to ROOT executed.", text)
        self.assertIn("Host: meow", text)
        self.assertIn("Rule: 5402 (Level 3)", text)
        self.assertIn("MITRE: T1548.003 (Privilege Escalation, Defense Evasion)", text)
        self.assertIn("Process: /usr/bin/true", text)
        self.assertIn("PID: 63214", text)
        self.assertIn("User: root", text)

    def test_format_does_not_invent_evidence(self):
        """Verify that missing fields are omitted rather than fabricated."""
        synthetic_record = IncidentRecord(
            source="synthetic",
            process_path="/usr/bin/curl",
            phase=None,  # No phase set
        )
        text_no_phase = IncidentFormatter.format_incident_text(synthetic_record, default_phase=None)
        # Should not have [Phase] prefix
        self.assertFalse(text_no_phase.startswith("["))
        self.assertEqual(text_no_phase, "Process execution: /usr/bin/curl.")
        # No fabricated MITRE, host, PID, or rule
        self.assertNotIn("MITRE", text_no_phase)
        self.assertNotIn("Host", text_no_phase)
        self.assertNotIn("PID", text_no_phase)
        self.assertNotIn("Rule", text_no_phase)


class TestSecBERTIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load frozen SecBERT encoder once for test verification."""
        cls.encoder = SecBERTStateEncoder()

    def test_encode_osquery_formatted_text(self):
        """Verify that OSQuery formatted text generates a valid 768-dim finite embedding."""
        rec = TelemetryAggregator.parse_osquery_process_event(REAL_OSQUERY_WHOAMI_EVENT)
        text = IncidentFormatter.format_incident_text(rec)

        embedding = self.encoder.encode_incident(text)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, torch.Size([768]))
        self.assertTrue(torch.isfinite(embedding).all())

    def test_encode_wazuh_formatted_text(self):
        """Verify that Wazuh formatted text generates a valid 768-dim finite embedding."""
        rec = TelemetryAggregator.parse_wazuh_alert(REAL_WAZUH_SUDO_ALERT)
        text = IncidentFormatter.format_incident_text(rec)

        embedding = self.encoder.encode_incident(text)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, torch.Size([768]))
        self.assertTrue(torch.isfinite(embedding).all())

    def test_encode_correlated_formatted_text(self):
        """Verify that correlated incident text generates a valid 768-dim finite embedding."""
        w_rec = TelemetryAggregator.parse_wazuh_alert(REAL_WAZUH_SUDO_ALERT)
        o_rec = TelemetryAggregator.parse_osquery_process_event(REAL_OSQUERY_WHOAMI_EVENT)
        correlated = TelemetryAggregator.correlate(w_rec, o_rec)
        text = IncidentFormatter.format_incident_text(correlated)

        embedding = self.encoder.encode_incident(text)
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, torch.Size([768]))
        self.assertTrue(torch.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
