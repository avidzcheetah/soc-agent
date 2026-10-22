# Responsibility: Deterministic Observable Telemetry Feature Extraction for Experiment 003
#
# Extracts a fixed-dimension, deterministic, normalized numerical feature vector
# derived strictly from observable IncidentRecord fields (process attributes,
# SIEM alerts, MITRE tactic tags).
#
# Constraints:
#   - Does NOT use is_benign, scenario_id, stage_id, hidden simulator state,
#     ground-truth labels, or action effectiveness metadata.
#   - Does NOT invent host roles, privilege levels, or network topology indicators
#     absent from genuine telemetry.
#   - Explicitly handles missing/None fields with dedicated presence indicator bits.
#   - Feature values are strictly finite and normalized in [0.0, 1.0].

from typing import Dict, List, Optional, Union, Any
import numpy as np
import logging

from src.telemetry.incident_formatter import IncidentRecord

logger = logging.getLogger(__name__)

# Fixed list of observable process binaries recognized in endpoint telemetry
OBSERVED_PROCESS_BINARIES: List[str] = [
    "sudo",
    "crontab",
    "sshd",
    "rsync",
    "nmap",
    "whoami",
    "curl",
    "mysqldump",
    "node",
    "python",
]

# Fixed list of MITRE Enterprise tactics recognized in SIEM alert telemetry
OBSERVED_MITRE_TACTICS: List[str] = [
    "initial access",
    "persistence",
    "privilege escalation",
    "defense evasion",
    "credential access",
    "discovery",
    "lateral movement",
    "collection",
    "command and control",
    "exfiltration",
]

# Canonical, deterministic feature names defining the 34-dimensional schema
TELEMETRY_FEATURE_SCHEMA: List[str] = (
    [
        "alert_level_norm",      # Index 0: Alert level normalized by 15.0 in [0, 1]
        "has_alert_level",       # Index 1: Presence bit for alert_level
        "uid_norm",              # Index 2: UID normalized by 2000.0 in [0, 1] (capped at 1.0)
        "is_uid_zero",           # Index 3: Binary indicator for UID == 0 (root)
        "has_uid",               # Index 4: Presence bit for UID
        "has_parent_pid",        # Index 5: Presence bit for PPID
        "has_cwd",               # Index 6: Presence bit for current working directory
        "is_cwd_tmp",            # Index 7: CWD located in /tmp, /dev/shm, or /var/tmp
        "is_cmdline_tmp",        # Index 8: Commandline referencing /tmp or /dev/shm
        "is_network_cmdline",    # Index 9: Observable network indicators in command line
    ]
    + [f"proc_{bin_name}" for bin_name in OBSERVED_PROCESS_BINARIES]  # Indices 10-19 (10 dims)
    + ["has_mitre"]                                                    # Index 20
    + [f"tactic_{t.replace(' ', '_')}" for t in OBSERVED_MITRE_TACTICS]  # Indices 21-30 (10 dims)
    + [
        "has_username",          # Index 31: Presence bit for username
        "user_is_root",          # Index 32: Username is 'root' or UID is 0
        "user_is_service",       # Index 33: Username indicates service/daemon/audit account
    ]
)

TELEMETRY_FEATURE_DIM: int = len(TELEMETRY_FEATURE_SCHEMA)  # Exactly 34


class TelemetryFeatureExtractor:
    """
    Deterministic feature extractor transforming observable incident telemetry
    into a standardized 34-dimensional feature vector.
    """

    def __init__(self) -> None:
        self.feature_names = list(TELEMETRY_FEATURE_SCHEMA)
        self.feature_dim = len(self.feature_names)
        assert self.feature_dim == 34, f"Expected 34 features, got {self.feature_dim}"

    def extract(self, record: Optional[Union[IncidentRecord, Dict[str, Any]]]) -> np.ndarray:
        """
        Extract a 34-dimensional float32 feature vector from an IncidentRecord
        or telemetry dictionary.

        Handles missing fields gracefully without inventing unobserved facts.

        Args:
            record: IncidentRecord or dictionary of observable telemetry fields.

        Returns:
            np.ndarray: 1D array of shape (34,), dtype np.float32, values in [0.0, 1.0].
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        if record is None:
            return features

        # Normalize access whether record is IncidentRecord or dict
        if isinstance(record, IncidentRecord):
            rec_dict = record.to_dict()
            mitre_tactics = [t.lower() for t in record.mitre_tactics] if record.mitre_tactics else []
            mitre_techniques = list(record.mitre_techniques) if record.mitre_techniques else []
        elif isinstance(record, dict):
            rec_dict = record
            raw_tactics = record.get("mitre_tactics", [])
            mitre_tactics = [t.lower() for t in raw_tactics] if raw_tactics else []
            mitre_techniques = list(record.get("mitre_techniques", []))
        else:
            return features

        # ── 1. Numerical & Presence Telemetry (Indices 0..5) ───
        alert_level = rec_dict.get("alert_level")
        if alert_level is not None and isinstance(alert_level, (int, float)):
            features[0] = float(np.clip(float(alert_level) / 15.0, 0.0, 1.0))
            features[1] = 1.0

        uid = rec_dict.get("uid")
        if uid is not None and isinstance(uid, (int, float)):
            features[4] = 1.0
            if uid == 0:
                features[3] = 1.0
                features[2] = 0.0
            elif uid > 0:
                features[2] = float(np.clip(float(uid) / 2000.0, 0.0, 1.0))

        parent_pid = rec_dict.get("parent_pid")
        if parent_pid is not None and isinstance(parent_pid, int):
            features[5] = 1.0

        # ── 2. Execution Context & Paths (Indices 6..9) ────────
        cwd = rec_dict.get("cwd")
        if cwd and isinstance(cwd, str):
            features[6] = 1.0
            cwd_lower = cwd.lower()
            if any(tmp_path in cwd_lower for tmp_path in ["/tmp", "/dev/shm", "/var/tmp"]):
                features[7] = 1.0

        cmdline = rec_dict.get("cmdline")
        if cmdline and isinstance(cmdline, str):
            cmd_lower = cmdline.lower()
            if any(tmp_path in cmd_lower for tmp_path in ["/tmp", "/dev/shm", "/var/tmp"]):
                features[8] = 1.0
            if any(net_token in cmd_lower for net_token in [
                "http://", "https://", "198.51.", "10.0.", ":443", ":8080", "--c2", "remote@"
            ]):
                features[9] = 1.0

        # ── 3. Observed Process Binaries (Indices 10..19) ──────
        proc_name = (rec_dict.get("process_name") or "").lower()
        proc_path = (rec_dict.get("process_path") or "").lower()

        for idx, bin_name in enumerate(OBSERVED_PROCESS_BINARIES):
            feat_idx = 10 + idx
            if bin_name == proc_name or (proc_path and bin_name in proc_path):
                features[feat_idx] = 1.0

        # ── 4. Observed MITRE Tactics (Indices 20..30) ─────────
        if mitre_tactics or mitre_techniques:
            features[20] = 1.0

        for idx, tactic in enumerate(OBSERVED_MITRE_TACTICS):
            feat_idx = 21 + idx
            if any(tactic in observed_tactic for observed_tactic in mitre_tactics):
                features[feat_idx] = 1.0

        # ── 5. User & Identity Context (Indices 31..33) ────────
        username = rec_dict.get("username")
        if username and isinstance(username, str):
            features[31] = 1.0
            user_lower = username.lower()
            if user_lower == "root" or (uid is not None and uid == 0):
                features[32] = 1.0
            if any(svc in user_lower for svc in ["service", "admin", "data", "audit", "runner", "ci"]):
                features[33] = 1.0
        elif uid is not None and uid == 0:
            features[32] = 1.0

        return features
