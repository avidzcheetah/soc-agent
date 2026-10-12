# Responsibility: Normalized Incident Schema & Canonical Text Formatting for SecBERT
#
# Defines the IncidentRecord dataclass and IncidentFormatter for converting
# raw endpoint/SIEM telemetry into canonical text representations compatible
# with the frozen SecBERT encoder (768-dim state representations).

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class IncidentRecord:
    """
    Normalized incident representation combining host endpoint telemetry
    (e.g., OSQuery bpf_process_events) and SIEM alerts (e.g., Wazuh alerts).

    Fields are strictly grounded in observable telemetry:
      - Host & Time provenance
      - Process & execution hierarchy (path, cmdline, pid, ppid, cwd, uid, gid)
      - SIEM alert metadata (rule ID, level, description, MITRE IDs/tactics)
      - Operational CISA Phase (e.g., 'Detection')
    """
    incident_id: Optional[str] = None
    timestamp: Optional[str] = None
    host: Optional[str] = None
    source: str = "unknown"  # "osquery", "wazuh", "correlated", "synthetic", etc.

    # Process execution attributes
    process_path: Optional[str] = None
    process_name: Optional[str] = None
    cmdline: Optional[str] = None
    pid: Optional[int] = None
    parent_pid: Optional[int] = None
    uid: Optional[int] = None
    gid: Optional[int] = None
    username: Optional[str] = None
    cwd: Optional[str] = None
    syscall: Optional[str] = None

    # SIEM / Alert attributes
    alert_id: Optional[str] = None
    alert_level: Optional[int] = None
    rule_id: Optional[str] = None
    rule_description: Optional[str] = None
    rule_groups: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)

    # CISA Phase
    phase: Optional[str] = None

    # Raw telemetry provenance
    raw_events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record into a serializable dictionary."""
        return {
            "incident_id": self.incident_id,
            "timestamp": self.timestamp,
            "host": self.host,
            "source": self.source,
            "process_path": self.process_path,
            "process_name": self.process_name,
            "cmdline": self.cmdline,
            "pid": self.pid,
            "parent_pid": self.parent_pid,
            "uid": self.uid,
            "gid": self.gid,
            "username": self.username,
            "cwd": self.cwd,
            "syscall": self.syscall,
            "alert_id": self.alert_id,
            "alert_level": self.alert_level,
            "rule_id": self.rule_id,
            "rule_description": self.rule_description,
            "rule_groups": list(self.rule_groups),
            "mitre_techniques": list(self.mitre_techniques),
            "mitre_tactics": list(self.mitre_tactics),
            "phase": self.phase,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IncidentRecord":
        """Construct an IncidentRecord from a dictionary."""
        return cls(
            incident_id=data.get("incident_id"),
            timestamp=data.get("timestamp"),
            host=data.get("host"),
            source=data.get("source", "unknown"),
            process_path=data.get("process_path"),
            process_name=data.get("process_name"),
            cmdline=data.get("cmdline"),
            pid=data.get("pid"),
            parent_pid=data.get("parent_pid"),
            uid=data.get("uid"),
            gid=data.get("gid"),
            username=data.get("username"),
            cwd=data.get("cwd"),
            syscall=data.get("syscall"),
            alert_id=data.get("alert_id"),
            alert_level=data.get("alert_level"),
            rule_id=data.get("rule_id"),
            rule_description=data.get("rule_description"),
            rule_groups=list(data.get("rule_groups", [])),
            mitre_techniques=list(data.get("mitre_techniques", [])),
            mitre_tactics=list(data.get("mitre_tactics", [])),
            phase=data.get("phase"),
            raw_events=list(data.get("raw_events", [])),
            metadata=dict(data.get("metadata", {})),
        )


class IncidentFormatter:
    """
    Builds canonical incident text representation strictly from verified evidence.
    Ensures exact compatibility with SecBERT tokenization and CISA phase conventions.
    """

    @staticmethod
    def format_incident_text(
        record: IncidentRecord,
        default_phase: Optional[str] = "Detection",
    ) -> str:
        """
        Assemble canonical incident text for a given IncidentRecord.

        Rules:
          1. Does NOT invent unobserved MITRE techniques, severity levels, or targets.
          2. Preserves exact observed values (rule ID, level, command, PID, user, host).
          3. Employs CISA phase prefix '[<Phase>] ' when justified or requested.

        Args:
            record: The normalized IncidentRecord.
            default_phase: Fallback CISA phase if record.phase is not set.

        Returns:
            str: Canonical incident string formatted for SecBERT.
        """
        clauses: List[str] = []

        # 1. Primary event headline
        if record.rule_description:
            clauses.append(f"Alert: {record.rule_description.strip()}")
        elif record.process_path:
            clauses.append(f"Process execution: {record.process_path.strip()}")
        elif record.cmdline:
            clauses.append(f"Process execution: {record.cmdline.strip()}")
        else:
            clauses.append("Telemetry event recorded")

        # 2. Host
        if record.host:
            clauses.append(f"Host: {record.host.strip()}")

        # 3. SIEM Rule provenance
        if record.rule_id:
            level_suffix = f" (Level {record.alert_level})" if record.alert_level is not None else ""
            clauses.append(f"Rule: {record.rule_id}{level_suffix}")

        # 4. MITRE ATT&CK Attribution (only if present in evidence)
        mitre_items: List[str] = []
        if record.mitre_techniques:
            mitre_items.append(", ".join(record.mitre_techniques))
        if record.mitre_tactics:
            mitre_items.append(f"({', '.join(record.mitre_tactics)})")
        if mitre_items:
            clauses.append(f"MITRE: {' '.join(mitre_items)}")

        # 5. Process execution path & commandline details
        # If headline was rule alert, show process path explicitly
        if record.rule_description and record.process_path:
            clauses.append(f"Process: {record.process_path.strip()}")

        # Show cmdline if distinct from headline / path
        if record.cmdline:
            cmd = record.cmdline.strip()
            # If path and cmdline are identical, avoid redundant repetition
            if record.process_path != cmd and not (record.rule_description is None and clauses[0].endswith(cmd)):
                clauses.append(f"Command: '{cmd}'")

        # 6. PID and Parent PID
        pid_items: List[str] = []
        if record.pid is not None:
            pid_items.append(f"PID: {record.pid}")
        if record.parent_pid is not None:
            pid_items.append(f"PPID: {record.parent_pid}")
        if pid_items:
            clauses.append(", ".join(pid_items))

        # 7. Identity & Privilege
        if record.username:
            clauses.append(f"User: {record.username.strip()}")
        elif record.uid is not None:
            clauses.append(f"UID: {record.uid}")

        # 8. Working Directory
        if record.cwd:
            clauses.append(f"CWD: {record.cwd.strip()}")

        # Assemble clauses with clean punctuation
        body = ". ".join(c.rstrip(".") for c in clauses if c)
        if not body.endswith("."):
            body += "."

        # CISA Phase prefix
        phase = record.phase or default_phase
        if phase:
            return f"[{phase}] {body}"
        return body

    @classmethod
    def to_secbert_input(
        cls,
        record: IncidentRecord,
        default_phase: Optional[str] = "Detection",
    ) -> str:
        """
        Alias for format_incident_text, guaranteeing compatibility with
        SecBERTStateEncoder.encode_incident().
        """
        return cls.format_incident_text(record, default_phase=default_phase)
