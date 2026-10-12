# Responsibility: Telemetry Aggregator for OSQuery and Wazuh Pipelines
#
# Parses raw OSQuery bpf_process_events and Wazuh alerts, extracts and normalizes
# key operational telemetry fields, performs correlation between alerts and endpoint
# process executions, and handles partial/malformed inputs safely.

import json
import os
import re
from typing import Any, Dict, List, Optional, Union
import logging

from src.telemetry.incident_formatter import IncidentRecord

logger = logging.getLogger(__name__)


def _safe_int(val: Any) -> Optional[int]:
    """Safely convert a value to an integer, returning None if conversion fails."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


class TelemetryAggregator:
    """
    Ingests and aggregates telemetry records from OSQuery daemon logs and Wazuh SIEM alerts.
    """

    @staticmethod
    def parse_osquery_process_event(
        raw_event: Union[str, Dict[str, Any]]
    ) -> Optional[IncidentRecord]:
        """
        Parse an OSQuery bpf_process_events record.

        Accepts either:
          - A JSON string representing an event log line.
          - A pre-parsed Python dictionary (either standard osquery format with 'columns'
            or flat column dictionary).

        Returns:
            IncidentRecord if parsing succeeds, or None if input is empty/invalid.
        """
        if not raw_event:
            return None

        data: Dict[str, Any] = {}
        if isinstance(raw_event, str):
            clean_str = raw_event.strip()
            if not clean_str:
                return None
            try:
                data = json.loads(clean_str)
            except (json.JSONDecodeError, ValueError) as err:
                logger.warning("Failed to decode OSQuery JSON: %s", err)
                return None
        elif isinstance(raw_event, dict):
            data = raw_event
        else:
            logger.warning("Unsupported OSQuery input type: %s", type(raw_event))
            return None

        # Check for nested 'columns' dictionary (standard osquery results log)
        cols: Dict[str, Any] = data.get("columns", data) if isinstance(data.get("columns"), dict) else data

        pid = _safe_int(cols.get("pid"))
        ppid = _safe_int(cols.get("parent") or cols.get("ppid"))
        uid = _safe_int(cols.get("uid"))
        gid = _safe_int(cols.get("gid"))

        path = cols.get("path")
        cmdline = cols.get("cmdline")
        cwd = cols.get("cwd")
        syscall = cols.get("syscall")
        proc_name = os.path.basename(path) if path else None

        host = data.get("hostIdentifier") or data.get("host")
        timestamp = data.get("calendarTime") or (str(data.get("unixTime")) if data.get("unixTime") else None)

        record = IncidentRecord(
            incident_id=f"osquery_{pid}_{data.get('unixTime', '')}" if pid else None,
            timestamp=str(timestamp) if timestamp else None,
            host=str(host) if host else None,
            source="osquery",
            process_path=str(path) if path else None,
            process_name=proc_name,
            cmdline=str(cmdline) if cmdline else None,
            pid=pid,
            parent_pid=ppid,
            uid=uid,
            gid=gid,
            cwd=str(cwd) if cwd else None,
            syscall=str(syscall) if syscall else None,
            phase="Detection",
            raw_events=[data],
        )

        return record

    @staticmethod
    def parse_wazuh_alert(
        raw_alert: Union[str, Dict[str, Any]]
    ) -> Optional[IncidentRecord]:
        """
        Parse a Wazuh SIEM alert record.

        Accepts either:
          - A JSON string representing a line from alerts.json.
          - A pre-parsed Python dictionary matching Wazuh alert schema.

        Returns:
            IncidentRecord if parsing succeeds, or None if input is empty/invalid.
        """
        if not raw_alert:
            return None

        data: Dict[str, Any] = {}
        if isinstance(raw_alert, str):
            clean_str = raw_alert.strip()
            if not clean_str:
                return None
            try:
                data = json.loads(clean_str)
            except (json.JSONDecodeError, ValueError) as err:
                logger.warning("Failed to decode Wazuh JSON: %s", err)
                return None
        elif isinstance(raw_alert, dict):
            data = raw_alert
        else:
            logger.warning("Unsupported Wazuh input type: %s", type(raw_alert))
            return None

        alert_id = data.get("id")
        timestamp = data.get("timestamp")

        # Host identification
        agent_info = data.get("agent")
        host = None
        if isinstance(agent_info, dict):
            host = agent_info.get("name") or agent_info.get("id") or agent_info.get("ip")

        # Rule extraction
        rule = data.get("rule", {})
        rule_id = str(rule.get("id")) if rule.get("id") is not None else None
        rule_level = _safe_int(rule.get("level"))
        rule_desc = rule.get("description")
        rule_groups = [str(g) for g in rule.get("groups", [])] if isinstance(rule.get("groups"), list) else []

        # MITRE ATT&CK extraction
        mitre_info = rule.get("mitre", {})
        mitre_techs = []
        mitre_tactics = []
        if isinstance(mitre_info, dict):
            raw_techs = mitre_info.get("id", [])
            if isinstance(raw_techs, list):
                mitre_techs = [str(t) for t in raw_techs]
            elif isinstance(raw_techs, str):
                mitre_techs = [raw_techs]

            raw_tactics = mitre_info.get("tactic", [])
            if isinstance(raw_tactics, list):
                mitre_tactics = [str(t) for t in raw_tactics]
            elif isinstance(raw_tactics, str):
                mitre_tactics = [raw_tactics]

        # Data block extraction
        data_block = data.get("data", {})
        cmdline = None
        cwd = None
        username = None
        if isinstance(data_block, dict):
            cmdline = data_block.get("command")
            cwd = data_block.get("pwd")
            username = data_block.get("dstuser") or data_block.get("srcuser")

        # Extract PID from full_log or syslog headers if available (e.g., sudo[63214])
        pid = None
        full_log = data.get("full_log", "")
        if isinstance(full_log, str) and full_log:
            pid_match = re.search(r"\[(\d{1,7})\]:", full_log)
            if pid_match:
                pid = _safe_int(pid_match.group(1))

        record = IncidentRecord(
            incident_id=str(alert_id) if alert_id else None,
            timestamp=str(timestamp) if timestamp else None,
            host=str(host) if host else None,
            source="wazuh",
            process_path=str(cmdline) if cmdline else None,
            cmdline=str(cmdline) if cmdline else None,
            pid=pid,
            username=str(username) if username else None,
            cwd=str(cwd) if cwd else None,
            alert_id=str(alert_id) if alert_id else None,
            alert_level=rule_level,
            rule_id=rule_id,
            rule_description=str(rule_desc) if rule_desc else None,
            rule_groups=rule_groups,
            mitre_techniques=mitre_techs,
            mitre_tactics=mitre_tactics,
            phase="Detection",
            raw_events=[data],
        )

        return record

    @classmethod
    def correlate(
        cls,
        wazuh_alert: Optional[Union[str, Dict[str, Any], IncidentRecord]] = None,
        osquery_event: Optional[Union[str, Dict[str, Any], IncidentRecord]] = None,
    ) -> Optional[IncidentRecord]:
        """
        Correlate a Wazuh alert and an OSQuery process execution event into a unified IncidentRecord.

        If only one source is provided, returns that source's parsed record.
        If both are provided, merges attributes, prioritizing OSQuery for low-level process details
        (pid, ppid, exact path, uid, gid) and Wazuh for SIEM detection details (rule, level, MITRE).
        """
        w_rec: Optional[IncidentRecord] = None
        if isinstance(wazuh_alert, IncidentRecord):
            w_rec = wazuh_alert
        elif wazuh_alert:
            w_rec = cls.parse_wazuh_alert(wazuh_alert)

        o_rec: Optional[IncidentRecord] = None
        if isinstance(osquery_event, IncidentRecord):
            o_rec = osquery_event
        elif osquery_event:
            o_rec = cls.parse_osquery_process_event(osquery_event)

        if w_rec is None and o_rec is None:
            return None
        if w_rec is None:
            return o_rec
        if o_rec is None:
            return w_rec

        # Merge both records
        merged_raw = []
        merged_raw.extend(w_rec.raw_events)
        merged_raw.extend(o_rec.raw_events)

        combined = IncidentRecord(
            incident_id=w_rec.incident_id or o_rec.incident_id,
            timestamp=w_rec.timestamp or o_rec.timestamp,
            host=o_rec.host or w_rec.host,
            source="correlated",
            # Process details: prefer osquery
            process_path=o_rec.process_path or w_rec.process_path,
            process_name=o_rec.process_name or w_rec.process_name,
            cmdline=o_rec.cmdline or w_rec.cmdline,
            pid=o_rec.pid if o_rec.pid is not None else w_rec.pid,
            parent_pid=o_rec.parent_pid if o_rec.parent_pid is not None else w_rec.parent_pid,
            uid=o_rec.uid if o_rec.uid is not None else w_rec.uid,
            gid=o_rec.gid if o_rec.gid is not None else w_rec.gid,
            username=w_rec.username or o_rec.username,
            cwd=o_rec.cwd or w_rec.cwd,
            syscall=o_rec.syscall or w_rec.syscall,
            # SIEM details from Wazuh
            alert_id=w_rec.alert_id,
            alert_level=w_rec.alert_level,
            rule_id=w_rec.rule_id,
            rule_description=w_rec.rule_description,
            rule_groups=list(w_rec.rule_groups),
            mitre_techniques=list(w_rec.mitre_techniques),
            mitre_tactics=list(w_rec.mitre_tactics),
            phase=w_rec.phase or o_rec.phase or "Detection",
            raw_events=merged_raw,
        )

        return combined

    @classmethod
    def aggregate_events(
        cls, events: List[Union[str, Dict[str, Any]]]
    ) -> List[IncidentRecord]:
        """
        Parse a heterogeneous list/stream of raw OSQuery and Wazuh telemetry entries.
        """
        records: List[IncidentRecord] = []
        for ev in events:
            if not ev:
                continue

            parsed_data: Dict[str, Any] = {}
            if isinstance(ev, str):
                try:
                    parsed_data = json.loads(ev)
                except Exception:
                    continue
            elif isinstance(ev, dict):
                parsed_data = ev
            else:
                continue

            # Heuristic detection of event type
            if "name" in parsed_data and "columns" in parsed_data:
                rec = cls.parse_osquery_process_event(parsed_data)
            elif "rule" in parsed_data or "agent" in parsed_data:
                rec = cls.parse_wazuh_alert(parsed_data)
            elif "syscall" in parsed_data or "pid" in parsed_data:
                rec = cls.parse_osquery_process_event(parsed_data)
            else:
                # Try Wazuh then OSQuery
                rec = cls.parse_wazuh_alert(parsed_data) or cls.parse_osquery_process_event(parsed_data)

            if rec is not None:
                records.append(rec)

        return records
