from enum import Enum

class V4ApiPaths(Enum):
    # Agents endpoints
    LIST_AGENTS = "/agents"
    LIST_OUTDATED_AGENTS = "/agents/outdated"
    LIST_AGENTS_DISTINCT = "/agents/stats/distinct"
    LIST_AGENTS_WITHOUT_GROUP = "/agents/no_group"
    DELETE_AGENTS = "/agents"  # Consider using HTTP DELETE on /agents
    ADD_AGENT = "/agents"      # Consider using HTTP POST on /agents
    GET_ACTIVE_CONFIGURATION = "/agents/{agent_id}/config/{component}/{configuration}"
    DELETE_AGENT_FROM_GROUPS = "/agents/{agent_id}/group"
    DELETE_AGENT_FROM_ONE_GROUP = "/agents/{agent_id}/group/{group_id}"
    ASSIGN_AGENT_TO_GROUP = "/agents/group"
    GET_KEY = "/agents/{agent_id}/key"
    RESTART_AGENT = "/agents/{agent_id}/restart"
    GET_DAEMON_STATS = "/agents/{agent_id}/daemons/stats"
    GET_AGENT_COMPONENT_STATS = "/agents/{agent_id}/stats/{component}"
    REMOVE_AGENTS_FROM_GROUP = "/agents/group"
    RESTART_AGENTS_IN_GROUP = "/agents/group/{group_id}/restart"
    ADD_AGENT_FULL = "/agents/insert"
    ADD_AGENT_QUICK = "/agents/insert/quick"
    RESTART_AGENTS_IN_NODE = "/agents/node/{node_id}/restart"
    FORCE_RECONNECT_AGENTS = "/agents/reconnect"
    RESTART_AGENTS = "/agents/restart"
    SUMMARIZE_AGENTS_OS = "/agents/summary/os"
    SUMMARIZE_AGENTS_STATUS = "/agents/summary/status"

    # Syscheck endpoints
    RUN_SCAN = "/syscheck"
    GET_SCAN_RESULTS = "/syscheck/{agent_id}"
    CLEAR_SCAN_RESULTS = "/syscheck/{agent_id}"
    GET_LAST_SCAN_DATETIME = "/syscheck/{agent_id}/last_scan"

    # Manager endpoints
    GET_WAZUH_STATUS = "/manager/status"
    GET_WAZUH_INFORMATION = "/manager/information"
    GET_WAZUH_CONFIGURATION = "/manager/configuration"
    UPDATE_WAZUH_CONFIGURATION = "/manager/configuration"
    GET_WAZUH_DAEMON_STATS = "/manager/daemons/stats"
    GET_WAZUH_STATS = "/manager/stats"
    GET_WAZUH_STATS_HOUR = "/manager/stats/hour"
    GET_WAZUH_STATS_WEEK = "/manager/stats/week"
    GET_WAZUH_LOGS = "/manager/logs"
    GET_WAZUH_LOGS_SUMMARY = "/manager/logs/summary"

    # Authentication endpoint
    GENERATE_TOKEN = "/security/user/authenticate"