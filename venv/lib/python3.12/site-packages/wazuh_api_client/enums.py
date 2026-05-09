from enum import Enum

class AgentStatus(Enum):
    ALL = "all"
    ACTIVE = "active"
    PENDING = "pending"
    NEVER_CONNECTED = "never_connected"
    DISCONNECTED = "disconnected"

    def __str__(self):
        return self.value

class GroupConfigStatus(Enum):
    SYNCED = "synced"
    NOT_SYNCED = "not_synced"

    def __str__(self):
        return self.value
    
class AgentComponent(Enum):
    AGENT = "agent"
    AGENTLESS = "agentless"
    ANALYSIS = "analysis"
    AUTH = "auth"
    COM = "com"
    CSYSLOG = "csyslog"
    INTEGRATOR = "integrator"
    LOGCOLLECTOR = "logcollector"
    MAIL = "mail"
    MONITOR = "monitor"
    REQUEST = "request"
    SYSCHECK = "syscheck"
    ROOTCHECK = "rootcheck"
    WDB = "wdb"
    WMODULES = "wmodules"
    RULE_TEST = "rule_test"


class AgentConfiguration(Enum):
    CLIENT = "client"
    BUFFER = "buffer"
    LABELS = "labels"
    INTERNAL = "internal"
    ANTI_TAMERING = "anti_tampering"
    AGENTLESS = "agentless"
    GLOBAL = "global"
    ACTIVE_RESPONSE = "active_response"
    ALERTS = "alerts"
    COMMAND = "command"
    RULES = "rules"
    DECODERS = "decoders"
    AUTH = "auth"
    LOGGING = "logging"
    REPORTS = "reports"
    ACTIVERESPONSE = "active-response"
    CLUSTER = "cluster"
    CSYSLOG = "csyslog"
    INTEGRATION = "integration"
    LOCALFILE = "localfile"
    SOCKET = "socket"
    REMOTE = "remote"
    SYSCHECK = "syscheck"
    ROOTCHECK = "rootcheck"
    WDB = "wdb"
    WMODULES = "wmodules"
    RULE_TEST = "rule_test"


class DaemonsList(Enum):
    WAZUH_ANALYSISD = "wazuh-analysisd"
    WAZUH_REMOTED = "wazuh-remoted"


class StatsComponent(Enum):
    LOGCOLLECTOR = "logcollector"
    AGENT = "agent"


class SysCheckScanType(Enum):
    FILE = "file"
    REGISTRY_KEY = "registry_key"
    REGISTRY_VALUE = "registry_value"
