from dataclasses import dataclass, field
from typing import List
from .enums import AgentStatus, GroupConfigStatus

@dataclass
class Error:
    code: int
    message: str
    remediation: str


@dataclass
class FailedItem:
    error: dict[str, dict[str, Error]]
    id: List[str] | List[int]


@dataclass
class OS:
    arch: str
    minor: str
    codename: str
    version: str
    platform: str
    uname: str
    name: str
    major: str



@dataclass
class Agent:
    os: OS
    group_config_status: GroupConfigStatus
    lastKeepAlive: str
    dateAdd: str
    node_name: str
    manager: str
    registerIp: str
    ip: str
    mergedSum: str
    group: List[str]
    configSum: str
    status: AgentStatus
    name: str
    id: str
    version: str
    status_code: int = 0


@dataclass
class ResponseData:
    total_affected_items: int
    failed_items: List[FailedItem]
    total_failed_items: int
    affected_items: List[Agent]


@dataclass
class AgentResponse:
    message: str
    error: int
    data: ResponseData


@dataclass
class AddAgentData:
    id: str
    key: str


@dataclass
class AddAgentResponse:
    data: AddAgentData
    error: int


@dataclass
class Server:
    address: str
    port: int
    max_retries: int
    retry_interval: int
    protocol: str


@dataclass
class Enrollment:
    enabled: str
    delay_after_enrollment: int
    port: int
    ssl_cipher: str
    auto_method: str


@dataclass
class Client:
    config_profile: str = field(metadata={"json": "config-profile"})
    notify_time: int
    time_reconnect: int = field(metadata={"json": "time-reconnect"})
    force_reconnect_interval: int
    ip_update_interval: int
    auto_restart: str
    remote_conf: str
    crypto_method: str
    server: List[Server]
    enrollment: List[Enrollment]


@dataclass
class AgentCofigurationData:
    client: Client


@dataclass
class AgentConfigurationResponse:
    data: AgentCofigurationData
    error: int