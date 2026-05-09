from typing import Optional, Any, List, Literal
from dataclasses import dataclass, field
from ..enums import (
    AgentStatus,
    GroupConfigStatus,
    AgentComponent,
    AgentConfiguration,
    StatsComponent,
)
from ..interfaces import AsyncClientInterface, ResourceManagerInterface
from ..client import AsyncRequestMaker
from ..endpoints.endpoints_v4 import V4ApiPaths
from ..query import ToDictDataClass, PaginationQueryParams, CommonQueryParams
from ..response import AgentResponse, AddAgentResponse, AgentConfigurationResponse


@dataclass(kw_only=True)
class OsQueryParameters(ToDictDataClass):
    platform: Optional[str] = None
    version: Optional[str] = None
    name: Optional[str] = None

    def to_query_dict(self) -> dict[str, dict[str, str]]:
        """ """
        query: dict[str, dict[str, str]] = {"os": {}}
        for key, value in vars(self).items():
            if value:
                query["os"][key] = value
        return query


@dataclass(kw_only=True)
class CommonListAgentsQueryParams(PaginationQueryParams, CommonQueryParams):

    def __post_init__(self):
        # Validation logic
        if self.offset < 0:
            raise ValueError("offset must be >= 0")
        if not (0 < self.limit <= 100000):
            raise ValueError("limit must be > 0 and <= 100000")


@dataclass(kw_only=True)
class ListAgentsQueryParams(CommonListAgentsQueryParams):
    agents_list: Optional[List[str]] = None
    status: Optional[List[AgentStatus]] = None
    older_than: Optional[str] = None
    os_query_parameters: Optional[OsQueryParameters] = None
    manager: Optional[str] = None
    version: Optional[str] = None
    group: Optional[str] = None
    node_name: Optional[str] = None
    name: Optional[str] = None
    ip: Optional[str] = None
    registerIP: Optional[str] = None
    group_config_status: Optional[GroupConfigStatus] = GroupConfigStatus.SYNCED
    distinct: bool = False  # Look for distinct values.

    def to_query_dict(self) -> dict[str, str | bool]:
        """Converts non-None parameters to a dictionary."""
        query: dict[str, str | bool] = {}

        for key, value in vars(self).items():
            if value is None or key == "os_query_parameters":
                continue

            if key == "select" and isinstance(value, list):
                query[key] = ",".join(value)
            elif key == "agents_list" and isinstance(value, list):
                query[key] = ",".join(value)
            elif key == "status" and isinstance(value, list):
                # Convert each AgentStatus enum to its value.
                query[key] = ",".join(
                    elem.value if isinstance(elem, AgentStatus) else str(elem)
                    for elem in value
                )
            elif isinstance(value, bool):
                query[key] = value
            else:
                query[key] = str(value)

        if self.os_query_parameters:
            os_dict = self.os_query_parameters.to_query_dict()
            for os_key, os_val in os_dict.get("os", {}).items():
                query[f"os.{os_key}"] = os_val
        return query


@dataclass(kw_only=True)
class ListAgentsDistinctQueryParams(CommonListAgentsQueryParams):
    fields: List[str]


@dataclass(kw_only=True)
class ListOutdatedAgentsQueryParams(CommonListAgentsQueryParams):
    pass


@dataclass(kw_only=True)
class ListAgentsWithoutGroupQueryParams(CommonListAgentsQueryParams):
    pass


@dataclass(kw_only=True)
class DeleteAgentsQueryParams(ToDictDataClass):
    pretty: Optional[bool] = False
    wait_for_complete: Optional[bool] = False
    older_than: Optional[str] = None
    os_query_parameters: Optional[OsQueryParameters] = None
    q: Optional[str] = None  # Query string (e.g. 'status=active')
    manager: Optional[str] = None
    version: Optional[str] = None
    group: Optional[str] = None
    node_name: Optional[str] = None
    name: Optional[str] = None
    ip: Optional[str] = None
    registerIP: Optional[str] = None
    agents_list: Optional[List[str]] = None
    status: Optional[List[AgentStatus]] = None
    purge: Optional[bool] = False


@dataclass(kw_only=True)
class AddAgentQueryParams(ToDictDataClass):
    pretty: Optional[bool] = False
    wait_for_complete: Optional[bool] = False


@dataclass(kw_only=True)
class AddAgentBodyParams(ToDictDataClass):
    name: str
    ip: Optional[str] = None


@dataclass(kw_only=True)
class DisconnectedTime(ToDictDataClass):
    enabled: bool = True
    value: str = "1h"


@dataclass(kw_only=True)
class AgentInsertForce(ToDictDataClass):
    enabled: bool = True
    disconnected_time: DisconnectedTime = field(default_factory=DisconnectedTime)
    after_registration_time: str = "1h"

class AgentsManager(ResourceManagerInterface):
    def __init__(self, client: AsyncClientInterface):
        """
        Initialize with a reference to the WazuhClient instance.
        """
        self.async_request_builder = AsyncRequestMaker(client)

    async def list(
        self, list_agent_params: Optional[ListAgentsQueryParams] = None, **kwargs
    ) -> AgentResponse:
        """
        Retrieve a list of agents.

        This method accepts either a ListAgentsQueryParams object or individual parameters as keyword arguments.

        Examples:
            # Using a dataclass
            params = ListAgentsQueryParams(status=["active"], limit=100)
            agents = await client.list(params)

            # Using keyword arguments
            agents = await client.list(status=["active"], limit=100)

        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.get_agents
        """
        endpoint = V4ApiPaths.LIST_AGENTS.value
        if not list_agent_params:
            list_agent_params = ListAgentsQueryParams()

        if kwargs:
            for param, value in kwargs.items():
                if not hasattr(ListAgentsQueryParams, param):
                    raise ValueError(
                        f"Invalid parameter: {param}, keywork argument must be one of : {list(ListAgentsQueryParams.__dataclass_fields__.keys())}"
                    )
                setattr(list_agent_params, param, value)
        params = list_agent_params.to_query_dict()
        res = await self.async_request_builder.get(endpoint, params)
        response = AgentResponse(**res)
        return response

    async def list_distinct(
        self,
        fields: Optional[List[str]] = None,
        list_agents_distinct_params: Optional[ListAgentsDistinctQueryParams] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        List all the different combinations that agents have for the selected fields.

        This method accepts either a ListAgentsDistinctQueryParams object or individual parameters as keyword arguments.

        One of `fields` or `list_agents_distinct_params` is mandatory, you must provide one. If both are given `fields' takes precedence.

        Examples:
            # Using a dataclass
            params = ListAgentsDistinctQueryParams(fields=["active"], limit=100)
            agents = await client.list(params)

            # Using keyword arguments
            agents = await client.list(fields=["active"], limit=100)

        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.get_agent_fields
        """
        params = None

        if not list_agents_distinct_params and not fields:
            raise ValueError(
                "At least one parameter should be provided: fields or an instance of ListAgentsDistinctQueryParams"
            )
        elif fields and list_agents_distinct_params:
            # fields takes precedence
            setattr(list_agents_distinct_params, "fields", fields)
        elif fields:
            list_agents_distinct_params = ListAgentsDistinctQueryParams(fields=fields)

        if kwargs:
            for param, value in kwargs.items():
                if not hasattr(ListAgentsDistinctQueryParams, param):
                    raise ValueError(
                        f"Invalid parameter: {param}, keywork argument must be one of : {list(ListAgentsDistinctQueryParams.__dataclass_fields__.keys())}"
                    )
                setattr(list_agents_distinct_params, param, value)

        if list_agents_distinct_params:
            params = list_agents_distinct_params.to_query_dict()
        res = await self.async_request_builder.get(
            V4ApiPaths.LIST_AGENTS_DISTINCT.value, params
        )
        response = AgentResponse(**res)
        return response

    async def list_outdated(
        self,
        list_outdated_agents_params: Optional[ListOutdatedAgentsQueryParams] = None,
        **kwargs,
    ) -> AgentResponse:
        """
        Return the list of outdated agents.

        This method accepts either a ListAgentsQueryParams object or individual parameters as keyword arguments.

        Examples:
            # Using a dataclass
            params = ListOutdatedAgentsQueryParams(limit=100)
            agents = await client.list(params)

            # Using keyword arguments
            agents = await client.list(limit=100)

        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.get_agent_outdated
        """
        params = None
        if not list_outdated_agents_params:
            list_outdated_agents_params = ListOutdatedAgentsQueryParams()

        if kwargs:
            for param, value in kwargs.items():
                if not hasattr(ListAgentsDistinctQueryParams, param):
                    raise ValueError(
                        f"Invalid parameter: {param}, keywork argument must be one of : {list(ListOutdatedAgentsQueryParams.__dataclass_fields__.keys())}"
                    )
                setattr(list_outdated_agents_params, param, value)
        if list_outdated_agents_params:
            params = list_outdated_agents_params.to_query_dict()

        res = await self.async_request_builder.get(
            V4ApiPaths.LIST_OUTDATED_AGENTS.value, params
        )
        response = AgentResponse(**res)
        return response

    async def list_without_group(
        self,
        list_agents_without_group_params: Optional[
            ListAgentsWithoutGroupQueryParams
        ] = None,
        **kwargs,
    ):
        """
        Return a list with all the available agents without an assigned group.

        This method accepts either a ListAgentsWithoutGroupQueryParams object or individual parameters as keyword arguments.

        Examples:
            # Using a dataclass
            params = ListAgentsWithoutGroupQueryParams(limit=100)
            agents = await client.list(params)

            # Using keyword arguments
            agents = await client.list(limit=100)

        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.get_agent_no_group
        """
        params = None
        if not list_agents_without_group_params:
            list_agents_without_group_params = ListAgentsWithoutGroupQueryParams()

        if kwargs:
            for param, value in kwargs.items():
                if not hasattr(ListAgentsDistinctQueryParams, param):
                    raise ValueError(
                        f"Invalid parameter: {param}, keywork argument must be one of : {list(ListAgentsWithoutGroupQueryParams.__dataclass_fields__.keys())}"
                    )
                setattr(list_agents_without_group_params, param, value)

        if list_agents_without_group_params:
            params = list_agents_without_group_params.to_query_dict()
        res = await self.async_request_builder.get(
            V4ApiPaths.LIST_AGENTS_WITHOUT_GROUP.value, params
        )
        response = AgentResponse(**res)
        return response

    async def delete(
        self,
        agents_list: List[str],
        status: List[AgentStatus],
        purge: bool = False,
        delete_agents_params: Optional[DeleteAgentsQueryParams] = None,
        **kwargs,
    ):
        """
        Delete all agents or a list of them based on optional criteria
        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.delete_agents
        """
        if not delete_agents_params:
            delete_agents_params = DeleteAgentsQueryParams()

        delete_agents_params.agents_list = agents_list
        delete_agents_params.status = status
        delete_agents_params.purge = purge

        if kwargs:
            for param, value in kwargs.items():
                if not hasattr(DeleteAgentsQueryParams, param):
                    raise ValueError(
                        f"Invalid parameter: {param}, keywork argument must be one of : {list(DeleteAgentsQueryParams.__dataclass_fields__.keys())}"
                    )
                setattr(delete_agents_params, param, value)

        res = await self.async_request_builder.delete(
            V4ApiPaths.DELETE_AGENTS.value, delete_agents_params
        )
        response = AgentResponse(**res)
        return response

    async def add(
        self,
        name: str,
        ip: str,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AddAgentResponse:
        """
        Add a new agent.
        """
        add_agent_request_body = AddAgentBodyParams(name=name, ip=ip)
        add_agent_query_params = AddAgentQueryParams(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.post(
            V4ApiPaths.ADD_AGENT.value,
            query_params=add_agent_query_params,
            body=add_agent_request_body.to_query_dict(),
        )
        response = AddAgentResponse(**res)
        return response

    async def get_active_configuration(
        self,
        agent_id: str,
        component: AgentComponent,
        configuration: AgentConfiguration,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentConfigurationResponse:
        """
        Return the active configuration the agent is currently using.
        This can be different from the configuration present in the configuration file,
        if it has been modified and the agent has not been restarted yet.

        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.add_agent
        """
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        path_parameters: dict[str, str | int] = dict(
            agent_id=str(agent_id),
            component=str(component),
            configuration=str(configuration),
        )
        res = await self.async_request_builder.get(
            V4ApiPaths.GET_ACTIVE_CONFIGURATION.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentConfigurationResponse(**res)
        return response

    async def remove_agent_from_one_or_more_groups(
        self,
        agent_id: str,
        pretty: bool = False,
        wait_for_complete: bool = False,
        groups_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
    ) -> AgentResponse:
        """
        Remove the agent from all groups or a list of them, one group given its name or id.
        The agent will automatically revert to the default group if it is removed from all its assigned groups.
        """
        if groups_list and group_id:
            raise ValueError(
                "Cannot provide a group_list and a group_id, can only provide one."
            )

        path_params: dict[str, str | int] = dict(agent_id=str(agent_id))
        params: dict[str, bool | list[str] | str] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )

        resource = V4ApiPaths.DELETE_AGENT_FROM_GROUPS
        if group_id:
            path_params["group_id"] = group_id
            resource = V4ApiPaths.DELETE_AGENT_FROM_ONE_GROUP
        elif groups_list:
            params["groups_list"] = groups_list

        res = await self.async_request_builder.delete(
            resource.value, query_params=params, path_params=path_params
        )
        response = AgentResponse(**res)
        return response

    async def assign_agent_to_group(
        self,
        agent_id: str,
        group_id: str,
        force_single_group: bool,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Assign an agent to a specified group
        """
        path_parameters: dict[str, str | int] = dict(
            agent_id=agent_id, group_id=group_id
        )
        params: dict[str, bool] = dict(
            pretty=pretty,
            wait_for_complete=wait_for_complete,
            force_single_group=force_single_group,
        )
        res = await self.async_request_builder.put(
            V4ApiPaths.ASSIGN_AGENT_TO_GROUP.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response

    async def get_key(
        self, agent_id: str, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Return the key of an agent.
        """
        path_parameters: dict[str, str | int] = dict(agent_id=agent_id)
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.get(
            V4ApiPaths.GET_KEY.value, query_params=params, path_params=path_parameters
        )
        response = AgentResponse(**res)
        return response

    async def restart_agent(
        self, agent_id: str, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Restart the specified agent.
        """
        path_parameters: dict[str, str | int] = dict(agent_id=agent_id)
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.put(
            V4ApiPaths.RESTART_AGENT.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response

    async def get_wazuh_daemon_stats(
        self,
        agent_id: str,
        daemons_list: Optional[List[str]] = None,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Get Wazuh daemon stats from an agent.
        """
        path_parameters: dict[str, str | int] = dict(agent_id=agent_id)
        params: dict[str, bool | list[str] | None] = dict(
            pretty=pretty,
            wait_for_complete=wait_for_complete,
            daemons_list=daemons_list,
        )
        res = await self.async_request_builder.get(
            V4ApiPaths.GET_DAEMON_STATS.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response

    async def get_agent_component_stats(
        self,
        agent_id: str,
        component: StatsComponent,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Return Wazuh's `component` statistical information from agent `agent_id`.
        """
        path_parameters: dict[str, str | int] = dict(
            agent_id=agent_id, component=str(component)
        )
        params: dict[str, bool] = dict(
            pretty=pretty,
            wait_for_complete=wait_for_complete,
        )
        res = await self.async_request_builder.get(
            V4ApiPaths.GET_AGENT_COMPONENT_STATS.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response

    def upgrade_agents(self):
        raise NotImplementedError()

    def upgrade_agents_custom(self):
        raise NotImplementedError()

    def check_user_permission_to_uninstall_agents(self):
        raise NotImplementedError()

    async def remove_agents_from_group(
        self,
        agents_list: List[str],
        group_id: str,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Remove all agents assignment or a list of them from the specified group.
        """
        params: dict[str, str | bool | List[str]] = dict(
            pretty=pretty,
            wait_for_complete=wait_for_complete,
            agents_list=agents_list,
            group_id=group_id,
        )
        res = await self.async_request_builder.delete(
            V4ApiPaths.REMOVE_AGENTS_FROM_GROUP.value, query_params=params
        )
        response = AgentResponse(**res)
        return response

    async def assign_agents_to_group(
        self,
        group_id: str,
        agents_list: List[str],
        force_single_group: bool = False,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Assign all agents or a list of them to the specified group.
        """
        params: dict[str, Any] = dict(
            pretty=pretty,
            wait_for_complete=wait_for_complete,
            agents_list=agents_list,
            group_id=group_id,
            force_single_group=force_single_group,
        )
        res = await self.async_request_builder.put(
            V4ApiPaths.ASSIGN_AGENT_TO_GROUP.value, query_params=params
        )
        response = AgentResponse(**res)
        return response

    async def restart_agents_in_group(
        self, group_id: str, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Restart all agents which belong to a given group.
        """
        path_parameters: dict[str, str | int] = dict(group_id=group_id)
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.put(
            V4ApiPaths.RESTART_AGENTS_IN_GROUP.value,
            path_params=path_parameters,
            query_params=params,
        )
        response = AgentResponse(**res)
        return response

    async def add_agent_full(
        self,
        id: str,
        key: str,
        name: str,
        ip: str,
        force: AgentInsertForce,
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AddAgentResponse:
        """
        Add an agent specifying its name, ID and IP.
        If an agent with the same name, the same ID or the same IP already exists, replace it using the force parameter.
        """
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        force_data = force.to_query_dict()
        body: dict[str, Any] = dict(id=id, key=key, name=name, ip=ip, force=force_data)
        res = await self.async_request_builder.post(
            V4ApiPaths.ADD_AGENT_FULL.value, query_params=params, body=body
        )
        response = AddAgentResponse(**res)
        return response

    async def add_agent_quick(
        self, agent_name: str, pretty: bool = False, wait_for_complete: bool = False
    ) -> AddAgentResponse:
        """
        Add a new agent with name `agent_name`. This agent will use any as `IP`.
        """
        params: dict[str, bool | str] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete, agent_name=agent_name
        )
        res = await self.async_request_builder.post(
            V4ApiPaths.ADD_AGENT_QUICK.value, query_params=params
        )
        response = AddAgentResponse(**res)
        return response

    async def restart_agents_in_node(
        self, node_id: str, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Restart all agents which belong to a specific given node.
        """
        path_parameters: dict[str, str | int] = dict(node_id=node_id)
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.put(
            V4ApiPaths.RESTART_AGENTS_IN_NODE.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response

    async def _restart_or_reconnect_agents(
        self,
        restart_or_reconnect: Literal["restart", "reconnect"],
        agents_list: List[str],
        pretty: bool = False,
        wait_for_complete: bool = False,
    ):
        params: dict[str, bool | List[str]] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete, agents_list=agents_list
        )
        if restart_or_reconnect == "restart":
            endpoint = V4ApiPaths.RESTART_AGENTS.value
        elif restart_or_reconnect == "reconnect":
            endpoint = V4ApiPaths.FORCE_RECONNECT_AGENTS.value
        else:
            raise ValueError(
                "`restart_or_reconnect` must be one of: restart or reconnect"
            )
        res = await self.async_request_builder.put(endpoint, query_params=params)
        return res

    async def force_reconnect_agents(
        self,
        agents_list: List[str],
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Force reconnect all agents or a list of them.
        """
        res = await self._restart_or_reconnect_agents(
            restart_or_reconnect="reconnect",
            pretty=pretty,
            agents_list=agents_list,
            wait_for_complete=wait_for_complete,
        )
        response = AgentResponse(**res)
        return response

    async def restart_agents(
        self,
        agents_list: List[str],
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Restart all agents or a list of them.
        """
        res = await self._restart_or_reconnect_agents(
            restart_or_reconnect="restart",
            pretty=pretty,
            agents_list=agents_list,
            wait_for_complete=wait_for_complete,
        )
        response = AgentResponse(**res)
        return response
    
    async def _summarize_agents_item(self, item: Literal["os", "status"], pretty: bool = False, wait_for_complete: bool = False
    ) -> dict[str, Any]:
        """
        Return a summary of the `item` of available agents, item is one of: os or status.
        """
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        if item == "os":
            endpoint = V4ApiPaths.SUMMARIZE_AGENTS_OS.value
        elif item == "status":
            endpoint = V4ApiPaths.SUMMARIZE_AGENTS_STATUS.value
        else:
            raise ValueError(
                "`item` must be one of: os or status."
            )
        res = await self.async_request_builder.get(
            endpoint, query_params=params
        )
        return res

    async def summarize_agents_os(
        self, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Return a summary of the OS of available agents
        """
        res = await self._summarize_agents_item(item="os", pretty=pretty, wait_for_complete=wait_for_complete)
        response = AgentResponse(**res)
        return response

    async def summarize_agents_status( self, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Return a summary of the connection and groups configuration synchronization statuses of available agents
        """
        res = await self._summarize_agents_item(item="status", pretty=pretty, wait_for_complete=wait_for_complete)
        response = AgentResponse(**res)
        return response

    async def get_upgrade_results(self):
        """
        Return the agents upgrade results
        https://documentation.wazuh.com/current/user-manual/api/reference.html#operation/api.controllers.agent_controller.get_agent_upgrade
        """
        raise NotImplementedError()

