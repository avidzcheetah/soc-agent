from dataclasses import dataclass
from typing import Optional, List
from ..query import CommonQueryParams, PaginationQueryParams
from ..enums import SysCheckScanType
from ..interfaces import AsyncClientInterface
from ..client import AsyncRequestMaker
from ..endpoints.endpoints_v4 import V4ApiPaths
from ..response import AgentResponse


@dataclass
class ScanResultParams(CommonQueryParams, PaginationQueryParams):
    file: Optional[str] = None
    arch: Optional[str] = None
    value_name: Optional[str] = None
    value_type: Optional[str] = None
    type: Optional[SysCheckScanType] = None
    summary: bool = False
    md5: Optional[str] = None
    sha1: Optional[str] = None
    sha256: Optional[str] = None
    hash: Optional[str] = None
    distinct: bool = False

    def to_query_dict(self) -> dict[str, str | int | bool]:
        """ """
        query: dict[str, str | int | bool] = {}
        for key, value in vars(self).items():
            if value and not isinstance(value, bool):
                query[key] = str(value)
        return query


class SysCheckManager:
    def __init__(self, client: AsyncClientInterface):
        """
        Initialize with a reference to the WazuhClient instance.
        """
        self.async_request_builder = AsyncRequestMaker(client)

    async def run_scan(
        self,
        agents_list: List[str],
        pretty: bool = False,
        wait_for_complete: bool = False,
    ) -> AgentResponse:
        """
        Run FIM scan in all agents
        """
        params: dict[str, bool | List[str]] = dict(
            agents_list=agents_list, pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.put(
            V4ApiPaths.RUN_SCAN.value, query_params=params
        )
        response = AgentResponse(**res)
        return response

    async def get_results(
        self, agent_id: str, params: Optional[ScanResultParams] = None, **kwargs
    ) -> AgentResponse:
        """
        Return FIM findings in the specified agent.
        """
        path_params: dict[str, str | int] = dict(agent_id=agent_id)
        if not params:
            params = ScanResultParams()
        if kwargs:
            for param, value in kwargs.items():
                if not hasattr(ScanResultParams, param):
                    raise ValueError(
                        f"Invalid parameter: {param}, keywork argument must be one of : {list(ScanResultParams.__dataclass_fields__.keys())}"
                    )
                setattr(params, param, value)
        res = await self.async_request_builder.get(
            V4ApiPaths.GET_SCAN_RESULTS.value,
            query_params=params.to_query_dict(),
            path_params=path_params,
        )
        response = AgentResponse(**res)
        return response

    async def clear_results(
        self, agent_id: str, pretty: bool = False, wait_for_complete: bool = False
    ):
        """
        Clear file integrity monitoring scan results for a specified agent.
        Only available for agents < 3.12.0, it doesn't apply for more recent ones
        """
        path_parameters: dict[str, str | int] = dict(agent_id=agent_id)
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.delete(
            V4ApiPaths.CLEAR_SCAN_RESULTS.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response

    async def get_last_scan_datetime(
        self, agent_id: str, pretty: bool = False, wait_for_complete: bool = False
    ) -> AgentResponse:
        """
        Return when the last syscheck scan started and ended. If the scan is still in progress the end date will be unknown.
        """
        path_parameters: dict[str, str | int] = dict(agent_id=agent_id)
        params: dict[str, bool] = dict(
            pretty=pretty, wait_for_complete=wait_for_complete
        )
        res = await self.async_request_builder.get(
            V4ApiPaths.GET_LAST_SCAN_DATETIME.value,
            query_params=params,
            path_params=path_parameters,
        )
        response = AgentResponse(**res)
        return response
