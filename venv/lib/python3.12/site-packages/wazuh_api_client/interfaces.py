from abc import ABC, abstractmethod
from typing import Coroutine, Any, Optional


class ClientInterface(ABC):
    @abstractmethod
    def build_endpoint(self, key: str) -> str:
        """
        Construct the full API endpoint URL using the mapping and provided parameters.
        """
        pass

    @abstractmethod
    def request(self, method: str, endpoint: str, **kwargs):
        """
        Helper method to make an HTTP request.
        """
        pass


class AsyncClientInterface(ABC):
    @abstractmethod
    def build_endpoint(
        self, key: str, params: Optional[dict[str, str | int]] = None
    ) -> str:
        """
        Construct the full API endpoint URL using the mapping and provided parameters.
        """
        pass

    @abstractmethod
    async def request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """
        Helper method to make an HTTP request.
        """
        pass


class AsyncRequestBuilderInterface(ABC):
    def __init__(self, client: AsyncClientInterface):
        pass

    @abstractmethod
    async def get(
        self, endpoint: str, query_params: Any, path_params: dict[str, str | int], **kwargs
    ) -> Any:
        pass

    @abstractmethod
    async def delete(
        self, endpoint: str, query_params: Any, path_params: dict[str, str | int], **kwargs
    ) -> Any:
        pass

    @abstractmethod
    async def post(
        self,
        endpoint: str,
        query_params: Any,
        body: dict[str, Any],
        path_params: dict[str, str | int],
        **kwargs
    ) -> Any:
        pass

    @abstractmethod
    async def put(
        self,
        endpoint: str,
        query_params: Any,
        path_params: dict[str, str | int],
        body: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        pass


class RequestBuilderInterface(ABC):
    def __init__(self, client: ClientInterface):
        pass

    @abstractmethod
    def get(self, endpoint_name: str, query_params: Any, **kwargs) -> Any:
        pass

    @abstractmethod
    def delete(self, endpoint_name: str, query_params: Any, **kwargs) -> Coroutine | None:
        pass


class ResourceManagerInterface:
    def __init__(self, client: AsyncClientInterface):
        """
        Initialize with a reference to the WazuhClient instance.
        """
        pass