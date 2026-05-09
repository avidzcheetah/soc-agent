import requests

from ssl import SSLContext
from httpx import AsyncClient, RequestError
from typing import Any, Optional

from .constants import DEFAULT_TIMEOUT, USER_AGENT
from .exceptions import WazuhError, WazuhConnectionError
from .utils import get_api_paths

from .interfaces import (
    ClientInterface,
    AsyncClientInterface,
    AsyncRequestBuilderInterface,
    RequestBuilderInterface,
)


class WazuhClient(ClientInterface):
    def __init__(
        self,
        base_url: str,
        version: str,
        username: str,
        password: str,
        verify: bool | None = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify
        self.session.headers.update({"User-Agent": USER_AGENT})

        # Detect or set the Wazuh version.
        self.version = version or self._detect_version()

        token = self._generate_token(username, password)
        self.session.headers.update({"Authorization": f"Bearer {token}"})

        try:
            self.api_paths = get_api_paths(self.version)
        except ValueError as ve:
            raise WazuhError(str(ve))

    def _generate_token(self, username: str, password: str) -> str:
        """ """
        generate_token_url = self.build_endpoint("/security/user/authenticate")
        response = self.session.post(
            generate_token_url, verify=False, auth=(username, password)
        )
        response.raise_for_status()
        token = response.json()["data"]["token"]
        return token

    def _detect_version(self) -> str:
        """
        Auto-detect the Wazuh version by calling an endpoint.
        Assumes that '/manager/info' returns JSON with a 'data' dict containing a 'version' field.
        """
        try:
            response = self.session.get(
                f"{self.base_url}/manager/info", timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            info = response.json()
            version = info.get("data", {}).get("version")
            if not version:
                raise WazuhError(
                    "Wazuh version information not found in manager info response."
                )
            return version
        except Exception as e:
            raise WazuhConnectionError("Failed to detect Wazuh version.") from e

    def build_endpoint(
        self, endpoint: str, params: Optional[dict[str, str | int]] = None
    ) -> str:
        """
        Construct the full API endpoint URL using the mapping and provided parameters.
        """
        res = self.base_url
        if params:
            for k, v in params.items():
                if not v:
                    del params[k]
            res += endpoint.format(**params)
        else:
            res += endpoint
        return res

    def request(self, method: str, endpoint: str, **kwargs):
        """
        Helper method to make an HTTP request.
        """
        try:
            response = self.session.request(
                method, endpoint, timeout=DEFAULT_TIMEOUT, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise WazuhConnectionError("HTTP request failed.") from e


class AsyncWazuhClient(AsyncClientInterface):
    def __init__(
        self,
        base_url: str,
        version: str,
        username: str,
        password: str,
        verify: SSLContext | str | bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.verify = verify
        self.username = username
        self.password = password
        self.version = version
        self.client: Optional[AsyncClient] = None
        self.authenticated = False
        self.api_paths: dict[str, str] = {}

    async def async_init(self):
        self.client = AsyncClient(
            base_url=self.base_url,
            headers={"User-Agent": USER_AGENT},
            verify=self.verify,
            timeout=DEFAULT_TIMEOUT,
        )
        # Optionally detect version if not provided.
        if not self.version:
            self.version = await self._detect_version()
        # Generate token and update headers.
        token = await self._generate_token(self.username, self.password)
        self.client.headers.update({"Authorization": f"Bearer {token}"})

        try:
            self.api_paths = get_api_paths(self.version)
        except ValueError as ve:
            raise WazuhError(str(ve))

        self.authenticated = True

    async def _generate_token(self, username: str, password: str) -> str:
        if self.client is None:
            raise RuntimeError("Async client is not initialized")

        url = self.build_endpoint("/security/user/authenticate")
        response = await self.client.post(url, auth=(username, password))
        response.raise_for_status()
        return response.json()["data"]["token"]

    async def _detect_version(self) -> str:
        if self.client is None:
            raise RuntimeError("Async client is not initialized")

        try:
            response = await self.client.get("/manager/info")
            response.raise_for_status()
            version = response.json().get("data", {}).get("version")
            if not version:
                raise WazuhError("Wazuh version not found in manager info response.")
            return version
        except Exception as e:
            raise WazuhConnectionError("Failed to detect Wazuh version.") from e

    def build_endpoint(
        self, endpoint: str, params: Optional[dict[str, str | int]] = None
    ) -> str:
        """
        Construct the full API endpoint URL using the mapping and provided parameters.
        """
        res = self.base_url
        if params:
            for k, v in params.items():
                if not v:
                    del params[k]
            res += endpoint.format(**params)
        else:
            res += endpoint
        return res

    async def request(self, method: str, endpoint: str, **kwargs):
        if self.client is None:
            raise RuntimeError("Async client is not initialized")

        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except RequestError as e:
            raise Exception("HTTP request failed.") from e

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def __aenter__(self):
        await self.async_init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class AsyncRequestMaker(AsyncRequestBuilderInterface):
    def __init__(self, client: AsyncClientInterface):
        self.client = client


    def _construct_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Construct a dictionary of params with only not None values.
        """
        res: dict[str, Any] = {}
        for key, value in params.items():
            if value:
                res[key] = value
        return res

    async def get(
        self,
        endpoint: str,
        query_params: Optional[dict[str, Any]] = None,
        path_params: Optional[dict[str, str | int]] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Make a get request and return a dictionary representing the result.
        """
        params = None
        if query_params:
            params = self._construct_params(query_params)
        endpoint = self.client.build_endpoint(endpoint, path_params)
        res = await self.client.request("GET", endpoint, params=params, **kwargs)
        return res

    async def delete(
        self,
        endpoint: str,
        query_params: Any,
        path_params: Optional[dict[str, str | int]] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Make an delete request and return a dictionary representing the result.
        """
        params = None
        if query_params:
            params = self._construct_params(query_params)
        endpoint = self.client.build_endpoint(endpoint, path_params)
        res = await self.client.request("DELETE", endpoint, params=params, **kwargs)
        return res

    async def post(
        self,
        endpoint: str,
        query_params: Any,
        body: Optional[dict[str, Any]] = None,
        path_params: Optional[dict[str, str | int]] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Make a post request and return a dictionary representing the result.
        """
        params = None
        if query_params:
            params = self._construct_params(query_params)
        endpoint = self.client.build_endpoint(endpoint, path_params)
        res = await self.client.request(
            "POST", endpoint, params=params, json=body, **kwargs
        )
        return res

    async def put(
        self,
        endpoint: str,
        query_params: Any,
        path_params: Optional[dict[str, str | int]] = None,
        body: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Make a put request and return a dictionary representing the result.
        """
        params = None
        if query_params:
            params = self._construct_params(query_params)
        endpoint = self.client.build_endpoint(endpoint, path_params)
        res = await self.client.request("PUT", endpoint, params=params, json=body, **kwargs)
        return res


class RequestMaker(RequestBuilderInterface):
    def __init__(self, client: ClientInterface):
        self.client = client

    def get(self, endpoint_name: str, query_params: Any, **kwargs) -> dict[str, Any]:
        """
        Make a get request and return a dictionary representing the result.
        """
        endpoint = self.client.build_endpoint(endpoint_name)
        params = query_params.to_query_dict()
        return self.client.request("GET", endpoint, params=params, **kwargs)
