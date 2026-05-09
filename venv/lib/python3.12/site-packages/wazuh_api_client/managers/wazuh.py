from ..interfaces import ResourceManagerInterface, AsyncClientInterface
from ..client import AsyncRequestMaker

class WazuhManager(ResourceManagerInterface):
    
    def __init__(self, client: AsyncClientInterface):
        self.async_request_builder = AsyncRequestMaker(client)

    