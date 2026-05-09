from .endpoints.endpoints_v4 import V4ApiPaths

def get_api_paths(version: str):
    """
    Return the API endpoints dictionary based on the provided Wazuh version.
    """
    if version.startswith("4"):
        return V4ApiPaths.__dict__
    else:
        raise ValueError(f"Unsupported Wazuh version: {version}")
