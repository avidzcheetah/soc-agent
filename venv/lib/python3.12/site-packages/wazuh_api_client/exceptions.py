class WazuhError(Exception):
    """Base exception class for Wazuh SDK errors."""
    pass

class WazuhConnectionError(WazuhError):
    """Exception raised for errors in the connection."""
    pass

class WazuhAuthenticationError(WazuhError):
    """Exception raised for authentication errors."""
    pass
