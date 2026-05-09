import platform

# Client configuration
DEFAULT_TIMEOUT = 30
DEFAULT_API_PORT = 55000
DEFAULT_PROTOCOL = "https"
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 0.3

# SDK information
SDK_NAME = "wazuh-sdk"
PYTHON_VERSION = platform.python_version()
SYSTEM = platform.system()
MACHINE = platform.machine()

# HTTP headers
USER_AGENT = f"{SDK_NAME}/0.1.0 (Python/{PYTHON_VERSION}; {SYSTEM}/{MACHINE})"

# Pagination and limits
DEFAULT_LIMIT = 500
DEFAULT_OFFSET = 0
MAX_LIMIT = 1000
