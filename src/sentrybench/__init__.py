"""SentryBench package entrypoint and metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sentrybench")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]

# Import built-ins so they self-register with the registry at import time.
from . import models, attacks, defenses, metrics  # noqa: E402,F401
