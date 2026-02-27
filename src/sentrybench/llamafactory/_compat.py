"""Python version compatibility shims for sentrybench.llamafactory.

Requires Python >= 3.11. Provides backports for any remaining edge cases.
"""
import sys

# All symbols below are native in Python 3.11+, but kept here for
# a single import point in case of future cross-version needs.
from enum import StrEnum       # noqa: F401  (3.11+)
from typing import Self        # noqa: F401  (3.11+)
from typing import NotRequired  # noqa: F401  (3.11+)
