"""Python version compatibility shims for sentrybench.llamafactory.

Provides backports of stdlib additions that require Python >= 3.11,
so the codebase runs on Python 3.10+.
"""
import sys

# StrEnum (3.11+)
if sys.version_info >= (3, 11):
    from enum import StrEnum  # noqa: F401
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport of enum.StrEnum for Python < 3.11."""
        def __new__(cls, value, *args, **kwargs):
            if not isinstance(value, str):
                raise TypeError(f"{value!r} is not a string")
            obj = str.__new__(cls, value)
            obj._value_ = value
            return obj

        def __str__(self) -> str:
            return self.value

# typing.Self (3.11+)
if sys.version_info >= (3, 11):
    from typing import Self  # noqa: F401
else:
    from typing_extensions import Self  # noqa: F401

# typing.NotRequired (3.11+)
if sys.version_info >= (3, 11):
    from typing import NotRequired  # noqa: F401
else:
    from typing_extensions import NotRequired  # noqa: F401
