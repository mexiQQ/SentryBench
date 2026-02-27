# SentryBench: only HuggingFace and dummy backends included.
from . import (
    dummy,
    huggingface,
)

try:
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore
    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
