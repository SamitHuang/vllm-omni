"""
vLLM-omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

from . import patch  # noqa: F401

try:  # Patch vLLM's SpeculativeMethod when available
    from . import speculative  # noqa: F401
except Exception as exc:  # pragma: no cover - best-effort patching
    import warnings

    warnings.warn(
        f"vLLM speculative patches skipped: {exc}",
        RuntimeWarning,
    )
from .config import OmniModelConfig

# Main entry points
try:
    from .entrypoints.omni_llm import OmniLLM
except Exception as exc:  # pragma: no cover - optional dependency
    OmniLLM = None  # type: ignore
    warnings.warn(
        f"OmniLLM import skipped: {exc}",
        RuntimeWarning,
    )

__all__ = [
    # Main components
    "OmniLLM",
    # "AsyncOmniLLM", # TODO: add AsyncOmniLLM back when it is implemented
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
