"""Abstract fall-detection model interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


class BaseFallModel(nn.Module, ABC):
    """Base class for clip-window fall classifiers.

    Subclasses consume a batch dict from :class:`~src.data.dataset.FallWindowDataset`
    and return unnormalized class logits of shape ``(batch, num_classes)``.
    """

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> Tensor:
        """Return logits for the input window batch."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Registry key for this architecture."""


MODEL_REGISTRY: dict[str, type[BaseFallModel]] = {}


def register_model(name: str):
    """Decorator that registers a :class:`BaseFallModel` subclass."""

    def decorator(cls: type[BaseFallModel]) -> type[BaseFallModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} is already registered as {MODEL_REGISTRY[name]!r}")
        MODEL_REGISTRY[name] = cls
        cls._registry_name = name
        return cls

    return decorator


def _ensure_builtin_models_loaded() -> None:
    """Import built-in models so their registry decorators run."""
    import src.models.bigru_hierarchical  # noqa: F401
    import src.models.bigru_hierarchical_v2  # noqa: F401


def build_model(name: str, config: dict[str, Any]) -> BaseFallModel:
    """Instantiate a registered model from the ``model`` section of a config."""
    if name not in MODEL_REGISTRY:
        _ensure_builtin_models_loaded()
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown model {name!r}. Available: {available}")
    return MODEL_REGISTRY[name](config)
