"""Registry for model adapters used by the combined pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelResult:
    name: str
    scores: object
    metadata: dict


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    run: Callable[..., ModelResult]


MODEL_REGISTRY: dict[str, ModelAdapter] = {}


def register_model(adapter: ModelAdapter) -> None:
    MODEL_REGISTRY[adapter.name] = adapter


def get_model(name: str) -> ModelAdapter:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{name}'. Available: {', '.join(sorted(MODEL_REGISTRY))}") from exc

