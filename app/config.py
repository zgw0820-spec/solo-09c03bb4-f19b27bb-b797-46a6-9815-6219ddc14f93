"""Runtime configuration read from environment variables.

The service is intentionally configured through plain environment variables so
that it can be dropped into Docker Compose without an extra configuration
library.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class ConfigurationError(RuntimeError):
    """Raised when an environment variable has an invalid value."""


def _read_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Process-wide settings.

    ``model_budget_bytes`` covers both resident model arrays and reservations
    held by in-progress loads.  ``inference_delay_ms`` simulates on-device
    scoring work; ``model_load_delay_ms`` simulates reading weights from slow
    storage and is mainly useful for exercising concurrent load handling.
    """

    model_dir: Path
    model_budget_bytes: int
    inference_delay_ms: int
    model_load_delay_ms: int

    @property
    def inference_delay_seconds(self) -> float:
        return self.inference_delay_ms / 1000.0

    @property
    def model_load_delay_seconds(self) -> float:
        return self.model_load_delay_ms / 1000.0


def load_settings() -> Settings:
    model_dir_raw = os.getenv("MODEL_DIR")
    model_dir = Path(model_dir_raw).expanduser().resolve() if model_dir_raw else DEFAULT_MODEL_DIR

    return Settings(
        model_dir=model_dir,
        model_budget_bytes=_read_int("MODEL_CACHE_BYTES", 640, minimum=1),
        inference_delay_ms=_read_int("INFERENCE_DELAY_MS", 250, minimum=0),
        model_load_delay_ms=_read_int("MODEL_LOAD_DELAY_MS", 0, minimum=0),
    )
