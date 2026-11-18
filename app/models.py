"""Model manifest parsing, validation and scoring."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ModelLoadError
from .features import FEATURE_NAMES

_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_SCORE_MAGNITUDE = 500.0


@dataclass(frozen=True)
class LinearModel:
    """A validated logistic-regression model with fixed weights."""

    model_id: str
    name: str
    device_type: str
    version: str
    description: str
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: float
    decision_threshold: float
    weights_bytes: int

    def score(self, features: np.ndarray) -> float:
        standardised = (features - self.feature_mean) / self.feature_scale
        logit = float(np.dot(standardised, self.coefficients) + self.intercept)
        logit = max(-_MAX_SCORE_MAGNITUDE, min(_MAX_SCORE_MAGNITUDE, logit))
        probability = 1.0 / (1.0 + math.exp(-logit))
        return float(probability)

    def is_anomalous(self, score: float) -> bool:
        return score >= self.decision_threshold


@dataclass(frozen=True)
class ModelDescriptor:
    """Manifest entry; enough to reserve budget before loading weights."""

    model_id: str
    name: str
    device_type: str
    version: str
    file_name: str
    declared_bytes: int


@dataclass(frozen=True)
class Manifest:
    model_dir: Path
    descriptors: dict[str, ModelDescriptor]
    feature_names: tuple[str, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_manifest(model_dir: Path) -> Manifest:
    manifest_path = model_dir / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"manifest not found: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {manifest_path}: {exc}") from exc

    _require(isinstance(raw, dict), "manifest must be a JSON object")
    _require(raw.get("schema_version") == 1, "manifest schema_version must be 1")

    feature_names = raw.get("feature_names")
    _require(
        isinstance(feature_names, list) and tuple(feature_names) == FEATURE_NAMES,
        f"manifest feature_names must equal {list(FEATURE_NAMES)}",
    )

    entries = raw.get("models")
    _require(isinstance(entries, list) and entries, "manifest models must be a non-empty list")

    descriptors: dict[str, ModelDescriptor] = {}
    for entry in entries:
        _require(isinstance(entry, dict), "each manifest model must be an object")
        model_id = entry.get("id")
        _require(
            isinstance(model_id, str) and bool(_MODEL_ID_PATTERN.fullmatch(model_id)),
            f"invalid model id: {model_id!r}",
        )
        _require(model_id not in descriptors, f"duplicate model id: {model_id}")

        name = entry.get("name")
        device_type = entry.get("device_type")
        version = entry.get("version")
        file_name = entry.get("file")
        weight_bytes = entry.get("weights_bytes")

        _require(isinstance(name, str) and name, f"{model_id}: name must be a non-empty string")
        _require(
            isinstance(device_type, str) and bool(_MODEL_ID_PATTERN.fullmatch(device_type)),
            f"{model_id}: invalid device_type",
        )
        _require(isinstance(version, str) and version, f"{model_id}: version must be set")
        _require(
            isinstance(file_name, str) and file_name == Path(file_name).name and file_name.endswith(".json"),
            f"{model_id}: file must be a plain .json file name",
        )
        _require(
            isinstance(weight_bytes, int) and not isinstance(weight_bytes, bool) and weight_bytes > 0,
            f"{model_id}: weights_bytes must be a positive integer",
        )

        descriptors[model_id] = ModelDescriptor(
            model_id=model_id,
            name=name,
            device_type=device_type,
            version=version,
            file_name=file_name,
            declared_bytes=weight_bytes,
        )

    return Manifest(model_dir=model_dir, descriptors=descriptors, feature_names=tuple(feature_names))


def _finite_array(value: Any, length: int, field: str, model_id: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelLoadError(model_id, f"{field} is not numeric") from exc
    if array.shape != (length,):
        raise ModelLoadError(model_id, f"{field} must contain {length} values")
    if not np.all(np.isfinite(array)):
        raise ModelLoadError(model_id, f"{field} contains non-finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def read_model_file(descriptor: ModelDescriptor, model_dir: Path) -> LinearModel:
    """Read and validate a model file.

    Returns the validated model.  Any problem is reported as ``ModelLoadError``
    so that the cache can clear its reservation and keep serving other models.
    """

    path = model_dir / descriptor.file_name
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelLoadError(descriptor.model_id, f"file not found: {descriptor.file_name}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(descriptor.model_id, f"cannot read {descriptor.file_name}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ModelLoadError(descriptor.model_id, "model file must contain a JSON object")

    if raw.get("model_id") != descriptor.model_id:
        raise ModelLoadError(descriptor.model_id, "model_id does not match manifest")
    if raw.get("version") != descriptor.version:
        raise ModelLoadError(descriptor.model_id, "version does not match manifest")
    if raw.get("features") != list(FEATURE_NAMES):
        raise ModelLoadError(descriptor.model_id, "feature schema does not match service schema")

    n_features = len(FEATURE_NAMES)
    coefficients = _finite_array(raw.get("coefficients"), n_features, "coefficients", descriptor.model_id)
    feature_mean = _finite_array(raw.get("feature_mean"), n_features, "feature_mean", descriptor.model_id)
    feature_scale = _finite_array(raw.get("feature_scale"), n_features, "feature_scale", descriptor.model_id)
    intercept = _finite_array([raw.get("intercept")], 1, "intercept", descriptor.model_id)[0]
    threshold_raw = raw.get("decision_threshold")
    if not isinstance(threshold_raw, (int, float)) or isinstance(threshold_raw, bool):
        raise ModelLoadError(descriptor.model_id, "decision_threshold must be a number")
    threshold = float(threshold_raw)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ModelLoadError(descriptor.model_id, "decision_threshold must be between 0 and 1")
    if np.any(feature_scale <= 0.0):
        raise ModelLoadError(descriptor.model_id, "feature_scale values must be positive")

    actual_bytes = int(
        coefficients.nbytes + feature_mean.nbytes + feature_scale.nbytes + intercept.nbytes
    )
    if actual_bytes != descriptor.declared_bytes:
        raise ModelLoadError(
            descriptor.model_id,
            f"resident size {actual_bytes} does not match manifest weights_bytes {descriptor.declared_bytes}",
        )

    name = raw.get("name", descriptor.name)
    device_type = raw.get("device_type", descriptor.device_type)
    description = raw.get("description", "")

    return LinearModel(
        model_id=descriptor.model_id,
        name=name if isinstance(name, str) else descriptor.name,
        device_type=device_type if isinstance(device_type, str) else descriptor.device_type,
        version=descriptor.version,
        description=description if isinstance(description, str) else "",
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        intercept=float(intercept),
        decision_threshold=threshold,
        weights_bytes=actual_bytes,
    )
