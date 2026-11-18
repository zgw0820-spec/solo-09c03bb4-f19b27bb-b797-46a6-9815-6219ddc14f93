"""Shared pytest fixtures."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.api import create_app
from app.config import Settings
from app.models import load_manifest

MODEL_BYTES = 296


def write_model(model_dir: Path, model_id: str, *, weights_bytes: int = MODEL_BYTES) -> None:
    n_features = 12
    payload = {
        "model_id": model_id,
        "name": model_id.title(),
        "device_type": model_id,
        "version": "1.0.0",
        "description": f"test model {model_id}",
        "features": [
            "vibration_mean",
            "vibration_rms",
            "vibration_peak",
            "vibration_std",
            "vibration_crest",
            "vibration_kurtosis",
            "current_mean",
            "current_rms",
            "current_peak",
            "current_std",
            "current_crest",
            "current_kurtosis",
        ],
        "feature_mean": [0.0] * n_features,
        "feature_scale": [1.0] * n_features,
        "coefficients": [0.1] * n_features,
        "intercept": -1.0,
        "decision_threshold": 0.5,
    }
    declared = int(
        np.asarray(payload["coefficients"], dtype=np.float64).nbytes
        + np.asarray(payload["feature_mean"], dtype=np.float64).nbytes
        + np.asarray(payload["feature_scale"], dtype=np.float64).nbytes
        + np.asarray(payload["intercept"], dtype=np.float64).nbytes
    )
    assert declared == weights_bytes, "test fixture assumes 296-byte weights"
    (model_dir / f"{model_id}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "models"
    directory.mkdir()
    ids = ["alpha", "beta", "gamma"]
    for model_id in ids:
        write_model(directory, model_id)
    manifest = {
        "schema_version": 1,
        "feature_names": [
            "vibration_mean",
            "vibration_rms",
            "vibration_peak",
            "vibration_std",
            "vibration_crest",
            "vibration_kurtosis",
            "current_mean",
            "current_rms",
            "current_peak",
            "current_std",
            "current_crest",
            "current_kurtosis",
        ],
        "models": [
            {
                "id": model_id,
                "name": model_id.title(),
                "device_type": model_id,
                "file": f"{model_id}.json",
                "version": "1.0.0",
                "weights_bytes": MODEL_BYTES,
            }
            for model_id in ids
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


@pytest.fixture
def manifest(model_dir: Path):
    return load_manifest(model_dir)


@pytest.fixture
def settings(model_dir: Path) -> Settings:
    return make_settings(model_dir)


@pytest_asyncio.fixture
async def client(settings: Settings, manifest) -> AsyncIterator[AsyncClient]:
    app = create_app(settings, manifest=manifest)
    transport = ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client


def make_settings(model_dir: Path, **overrides) -> Settings:
    values = {
        "model_dir": model_dir,
        "model_budget_bytes": 2048,
        "inference_delay_ms": 0,
        "model_load_delay_ms": 0,
    }
    values.update(overrides)
    return Settings(**values)
