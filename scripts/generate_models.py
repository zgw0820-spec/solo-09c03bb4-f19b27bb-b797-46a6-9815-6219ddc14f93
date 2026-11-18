#!/usr/bin/env python3
"""Generate the three fixed-weight anomaly models.

The artifacts are committed to the repository so that the runtime only ever
reads weights; this script documents how they were produced and lets the
fixture bytes be regenerated deterministically.

Each model is a logistic regression over the 12 features returned by
``app.features.extract_features``.  The three models share the feature schema
but use different standardisation statistics and coefficients tailored to the
signal characteristics of their device family.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES = [
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
]

# Standardisation statistics and coefficients.  Statistics are expressed in
# engineering units (mm/s for vibration, A for current).
MODELS: list[dict[str, Any]] = [
    {
        "id": "pump",
        "name": "离心泵",
        "device_type": "pump",
        "description": "离心泵振动与驱动电流联合异常模型",
        "version": "1.0.0",
        "feature_mean": [1.8, 2.6, 6.2, 1.9, 2.45, 3.0, 12.0, 13.2, 20.0, 3.0, 1.75, 3.0],
        "feature_scale": [1.1, 1.4, 2.8, 0.9, 0.55, 1.2, 4.5, 4.8, 7.0, 1.8, 0.40, 1.2],
        "coefficients": [0.10, 1.35, 0.65, 0.95, 0.55, 1.20, -0.10, 0.55, 0.35, 0.45, 0.30, 0.20],
        "intercept": -1.35,
        "decision_threshold": 0.50,
    },
    {
        "id": "fan",
        "name": "引风机",
        "device_type": "fan",
        "description": "引风机轴承振动为主、电流为辅的异常模型",
        "version": "1.0.0",
        "feature_mean": [1.2, 1.8, 4.5, 1.25, 2.60, 3.2, 8.5, 9.4, 14.5, 2.2, 1.65, 3.1],
        "feature_scale": [0.8, 1.0, 2.2, 0.7, 0.50, 1.5, 3.2, 3.5, 5.0, 1.4, 0.35, 1.4],
        "coefficients": [0.05, 0.95, 0.40, 1.45, 0.80, 1.60, -0.20, 0.35, 0.25, 0.30, 0.20, 0.15],
        "intercept": -1.20,
        "decision_threshold": 0.50,
    },
    {
        "id": "compressor",
        "name": "螺杆压缩机",
        "device_type": "compressor",
        "description": "螺杆压缩机振动与电流多特征异常模型",
        "version": "1.0.0",
        "feature_mean": [2.4, 3.3, 8.0, 2.5, 2.35, 3.4, 24.0, 26.0, 38.0, 5.5, 1.55, 3.5],
        "feature_scale": [1.4, 1.8, 3.5, 1.2, 0.60, 1.8, 8.0, 8.5, 12.0, 3.0, 0.45, 1.6],
        "coefficients": [0.15, 1.10, 0.55, 0.85, 0.45, 0.90, 0.10, 0.80, 0.55, 0.65, 0.35, 0.25],
        "intercept": -1.45,
        "decision_threshold": 0.50,
    },
]


def weights_bytes(artifact: dict[str, Any]) -> int:
    """Byte count of the arrays that stay resident in the cache."""

    total = np.asarray(artifact["coefficients"], dtype=np.float64).nbytes
    total += np.asarray(artifact["feature_mean"], dtype=np.float64).nbytes
    total += np.asarray(artifact["feature_scale"], dtype=np.float64).nbytes
    total += np.asarray(artifact["intercept"], dtype=np.float64).nbytes
    return int(total)


def main() -> int:
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_models = []
    for definition in MODELS:
        array = np.asarray(definition["coefficients"], dtype=np.float64)
        if array.shape != (len(FEATURE_NAMES),):
            raise SystemExit(f"{definition['id']}: expected {len(FEATURE_NAMES)} coefficients")
        if len(definition["feature_mean"]) != len(FEATURE_NAMES):
            raise SystemExit(f"{definition['id']}: feature_mean length mismatch")
        if len(definition["feature_scale"]) != len(FEATURE_NAMES):
            raise SystemExit(f"{definition['id']}: feature_scale length mismatch")
        if any(scale <= 0 for scale in definition["feature_scale"]):
            raise SystemExit(f"{definition['id']}: feature_scale must be positive")

        artifact = {
            "model_id": definition["id"],
            "name": definition["name"],
            "device_type": definition["device_type"],
            "description": definition["description"],
            "version": definition["version"],
            "features": FEATURE_NAMES,
            "feature_mean": definition["feature_mean"],
            "feature_scale": definition["feature_scale"],
            "coefficients": definition["coefficients"],
            "intercept": definition["intercept"],
            "decision_threshold": definition["decision_threshold"],
        }
        file_name = f"{definition['id']}.json"
        (output_dir / file_name).write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        manifest_models.append(
            {
                "id": definition["id"],
                "name": definition["name"],
                "device_type": definition["device_type"],
                "file": file_name,
                "version": definition["version"],
                "weights_bytes": weights_bytes(artifact),
            }
        )

    manifest = {
        "schema_version": 1,
        "feature_names": FEATURE_NAMES,
        "models": manifest_models,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
