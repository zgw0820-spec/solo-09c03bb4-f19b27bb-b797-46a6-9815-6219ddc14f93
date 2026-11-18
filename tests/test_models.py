"""Tests for manifest parsing and model-file validation."""

import json

import pytest

from app.errors import ModelLoadError
from app.models import load_manifest, read_model_file


def test_manifest_and_model_round_trip(manifest, model_dir):
    assert set(manifest.descriptors) == {"alpha", "beta", "gamma"}
    model = read_model_file(manifest.descriptors["alpha"], model_dir)
    assert model.model_id == "alpha"
    assert model.weights_bytes == manifest.descriptors["alpha"].declared_bytes
    assert model.decision_threshold == 0.5


@pytest.mark.parametrize(
    "mutation",
    ["model_id", "version", "features", "coefficients", "scale", "size", "threshold"],
)
def test_invalid_model_files_are_rejected(manifest, model_dir, mutation):
    path = model_dir / "alpha.json"
    raw = json.loads(path.read_text())
    if mutation == "model_id":
        raw["model_id"] = "other"
    elif mutation == "version":
        raw["version"] = "9.9.9"
    elif mutation == "features":
        raw["features"] = raw["features"][:-1]
    elif mutation == "coefficients":
        raw["coefficients"] = [1.0, 2.0]
    elif mutation == "scale":
        raw["feature_scale"][0] = 0.0
    elif mutation == "size":
        raw["coefficients"] = [0.1] * 12  # still 296 bytes: patch manifest instead
    elif mutation == "threshold":
        raw["decision_threshold"] = 1.5
    path.write_text(json.dumps(raw))

    if mutation == "size":
        manifest_patch = json.loads((model_dir / "manifest.json").read_text())
        manifest_patch["models"][0]["weights_bytes"] = 999
        (model_dir / "manifest.json").write_text(json.dumps(manifest_patch))
        manifest = load_manifest(model_dir)

    with pytest.raises(ModelLoadError):
        read_model_file(manifest.descriptors["alpha"], model_dir)


def test_manifest_rejects_duplicate_and_unsafe_ids(model_dir):
    raw = json.loads((model_dir / "manifest.json").read_text())
    raw["models"].append(dict(raw["models"][0]))
    (model_dir / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(model_dir)


def test_manifest_rejects_path_traversal(model_dir):
    raw = json.loads((model_dir / "manifest.json").read_text())
    raw["models"][0]["file"] = "../outside.json"
    (model_dir / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(model_dir)


def test_manifest_rejects_feature_schema_drift(model_dir):
    raw = json.loads((model_dir / "manifest.json").read_text())
    raw["feature_names"] = raw["feature_names"][:-1]
    (model_dir / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_manifest(model_dir)
