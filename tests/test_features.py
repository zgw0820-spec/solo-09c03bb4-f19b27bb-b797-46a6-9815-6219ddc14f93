import math

import numpy as np
import pytest

from app.features import FEATURE_NAMES, extract_features


def test_feature_vector_is_deterministic_and_ordered():
    vibration = [1.0, -1.0, 2.0, -2.0]
    current = [10.0, 12.0, 10.0, 12.0]

    first = extract_features(vibration, current)
    second = extract_features(np.asarray(vibration), np.asarray(current))

    assert first.shape == (len(FEATURE_NAMES),)
    assert np.array_equal(first, second)

    values = dict(zip(FEATURE_NAMES, first, strict=True))
    assert values["vibration_mean"] == pytest.approx(0.0)
    assert values["vibration_rms"] == pytest.approx(math.sqrt(2.5))
    assert values["vibration_peak"] == pytest.approx(2.0)
    assert values["current_mean"] == pytest.approx(11.0)
    assert values["current_peak"] == pytest.approx(12.0)


def test_constant_signal_does_not_produce_nan():
    features = extract_features([3.0, 3.0, 3.0], [7.0, 7.0, 7.0])
    assert np.all(np.isfinite(features))
    values = dict(zip(FEATURE_NAMES, features, strict=True))
    assert values["vibration_crest"] == pytest.approx(1.0)
    assert values["vibration_kurtosis"] == pytest.approx(0.0)
    assert values["current_crest"] == pytest.approx(1.0)
    assert values["current_kurtosis"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "vibration,current",
    [
        ([], [1.0]),
        ([1.0], []),
        ([1.0, float("nan")], [1.0]),
        ([float("inf")], [1.0]),
        ([[1.0, 2.0]], [1.0]),
    ],
)
def test_invalid_measurements_are_rejected(vibration, current):
    with pytest.raises(ValueError):
        extract_features(vibration, current)


def test_kurtosis_is_clipped_for_extreme_spikes():
    samples = [0.0] * 999 + [1.0e12]
    features = extract_features(samples, samples)
    assert np.all(np.isfinite(features))
    assert np.max(features) <= 1.0e9
