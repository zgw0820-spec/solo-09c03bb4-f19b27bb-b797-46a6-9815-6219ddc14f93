"""Deterministic feature extraction shared by all anomaly models.

Each signal is reduced to summary statistics that are meaningful for both
vibration and current waveforms.  The models standardise these features with
their own stored ``feature_mean`` / ``feature_scale`` arrays, so the values are
intentionally left in their engineering units here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
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
)

# Kurtosis can grow without bound for nearly constant signals; clipping keeps
# the downstream dot product finite without materially changing anomaly scores.
_FEATURE_CLIP = 1.0e9


def _signal_stats(signal: np.ndarray) -> tuple[float, float, float, float, float, float]:
    absolute = np.abs(signal)
    mean = float(np.mean(signal))
    rms = float(np.sqrt(np.mean(np.square(signal))))
    peak = float(np.max(absolute))
    std = float(np.std(signal))

    crest = peak / rms if rms > 1.0e-12 else 0.0

    if std > 1.0e-12:
        centred = signal - mean
        kurtosis = float(np.mean(np.power(centred, 4)) / np.power(std, 4))
    else:
        kurtosis = 0.0

    return mean, rms, peak, std, crest, kurtosis


def extract_features(vibration: Sequence[float], current: Sequence[float]) -> np.ndarray:
    """Return the fixed 12-element feature vector for one measurement window."""

    vibration_array = np.asarray(vibration, dtype=np.float64)
    current_array = np.asarray(current, dtype=np.float64)

    if vibration_array.ndim != 1 or current_array.ndim != 1:
        raise ValueError("vibration and current must be one-dimensional")
    if vibration_array.size == 0 or current_array.size == 0:
        raise ValueError("vibration and current must contain at least one sample")
    if not np.all(np.isfinite(vibration_array)) or not np.all(np.isfinite(current_array)):
        raise ValueError("measurements must be finite")

    values = np.concatenate((_signal_stats(vibration_array), _signal_stats(current_array)))
    values = np.clip(values, -_FEATURE_CLIP, _FEATURE_CLIP)
    return np.ascontiguousarray(values, dtype=np.float64)
