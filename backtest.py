from __future__ import annotations

import math

from predictive_intelligence_platform import mae, moving_average_forecast


def walk_forward_moving_average(
    values: list[float],
    *,
    window: int,
    min_train: int | None = None,
) -> dict:
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError("window must be a positive integer")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in values):
        raise ValueError("time-series values must be finite numbers")
    if min_train is not None and (not isinstance(min_train, int) or isinstance(min_train, bool) or min_train < 1):
        raise ValueError("min_train must be a positive integer")
    min_train = window if min_train is None else max(min_train, window)
    if len(values) <= min_train:
        raise ValueError("series is too short for backtest")

    actual: list[float] = []
    predicted: list[float] = []
    indices: list[int] = []

    for i in range(min_train, len(values)):
        history = values[:i]
        prediction = moving_average_forecast(history, window)
        indices.append(i)
        actual.append(float(values[i]))
        predicted.append(float(prediction))

    return {
        "indices": indices,
        "actual": actual,
        "predicted": predicted,
        "mae": mae(actual, predicted),
        "window": window,
    }


def compare_windows(values: list[float], windows: list[int]) -> list[dict]:
    if not windows or any(not isinstance(window, int) or isinstance(window, bool) or window < 1 for window in windows):
        raise ValueError("windows must be a non-empty list of positive integers")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique")
    results = [
        walk_forward_moving_average(values, window=window)
        for window in windows
    ]
    return sorted(results, key=lambda row: (row["mae"], row["window"]))
