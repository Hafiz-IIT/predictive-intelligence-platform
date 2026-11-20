from __future__ import annotations
from math import isfinite
from predictive_intelligence_platform import mae, moving_average_forecast

def _validate_values(values):
    if not isinstance(values, list) or not values or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in values
    ):
        raise ValueError("values must be a non-empty list of finite numbers")

def walk_forward_moving_average(values: list[float], *, window: int, min_train: int | None = None) -> dict:
    _validate_values(values)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if min_train is not None and (isinstance(min_train, bool) or not isinstance(min_train, int) or min_train < 1):
        raise ValueError("min_train must be a positive integer")
    min_train = window if min_train is None else max(min_train, window)
    if len(values) <= min_train: raise ValueError("series is too short for backtest")
    actual, predicted, indices = [], [], []
    for i in range(min_train, len(values)):
        prediction = moving_average_forecast(values[:i], window)
        indices.append(i); actual.append(float(values[i])); predicted.append(float(prediction))
    return {"indices": indices, "actual": actual, "predicted": predicted,
            "mae": mae(actual, predicted), "window": window}

def compare_windows(values: list[float], windows: list[int]) -> list[dict]:
    _validate_values(values)
    if not isinstance(windows, list) or not windows:
        raise ValueError("windows must be a non-empty list")
    if any(isinstance(w, bool) or not isinstance(w, int) for w in windows):
        raise ValueError("windows must contain integers")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique")
    results = [walk_forward_moving_average(values, window=window) for window in windows]
    return sorted(results, key=lambda row: (row["mae"], row["window"]))
