from __future__ import annotations
from predictive_intelligence_platform import _finite_series, _positive_int, mae, moving_average_forecast

def walk_forward_moving_average(values: list[float], *, window: int, min_train: int | None = None) -> dict:
    data = _finite_series(values)
    _positive_int(window, "window")
    if min_train is not None:
        _positive_int(min_train, "min_train")
    min_train = window if min_train is None else max(min_train, window)
    if len(data) <= min_train:
        raise ValueError("series is too short for backtest")
    actual, predicted, indices = [], [], []
    for i in range(min_train, len(data)):
        prediction = moving_average_forecast(data[:i], window)
        indices.append(i)
        actual.append(data[i])
        predicted.append(float(prediction))
    return {"indices": indices, "actual": actual, "predicted": predicted,
            "mae": mae(actual, predicted), "window": window}

def compare_windows(values: list[float], windows: list[int]) -> list[dict]:
    _finite_series(values)
    if not isinstance(windows, list) or not windows:
        raise ValueError("windows must be a non-empty list")
    for window in windows:
        _positive_int(window, "window")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must be unique")
    results = [walk_forward_moving_average(values, window=window) for window in windows]
    return sorted(results, key=lambda row: (row["mae"], row["window"]))
