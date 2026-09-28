from __future__ import annotations

from predictive_intelligence_platform import mae, moving_average_forecast


def walk_forward_moving_average(
    values: list[float],
    *,
    window: int,
    min_train: int | None = None,
) -> dict:
    if window < 1:
        raise ValueError("window must be >= 1")
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
    results = [
        walk_forward_moving_average(values, window=window)
        for window in windows
    ]
    return sorted(results, key=lambda row: (row["mae"], row["window"]))
