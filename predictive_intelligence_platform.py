from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


def _finite_series(values: list[float], name: str = "values") -> list[float]:
    try:
        converted = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain finite numbers") from None
    if not all(isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain finite numbers")
    return converted


def _positive_int(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def moving_average_forecast(values: list[float], window: int) -> float:
    _positive_int(window, "window")
    data = _finite_series(values)
    if len(data) < window:
        raise ValueError("not enough values for window")
    return sum(data[-window:]) / window


def exponential_smoothing(values: list[float], alpha: float) -> float:
    data = _finite_series(values)
    if not data:
        raise ValueError("values required")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    level = data[0]
    for value in data[1:]:
        level = alpha * value + (1 - alpha) * level
    return level


@dataclass(frozen=True)
class Anomaly:
    index: int
    value: float
    z_score: float | None


def rolling_anomalies(values: list[float], *, window: int = 5, threshold: float = 3.0) -> list[Anomaly]:
    _positive_int(window, "window", minimum=2)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and non-negative")
    data = _finite_series(values)
    out: list[Anomaly] = []
    for i in range(window, len(data)):
        history = data[i-window:i]
        mean = sum(history) / window
        variance = sum((x - mean) ** 2 for x in history) / window
        std = sqrt(variance)
        # A z-score is undefined when historical variance is zero.
        z = None if std == 0 and data[i] != mean else (0.0 if std == 0 else (data[i] - mean) / std)
        if z is None or abs(z) >= threshold:
            out.append(Anomaly(i, data[i], z))
    return out


def mae(actual: list[float], predicted: list[float]) -> float:
    actual_data = _finite_series(actual, "actual")
    predicted_data = _finite_series(predicted, "predicted")
    if len(actual_data) != len(predicted_data) or not actual_data:
        raise ValueError("equal non-empty series required")
    return sum(abs(a - p) for a, p in zip(actual_data, predicted_data)) / len(actual_data)


if __name__ == "__main__":
    series = [10, 11, 10, 12, 11, 50]
    print("next:", moving_average_forecast(series[:-1], 3))
    print("anomalies:", rolling_anomalies(series, window=5, threshold=3))
