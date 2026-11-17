from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, isfinite, copysign


def moving_average_forecast(values: list[float], window: int) -> float:
    if not isinstance(window, int) or isinstance(window, bool) or window < 1 or len(values) < window:
        raise ValueError("window must be a positive integer with enough observations")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) for value in values):
        raise ValueError("time-series values must be finite numbers")
    return sum(values[-window:]) / window


def exponential_smoothing(values: list[float], alpha: float) -> float:
    if not values:
        raise ValueError("values required")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) for value in values):
        raise ValueError("time-series values must be finite numbers")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1 - alpha) * level
    return level


@dataclass(frozen=True)
class Anomaly:
    index: int
    value: float
    z_score: float


def rolling_anomalies(values: list[float], *, window: int = 5, threshold: float = 3.0) -> list[Anomaly]:
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError("window must be an integer >= 2")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value) for value in values):
        raise ValueError("time-series values must be finite numbers")
    out: list[Anomaly] = []
    for i in range(window, len(values)):
        history = values[i-window:i]
        mean = sum(history) / window
        variance = sum((x - mean) ** 2 for x in history) / window
        std = sqrt(variance)
        if std == 0:
            # A finite sentinel flags deviation from a constant history without
            # leaking infinity into metrics, JSON, or downstream visualizations.
            z = copysign(float(threshold), values[i] - mean) if values[i] != mean else 0.0
        else:
            z = (values[i] - mean) / std
        if abs(z) >= threshold:
            out.append(Anomaly(i, values[i], z))
    return out


def mae(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("equal non-empty series required")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value)
           for value in [*actual, *predicted]):
        raise ValueError("actual and predicted values must be finite numbers")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


if __name__ == "__main__":
    series = [10, 11, 10, 12, 11, 50]
    print("next:", moving_average_forecast(series[:-1], 3))
    print("anomalies:", rolling_anomalies(series, window=5, threshold=3))
