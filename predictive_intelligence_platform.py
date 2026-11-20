from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt

def _series(values: list[float]) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError("values must be a non-empty list")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in values):
        raise ValueError("series values must be finite numbers")
    return [float(v) for v in values]

def _positive_int(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value

def moving_average_forecast(values: list[float], window: int) -> float:
    values = _series(values)
    window = _positive_int("window", window)
    if len(values) < window: raise ValueError("not enough values for window")
    return sum(values[-window:]) / window

def exponential_smoothing(values: list[float], alpha: float) -> float:
    values = _series(values)
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    level = values[0]
    for value in values[1:]: level = alpha * value + (1 - alpha) * level
    return level

@dataclass(frozen=True)
class Anomaly:
    index: int
    value: float
    z_score: float

def rolling_anomalies(values: list[float], *, window: int = 5, threshold: float = 3.0) -> list[Anomaly]:
    values = _series(values)
    window = _positive_int("window", window, 2)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    out: list[Anomaly] = []
    for i in range(window, len(values)):
        history = values[i-window:i]
        mean = sum(history) / window
        variance = sum((x - mean) ** 2 for x in history) / window
        std = sqrt(variance)
        z = (values[i] - mean) / std if std else (float("inf") if values[i] != mean else 0.0)
        if abs(z) >= threshold: out.append(Anomaly(i, values[i], z))
    return out

def mae(actual: list[float], predicted: list[float]) -> float:
    actual, predicted = _series(actual), _series(predicted)
    if len(actual) != len(predicted): raise ValueError("equal non-empty series required")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)

if __name__ == "__main__":
    series = [10, 11, 10, 12, 11, 50]
    print("next:", moving_average_forecast(series[:-1], 3))
    print("anomalies:", rolling_anomalies(series, window=5, threshold=3))
