from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


def moving_average_forecast(values: list[float], window: int) -> float:
    if window < 1 or len(values) < window:
        raise ValueError("not enough values for window")
    return sum(values[-window:]) / window


def exponential_smoothing(values: list[float], alpha: float) -> float:
    if not values:
        raise ValueError("values required")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
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
    if window < 2:
        raise ValueError("window must be >= 2")
    out: list[Anomaly] = []
    for i in range(window, len(values)):
        history = values[i-window:i]
        mean = sum(history) / window
        variance = sum((x - mean) ** 2 for x in history) / window
        std = sqrt(variance)
        if std == 0:
            z = float("inf") if values[i] != mean else 0.0
        else:
            z = (values[i] - mean) / std
        if abs(z) >= threshold:
            out.append(Anomaly(i, values[i], z))
    return out


def mae(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("equal non-empty series required")
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


if __name__ == "__main__":
    series = [10, 11, 10, 12, 11, 50]
    print("next:", moving_average_forecast(series[:-1], 3))
    print("anomalies:", rolling_anomalies(series, window=5, threshold=3))
