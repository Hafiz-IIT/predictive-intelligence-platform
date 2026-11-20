from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite, sqrt


def _finite_series(values: list[float], name: str = "values") -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must be a non-empty list or tuple")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError(f"{name} must contain numeric values")
    try:
        converted = [float(value) for value in values]
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
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
    sample = data[-window:]
    scale = max(abs(value) for value in sample)
    if scale == 0:
        return 0.0
    return scale * (fsum(value / scale for value in sample) / window)


def exponential_smoothing(values: list[float], alpha: float) -> float:
    data = _finite_series(values)
    if not data:
        raise ValueError("values required")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be finite and in (0, 1]")
    try:
        alpha = float(alpha)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("alpha must be finite and in (0, 1]") from exc
    if not isfinite(alpha) or not 0 < alpha <= 1:
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
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("threshold must be finite and positive")
    try:
        threshold = float(threshold)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("threshold must be finite and positive") from exc
    if not isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    data = _finite_series(values)
    out: list[Anomaly] = []
    for i in range(window, len(data)):
        history = data[i-window:i]
        # Include the candidate in the scale so current/history ratios stay bounded.
        scale = max(max(abs(value) for value in history), abs(data[i]))
        if scale == 0:
            mean_scaled = 0.0
            std_scaled = 0.0
            current_scaled = 0.0
        else:
            scaled = [value / scale for value in history]
            mean_scaled = fsum(scaled) / window
            variance_scaled = fsum((value - mean_scaled) ** 2 for value in scaled) / window
            std_scaled = sqrt(variance_scaled)
            current_scaled = data[i] / scale
        # A nonzero observation after a zero-variance history is represented by None,
        # not infinity, so result artifacts remain valid JSON.
        z = None if std_scaled == 0 and current_scaled != mean_scaled else (
            0.0 if std_scaled == 0 else (current_scaled - mean_scaled) / std_scaled
        )
        if z is None or abs(z) >= threshold:
            out.append(Anomaly(i, data[i], z))
    return out


def mae(actual: list[float], predicted: list[float]) -> float:
    actual_data = _finite_series(actual, "actual")
    predicted_data = _finite_series(predicted, "predicted")
    if len(actual_data) != len(predicted_data) or not actual_data:
        raise ValueError("equal non-empty series required")
    errors = [abs(a - p) for a, p in zip(actual_data, predicted_data)]
    if any(not isfinite(error) for error in errors):
        raise OverflowError("absolute prediction error is outside finite numeric range")
    return fsum(errors) / len(errors)


if __name__ == "__main__":
    series = [10, 11, 10, 12, 11, 50]
    print("next:", moving_average_forecast(series[:-1], 3))
    print("anomalies:", rolling_anomalies(series, window=5, threshold=3))
