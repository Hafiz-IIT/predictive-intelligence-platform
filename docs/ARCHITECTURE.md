# Architecture

Numeric series → moving-average / exponential-smoothing baseline → rolling anomaly detector → error metrics and event report.

## Invariants
1. Forecast functions reject invalid inputs.
2. Anomaly scoring uses only prior-window history.
3. Error metrics compare equal-length non-empty series.
