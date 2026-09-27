# Architecture

```mermaid
flowchart LR
    N0[numeric series] --> N1
    N1[baseline forecast/smoothing] --> N2
    N2[rolling history] --> N3
    N3[anomaly score] --> N4
    N4[evaluation]
```

## Forecast baselines
Moving average and exponential smoothing provide interpretable predictions.

## Anomaly baseline
Rolling z-score compares a new observation with recent history.

## Evaluation
MAE provides a basic common scale for forecast comparison.

## Future layer
More complex models should be added only with side-by-side baseline evaluation.

## Design principle
A complex model should earn its complexity by beating simple baselines on frozen data.
