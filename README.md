# Predictive Intelligence Platform

A small baseline toolkit for forecasting and anomaly detection on numeric time series.

## Implemented
- moving-average forecast
- exponential smoothing
- rolling z-score anomaly detection
- train/test error calculation (MAE)
- deterministic tests

## Run
```bash
python -m unittest discover -s tests -v
python predictive_intelligence_platform.py
```

## Scope
This repository provides transparent statistical baselines on synthetic/example data. It is not a production forecasting service and does not claim domain-specific predictive accuracy.
