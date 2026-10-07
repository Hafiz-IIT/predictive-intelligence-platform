# Predictive Intelligence Platform

<p align="center"><strong>Transparent Forecasting Before Complex ML</strong><br/><sub>Baselines, walk-forward evaluation and anomaly detection for numeric time series.</sub></p>

<p align="center"><img src="https://img.shields.io/badge/status-reproducible%20prototype-blue" alt="Prototype"/> <img src="https://img.shields.io/badge/focus-time--series-orange" alt="Time series"/></p>

## Question

**What can a simple, inspectable forecasting baseline establish before a project reaches for a complex model?**

```
Time series
   ↓
Moving average / smoothing
   ↓
Walk-forward evaluation
   ↓
MAE comparison
   +
Rolling anomaly detection
```

## Try it

```bash
python predictive_intelligence_platform.py
python -m unittest discover -s tests -v
```

`backtest.py` adds walk-forward evaluation and automatic comparison of forecast windows.

## Implemented

- moving-average forecast
- exponential smoothing
- rolling z-score anomalies
- MAE metric
- walk-forward backtesting
- window comparison
- deterministic CI

## Research boundary

Baseline forecasting toolkit only. No claim of predictive superiority or production forecasting accuracy.

Related: [Privacy-Aware Recommender Lab](https://github.com/Hafiz-IIT/privacy-aware-recommender-lab)
