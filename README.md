# Predictive Intelligence Platform

> **Transparent forecasting and anomaly-detection baselines before the word ‘AI’ hides the assumptions.**

Many predictive-system concepts jump straight to complex models without a reproducible baseline. This repository keeps forecasting and anomaly detection deliberately transparent so future ML methods must demonstrate measurable improvement.

## Implemented
- moving-average forecast
- exponential smoothing
- rolling z-score anomaly detection
- zero-variance anomaly handling
- mean absolute error evaluation
- input validation

## Run
```bash
python -m unittest discover -s tests -v
python predictive_intelligence_platform.py
```

## Repository map
- `predictive_intelligence_platform.py` — implementation
- `tests/` — tests
- `examples/` — example input
- `docs/architecture.md` — architecture
- `docs/research-agenda.md` — experiments + research lineage
- `STATUS.md` — claims boundary
- `CITATION.cff` — citation

## Pipeline
**numeric series → baseline forecast/smoothing → rolling history → anomaly score → evaluation**

## Research lineage
This repo comes from the older Predictive Intelligence Platform, trade forecasting, risk/anomaly detection, city prediction, and operational analytics ideas.

## Evaluation direction
Benchmark the baselines on synthetic trend/seasonality/shift/outlier series, then compare future learned models against the same frozen splits and metrics.

## Maturity
**Research prototype.** These are statistical baselines, not a production forecasting service, not a trained AI platform, and not evidence of predictive performance on a real business domain.
