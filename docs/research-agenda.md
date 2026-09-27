# Research agenda

## Central question
**When do simple statistical baselines fail enough to justify more complex predictive models in operational time series?**

## Hypotheses
### H1
Moving averages perform poorly under rapid trend shifts but remain competitive on locally stable series.

### H2
Exponential smoothing reacts faster to recent changes as alpha increases, at the cost of noise sensitivity.

### H3
Rolling z-score detectors produce brittle behavior when the recent window has near-zero variance.

## Experiments
1. Generate stationary, trending, seasonal, regime-shift, and spike-contaminated synthetic series.
2. Sweep moving-average window and smoothing alpha.
3. Compare anomaly performance under changing variance and future robust alternatives.

## Metrics
- MAE
- detection precision/recall on synthetic anomalies
- adaptation delay
- false-alarm rate
- parameter sensitivity

## Historical/candidate paper lineage
- **AI for Climate Change: Modeling Micro-Level Energy Efficiency**
- **Smart Urban Infrastructures: AI-Enabled City Optimization**

Research directions only; not publication claims.
