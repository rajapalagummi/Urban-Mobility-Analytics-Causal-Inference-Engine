# NYC Rides Marketplace Analytics

> Five empirical analyses on NYC TLC Yellow Taxi data: causal policy impact (DiD), switchback experiment simulation, demand forecasting (SARIMAX vs LightGBM), and cohort survival analysis — with a Streamlit dashboard surfacing marketplace KPIs.

---

## Research Questions

1. Did the 2019 TLC minimum wage policy causally alter tip ratios, fares, and trip patterns across Manhattan vs outer boroughs?
2. Can a switchback experiment detect a 15% surge pricing effect in a two-sided marketplace with interference?
3. Does LightGBM with lag features outperform SARIMAX on hourly demand forecasting?
4. What is the retention lifecycle of pickup zones, and what factors predict early churn?

---

## Methods

**Policy Impact — Difference-in-Differences**
Treatment: Manhattan zones post February 2019 wage policy. Control: Outer borough zones. Parallel trends tested before estimation. Extension: synthetic post-period for January 2025 congestion pricing.

**Switchback Simulation**
Monte Carlo simulation across zone-hour periods. Addresses marketplace interference — the reason standard A/B tests are biased in two-sided networks. Power curve computed across effect sizes 5%-25%.

**Demand Forecasting — Walk-Forward Backtest**
SARIMAX baseline vs LightGBM with lag features (1h, 2h, 3h, 24h, 168h lags + rolling means). Rolling window backtesting with MAPE and RMSE evaluated per fold.

**Cohort Survival — Kaplan-Meier & Cox PH**
Zone-level survival analysis measuring retention decay. Cox proportional hazards model identifies which features are protective vs churn-accelerating.

---

## Stack

| Component | Technology |
|---|---|
| Data | NYC TLC Yellow Taxi (public, free) |
| Query Engine | DuckDB (local, 1B+ row capable) |
| Causal Inference | SciPy, statsmodels |
| Forecasting | statsmodels SARIMAX, LightGBM |
| Survival Analysis | lifelines (KM + Cox PH) |
| Dashboard | Streamlit + Plotly |
| Experiment Tracking | MLflow |
| CI/CD | GitHub Actions |

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python main.py

streamlit run dashboard/app.py

pytest tests/ -v
```

---

## Key Results

| Analysis | Metric | Value |
|---|---|---|
| DiD — Tip Ratio | Estimate | TBD after run |
| DiD — Fare | Estimate | TBD after run |
| Switchback | Power at 15% effect | TBD after run |
| Forecast | LGB MAPE improvement | TBD after run |
| Survival | Median zone retention | TBD after run |
