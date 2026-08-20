import json
import mlflow
from pathlib import Path
from data.ingest import fetch_data, build_database
from analysis.causal import run_causal_analysis
from analysis.switchback import run_switchback_analysis
from analysis.forecast import run_forecast_comparison
from analysis.cohort import run_cohort_analysis

Path("data").mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("nyc_rides")

print("=== NYC Rides Marketplace Analytics ===")

print("\n[1/5] Fetching TLC data...")
fetch_data()
build_database()

print("\n[2/5] Running causal analysis (DiD)...")
causal_results, _ = run_causal_analysis()

print("\n[3/5] Running switchback simulation...")
switchback_results = run_switchback_analysis()

print("\n[4/5] Running demand forecasting...")
forecast_results = run_forecast_comparison()

print("\n[5/5] Running cohort survival analysis...")
cohort_results = run_cohort_analysis()

results = {
    "causal": causal_results,
    "switchback": switchback_results,
    "forecast": forecast_results,
    "cohort": cohort_results,
}

with open("data/results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n=== Results ===")
for outcome in ["avg_tip_ratio", "avg_fare", "avg_distance"]:
    r = causal_results.get(outcome, {})
    print(f"DiD {outcome}: {r.get('did_estimate')} (p={r.get('p_value')})")

sb = switchback_results.get("primary", {})
print(f"Switchback power: {sb.get('statistical_power')}")
print(f"Forecast LGB improvement: {forecast_results.get('mape_improvement_pct')}%")
km = cohort_results.get("kaplan_meier", {})
print(f"Median zone survival: {km.get('median_survival_months')} months")
print("\nResults saved to data/results.json")
print("Run dashboard: streamlit run dashboard/app.py")
