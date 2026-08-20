import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="NYC Rides Marketplace Analytics", layout="wide")
st.title("NYC Rides Marketplace Analytics")
st.caption("TLC Yellow Taxi — Policy Impact, Surge Simulation, Demand Forecasting, Cohort Survival")

RESULTS_PATH = Path("data/results.json")

if not RESULTS_PATH.exists():
    st.warning("Run `python main.py` first to generate results.")
    st.stop()

with open(RESULTS_PATH) as f:
    results = json.load(f)

tab1, tab2, tab3, tab4 = st.tabs([
    "Policy Impact (DiD)",
    "Switchback Simulation",
    "Demand Forecasting",
    "Cohort Survival"
])

with tab1:
    st.subheader("2019 TLC Minimum Wage Policy — Difference-in-Differences")
    causal = results.get("causal", {})

    outcomes = ["avg_tip_ratio", "avg_fare", "avg_distance", "avg_duration"]
    rows = []
    for outcome in outcomes:
        r = causal.get(outcome, {})
        if r:
            rows.append({
                "Outcome": outcome.replace("avg_", "").replace("_", " ").title(),
                "DiD Estimate": r.get("did_estimate"),
                "p-value": r.get("p_value"),
                "Significant": "✅" if r.get("significant") else "❌",
                "Treatment Pre": r.get("treat_pre"),
                "Treatment Post": r.get("treat_post"),
            })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        fig = px.bar(
            pd.DataFrame(rows),
            x="Outcome", y="DiD Estimate",
            color="Significant",
            color_discrete_map={"✅": "#2ecc71", "❌": "#e74c3c"},
            title="DiD Estimates by Outcome — 2019 Wage Policy"
        )
        st.plotly_chart(fig, use_container_width=True)

    congestion = causal.get("congestion_extension", {})
    if congestion:
        st.subheader("Congestion Pricing Extension (Synthetic Post-Period)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Surcharge", f"${congestion.get('congestion_surcharge')}")
        col2.metric("Est. Fare Increase", f"${congestion.get('estimated_fare_increase')}")
        col3.metric("Est. Demand Change", f"{congestion.get('estimated_demand_reduction_pct')}%")
        st.caption(congestion.get("note", ""))

with tab2:
    st.subheader("Switchback Experiment Simulation — 15% Surge Multiplier")
    sb = results.get("switchback", {})
    primary = sb.get("primary", {})

    if primary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Statistical Power", f"{primary.get('statistical_power', 0)*100:.1f}%")
        col2.metric("Mean Estimated Effect", f"{primary.get('mean_estimated_effect', 0)*100:.1f}%")
        col3.metric("95% CI Lower", f"{primary.get('ci_lower', 0)*100:.1f}%")
        col4.metric("95% CI Upper", f"{primary.get('ci_upper', 0)*100:.1f}%")

    curve = sb.get("power_curve", [])
    if curve:
        curve_df = pd.DataFrame(curve)
        fig = px.line(
            curve_df, x="treatment_effect", y="statistical_power",
            title="Statistical Power vs Treatment Effect Size",
            labels={"treatment_effect": "Effect Size", "statistical_power": "Power"},
            markers=True
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                      annotation_text="80% power threshold")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Monte Carlo switchback simulation accounts for marketplace interference — standard A/B tests are biased in two-sided networks.")

with tab3:
    st.subheader("Hourly Demand Forecasting — SARIMAX vs LightGBM")
    fc = results.get("forecast", {})

    if fc:
        col1, col2, col3 = st.columns(3)
        col1.metric("SARIMA MAPE", f"{fc.get('sarima_avg_mape', 0)*100:.1f}%")
        col2.metric("LightGBM MAPE", f"{fc.get('lgb_avg_mape', 0)*100:.1f}%")
        improvement = fc.get("mape_improvement_pct")
        col3.metric("LGB Improvement", f"{improvement:.1f}%" if improvement else "N/A",
                    delta="LGB wins" if fc.get("lgb_wins") else "SARIMA wins")

        models = ["SARIMAX Baseline", "LightGBM + Lag Features"]
        mapes = [fc.get("sarima_avg_mape", 0), fc.get("lgb_avg_mape", 0)]
        fig = px.bar(
            x=models, y=[m * 100 for m in mapes],
            labels={"x": "Model", "y": "MAPE (%)"},
            title="Walk-Forward Backtest — Mean Absolute Percentage Error",
            color=models,
            color_discrete_sequence=["#e74c3c", "#2ecc71"]
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Zone Cohort Survival Analysis — Kaplan-Meier & Cox PH")
    cohort = results.get("cohort", {})
    km = cohort.get("kaplan_meier", {})
    cox = cohort.get("cox", {})

    if km:
        col1, col2, col3 = st.columns(3)
        col1.metric("Median Survival", f"{km.get('median_survival_months', 0):.1f} months")
        col2.metric("Survival at 6 months", f"{km.get('survival_at_6m', 0)*100:.1f}%")
        col3.metric("Survival at 12 months", f"{km.get('survival_at_12m', 0)*100:.1f}%")

        fare_data = {
            "Group": ["High Fare Zones", "Low Fare Zones"],
            "Median Survival (months)": [
                km.get("high_fare_median_survival", 0),
                km.get("low_fare_median_survival", 0)
            ]
        }
        fig = px.bar(fare_data, x="Group", y="Median Survival (months)",
                     title="Survival by Fare Tier",
                     color="Group",
                     color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig, use_container_width=True)

    if cox:
        st.subheader("Cox Proportional Hazards — Feature Impact on Churn Risk")
        cox_rows = [
            {"Feature": "Avg Fare", "Hazard Ratio": cox.get("fare_hazard_ratio"),
             "p-value": cox.get("fare_p_value")},
            {"Feature": "Avg Monthly Trips", "Hazard Ratio": cox.get("trips_hazard_ratio"),
             "p-value": cox.get("trips_p_value")},
        ]
        st.dataframe(pd.DataFrame(cox_rows), use_container_width=True)
        st.caption("Hazard ratio > 1 = higher churn risk. Hazard ratio < 1 = lower churn risk (protective factor).")
