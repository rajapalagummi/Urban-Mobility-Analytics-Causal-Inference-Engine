import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from data.ingest import get_con


def load_cohort_data(con):
    return con.execute("""
        WITH first_trips AS (
            SELECT
                pickup_zone,
                DATE_TRUNC('month', MIN(pickup_dt)) AS cohort_month,
                MIN(pickup_dt) AS first_trip
            FROM trips
            GROUP BY pickup_zone
        ),
        monthly_activity AS (
            SELECT
                t.pickup_zone,
                DATE_TRUNC('month', t.pickup_dt) AS activity_month,
                COUNT(*) AS trips_in_month,
                AVG(t.fare_amount) AS avg_fare
            FROM trips t
            GROUP BY t.pickup_zone, DATE_TRUNC('month', t.pickup_dt)
        )
        SELECT
            f.pickup_zone,
            f.cohort_month,
            m.activity_month,
            DATEDIFF('month', f.cohort_month, m.activity_month) AS months_since_first,
            m.trips_in_month,
            m.avg_fare
        FROM first_trips f
        JOIN monthly_activity m ON f.pickup_zone = m.pickup_zone
        WHERE m.activity_month >= f.cohort_month
        ORDER BY f.pickup_zone, m.activity_month
    """).df()


def build_survival_dataset(df):
    zone_last = df.groupby("pickup_zone")["months_since_first"].max().reset_index()
    zone_last.columns = ["pickup_zone", "duration"]

    max_obs = df["months_since_first"].max()
    zone_last["event"] = (zone_last["duration"] < max_obs).astype(int)

    zone_features = df.groupby("pickup_zone").agg(
        avg_fare=("avg_fare", "mean"),
        avg_monthly_trips=("trips_in_month", "mean"),
    ).reset_index()

    return zone_last.merge(zone_features, on="pickup_zone")


def run_kaplan_meier(survival_df):
    kmf = KaplanMeierFitter()
    kmf.fit(survival_df["duration"], event_observed=survival_df["event"])

    median_survival = kmf.median_survival_time_
    at_6m = kmf.survival_function_at_times([6]).values[0]
    at_12m = kmf.survival_function_at_times([12]).values[0]

    high_fare = survival_df[survival_df["avg_fare"] >= survival_df["avg_fare"].median()]
    low_fare = survival_df[survival_df["avg_fare"] < survival_df["avg_fare"].median()]

    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    kmf_high.fit(high_fare["duration"], event_observed=high_fare["event"])
    kmf_low.fit(low_fare["duration"], event_observed=low_fare["event"])

    return {
        "median_survival_months": float(median_survival),
        "survival_at_6m": round(float(at_6m), 4),
        "survival_at_12m": round(float(at_12m), 4),
        "high_fare_median_survival": float(kmf_high.median_survival_time_),
        "low_fare_median_survival": float(kmf_low.median_survival_time_),
    }


def run_cox(survival_df):
    df = survival_df[["duration", "event", "avg_fare", "avg_monthly_trips"]].dropna()
    df = df[df["duration"] > 0]

    cph = CoxPHFitter()
    cph.fit(df, duration_col="duration", event_col="event")

    summary = cph.summary[["coef", "exp(coef)", "p"]].round(4).to_dict()
    return {
        "fare_coef": summary["coef"].get("avg_fare"),
        "fare_hazard_ratio": summary["exp(coef)"].get("avg_fare"),
        "fare_p_value": summary["p"].get("avg_fare"),
        "trips_coef": summary["coef"].get("avg_monthly_trips"),
        "trips_hazard_ratio": summary["exp(coef)"].get("avg_monthly_trips"),
        "trips_p_value": summary["p"].get("avg_monthly_trips"),
    }


def run_cohort_analysis():
    con = get_con()
    df = load_cohort_data(con)
    con.close()

    survival_df = build_survival_dataset(df)

    print("Running Kaplan-Meier...")
    km_results = run_kaplan_meier(survival_df)
    print(f"Median survival: {km_results['median_survival_months']} months")
    print(f"Survival at 6m: {km_results['survival_at_6m']}, 12m: {km_results['survival_at_12m']}")

    print("Running Cox PH...")
    cox_results = run_cox(survival_df)
    print(f"Fare hazard ratio: {cox_results['fare_hazard_ratio']}, p={cox_results['fare_p_value']}")

    return {"kaplan_meier": km_results, "cox": cox_results}


if __name__ == "__main__":
    run_cohort_analysis()
