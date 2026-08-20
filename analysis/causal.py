import numpy as np
import pandas as pd
import duckdb
from scipy import stats
from data.ingest import get_con


def load_did_data(con):
    return con.execute("""
        SELECT
            year,
            month,
            post_wage_policy,
            CASE WHEN pickup_borough IN ('Manhattan') THEN 1 ELSE 0 END AS is_manhattan,
            AVG(trip_distance) AS avg_distance,
            AVG(duration_min) AS avg_duration,
            AVG(tip_ratio) AS avg_tip_ratio,
            AVG(fare_amount) AS avg_fare,
            COUNT(*) AS trip_count
        FROM trips_with_zones
        WHERE year IN (2018, 2019, 2020)
          AND duration_min BETWEEN 1 AND 120
          AND fare_amount BETWEEN 2.5 AND 200
        GROUP BY year, month, post_wage_policy, is_manhattan
        ORDER BY year, month
    """).df()


def run_did(df, outcome):
    treat_post = df[(df["is_manhattan"] == 1) & (df["post_wage_policy"] == 1)][outcome].mean()
    treat_pre = df[(df["is_manhattan"] == 1) & (df["post_wage_policy"] == 0)][outcome].mean()
    ctrl_post = df[(df["is_manhattan"] == 0) & (df["post_wage_policy"] == 1)][outcome].mean()
    ctrl_pre = df[(df["is_manhattan"] == 0) & (df["post_wage_policy"] == 0)][outcome].mean()

    did_estimate = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)

    treat_series = df[df["is_manhattan"] == 1][outcome]
    ctrl_series = df[df["is_manhattan"] == 0][outcome]
    se = np.sqrt(treat_series.var() / len(treat_series) + ctrl_series.var() / len(ctrl_series))
    t_stat = did_estimate / (se + 1e-9)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "outcome": outcome,
        "did_estimate": round(did_estimate, 6),
        "se": round(se, 6),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "treat_pre": round(treat_pre, 4),
        "treat_post": round(treat_post, 4),
        "ctrl_pre": round(ctrl_pre, 4),
        "ctrl_post": round(ctrl_post, 4),
    }


def test_parallel_trends(df, outcome):
    pre_df = df[df["post_wage_policy"] == 0].copy()
    treat = pre_df[pre_df["is_manhattan"] == 1].groupby(["year", "month"])[outcome].mean()
    ctrl = pre_df[pre_df["is_manhattan"] == 0].groupby(["year", "month"])[outcome].mean()
    n = min(len(treat), len(ctrl))
    if n < 3:
        return {"p_value": None, "satisfied": None}
    t, p = stats.ttest_ind(np.diff(treat.values[:n]), np.diff(ctrl.values[:n]))
    return {"p_value": round(float(p), 4), "satisfied": p > 0.05}


def synthetic_congestion_extension(df):
    np.random.seed(42)
    base_manhattan = df[df["is_manhattan"] == 1]["avg_fare"].mean()
    congestion_surcharge = 9.0
    synthetic_post = df[df["is_manhattan"] == 1].copy()
    synthetic_post["avg_fare"] = synthetic_post["avg_fare"] + congestion_surcharge * np.random.uniform(0.6, 0.9, len(synthetic_post))
    synthetic_post["synthetic_period"] = "post_congestion"

    demand_elasticity = -0.15
    synthetic_post["trip_count"] = (synthetic_post["trip_count"] * (1 + demand_elasticity)).astype(int)

    return {
        "method": "synthetic_extension",
        "congestion_surcharge": congestion_surcharge,
        "assumed_demand_elasticity": demand_elasticity,
        "estimated_fare_increase": round(congestion_surcharge * 0.75, 2),
        "estimated_demand_reduction_pct": round(demand_elasticity * 100, 1),
        "note": "Synthetic post-period — congestion pricing went live Jan 2025, pre-policy data used as baseline",
    }


def run_causal_analysis():
    con = get_con()
    df = load_did_data(con)
    con.close()

    outcomes = ["avg_tip_ratio", "avg_fare", "avg_distance", "avg_duration"]
    results = {}

    for outcome in outcomes:
        parallel = test_parallel_trends(df, outcome)
        did = run_did(df, outcome)
        did["parallel_trends"] = parallel
        results[outcome] = did
        print(f"{outcome}: DiD={did['did_estimate']}, p={did['p_value']}, sig={did['significant']}")

    congestion = synthetic_congestion_extension(df)
    results["congestion_extension"] = congestion

    return results, df


if __name__ == "__main__":
    results, df = run_causal_analysis()
