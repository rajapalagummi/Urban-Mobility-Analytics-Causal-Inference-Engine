import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_did_estimate_direction():
    from analysis.causal import run_did
    df = pd.DataFrame({
        "is_manhattan": [1, 1, 0, 0, 1, 1, 0, 0],
        "post_wage_policy": [0, 0, 0, 0, 1, 1, 1, 1],
        "avg_tip_ratio": [0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1],
    })
    result = run_did(df, "avg_tip_ratio")
    assert result["did_estimate"] == pytest.approx(0.05, abs=0.01)
    assert "p_value" in result
    assert "significant" in result


def test_did_returns_all_fields():
    from analysis.causal import run_did
    df = pd.DataFrame({
        "is_manhattan": [1, 0, 1, 0],
        "post_wage_policy": [0, 0, 1, 1],
        "avg_fare": [10.0, 10.0, 12.0, 10.5],
    })
    result = run_did(df, "avg_fare")
    for field in ["did_estimate", "se", "t_stat", "p_value", "significant"]:
        assert field in result


def test_switchback_power_between_0_and_1():
    from analysis.switchback import simulate_switchback
    df = pd.DataFrame({
        "pickup_zone": list(range(20)) * 5,
        "trip_count": np.random.randint(10, 100, 100),
    })
    zone_means = df.groupby("pickup_zone")["trip_count"].mean()

    import unittest.mock as mock
    with mock.patch("analysis.switchback.load_zone_hourly", return_value=df):
        result = simulate_switchback(df, treatment_effect=0.15, n_periods=12, n_simulations=50)
    assert 0 <= result["statistical_power"] <= 1
    assert result["n_simulations"] == 50


def test_lag_features_created():
    from analysis.forecast import make_lag_features
    df = pd.DataFrame({
        "trip_count": range(200),
        "avg_fare": [10.0] * 200,
        "hour": [12] * 200,
        "dow": [1] * 200,
        "month": [1] * 200,
    })
    result = make_lag_features(df, lags=[1, 24])
    assert "lag_1" in result.columns
    assert "lag_24" in result.columns
    assert "rolling_24h" in result.columns
    assert len(result) < len(df)


def test_survival_dataset_has_required_cols():
    from analysis.cohort import build_survival_dataset
    df = pd.DataFrame({
        "pickup_zone": [1, 1, 1, 2, 2],
        "months_since_first": [0, 1, 2, 0, 1],
        "trips_in_month": [10, 8, 5, 12, 10],
        "avg_fare": [12.0, 11.0, 10.0, 15.0, 14.0],
    })
    result = build_survival_dataset(df)
    for col in ["pickup_zone", "duration", "event", "avg_fare"]:
        assert col in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
