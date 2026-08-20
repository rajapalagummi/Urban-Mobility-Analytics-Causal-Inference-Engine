import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
from data.ingest import get_con


def load_hourly_demand(con, zone_id=161):
    return con.execute(f"""
        SELECT
            DATE_TRUNC('hour', pickup_dt) AS hour_ts,
            COUNT(*) AS trip_count,
            AVG(fare_amount) AS avg_fare,
            EXTRACT(hour FROM pickup_dt) AS hour,
            EXTRACT(dow FROM pickup_dt) AS dow,
            EXTRACT(month FROM pickup_dt) AS month
        FROM trips
        WHERE pickup_zone = {zone_id}
          AND year IN (2018, 2019)
        GROUP BY DATE_TRUNC('hour', pickup_dt), EXTRACT(hour FROM pickup_dt),
                 EXTRACT(dow FROM pickup_dt), EXTRACT(month FROM pickup_dt)
        ORDER BY hour_ts
    """).df()


def make_lag_features(df, target="trip_count", lags=[1, 2, 3, 24, 168]):
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target].shift(lag)
    df["rolling_24h"] = df[target].shift(1).rolling(24, min_periods=1).mean()
    df["rolling_7d"] = df[target].shift(1).rolling(168, min_periods=1).mean()
    return df.dropna()


def walk_forward_sarima(series, n_test=168, step=24):
    results = []
    for start in range(0, n_test - step, step):
        train = series.iloc[:-(n_test - start)]
        test = series.iloc[-(n_test - start):-(n_test - start - step) or None]
        try:
            m = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24),
                        enforce_stationarity=False, enforce_invertibility=False)
            fit = m.fit(disp=False)
            pred = fit.forecast(step)
            mape = mean_absolute_percentage_error(test.clip(1), pred.clip(1))
            results.append({"mape": mape, "rmse": np.sqrt(mean_squared_error(test, pred))})
        except Exception:
            continue
    return results


def walk_forward_lgb(df, target="trip_count", n_test=168, step=24):
    feature_cols = [c for c in df.columns if c not in [target, "hour_ts"]]
    results = []
    for start in range(0, n_test - step, step):
        train = df.iloc[:-(n_test - start)]
        test = df.iloc[-(n_test - start):-(n_test - start - step) or None]
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05,
                                   num_leaves=31, random_state=42, verbose=-1)
        model.fit(train[feature_cols], train[target])
        pred = model.predict(test[feature_cols]).clip(0)
        mape = mean_absolute_percentage_error(test[target].clip(1), pred.clip(1))
        results.append({"mape": mape, "rmse": np.sqrt(mean_squared_error(test[target], pred))})
    return results


def run_forecast_comparison():
    con = get_con()
    df = load_hourly_demand(con)
    con.close()

    if len(df) < 200:
        print("Insufficient data for zone 161, using zone 237")
        con = get_con()
        df = load_hourly_demand(con, zone_id=237)
        con.close()

    df_features = make_lag_features(df)
    series = df_features["trip_count"]

    print("Running SARIMA walk-forward backtest...")
    sarima_results = walk_forward_sarima(series)

    print("Running LightGBM walk-forward backtest...")
    lgb_results = walk_forward_lgb(df_features)

    sarima_mape = np.mean([r["mape"] for r in sarima_results]) if sarima_results else None
    lgb_mape = np.mean([r["mape"] for r in lgb_results]) if lgb_results else None

    improvement = None
    if sarima_mape and lgb_mape:
        improvement = (sarima_mape - lgb_mape) / sarima_mape * 100

    result = {
        "sarima_avg_mape": round(sarima_mape, 4) if sarima_mape else None,
        "lgb_avg_mape": round(lgb_mape, 4) if lgb_mape else None,
        "mape_improvement_pct": round(improvement, 2) if improvement else None,
        "lgb_wins": lgb_mape < sarima_mape if (lgb_mape and sarima_mape) else None,
        "n_folds": len(sarima_results),
    }

    print(f"SARIMA MAPE: {result['sarima_avg_mape']}, LGB MAPE: {result['lgb_avg_mape']}, "
          f"Improvement: {result['mape_improvement_pct']}%")
    return result


if __name__ == "__main__":
    run_forecast_comparison()
