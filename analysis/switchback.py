import numpy as np
import pandas as pd
from scipy import stats
from data.ingest import get_con


def load_zone_hourly(con):
    return con.execute("""
        SELECT
            pickup_zone,
            pickup_borough,
            hour,
            dow,
            year,
            month,
            COUNT(*) AS trip_count,
            AVG(fare_amount) AS avg_fare,
            AVG(tip_ratio) AS avg_tip_ratio
        FROM trips_with_zones
        WHERE year = 2019
        GROUP BY pickup_zone, pickup_borough, hour, dow, year, month
        ORDER BY pickup_zone, hour
    """).df()


def simulate_switchback(df, treatment_effect=0.15, n_periods=48, n_simulations=1000):
    zone_means = df.groupby("pickup_zone")["trip_count"].mean()
    zones = zone_means.index.tolist()
    n_zones = len(zones)

    detected = 0
    estimates = []

    for _ in range(n_simulations):
        assignments = np.zeros((n_zones, n_periods))
        for t in range(n_periods):
            assignments[:, t] = np.random.binomial(1, 0.5, n_zones)

        treat_outcomes = []
        ctrl_outcomes = []

        for z_idx, zone in enumerate(zones):
            base = zone_means[zone]
            for t in range(n_periods):
                noise = np.random.normal(0, base * 0.1)
                if assignments[z_idx, t] == 1:
                    treat_outcomes.append(base * (1 + treatment_effect) + noise)
                else:
                    ctrl_outcomes.append(base + noise)

        t_stat, p_value = stats.ttest_ind(treat_outcomes, ctrl_outcomes)
        if p_value < 0.05:
            detected += 1
        estimates.append(np.mean(treat_outcomes) / np.mean(ctrl_outcomes) - 1)

    power = detected / n_simulations
    mean_estimate = np.mean(estimates)
    ci_lower = np.percentile(estimates, 2.5)
    ci_upper = np.percentile(estimates, 97.5)

    return {
        "treatment_effect": treatment_effect,
        "n_periods": n_periods,
        "n_simulations": n_simulations,
        "n_zones": n_zones,
        "statistical_power": round(power, 4),
        "mean_estimated_effect": round(mean_estimate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "mde_achieved": treatment_effect,
    }


def power_curve(df, effects=None, n_periods=48, n_simulations=500):
    if effects is None:
        effects = [0.05, 0.10, 0.15, 0.20, 0.25]
    return [simulate_switchback(df, e, n_periods, n_simulations) for e in effects]


def run_switchback_analysis():
    con = get_con()
    df = load_zone_hourly(con)
    con.close()

    print("Running switchback simulation for 15% surge effect...")
    primary = simulate_switchback(df, treatment_effect=0.15)
    print(f"Power: {primary['statistical_power']}, CI: [{primary['ci_lower']}, {primary['ci_upper']}]")

    print("Computing power curve...")
    curve = power_curve(df)
    for r in curve:
        print(f"Effect={r['treatment_effect']}: power={r['statistical_power']}")

    return {"primary": primary, "power_curve": curve}


if __name__ == "__main__":
    run_switchback_analysis()
