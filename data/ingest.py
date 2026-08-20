import os
import duckdb
import requests
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
DB_PATH = "data/nyc_rides.duckdb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TLC_URLS = {
    "2018_01": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2018-01.parquet",
    "2018_06": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2018-06.parquet",
    "2019_01": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-01.parquet",
    "2019_06": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-06.parquet",
    "2019_09": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-09.parquet",
    "2019_12": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2019-12.parquet",
    "2020_01": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-01.parquet",
    "2020_06": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2020-06.parquet",
}

ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"


def download_file(url, path):
    if path.exists():
        return
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


def fetch_data():
    for key, url in TLC_URLS.items():
        path = RAW_DIR / f"{key}.parquet"
        print(f"Fetching {key}...")
        download_file(url, path)

    zone_path = RAW_DIR / "zone_lookup.csv"
    download_file(ZONE_LOOKUP_URL, zone_path)
    print("All files downloaded")


def build_database():
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE OR REPLACE TABLE trips AS
        SELECT
            tpep_pickup_datetime AS pickup_dt,
            tpep_dropoff_datetime AS dropoff_dt,
            passenger_count,
            trip_distance,
            PULocationID AS pickup_zone,
            DOLocationID AS dropoff_zone,
            fare_amount,
            tip_amount,
            total_amount,
            CAST(strftime(tpep_pickup_datetime, '%Y') AS INTEGER) AS year,
            CAST(strftime(tpep_pickup_datetime, '%m') AS INTEGER) AS month,
            CAST(strftime(tpep_pickup_datetime, '%H') AS INTEGER) AS hour,
            EXTRACT(dow FROM tpep_pickup_datetime) AS dow,
            DATEDIFF('minute', tpep_pickup_datetime, tpep_dropoff_datetime) AS duration_min,
            CASE WHEN tip_amount > 0 THEN tip_amount / fare_amount ELSE 0 END AS tip_ratio,
            CASE
                WHEN CAST(strftime(tpep_pickup_datetime, '%Y-%m') AS VARCHAR) >= '2019-02'
                THEN 1 ELSE 0
            END AS post_wage_policy
        FROM read_parquet('data/raw/*.parquet')
        WHERE fare_amount > 2.5
          AND trip_distance > 0
          AND trip_distance < 100
          AND tpep_pickup_datetime >= '2018-01-01'
          AND tpep_pickup_datetime < '2021-01-01'
          AND passenger_count > 0
    """)

    con.execute("""
        CREATE OR REPLACE TABLE zones AS
        SELECT * FROM read_csv_auto('data/raw/zone_lookup.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE trips_with_zones AS
        SELECT t.*, z.Zone AS pickup_zone_name, z.Borough AS pickup_borough
        FROM trips t
        LEFT JOIN zones z ON t.pickup_zone = z.LocationID
    """)

    row_count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Database built: {row_count:,} trips loaded")
    con.close()
    return row_count


def get_con():
    return duckdb.connect(DB_PATH)


if __name__ == "__main__":
    fetch_data()
    build_database()
