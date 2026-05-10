#!/usr/bin/env python3
"""Feature engineering for time series — Polars + DuckDB rewrite."""

import sys
import logging
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core import build_lagged_matrix, make_supervised_from_series, add_calendar_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
OUTPUT_DIR = Path(__file__).parent.parent / "images"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # --- synthetic monthly energy series ---
    rng = np.random.default_rng(42)
    n = 120
    t = np.arange(n)
    values = 1000 + 50 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 20, n)
    start = datetime(2015, 1, 1)
    dates = [start + timedelta(days=30 * i) for i in range(n)]

    series = pl.Series("energy_mwh", values.tolist())
    date_series = pl.Series("date", dates)

    # --- Demo 1: lag matrix (lags 1, 2, 3, 12) ---
    logging.info("=== Lag matrix (lags 1,2,3,12) ===")
    lagged = build_lagged_matrix(series, lags=[1, 2, 3, 12])
    logging.info(f"Shape: {lagged.shape}")
    logging.info(f"\n{lagged.head(5)}")

    # --- Demo 2: supervised (X, y) split ---
    logging.info("\n=== Supervised split (max lag = 6) ===")
    X, y = make_supervised_from_series(series, lags=6)
    logging.info(f"X shape: {X.shape}  |  y length: {y.len()}")
    logging.info(f"Feature columns: {X.columns}")

    # --- Demo 3: calendar features ---
    logging.info("\n=== Calendar features ===")
    df = pl.DataFrame({"date": dates, "energy_mwh": values.tolist()})
    df_cal = add_calendar_features(df, date_col="date")
    logging.info(f"\n{df_cal.select(['date', 'year', 'month', 'day_of_week']).head(5)}")

    # save lag matrix for inspection
    lagged.write_csv(OUTPUT_DIR / "lag_matrix.csv")
    logging.info(f"\nLag matrix saved → {OUTPUT_DIR / 'lag_matrix.csv'}")


if __name__ == "__main__":
    main()
