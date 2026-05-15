"""Time series feature engineering using Polars and DuckDB.

build_lagged_matrix: replaces pandas .shift() loop with DuckDB LAG() window functions.
add_calendar_features: adds datetime components as ML-ready columns.
"""

from __future__ import annotations

import logging
from typing import Sequence, Union

import duckdb
import polars as pl

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def build_lagged_matrix(
    series: pl.Series,
    lags: Union[int, Sequence[int]],
    date_series: pl.Series | None = None,
) -> pl.DataFrame:
    """Build a lagged design matrix via DuckDB LAG() window functions.

    Each lag becomes one column (y_lag_1, y_lag_2, …). Rows where any
    lag is null (the warm-up period) are dropped, matching the pandas
    .dropna() behaviour of the original.
    """
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = sorted({int(l) for l in lags if int(l) > 0})

    lag_exprs = ",\n            ".join(
        f"LAG(y, {lag}) OVER (ORDER BY idx) AS y_lag_{lag}" for lag in lag_list
    )

    idx = list(range(len(series)))
    pl.DataFrame({"idx": idx, "y": series})

    result = (
        duckdb.sql(f"""
            SELECT y, {lag_exprs}
            FROM df
            ORDER BY idx
        """)
        .pl()
        .drop_nulls()
    )

    logger.info(
        "Built lagged matrix: %d rows, %d lag columns",
        result.height,
        len(lag_list),
    )
    return result


def make_supervised_from_series(
    series: pl.Series,
    lags: Union[int, Sequence[int]],
) -> tuple[pl.DataFrame, pl.Series]:
    """Return (X, y) for supervised learning from a univariate time series."""
    lagged = build_lagged_matrix(series, lags=lags)
    X = lagged.drop("y")
    y = lagged["y"]
    return X, y


def add_calendar_features(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """Add year, month, day-of-week, and hour columns from a datetime column."""
    return duckdb.sql(f"""
        SELECT
            *,
            YEAR("{date_col}")        AS year,
            MONTH("{date_col}")       AS month,
            DAYOFWEEK("{date_col}")   AS day_of_week,
            HOUR("{date_col}")        AS hour
        FROM df
    """).pl()
