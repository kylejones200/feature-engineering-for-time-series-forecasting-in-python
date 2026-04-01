"""Feature engineering utilities for time series examples."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_lagged_matrix(
    series: pd.Series,
    lags: int | Sequence[int],
) -> pd.DataFrame:
    """Build a lagged design matrix from a univariate time series.

    This is useful for turning a time series into a supervised learning
    problem where past values are used to predict the future.

    Args:
        series: Univariate time series indexed by time.
        lags: Single integer specifying the maximum lag, or an explicit
            sequence of lags to include (e.g. ``[1, 2, 7]``).

    Returns:
        DataFrame whose columns are lagged versions of ``series``.
        Rows with insufficient history are dropped.
    """
    if isinstance(lags, int):
        lag_list = list(range(1, lags + 1))
    else:
        lag_list = sorted(set(int(l) for l in lags if int(l) > 0))

    df = pd.DataFrame({"y": series})
    for lag in lag_list:
        df[f"y_lag_{lag}"] = df["y"].shift(lag)

    result = df.dropna().copy()
    logger.info(
        "Built lagged matrix with %d rows and %d feature columns",
        len(result),
        len(result.columns) - 1,
    )
    return result


def make_supervised_from_series(
    series: pd.Series,
    lags: int | Sequence[int],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create (X, y) arrays for supervised learning from a time series.

    Args:
        series: Univariate time series used as both features and target.
        lags: Single integer for maximum lag, or explicit lag list.

    Returns:
        Tuple ``(X, y)`` where:

        - ``X`` is a DataFrame of lagged features.
        - ``y`` is the aligned target series.
    """
    lagged = build_lagged_matrix(series, lags=lags)
    X = lagged.drop(columns=["y"])
    y = lagged["y"]
    return X, y


