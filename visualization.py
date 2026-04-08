"""Common visualization utilities for time series examples.

These helpers implement a minimalist plotting style and are aligned with the
API used in ``MINIMALIST_PLOTS_README.py``.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _apply_minimalist_style(ax: plt.Axes) -> None:
    """Apply a simple minimalist style to a Matplotlib axis."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def plot_time_series_with_groups(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    group_labels: Optional[Mapping[str, str]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    colors: Optional[Sequence[str]] = None,
    linestyles: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a univariate time series, optionally grouped by a categorical column."""
    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    _apply_minimalist_style(ax)

    x = df[x_col]
    y = df[y_col]

    if group_col is None:
        ax.plot(x, y, color=colors[0] if colors else "#1f77b4", linestyle="-")
    else:
        groups = df[group_col].astype(str)
        unique_groups = groups.unique()
        for idx, g in enumerate(unique_groups):
            mask = groups == g
            label = group_labels[g] if group_labels and g in group_labels else str(g)
            color = colors[idx % len(colors)] if colors else None
            linestyle = linestyles[idx % len(linestyles)] if linestyles else "-"
            ax.plot(x[mask], y[mask], label=label, color=color, linestyle=linestyle)
        ax.legend(frameon=False)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.tight_layout()

    if save_path:
        logger.info("Saving time series plot to '%s'", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_trend_line(
    x: Iterable[float],
    trend_values: Iterable[float],
    trend_label: str = "Trend",
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a trend line against a time or index axis."""
    x_arr = np.asarray(list(x))
    trend_arr = np.asarray(list(trend_values))

    fig: plt.Figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    _apply_minimalist_style(ax)
    ax.plot(x_arr, trend_arr, label=trend_label, color="#1f77b4")
    ax.legend(frameon=False)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    fig.tight_layout()

    if save_path:
        logger.info("Saving trend plot to '%s'", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_detrended_data(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    trend_values: Sequence[float],
    group_col: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot detrended series (original minus trend)."""
    detrended = df[y_col].values - np.asarray(trend_values)
    tmp = df.copy()
    tmp["detrended"] = detrended

    fig, ax = plot_time_series_with_groups(
        tmp,
        x_col=x_col,
        y_col="detrended",
        group_col=group_col,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel or f"{y_col} (detrended)",
        save_path=None,
    )

    if save_path:
        logger.info("Saving detrended plot to '%s'", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_forecast(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    trend_model,
    n_years_ahead: int = 10,
    step_size: int = 1,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plot a simple forecast with normal-based confidence intervals."""
    x = df[x_col].values.reshape(-1, 1)
    y = df[y_col].values

    # In-sample fit and residuals
    y_hat = trend_model.predict(x)
    residuals = y - y_hat
    sigma = residuals.std(ddof=1)

    last_x = x.max()
    future_x = np.arange(last_x + step_size, last_x + step_size * (n_years_ahead + 1), step_size)
    future_x_2d = future_x.reshape(-1, 1)
    forecast = trend_model.predict(future_x_2d)

    z = 1.96  # approx 95% CI
    lower = forecast - z * sigma
    upper = forecast + z * sigma

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_minimalist_style(ax)

    ax.plot(df[x_col], df[y_col], label="Observed", color="#1f77b4")
    ax.plot(future_x, forecast, label="Forecast", color="#ff7f0e")
    ax.fill_between(future_x, lower, upper, color="#ff7f0e", alpha=0.2, label="95% CI")
    ax.legend(frameon=False)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel(x_col)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel(y_col)

    fig.tight_layout()

    if save_path:
        logger.info("Saving forecast plot to '%s'", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax, future_x, forecast, lower, upper


def plot_statistical_decomposition(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    period: int,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray, seasonal_decompose]:
    """Perform and plot a classical seasonal decomposition."""
    series = df.set_index(x_col)[y_col].asfreq("D")
    decomposition = seasonal_decompose(series, model="additive", period=period)

    fig = decomposition.plot()
    axes = np.asarray(fig.axes)
    for ax in axes:
        _apply_minimalist_style(ax)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        fig.tight_layout()

    if save_path:
        logger.info("Saving decomposition plot to '%s'", save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, decomposition


