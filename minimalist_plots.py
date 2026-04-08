"""Backwards-compatible wrapper around the shared visualization utilities.

The examples in ``MINIMALIST_PLOTS_README.py`` import from ``minimalist_plots``.
To keep that API stable while centralizing logic, this module simply re-exports
the plotting helpers from ``visualization.py``.
"""

from visualization import (
    plot_detrended_data,
    plot_forecast,
    plot_statistical_decomposition,
    plot_time_series_with_groups,
    plot_trend_line,
)

