# Description: Short example for MINIMALIST PLOTS README.



# Create sample data

from minimalist_plots import (
    plot_time_series_with_groups,
    plot_trend_line,
    plot_detrended_data,
    plot_forecast,
    plot_statistical_decomposition
)
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'Year': np.arange(2000, 2023),
    'Value': 50 + 0.5 * np.arange(23) + np.random.randn(23) * 2
})

# Plot time series
plot_time_series_with_groups(
    df,
    x_col='Year',
    y_col='Value',
    title='My Time Series',
    xlabel='Year',
    ylabel='Value',
    save_path='my_plot.png'
)

df['Group'] = ['A' if i % 2 == 0 else 'B' for i in range(len(df))]

plot_time_series_with_groups(
    df,
    x_col='Year',
    y_col='Value',
    group_col='Group',
    group_labels={'A': 'Group A', 'B': 'Group B'},
    title='Time Series by Group',
    save_path='grouped_plot.png'
)


model = LinearRegression()
model.fit(df[['Year']], df['Value'])
trend = model.predict(df[['Year']])

plot_trend_line(
    df['Year'],
    trend,
    trend_label='Trend',
    title='Long-term Trend',
    save_path='trend.png'
)

plot_detrended_data(
    df,
    x_col='Year',
    y_col='Value',
    trend_values=trend,
    group_col='Group',  # optional
    title='Detrended Data',
    save_path='detrended.png'
)

fig, ax, future_x, forecast, lower, upper = plot_forecast(
    df,
    x_col='Year',
    y_col='Value',
    trend_model=model,
    n_years_ahead=10,
    step_size=1,
    title='Forecast',
    save_path='forecast.png'
)

fig, axes, decomposition = plot_statistical_decomposition(
    df,
    x_col='Year',
    y_col='Value',
    period=2,
    title='Statistical Decomposition',
    save_path='decomposition.png'
)

plot_time_series_with_groups(
    df,
    x_col='Year',
    y_col='Value',
    colors=['#1f77b4', '#ff7f0e'],  # Custom colors
    linestyles=['-', '--'],         # Custom line styles
    save_path='custom_plot.png'
)
