# Description: Short example for Feature Engineering for Time Series Forecasting in Python.



from data_io import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Example Time Series
y = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
# Min-Max Scaling
scaler = MinMaxScaler()
y_minmax = scaler.fit_transform(y)
# Standardization
scaler_std = StandardScaler()
y_std = scaler_std.fit_transform(y)
logger.info("Min-Max Scaled:", y_minmax.flatten())
logger.info("Standardized:", y_std.flatten())


###
# To avoid leakage, apply the fit_transform to training data only
# use transform on the test values so they are in the same scale
###

scaler.fit_transform(train['values'])  # Train values only
scaler.transform(test['values'])

# Example Time Series
y = pd.Series([10, 12, 15, 19, 24])
# First Difference
y_diff = y.diff().dropna()
logger.info("First Difference:", y_diff.values)

# Example Time Series
y = np.array([10, 12, 15, 19, 24])
# First Derivative (Rate of Change)
dy = np.gradient(y)
logger.info("First Derivative (Rate of Change):", dy)
# Second Derivative (Acceleration)
d2y = np.gradient(dy)
logger.info("Second Derivative (Acceleration):", d2y)

# Simulated Time Series Data
np.random.seed(42)
data = pd.Series(np.cumsum(np.random.randn(200))) # Random walk time series
# Create Features: Lagged Values and Rate of Change
df = pd.DataFrame({
'value': data,
'lag_1': data.shift(1),
'lag_2': data.shift(2),
'rate_of_change': data.diff()
}).dropna()

df['rolling_mean'] = df['values'].shift(1).rolling(7, min_periods=1).mean()

result = seasonal_decompose(y, period=12)
result.seasonal.head()

# Adding Time Features
df.index = pd.date_range(start="2023–01", periods=len(df), freq="M")
df_features = pd.DataFrame({"month": df.index.month, "year": df.index.year})
logger.info(df_features.head())


# Load the ERCOT data
df = read_csv("ercot_load_data.csv")
df['date'] = pd.to_datetime(df['date'])  # Ensure 'date' is in datetime format
df['values'] = pd.to_numeric(df['values'], errors='coerce')  # Convert 'values' to numeric
df = df.sort_values('date')  # Sort by date

# Drop rows with missing or NaN values
df = df.dropna()

# Resample the data to hourly frequency (mean aggregation)
df = df.set_index('date').resample('H').mean().reset_index()  # Resample to hourly frequency

# Define hold-out period (e.g., last 24 hours)
hold_out_hours = 24  # Hold-out size (24 hours = 1 day)
train = df.iloc[:-hold_out_hours]
hold_out = df.iloc[-hold_out_hours:]

"""
Feature Engineering for Time Series 

All transformations respect temporal order and use only available information.
"""


# Configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
np.random.seed(42)

# Generate synthetic time series
dates = pd.date_range('2023-01-01', '2025-10-31', freq='D')
n = len(dates)
trend = np.linspace(100, 150, n)
seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
noise = np.random.normal(0, 5, n)

df = pd.DataFrame({'values': trend + seasonal + noise}, index=dates)

# Temporal train/test split
split_idx = int(len(df) * 0.9)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

logger.info(f"Train: {len(train)} samples | Test: {len(test)} samples")

# Scaling: fit on train, transform both
scaler = StandardScaler()
train['scaled'] = scaler.fit_transform(train[['values']])
test['scaled'] = scaler.transform(test[['values']])

# Differencing
train['diff'] = train['values'].diff()
test['diff'] = test['values'].diff()

# Gradients
train['gradient'] = np.gradient(train['values'])
test['gradient'] = np.gradient(test['values'])

# Rolling statistics (time-safe with shift)
train['rolling_mean'] = train['values'].shift(1).rolling(3, min_periods=1).mean()
test['rolling_mean'] = test['values'].shift(1).rolling(3, min_periods=1).mean()

# Lag features
for lag in [1, 2, 7]:
    train[f'lag_{lag}'] = train['values'].shift(lag)
    test[f'lag_{lag}'] = test['values'].shift(lag)

# Calendar features
train['month'] = train.index.month
train['dayofweek'] = train.index.dayofweek
test['month'] = test.index.month
test['dayofweek'] = test.index.dayofweek

# Seasonal decomposition (train only)
seasonal_model = seasonal_decompose(train['values'], period=365, model='additive')
train['seasonal'] = seasonal_model.seasonal
train['trend'] = seasonal_model.trend

# Feature matrix
features = ['scaled', 'diff', 'gradient', 'rolling_mean', 
            'lag_1', 'lag_2', 'lag_7', 'month', 'dayofweek']

X_train = train[features].dropna()
y_train = train.loc[X_train.index, 'values']

X_test = test[features].dropna()
y_test = test.loc[X_test.index, 'values']

logger.info(f"Features: {len(features)} | Train: {len(X_train)} | Test: {len(X_test)}")

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Original values
axes[0].plot(train.index, train['values'], label='Train', linewidth=1.2, color='#2c3e50')
axes[0].plot(test.index, test['values'], label='Test', linewidth=1.2, color='#e74c3c')
axes[0].axvline(train.index[-1], color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[0].set_ylabel('Original Values')
axes[0].set_title('Time Series Feature Engineering', fontweight='bold')
axes[0].legend(frameon=False, loc='upper left')

# Scaled values
axes[1].plot(train.index, train['scaled'], label='Train (Scaled)', linewidth=1.2, color='#2c3e50')
axes[1].plot(test.index, test['scaled'], label='Test (Scaled)', linewidth=1.2, color='#e74c3c')
axes[1].axvline(train.index[-1], color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[1].set_ylabel('Standardized Values')
axes[1].legend(frameon=False, loc='upper left')

# Rolling mean
axes[2].plot(train.index, train['values'], alpha=0.3, linewidth=1, color='#2c3e50')
axes[2].plot(train.index, train['rolling_mean'], label='Train (3-day MA)', linewidth=1.2, color='#3498db')
axes[2].plot(test.index, test['values'], alpha=0.3, linewidth=1, color='#e74c3c')
axes[2].plot(test.index, test['rolling_mean'], label='Test (3-day MA)', linewidth=1.2, color='#e67e22')
axes[2].axvline(train.index[-1], color='gray', linestyle='--', alpha=0.5, linewidth=1)
axes[2].set_ylabel('Values')
axes[2].set_xlabel('Date')
axes[2].legend(frameon=False, loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig('feature_engineering_synthetic_data.png', dpi=150, bbox_inches='tight')
logger.info("Saved: feature_engineering_synthetic_data.png")

"""
Exponential Smoothing forecast example
Assumes train and hold_out DataFrames already exist
"""



# Fit model
model = ExponentialSmoothing(
    train['values'],
    trend='additive',
    seasonal='additive',
    seasonal_periods=24
)
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=len(hold_out))
hold_out['forecast'] = forecast.values

# Evaluate
mape = mean_absolute_percentage_error(hold_out['values'], hold_out['forecast'])
logger.info(f"MAPE: {mape * 100:.2f}%")

# Visualize
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(train['date'], train['values'], label='Train', color='#2c3e50', linewidth=1.2)
ax.plot(hold_out['date'], hold_out['values'], label='Test (Actual)', color='#27ae60', linewidth=1.2)
ax.plot(hold_out['date'], hold_out['forecast'], label='Forecast', color='#e74c3c', 
        linestyle='--', linewidth=1.2)

ax.set_title(f'Exponential Smoothing Forecast (MAPE: {mape * 100:.2f}%)', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Values')
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig('forecast_holdout_synthetic_data.png', dpi=150, bbox_inches='tight')
logger.info("Saved: forecast_holdout_synthetic_data.png")

df['weight_lag1'] = df['weight'].shift(1)
df['weight_derivative'] = df['weight'].diff()
df.head()

"""
Time Series as Supervised Learning - Synthetic Temperature Data
Sliding window approach with Random Forest for temperature forecasting
"""


np.random.seed(42)

def create_features(series, window_size=7):
    """Create lagged features for supervised learning."""
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

# Generate synthetic temperature data
n_samples = 1000
time = np.arange(n_samples)
seasonal = 10 * np.sin(2 * np.pi * time / 365)
trend = 0.01 * time
noise = np.random.randn(n_samples) * 2
temperature = 20 + seasonal + trend + noise

# Create features with 7-day window
window_size = 7
X, y = create_features(temperature, window_size)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

logger.info(f"Training samples: {len(X_train)}")
logger.info(f"Test samples: {len(X_test)}")

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

logger.info(f"\nTrain RMSE: {train_rmse:.2f}°C")
logger.info(f"Test RMSE:  {test_rmse:.2f}°C")

# Time series CV
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for train_idx, val_idx in tscv.split(X_train):
    cv_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    cv_model.fit(X_train[train_idx], y_train[train_idx])
    cv_pred = cv_model.predict(X_train[val_idx])
    cv_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], cv_pred)))

logger.info(f"CV RMSE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}°C")

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

test_indices = np.arange(train_size, train_size + len(y_test))
ax1.plot(test_indices, y_test, label='Actual', linewidth=2, alpha=0.7)
ax1.plot(test_indices, y_pred_test, label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Temperature Forecasting: Actual vs Predicted')
ax1.legend()

residuals = y_test - y_pred_test
ax2.scatter(y_pred_test, residuals, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Temperature (°C)')
ax2.set_ylabel('Residuals (°C)')
ax2.set_title('Residual Plot')

plt.tight_layout()
plt.savefig('supervised_learning_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance
importances = model.feature_importances_
feature_names = [f'Day-{i+1}' for i in range(window_size)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(feature_names, importances)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance: Which Days Matter Most?')
plt.tight_layout()
plt.savefig('sliding_window_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Multi-step forecasting
forecast_horizon = 3
X_multi, y_multi = [], []
for i in range(len(temperature) - window_size - forecast_horizon + 1):
    X_multi.append(temperature[i:i + window_size])
    y_multi.append(temperature[i + window_size:i + window_size + forecast_horizon])
X_multi, y_multi = np.array(X_multi), np.array(y_multi)

train_size_multi = int(len(X_multi) * 0.8)
X_train_multi = X_multi[:train_size_multi]
y_train_multi = y_multi[:train_size_multi]
X_test_multi = X_multi[train_size_multi:]
y_test_multi = y_multi[train_size_multi:]

model_multi = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = model_multi.predict(X_test_multi)

logger.info(f"\nMulti-step forecasting:")
for h in range(forecast_horizon):
    horizon_rmse = np.sqrt(mean_squared_error(y_test_multi[:, h], y_pred_multi[:, h]))
    logger.info(f"Day +{h+1} RMSE: {horizon_rmse:.2f}°C")

logger.info('\nOutputs: supervised_learning_forecast.png, sliding_window_feature_importance.png')
