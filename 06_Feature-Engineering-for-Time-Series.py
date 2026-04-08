import logging

logger = logging.getLogger(__name__)

# Extracted code from '06_Feature-Engineering-for-Time-Series.md'
# Blocks appear in the same order as in the markdown article.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Load energy use data and aggregate across all MSN codes
data_path = BASE_DIR / "data" / "use_OK.csv"
df = pd.read_csv(data_path)

use_cols = [col for col in df.columns if col.isdigit()]
year_totals = df[use_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)

df_base = pd.DataFrame(
    {"value": year_totals.values},
    index=pd.to_datetime(year_totals.index, format="%Y"),
).sort_index()
logger.info(f"Time series length: {len(df_base)}")
logger.info(f"Date range: {df_base.index.min()} to {df_base.index.max()}")

def create_temporal_features(df):
    """Create temporal embedding features"""
    features = df.copy()
    
    # Extract time components
    features['year'] = features.index.year
    features['month'] = features.index.month if hasattr(features.index, 'month') else 1
    features['day_of_year'] = features.index.dayofyear if hasattr(features.index, 'dayofyear') else 1
    
    # Cyclical encoding for year (captures long-term cycles)
    max_year = features['year'].max()
    min_year = features['year'].min()
    year_range = max_year - min_year
    features['year_sin'] = np.sin(2 * np.pi * (features['year'] - min_year) / year_range)
    features['year_cos'] = np.cos(2 * np.pi * (features['year'] - min_year) / year_range)
    
    # Cyclical encoding for month (if monthly data)
    if 'month' in features.columns:
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Fourier features for seasonality (multiple frequencies)
    for freq in [1, 2, 4]:  # Annual, semi-annual, quarterly
        features[f'fourier_sin_{freq}'] = np.sin(2 * np.pi * freq * features['year'] / year_range)
        features[f'fourier_cos_{freq}'] = np.cos(2 * np.pi * freq * features['year'] / year_range)
    
    # Time since start (trend feature)
    features['time_since_start'] = (features['year'] - min_year)
    
    # Decade and century (categorical-like features)
    features['decade'] = (features['year'] // 10) * 10
    features['century'] = (features['year'] // 100) * 100
    
    return features

# Create temporal features
df_features = create_temporal_features(df_base)
logger.info(f"\nTemporal features created: {len([c for c in df_features.columns if c not in ['value']])}")
logger.info(f"Feature columns: {[c for c in df_features.columns if c != 'value']}")

def create_lag_features(df, target_col='value', lags=[1, 2, 3, 4, 5, 6, 12, 24]):
    """Create lag features with multiple windows"""
    features = df.copy()
    
    for lag in lags:
        if lag <= len(features):
            features[f'lag_{lag}'] = features[target_col].shift(lag)
    
    # Lag differences (capture changes)
    for lag in [1, 2, 3]:
        if f'lag_{lag}' in features.columns:
            features[f'lag_diff_{lag}'] = features[target_col] - features[f'lag_{lag}']
            features[f'lag_pct_change_{lag}'] = features[target_col].pct_change(lag)
    
    return features

# Add lag features
df_features = create_lag_features(df_features)
logger.info(f"\nLag features added")
logger.info(f"Total features: {len(df_features.columns)}")

def create_rolling_features(df, target_col='value', windows=[3, 6, 12, 24]):
    """Create rolling statistical features"""
    features = df.copy()
    
    for window in windows:
        if window <= len(features):
            # Rolling mean (trend)
            features[f'rolling_mean_{window}'] = features[target_col].rolling(window, min_periods=1).mean()
            
            # Rolling std (volatility)
            features[f'rolling_std_{window}'] = features[target_col].rolling(window, min_periods=1).std()
            
            # Rolling min/max (range)
            features[f'rolling_min_{window}'] = features[target_col].rolling(window, min_periods=1).min()
            features[f'rolling_max_{window}'] = features[target_col].rolling(window, min_periods=1).max()
            
            # Rolling median (robust to outliers)
            features[f'rolling_median_{window}'] = features[target_col].rolling(window, min_periods=1).median()
            
            # Distance from rolling mean (anomaly indicator)
            features[f'dist_from_mean_{window}'] = features[target_col] - features[f'rolling_mean_{window}']
            
            # Coefficient of variation
            features[f'cv_{window}'] = features[f'rolling_std_{window}'] / (features[f'rolling_mean_{window}'] + 1e-10)
    
    return features

# Add rolling features
df_features = create_rolling_features(df_features)
logger.info(f"\nRolling features added")
logger.info(f"Total features: {len(df_features.columns)}")

def create_change_features(df, target_col='value'):
    """Create change and rate features"""
    features = df.copy()
    
    # First difference
    features['diff_1'] = features[target_col].diff(1)
    features['diff_2'] = features[target_col].diff(2)
    features['diff_3'] = features[target_col].diff(3)
    
    # Percentage change
    features['pct_change_1'] = features[target_col].pct_change(1)
    features['pct_change_2'] = features[target_col].pct_change(2)
    features['pct_change_3'] = features[target_col].pct_change(3)
    
    # Log difference (for multiplicative processes)
    features['log_diff'] = np.log(features[target_col] + 1e-10) - np.log(features[target_col].shift(1) + 1e-10)
    
    # Acceleration (second derivative)
    features['acceleration'] = features['diff_1'].diff(1)
    
    # Year-over-year change (if annual data)
    if len(features) > 1:
        features['yoy_change'] = features[target_col] - features[target_col].shift(1)
        features['yoy_pct'] = features[target_col].pct_change(1)
    
    # Cumulative sum (trend indicator)
    features['cumsum'] = features[target_col].cumsum()
    
    return features

# Add change features
df_features = create_change_features(df_features)
logger.info(f"\nChange features added")
logger.info(f"Total features: {len(df_features.columns)}")

def create_domain_features(df, target_col='value'):
    """Create domain-specific features for energy consumption"""
    features = df.copy()
    
    # Energy-specific features
    # Growth rate (exponential growth indicator)
    features['growth_rate'] = features[target_col] / (features[target_col].shift(1) + 1e-10)
    
    # Energy intensity (if population data available)
    # features['energy_per_capita'] = features[target_col] / population_data
    
    # Economic cycle indicators (proxy using rolling statistics)
    features['economic_cycle'] = (features[target_col] - features['rolling_mean_12']) / (features['rolling_std_12'] + 1e-10)
    
    # Policy period indicators (example: post-2000 environmental policies)
    features['post_2000'] = (features.index.year >= 2000).astype(int)
    features['post_2010'] = (features.index.year >= 2010).astype(int)
    
    # Technology adoption phases (proxy using time since start)
    features['tech_phase'] = pd.cut(
        features['time_since_start'],
        bins=[0, 20, 40, 60, 100],
        labels=['Early', 'Mid', 'Late', 'Modern']
    )
    features['tech_phase'] = features['tech_phase'].cat.codes
    
    return features

# Add domain features
df_features = create_domain_features(df_features)
logger.info(f"\nDomain features added")
logger.info(f"Total features: {len(df_features.columns)}")

def create_external_features(df, energy_data_path=None):
    """Create features from external data sources"""
    features = df.copy()
    
    # Example: Add production data as external regressor
    if energy_data_path:
        try:
            pr_df = pd.read_csv(energy_data_path)
            # Process and merge production data
            # This is a placeholder - actual implementation would merge on year
            pass
        except:
            pass
    
    # Economic indicators (proxy using time-based features)
    # In practice, you'd merge actual GDP, unemployment, etc.
    features['economic_growth_proxy'] = features['value'].pct_change(1).rolling(3).mean()
    
    # Technology adoption (proxy using time)
    features['tech_adoption'] = features['time_since_start'] ** 0.5  # Diminishing returns
    
    # Seasonal economic factors (if monthly/quarterly data)
    # features['quarter'] = features.index.quarter if hasattr(features.index, 'quarter') else 1
    
    return features

# Add external features
# df_features = create_external_features(df_features, "geospatial/datasets/pr_OK.csv")
logger.info(f"\nExternal features framework ready")

from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Prepare data for feature selection
feature_cols = [c for c in df_features.columns if c != 'value']
X = df_features[feature_cols].fillna(0)  # Simple imputation for demo
y = df_features['value'].fillna(method='ffill')

# Remove rows with NaN target
valid_idx = ~y.isna()
X = X[valid_idx]
y = y[valid_idx]

logger.info(f"Features for selection: {len(feature_cols)}")
logger.info(f"Valid samples: {len(X)}")

# Method 1: Mutual Information
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

logger.info("\nTop 10 features by Mutual Information:")
logger.info(mi_df.head(10))

# Method 2: Random Forest Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

logger.info("\nTop 10 features by Random Forest Importance:")
logger.info(rf_importance.head(10))

# Select top features
top_n = 20
top_features_mi = mi_df.head(top_n)['feature'].tolist()
top_features_rf = rf_importance.head(top_n)['feature'].tolist()

# Combine both methods
selected_features = list(set(top_features_mi + top_features_rf))
logger.info(f"\nSelected {len(selected_features)} features")

# Visualize feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mutual Information
top_mi = mi_df.head(15)
axes[0].barh(range(len(top_mi)), top_mi['mi_score'].values, color='#1f77b4', alpha=0.8)
axes[0].set_yticks(range(len(top_mi)))
axes[0].set_yticklabels(top_mi['feature'].values)
axes[0].set_xlabel('Mutual Information Score', fontsize=11)
axes[0].set_title('Top Features by Mutual Information', fontsize=13, fontweight='bold')
# Random Forest Importance
top_rf = rf_importance.head(15)
axes[1].barh(range(len(top_rf)), top_rf['importance'].values, color='#ff7f0e', alpha=0.8)
axes[1].set_yticks(range(len(top_rf)))
axes[1].set_yticklabels(top_rf['feature'].values)
axes[1].set_xlabel('Feature Importance', fontsize=11)
axes[1].set_title('Top Features by Random Forest', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Baseline: Only lag features
X_baseline = df_features[['lag_1', 'lag_2', 'lag_3']].fillna(0)
y_baseline = df_features['value'].fillna(method='ffill')
valid_baseline = ~y_baseline.isna()
X_baseline = X_baseline[valid_baseline]
y_baseline = y_baseline[valid_baseline]

# Advanced: Selected features
X_advanced = df_features[selected_features].fillna(0)
y_advanced = df_features['value'].fillna(method='ffill')
valid_advanced = ~y_advanced.isna()
X_advanced = X_advanced[valid_advanced]
y_advanced = y_advanced[valid_advanced]

# Split
X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
    X_baseline, y_baseline, test_size=0.2, random_state=42, shuffle=False
)

X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
    X_advanced, y_advanced, test_size=0.2, random_state=42, shuffle=False
)

# Train models
model_baseline = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_baseline.fit(X_b_train, y_b_train)

model_advanced = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_advanced.fit(X_a_train, y_a_train)

# Evaluate
pred_baseline = model_baseline.predict(X_b_test)
pred_advanced = model_advanced.predict(X_a_test)

mae_baseline = mean_absolute_error(y_b_test, pred_baseline)
mae_advanced = mean_absolute_error(y_a_test, pred_advanced)

rmse_baseline = np.sqrt(mean_squared_error(y_b_test, pred_baseline))
rmse_advanced = np.sqrt(mean_squared_error(y_a_test, pred_advanced))

logger.info("\n" + "="*60)
logger.info("FEATURE ENGINEERING IMPACT")
logger.info("="*60)
logger.info(f"{'Model':<20} {'MAE':<15} {'RMSE':<15}")
logger.info("-"*60)
logger.info(f"{'Baseline (3 lags)':<20} {mae_baseline:<15.2f} {rmse_baseline:<15.2f}")
logger.info(f"{'Advanced Features':<20} {mae_advanced:<15.2f} {rmse_advanced:<15.2f}")
logger.info(f"{'Improvement':<20} {(1-mae_advanced/mae_baseline)*100:<15.1f}% {(1-rmse_advanced/rmse_baseline)*100:<15.1f}%")

# Complete code for reproducibility
# All imports, data loading, feature engineering, selection, and evaluation
# See individual code blocks above for full implementation
