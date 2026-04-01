# Feature Engineering for Time Series: Advanced Techniques Beyond Lag Features Feature engineering makes or breaks time series models. We explore advanced techniques beyond simple lags: temporal embeddings, domain-specific features, external regressors, and statistical features using Oklahoma energy consumption data.

### Feature Engineering for Time Series: Advanced Techniques Beyond Lag Features
Feature engineering is the difference between good and great time series models. Simple lag features capture basic patterns, but advanced techniques reveal hidden relationships that dramatically improve forecasting accuracy.

We explore sophisticated feature engineering methods using Oklahoma energy consumption data from 1960-2023. Beyond basic lags, we create temporal embeddings, domain-specific features, external regressors, and statistical features that capture complex patterns.

### Why Advanced Feature Engineering?
Simple lag features work for basic patterns. But real-world time series have:
- Complex seasonality Multiple seasonal patterns at different frequencies
- External factors Economic indicators, weather, policy changes
- Non-linear relationships Interactions between features
- Temporal structure Time-of-day, day-of-week, month effects

Advanced features capture these patterns, improving model accuracy significantly.

### Dataset: Oklahoma Energy Use
We use energy consumption data to demonstrate feature engineering.


The series contains **64 annual observations from 1960–2023**, aggregated from Oklahoma energy use data. This gives enough history to build a rich set of temporal, lagged, rolling, and change-based features while still keeping the feature space interpretable.
### Technique 1: Temporal Embeddings
Temporal embeddings encode time information in ways that capture cyclical patterns.


Cyclical encoding ensures that December (month 12) is close to January (month 1), which linear encoding misses.

### Technique 2: Lag Features with Multiple Windows
Multiple lag windows capture patterns at different time scales.


Multiple lag windows capture short-term and long-term dependencies simultaneously.

### Technique 3: Rolling Statistics
Rolling statistics capture local trends and volatility.


Rolling statistics capture local patterns that global statistics miss.

### Technique 4: Change and Rate Features
Change features capture momentum and acceleration.


Change features capture momentum and turning points in the time series.

### Technique 5: Domain-Specific Features
Domain knowledge creates powerful features specific to your problem.


Domain features leverage expert knowledge to capture problem-specific patterns.

### Technique 6: External Regressors
External regressors incorporate information from other data sources.


External regressors incorporate domain knowledge from related datasets.

### Feature Selection
Not all features are useful. We select the most important ones.


Feature selection identifies the most informative features, reducing overfitting and improving model performance. In our experiment, we started from **75 engineered features** and selected **25** using a combination of **mutual information** and **Random Forest feature importance**.

### Feature Importance Visualization
We visualize feature importance to understand what drives predictions.


Visualization reveals which features matter most for forecasting. On the Oklahoma energy series, the most influential features include:

- Short- and medium-horizon **rolling maxima and means** (e.g., 3-, 6-, 12-, and 24-year windows)  
- Recent **lags** (`lag_1`, `lag_3`, `lag_4`, `lag_5`)  
- Low-order **Fourier terms** (such as `fourier_sin_1`) capturing smooth cyclic structure  

The plot saved as `feature_importance.png` highlights how these rolling and lag features dominate both mutual information scores and tree-based importances.

### Impact on Model Performance
We test how advanced features improve model performance.


Using a simple Gradient Boosting model, we compared a **baseline feature set** (3 lags) against the **full engineered feature set**. On the Oklahoma energy series:

| Feature Set          | MAE        | RMSE       |
|----------------------|-----------:|-----------:|
| Baseline (3 lags)    | 617,104.40 | 843,327.59 |
| Advanced features    | 406,108.33 | 494,795.88 |

That corresponds to roughly a **34% reduction in MAE** and a **41% reduction in RMSE** when moving from simple lags to the richer engineered feature set.

### Best Practices
- Start simple, add complexity Begin with basic features, add advanced ones incrementally
- Domain knowledge matters Leverage expert knowledge to create meaningful features
- Feature selection is crucial Too many features can hurt performance
- Handle missing values Rolling features create NaNs—handle them carefully
- Scale features Normalize features for neural networks and distance-based models
- Validate feature importance Use multiple methods to identify important features

### Conclusion
Advanced feature engineering dramatically improves time series forecasting. Temporal embeddings, rolling statistics, change features, and domain-specific knowledge create a rich feature set that captures complex patterns. For Oklahoma energy consumption forecasting, these features reduced error by roughly one-third in MAE and over 40% in RMSE compared to a simple lag-only baseline.

The key is combining multiple techniques: temporal structure, statistical patterns, domain knowledge, and external information. This comprehensive approach unlocks the full potential of machine learning for time series forecasting.


