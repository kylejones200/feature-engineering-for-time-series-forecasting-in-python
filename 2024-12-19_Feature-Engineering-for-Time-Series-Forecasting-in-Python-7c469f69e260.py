import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


def create_features(series, window_size=7):
    """Create lagged features for supervised learning."""
    X, y = ([], [])
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return (np.array(X), np.array(y))


def generate_synthetic_temperature_data() -> None:
    n_samples = 1000

    time = np.arange(n_samples)

    seasonal = 10 * np.sin(2 * np.pi * time / 365)

    trend = 0.01 * time

    noise = np.random.randn(n_samples) * 2

    temperature = 20 + seasonal + trend + noise

    window_size = 7

    X, y = create_features(temperature, window_size)

    train_size = int(len(X) * 0.8)

    X_train, X_test = (X[:train_size], X[train_size:])

    y_train, y_test = (y[:train_size], y[train_size:])

    logger.info(f"Training samples: {len(X_train)}")

    logger.info(f"Test samples: {len(X_test)}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)

    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    logger.info(f"\nTrain RMSE: {train_rmse:.2f}°C")

    logger.info(f"Test RMSE:  {test_rmse:.2f}°C")

    tscv = TimeSeriesSplit(n_splits=config.get("cv", {}).get("n_splits", 5))

    cv_scores = []

    for train_idx, val_idx in tscv.split(X_train):
        cv_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        cv_model.fit(X_train[train_idx], y_train[train_idx])
        cv_pred = cv_model.predict(X_train[val_idx])
        cv_scores.append(np.sqrt(mean_squared_error(y_train[val_idx], cv_pred)))

    logger.info(f"CV RMSE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}°C")


def plot_results() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    test_indices = np.arange(train_size, train_size + len(y_test))

    ax1.plot(test_indices, y_test, label="Actual", linewidth=2, alpha=0.7)

    ax1.plot(test_indices, y_pred_test, label="Predicted", linewidth=2, alpha=0.7, linestyle="--")

    ax1.set_ylabel("Temperature (°C)")

    ax1.set_title("Temperature Forecasting: Actual vs Predicted")

    ax1.legend()

    residuals = y_test - y_pred_test

    ax2.scatter(y_pred_test, residuals, alpha=0.5)

    ax2.axhline(y=0, color="r", linestyle="--")

    ax2.set_xlabel("Predicted Temperature (°C)")

    ax2.set_ylabel("Residuals (°C)")

    ax2.set_title("Residual Plot")

    plt.tight_layout()

    plt.savefig("supervised_learning_forecast.png", dpi=150, bbox_inches="tight")

    plt.close()

    importances = model.feature_importances_

    feature_names = [f"Day-{i + 1}" for i in range(window_size)]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(feature_names, importances)

    ax.set_xlabel("Importance")

    ax.set_title("Feature Importance: Which Days Matter Most?")

    plt.tight_layout()

    plt.savefig("sliding_window_feature_importance.png", dpi=150, bbox_inches="tight")

    plt.close()

    forecast_horizon = 3

    X_multi, y_multi = ([], [])

    for i in range(len(temperature) - window_size - forecast_horizon + 1):
        X_multi.append(temperature[i : i + window_size])
        y_multi.append(temperature[i + window_size : i + window_size + forecast_horizon])

    X_multi, y_multi = (np.array(X_multi), np.array(y_multi))

    train_size_multi = int(len(X_multi) * 0.8)

    X_train_multi = X_multi[:train_size_multi]

    y_train_multi = y_multi[:train_size_multi]

    X_test_multi = X_multi[train_size_multi:]

    y_test_multi = y_multi[train_size_multi:]

    model_multi = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    model_multi.fit(X_train_multi, y_train_multi)

    y_pred_multi = model_multi.predict(X_test_multi)

    logger.info("\nMulti-step forecasting:")

    for h in range(forecast_horizon):
        horizon_rmse = np.sqrt(mean_squared_error(y_test_multi[:, h], y_pred_multi[:, h]))
        logger.info(f"Day +{h + 1} RMSE: {horizon_rmse:.2f}°C")

    logger.info(
        "\nOutputs: supervised_learning_forecast.png, sliding_window_feature_importance.png"
    )


def main() -> None:
    generate_synthetic_temperature_data()
    plot_results()


if __name__ == "__main__":
    main()
