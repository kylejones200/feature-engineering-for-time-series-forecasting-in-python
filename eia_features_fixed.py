import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
plt.rcParams.update({'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    n_splits: int = 5
    horizon: int = 12
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def build_features(y: pd.Series, season: int) -> pd.DataFrame:
    df = pd.DataFrame({'y': y})
    # Lags 1..12
    for k in range(1, season + 1):
        df[f'lag{k}'] = df['y'].shift(k)
    # Rolling windows on lag1
    for w in (3, 6, 12):
        df[f'roll_mean_{w}'] = df['y'].rolling(w).mean()
        df[f'roll_std_{w}'] = df['y'].rolling(w).std()
    # Calendar features
    m = df.index.month
    df['sin12'] = np.sin(2 * np.pi * m / 12.0)
    df['cos12'] = np.cos(2 * np.pi * m / 12.0)
    return df.dropna()


def rolling_origin_importance(y: pd.Series, cfg: Config):
    Xy = build_features(y, cfg.season)
    features = [c for c in Xy.columns if c != 'y']
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    idx = np.arange(len(Xy))
    importances = np.zeros(len(features), dtype=float)
    maes = []
    last_true, last_pred = None, None
    for tr, te in tscv.split(idx):
        # Forecast next horizon deterministically by taking test span as the next periods
        end = tr[-1]
        X_tr = Xy.iloc[: end + 1][features]
        y_tr = Xy.iloc[: end + 1]['y']
        X_te = Xy.iloc[end + 1 : end + 1 + cfg.horizon][features]
        y_te = Xy.iloc[end + 1 : end + 1 + cfg.horizon]['y']
        if len(X_te) == 0:
            continue
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_tr, y_tr)
        yhat = model.predict(X_te)
        maes.append(mean_absolute_error(y_te.values, yhat))
        importances += model.feature_importances_
        last_true, last_pred = y_te, pd.Series(yhat, index=y_te.index)
    importances /= max(1, len(maes))
    imp = pd.Series(importances, index=features).sort_values(ascending=False)
    return float(np.mean(maes)) if maes else np.nan, imp, last_true, last_pred


def main():
    cfg = Config()
    y = load_series(cfg)
    mean_mae, imp, y_true, y_pred = rolling_origin_importance(y, cfg)
    print(f"RF feature baseline mean MAE: {mean_mae}")
    print(imp.head(10).to_string())

    # Importance figure
    plt.figure(figsize=(10,5))
    imp.head(15)[::-1].plot(kind='barh')
    plt.title('Top feature importances')
    save_fig('eia_features.png')

if __name__ == '__main__':
    main()
