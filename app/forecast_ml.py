# app/forecast_ml.py

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.base import RegressorMixin
except ImportError:
    RandomForestRegressor = None
    LinearRegression = None
    RegressorMixin = object  # Dummy base to keep type hints valid


# ---------------------------
# 1) Build daily net delta series
# ---------------------------

def build_daily_net_series(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily net delta series from raw transactions.
    Expects columns: ['date', 'amount'].
    Returns DataFrame with ['date', 'delta'] sorted by date.
    """
    if tx.empty:
        return pd.DataFrame(columns=["date", "delta"])

    df = tx.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    daily = (
        df.groupby("date", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "delta"})
        .sort_values("date")
    )

    return daily


# ---------------------------
# 2) Feature engineering
# ---------------------------

def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Add ML features to the daily delta series.
    Input: ['date', 'delta']
    Output: ['date', 'delta', 'day_of_week', 'is_weekend', 'lag_1', 'lag_7', 'rm7', 'rm30']
    """
    df = daily.copy()
    if df.empty:
        return df

    df["date_dt"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date_dt"].dt.weekday  # 0–6
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df = df.sort_values("date_dt")
    df["lag_1"] = df["delta"].shift(1)
    df["lag_7"] = df["delta"].shift(7)

    df["rm7"] = df["delta"].shift(1).rolling(window=7, min_periods=1).mean()
    df["rm30"] = df["delta"].shift(1).rolling(window=30, min_periods=1).mean()

    df[["lag_1", "lag_7", "rm7", "rm30"]] = df[["lag_1", "lag_7", "rm7", "rm30"]].fillna(0.0)
    return df


def get_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    From feature-augmented df, build X and y for model training.
    """
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    feature_cols = ["day_of_week", "is_weekend", "lag_1", "lag_7", "rm7", "rm30"]
    X = df[feature_cols].copy()
    y = df["delta"].astype(float).copy()

    return X, y


# ---------------------------
# 3) Train model
# ---------------------------

def train_delta_model(
    daily: pd.DataFrame,
    model_type: Literal["rf", "linear"] = "rf",
    min_samples: int = 5,
) -> Optional[RegressorMixin]:
    """
    Train a regression model for daily deltas.

    Parameters
    ----------
    daily : DataFrame
        ['date', 'delta']
    model_type : str
        'rf' (RandomForestRegressor) or 'linear' (LinearRegression)
    min_samples : int
        Minimum number of points required to train

    Returns
    -------
    Trained model or None if not enough data or sklearn unavailable.
    """
    if RandomForestRegressor is None or LinearRegression is None:
        return None

    if daily.shape[0] < min_samples:
        return None

    feats = add_features(daily)
    X, y = get_feature_matrix(feats)

    if X.empty:
        return None

    if model_type == "linear":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X, y)
    return model


# ---------------------------
# 4) Forecast with trained model
# ---------------------------

def _make_single_feature_row(
    day: datetime.date,
    last_deltas: pd.Series,
) -> pd.DataFrame:
    """
    Build one feature row for a future day, based on recent deltas.
    """
    day_dt = pd.to_datetime(day)
    day_of_week = day_dt.weekday()
    is_weekend = int(day_of_week >= 5)

    lag_1 = float(last_deltas.iloc[-1]) if len(last_deltas) >= 1 else 0.0
    lag_7 = float(last_deltas.iloc[-7]) if len(last_deltas) >= 7 else 0.0
    rm7 = float(last_deltas.tail(7).mean()) if len(last_deltas) >= 1 else 0.0
    rm30 = float(last_deltas.tail(30).mean()) if len(last_deltas) >= 1 else 0.0

    return pd.DataFrame({
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "rm7": [rm7],
        "rm30": [rm30],
    })


def forecast_balance_ml(
    tx_history: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    tx_future: Optional[pd.DataFrame] = None,
    model_type: Literal["rf", "linear"] = "rf",
) -> pd.DataFrame:
    """
    ML-based forecast of daily balances.

    Parameters
    ----------
    tx_history : DataFrame
        Historical transactions ['date', 'amount'].
    start_balance : float
        Starting balance for the forecast.
    horizon_days : int
        Days ahead to forecast.
    tx_future : DataFrame, optional
        Explicit future transactions (e.g., rent or scheduled payments).
    model_type : str
        'rf' or 'linear'

    Returns
    -------
    DataFrame with ['date', 'delta', 'balance', 'source'].
    """
    today = datetime.today().date()

    # Normalize historical data
    hist = tx_history.copy()
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"]).dt.date
        hist["amount"] = pd.to_numeric(hist["amount"], errors="coerce").fillna(0.0)
    else:
        hist = pd.DataFrame(columns=["date", "amount"])

    daily_hist = build_daily_net_series(hist)
    model = train_delta_model(daily_hist, model_type=model_type)

        # === DEBUG INFO ===
    print("🔍 DEBUG INFO (forecast_balance_ml)")
    print(f"Model trained: {model is not None}")
    print(f"Daily hist shape: {daily_hist.shape}")
    print("Daily hist head:")
    print(daily_hist.head())
    if hasattr(model, 'predict'):
        print("✅ Model has predict() method")
    else:
        print("⚠️ Model has no predict() method (check train_delta_model)")
    print("=============================")

    # Map of explicit future transactions
    future_map = {}
    if tx_future is not None and not tx_future.empty:
        fut = tx_future.copy()
        fut["date"] = pd.to_datetime(fut["date"]).dt.date
        fut["amount"] = pd.to_numeric(fut["amount"], errors="coerce").fillna(0.0)
        future_map = fut.groupby("date")["amount"].sum().to_dict()

    rows = []
    balance = start_balance

    last_deltas = (
        daily_hist["delta"].astype(float).reset_index(drop=True)
        if not daily_hist.empty
        else pd.Series([], dtype=float)
    )

    for i in range(horizon_days):
        day = today + timedelta(days=i)

        if day in future_map:
            delta = float(future_map[day])
            source = "explicit_tx"
        elif model is not None and len(last_deltas) > 0:
            X_day = _make_single_feature_row(day, last_deltas)
            delta = float(model.predict(X_day)[0])
            source = "ml_model"
        else:
            delta = 0.0
            source = "fallback_zero"

        balance += delta
        last_deltas = pd.concat([last_deltas, pd.Series([delta])], ignore_index=True)

        rows.append({
            "date": day,
            "delta": delta,
            "balance": balance,
            "source": source,
        })

    forecast_df = pd.DataFrame(rows)
    return forecast_df

if __name__ == "__main__":
    import os

    # === Demo / sanity check ===
    print("🧠 Running standalone test for forecast_balance_ml...")

    # 1️⃣ Locate your CSV
    csv_path = os.path.join("data", "transactions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV not found: {csv_path}")

    # 2️⃣ Load transaction data
    tx_history = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(tx_history)} transactions")

    # 3️⃣ Basic cleanup
    tx_history["date"] = pd.to_datetime(tx_history["date"]).dt.date
    tx_history["amount"] = pd.to_numeric(tx_history["amount"], errors="coerce").fillna(0.0)

    # 4️⃣ Calculate start balance (priority == 0)
    if "priority" in tx_history.columns:
        start_balance = float(tx_history.loc[tx_history["priority"] == 0, "amount"].sum())
    else:
        start_balance = 5000.0  # fallback example
    print(f"💰 Start balance: {start_balance:,.2f}")

    # 5️⃣ Run the ML forecast
    forecast_df = forecast_balance_ml(
        tx_history=tx_history,
        start_balance=start_balance,
        horizon_days=14,
        model_type="rf"
    )

    # 6️⃣ Display results
    print("\n=== Forecast preview ===")
    print(forecast_df.head(10))
    print(f"\n📈 Final balance after {len(forecast_df)} days: {forecast_df.iloc[-1]['balance']:,.2f}")

    # 7️⃣ Save forecast to file
    out_path = "forecast_debug.csv"
    forecast_df.to_csv(out_path, index=False)
    print(f"✅ Forecast saved to: {out_path}")
