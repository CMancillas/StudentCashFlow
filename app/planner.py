"""
planner.py
----------
Adapted for Student Cashflow Agent

Works directly with CSVs containing:
type, description, amount, date, priority

Main features:
1. Forecasts daily balance given current balance and horizon
2. Plans payments respecting a minimum balance buffer
3. Optimized planner: prioritizes critical expenses and postpones low-priority ones
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Tuple, List
from .forecast_ml import forecast_balance_ml
import pandas as pd

# ---------------------------------------------------------------------
#  FORECAST
# ---------------------------------------------------------------------

def forecast_balance(
        tx: pd.DataFrame,
        start_balance: float,
        horizon_days: int = 14,
) -> pd.DataFrame:
    """Builds a day-by-day balance projection from today's date."""
    if tx.empty:
        today = datetime.today().date()
        return pd.DataFrame(
            {"date": [today + timedelta(days=i) for i in range(horizon_days)],
             "balance": [start_balance] * horizon_days}
        )

    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.date
    today = datetime.today().date()
    daily_totals = tx.groupby("date")["amount"].sum().to_dict()

    rows = []
    balance = start_balance

    for i in range(horizon_days):
        day = today + timedelta(days=i)
        delta = daily_totals.get(day, 0.0)
        balance += delta
        rows.append({"date": day, "delta": delta, "balance": balance})

    forecast_df = pd.DataFrame(rows)
    forecast_df["type"] = tx["type"].iloc[0]

    return forecast_df


# ---------------------------------------------------------------------
#  BASIC PLANNER
# ---------------------------------------------------------------------

def plan_payments(
    tx: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    min_buffer: float = 1000.0,
) -> pd.DataFrame:
    """
    Creates a simple payment plan using available balance and buffer.
    - Pay incomes immediately
    - Pay expenses if they don’t break the buffer; otherwise postpone
    """
    if tx.empty:
        return pd.DataFrame(columns=[
            "pay_on", "description", "amount", "priority",
            "balance_after", "status", "reason", "type"
        ])

    today = datetime.today().date()
    end_date = today + timedelta(days=horizon_days)

    df = tx.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[(df["date"] >= today) & (df["date"] <= end_date)]

    if df.empty:
        return pd.DataFrame(columns=[
            "pay_on", "description", "amount", "priority",
            "balance_after", "status", "reason", "type"
        ])

    # Sort: priority ascending (0 = highest), then date
    df = df.sort_values(["priority", "date"]).reset_index(drop=True)

    balance = start_balance
    plan_rows: List[dict] = []

    for _, row in df.iterrows():
        pay_on = row["date"]
        desc = row["description"]
        amt = float(row["amount"])
        prio = int(row["priority"])
        typ = row["type"]

        if typ == "income":
            balance += amt
            plan_rows.append({
                "pay_on": pay_on,
                "description": desc,
                "amount": amt,
                "priority": prio,
                "balance_after": balance,
                "status": "applied",
                "reason": "income",
                "type": typ
            })
        else:  # expense
            projected_balance = balance + amt  # amt is negative
            if projected_balance >= min_buffer:
                balance = projected_balance
                plan_rows.append({
                    "pay_on": pay_on,
                    "description": desc,
                    "amount": amt,
                    "priority": prio,
                    "balance_after": balance,
                    "status": "scheduled",
                    "reason": "within_buffer",
                })
            else:
                plan_rows.append({
                    "pay_on": pay_on,
                    "description": desc,
                    "amount": amt,
                    "priority": prio,
                    "balance_after": balance,
                    "status": "postponed",
                    "reason": "not_enough_after_buffer",
                    "type": typ,
                })

    
    return pd.DataFrame(plan_rows)


# ---------------------------------------------------------------------
#  OPTIMIZED PLANNER
# ---------------------------------------------------------------------

import pandas as pd
from typing import List

def optimize_payments(tx: pd.DataFrame, start_balance: float, min_buffer: float) -> pd.DataFrame:
    """
    Returns only the payments that can be applied without bajar del min_buffer.
    """
    if tx.empty:
        return pd.DataFrame(columns=[
            "pay_on", "description", "amount", "priority",
            "balance_after", "status", "reason", "type"
        ])

    df = tx.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["amount"] = df["amount"].astype(float)
    df["priority"] = df["priority"].astype(int)

    plan_rows: List[dict] = []

    # --- Apply all incomes ---
    incomes = df[df["type"] == "income"].sort_values("date")
    expenses = df[df["type"] == "expense"].sort_values(["priority", "date"])

    balance = start_balance
    for _, row in incomes.iterrows():
        balance += row["amount"]
        plan_rows.append({
            "pay_on": row["date"],
            "description": row["description"],
            "amount": row["amount"],
            "priority": row["priority"],
            "balance_after": balance,
            "status": "applied",
            "reason": "income",
            "type": "income"
        })

    # --- Apply mandatory payments (priority 1 & 2) ---
    must_pay_mask = expenses["priority"].isin([1, 2])
    must_pay = expenses[must_pay_mask].reset_index()

    available_balance = balance - min_buffer

    applied_mandatory = []
    for _, row in must_pay.iterrows():
        amt = abs(row["amount"])
        if amt <= available_balance:
            balance += row["amount"]  # amount < 0
            available_balance -= amt
            applied_mandatory.append(row.name)
            plan_rows.append({
                "pay_on": row["date"],
                "description": row["description"],
                "amount": row["amount"],
                "priority": row["priority"],
                "balance_after": balance,
                "status": "scheduled",
                "reason": f"priority_{row['priority']}_mandatory",
                "type": "expense"
            })
        # si no hay suficiente balance, simplemente no se aplica (se ignora)

    # --- Knapsack para pagos opcionales ---
    optional = expenses[~must_pay_mask].reset_index(drop=True)
    n = len(optional)
    dp = [0] * (int(available_balance) + 1)
    decision = [-1] * (int(available_balance) + 1)

    if n > 0 and available_balance > 0:
        max_priority = optional["priority"].max()
        values = (max_priority - optional["priority"] + 1) ** 3

        for i in range(n):
            amt = int(abs(optional.loc[i, "amount"]))
            val = values[i]
            if amt <= available_balance:
                for b in range(int(available_balance), amt - 1, -1):
                    if dp[b - amt] + val > dp[b]:
                        dp[b] = dp[b - amt] + val
                        decision[b] = i

        # Reconstruir pagos aplicados
        rem_balance = int(available_balance)
        applied_indices = []
        while rem_balance > 0 and decision[rem_balance] != -1:
            idx = decision[rem_balance]
            if idx not in applied_indices:
                applied_indices.append(idx)
            rem_balance -= int(abs(optional.loc[idx, "amount"]))

        for i in applied_indices:
            row = optional.loc[i]
            balance += row["amount"]
            plan_rows.append({
                "pay_on": row["date"],
                "description": row["description"],
                "amount": row["amount"],
                "priority": row["priority"],
                "balance_after": balance,
                "status": "scheduled",
                "reason": "selected_by_knapsack",
                "type": "expense"
            })

    # --- Solo devolvemos pagos aplicables ---
    result_df = pd.DataFrame(plan_rows).sort_values("pay_on").reset_index(drop=True)
    return result_df

# ---------------------------------------------------------------------
#  WRAPPERS
# ---------------------------------------------------------------------

def build_forecast_and_plan(
    classified_tx: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    min_buffer: float = 1000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Standard version without optimization."""
    forecast_df = forecast_balance(classified_tx, start_balance, horizon_days)
    plan_df = plan_payments(classified_tx, start_balance, horizon_days, min_buffer)
    return forecast_df, plan_df


def build_optimized_forecast_and_plan(
    classified_tx: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    min_buffer: float = 1000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optimized version using smarter selection logic."""
    forecast_df = forecast_balance(classified_tx, start_balance, horizon_days)
    plan_df = optimize_payments(classified_tx, start_balance, min_buffer)
    return forecast_df, plan_df

def build_ml_forecast_and_plan(
    classified_tx: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    min_buffer: float = 1000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ML-based version that uses a trained model to predict balance deltas.
    If training or prediction fails, falls back to the classic forecast.
    """
    try:
        forecast_df = forecast_balance_ml(
            tx_history=classified_tx,
            start_balance=start_balance,
            horizon_days=horizon_days,
            model_type="rf"
        )

        # Validate forecast
        if forecast_df is None or forecast_df.empty or "balance" not in forecast_df.columns:
            raise ValueError("Forecast ML returned empty or invalid DataFrame")

        plan_df = optimize_payments(classified_tx, start_balance, min_buffer)

    except Exception as e:
        import streamlit as st
        st.warning(f"⚠️ ML forecast failed: {e}. Falling back to classic mode.")
        forecast_df, plan_df = build_optimized_forecast_and_plan(
            classified_tx, start_balance, horizon_days, min_buffer
        )

    return forecast_df, plan_df
# ---------------------------------------------------------------------
#  TEST MODE
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("./data/transactions.csv")
    print("=== BASIC PLAN ===")
    f1, p1 = build_forecast_and_plan(df, 5000, 14, 1000)
    print(p1)
    print("\n=== OPTIMIZED PLAN ===")
    f2, p2 = build_optimized_forecast_and_plan(df, 5000, 14, 1000)
    print(p2)
