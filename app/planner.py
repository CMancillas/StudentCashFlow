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
def optimize_payments(tx: pd.DataFrame, start_balance: float, min_buffer: float) -> pd.DataFrame:
    """
    Returns ONLY the expense transactions that will be paid.

    Conventions:
    - start_balance ALREADY includes all incomes (priority 0 / type 'income').
    - Rows with priority 0 are NOT returned.

    Rules:
    - Expenses with priority == 1:
        -> Always paid (mandatory).
    - Expenses with priority >= 2:
        -> 0/1 Knapsack:
            * cost = |amount|
            * capacity = start_balance - min_buffer - sum(|mandatory|)
            (minimum 0)
            * maximize value by favoring higher priority (2 > 3 > 4 > ...).
    - The final balance is calculated only with the selected expenses.
    (If mandatory expenses alone exceed the capacity, they are still accepted by design.)

    Output:
    - ONLY selected expense transactions:
        columns: pay_on, description, amount, priority,
                balance_after, status, reason, type
    """

    if tx.empty:
        return pd.DataFrame(columns=[
            "pay_on", "description", "amount", "priority",
            "balance_after", "status", "reason", "type",
        ])

    df = tx.copy()

    if "date" not in df.columns and "pay_on" in df.columns:
        df = df.rename(columns={"pay_on": "date"})
    if "date" not in df.columns:
        raise ValueError("Se requiere columna 'date' o 'pay_on' en tx.")

    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["priority"] = df["priority"].astype(int)

    df = df[df["type"] == "expense"].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "pay_on", "description", "amount", "priority",
            "balance_after", "status", "reason", "type",
        ])

    df = (
        df.sort_values(["date", "priority"])
          .reset_index()
          .rename(columns={"index": "orig_index"})
    )

    mandatory = df[df["priority"] == 1].copy()
    optional = df[df["priority"] >= 2].copy()

    total_capacity = max(0.0, start_balance - min_buffer)

    mandatory_cost = mandatory["amount"].abs().sum()

    capacity_optional = max(0.0, total_capacity - mandatory_cost)

    optional_selected = pd.DataFrame(columns=df.columns)

    if capacity_optional > 0 and not optional.empty:
        opt = optional.copy()

        opt["cost"] = opt["amount"].abs()
        scale = 1
        opt["cost_int"] = (opt["cost"] / scale).astype(int)
        C = int(capacity_optional / scale)

        max_p = opt["priority"].max()
        opt["value"] = (max_p - opt["priority"] + 1) ** 3

        n = len(opt)
        costs = opt["cost_int"].tolist()
        values = opt["value"].tolist()
        orig_idx = opt["orig_index"].tolist()

        dp = [[0.0] * (C + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            cost = costs[i - 1]
            val = values[i - 1]
            for c in range(C + 1):
                best = dp[i - 1][c]
                if cost <= c:
                    cand = dp[i - 1][c - cost] + val
                    if cand > best:
                        best = cand
                dp[i][c] = best

        chosen = set()
        c = C
        for i in range(n, 0, -1):
            if dp[i][c] != dp[i - 1][c]:
                idx = i - 1
                chosen.add(orig_idx[idx])
                c -= costs[idx]

        optional_selected = opt[opt["orig_index"].isin(chosen)]


    selected = pd.concat(
        [mandatory, optional_selected],
        axis=0
    ).drop_duplicates(subset=["orig_index"]).sort_values(["date", "priority"])

    balance = start_balance
    rows: List[dict] = []

    for _, row in selected.iterrows():
        pay_on = row["date"].date()
        desc = row["description"]
        amt = float(row["amount"])   # negativo
        prio = int(row["priority"])
        typ = row["type"]

        if prio == 1:
            balance += amt
            status = "scheduled"
            reason = "priority_1_mandatory"
        else:
            balance += amt
            status = "scheduled"
            reason = "selected_by_knapsack"

        rows.append({
            "pay_on": pay_on,
            "description": desc,
            "amount": amt,
            "priority": prio,
            "balance_after": balance,
            "status": status,
            "reason": reason,
            "type": typ,
        })

    result = pd.DataFrame(rows)


    return result

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
    df = pd.read_csv("./data/transactions2.csv")
    print("=== BASIC PLAN ===")
    f1, p1 = build_forecast_and_plan(df, 5000, 14, 1000)
    print(p1)
    print("\n=== OPTIMIZED PLAN ===")
    f2, p2 = build_optimized_forecast_and_plan(df, 5000, 14, 1000)
    plan = optimize_payments(df, start_balance=14500, min_buffer=3660)
    print(plan["amount"].sum())            
    print(14500 + plan["amount"].sum())    

    print(p2)
