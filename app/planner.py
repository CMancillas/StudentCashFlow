"""
planner.py
----------
Takes the classified transactions and tries to answer two questions:

1) Forecast: 
    "If I start with BALANCE on DAY 0, what will my balance look like for the next N days when I apply incomes and expenses on their dates?"

2) Payment Plan:
    "Given my future/known payments (rent, tuition, utilities, subs), which ones can I actually pay on/before their due date WITHOUT going below 
     below a minimum buffer?"

This is NOT meant to be perfect finance logic. It's a student MVP:
- sort by priority
- then by due date
- try to pay
- if not enough money (balance - amount < min_buffer), mark it as postponed
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Tuple, List

import pandas as pd

def forecast_balance(
        tx: pd.DataFrame,
        start_balance: float,
        horizon_days: int = 14,
) -> pd.DataFrame:
    """
    Build a daily balance projection for the next `horizon_days`.

    Rules:
    - We look at tx["date"] (not due_date) to know on which day to apply each tx.
    - Positive amounts increase balance, negative amounts decrease it.
    - If some day has no transactions, balance stays the same.
    """

    if tx.empty:
        # if no transactions, just return flat balance
        today = datetime.today().date()
        rows = []
        for i in range(horizon_days):
            day = today + timedelta(days=i)
            rows.append({"date": day, "balance": start_balance})
        return pd.DataFrame(rows)
    
    # make sure date is datetime.date
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.date

    today = datetime.today().date()

    # group transactions by day, sum amounts
    daily_totals = (
        tx.groupby("date")["amount"].sum().to_dict()
    ) # {date: net_amount_that_day}

    rows: List[dict] = []
    balance: float = start_balance

    for i in range(horizon_days):
        day = today + timedelta(days=i)
        day_amount = daily_totals.get(day, 0.0)
        balance = balance + day_amount
        rows.append(
            {
                "date": day,
                "delta": day_amount,
                "balance": balance,
            }
        )
    
    forecast_df = pd.DataFrame(rows)
    return forecast_df

def plan_payments(
    tx: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    min_buffer: float = 1000.0,
) -> pd.DataFrame:
    """
    Create a simple payment plan.

    Input:
    - tx: classified transactions (they already have category, priority, due_date)
    - start_balance: how much money we have right now
    - horizon_days: only consider payments inside this window
    - min_buffer: we should NEVER let balance fall below this

    Output columns:
    - pay_on (date we intend to pay)
    - description
    - amount
    - category
    - priority
    - balance_after
    - status: "scheduled" | "postponed" | "ignored"
    - reason
    """
    if tx.empty:
        return pd.DataFrame(
            columns=[
                "pay_on",
                "description",
                "amount",
                "category",
                "priority",
                "balance_after",
                "status",
                "reason",
            ]
        )
    
    today = datetime.today().date()
    end_date = today + timedelta(days=horizon_days)

    # copy to avoid mutating original
    df = tx.copy()

    # which date to use for "when is this due?"
    # rule: if there's a due_date, use that; otherwise use the tx date
    df["effective_date"] = df["due_date"].fillna(df["date"])
    df["effective_date"] = pd.to_datetime(df["effective_date"]).dt.date

    # filter only things in our horizon
    df = df[(df["effective_date"] >= today) & (df["effective_date"] <= end_date)]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "pay_on",
                "description",
                "amount",
                "category",
                "priority",
                "balance_after",
                "status",
                "reason",
            ]
        )
    
    # sort by priority first (1 = more urgent), then by date
    df = df.sort_values(["priority", "effective_date"]).reset_index(drop=True)

    balance: float = start_balance
    plan_rows: List[dict] = []

    for _, row in df.iterrows():
        desc: str = str(row["description"])
        amt: float = float(row["amount"])
        cat: str = str(row["category"])
        prio: int = int(row["priority"])
        pay_on = row["effective_date"]

        # incomes: we just add them and mark them as "applied"
        if amt > 0:
            balance = balance + amt
            plan_rows.append(
                {
                    "pay_on": pay_on,
                    "description": desc,
                    "amount": amt,
                    "category": cat,
                    "priority": prio,
                    "balance_after": balance,
                    "status": "applied",
                    "reason": "income",
                }
             
            )
            continue

        # expenses: check if we can pay without breaking the buffer
        projected_balance = balance + amt # amt is negative
        if projected_balance >= min_buffer:
            # we can pay it
            balance = projected_balance
            plan_rows.append(
                {
                    "pay_on": pay_on,
                    "description": desc,
                    "amount": amt,
                    "category": cat,
                    "priority": prio,
                    "balance_after": balance,
                    "status": "scheduled",
                    "reason": "within_buffer",                    
                }
            )
        else:
            # we can't pay, would break buffer
            plan_rows.append(
                {
                    "pay_on": pay_on,
                    "description": desc,
                    "amount": amt,
                    "category": cat,
                    "priority": prio,
                    "balance_after": balance,
                    "status": "postponed",
                    "reason": "not_enough_after_buffer",    
                }
            )
    
    plan_df = pd.DataFrame(plan_rows)
    return plan_df

def build_forecast_and_plan(
    classified_tx: pd.DataFrame,
    start_balance: float,
    horizon_days: int = 14,
    min_buffer: float = 1000.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience helper: given the already-classified transactions,
    return (forecast_df, plan_df).
    """
    forecast_df: pd.DataFrame = forecast_balance(
        classified_tx, start_balance=start_balance, horizon_days=horizon_days
    )
    plan_df: pd.DataFrame = plan_payments(
        classified_tx,
        start_balance=start_balance,
        horizon_days=horizon_days,
        min_buffer=min_buffer,
    )
    return forecast_df, plan_df

if __name__ == "__main__":
    # quick manual test
    from ingest import load_all_transactions
    from classify import classify_transactions

    tx_raw = load_all_transactions()
    tx_cls = classify_transactions(tx_raw)

    forecast, plan = build_forecast_and_plan(
        tx_cls,
        start_balance=5000.0,
        horizon_days=14,
        min_buffer=1000.0,
    )

    print("=== FORECAST ===")
    print(forecast)
    print("\n=== PLAN ===")
    print(plan)