"""
summarize.py
------------
Builds a short, human-readable summary (no AI) from:
- forecast (daily balance)
- plan (scheduled/postponed/applied items)

Goal: print something like:
"Projected balance ... Scheduled payments ... Suggestions ..."
"""

from __future__ import annotations
from typing import Tuple, List
from datetime import date

import pandas as pd

def _fmt_money(x: float) -> str:
    """ Format numbers as MXN with no AI dependency."""
    return f"${x:,.2f} MXN"

def _safe_min(series: pd.Series) -> float | None:
    return float(series.min()) if not series.empty else None

def _safe_max(series: pd.Series) -> float | None:
    return float(series.max()) if not series.empty else None

def generate_summary(
        forecast_df: pd.DataFrame,
        plan_df: pd.DataFrame,
        start_balance: float,
        min_buffer: float,
) -> str:
    """
    Create a concise, deterministic summary in plain language.
    """

    # ---- Forecast stats
    end_balance: float | None = None
    min_balance: float | None = None

    if not forecast_df.empty:
        end_balance = float(forecast_df.iloc[-1]["balance"])
        min_balance = _safe_min(forecast_df["balance"])

    # ---- Plan stats
    scheduled = plan_df[plan_df["status"] == "scheduled"] if not plan_df.empty else pd.DataFrame()
    postponed = plan_df[plan_df["status"] == "postponed"] if not plan_df.empty else pd.DataFrame()

    total_scheduled = float(scheduled["amount"].sum()) if not scheduled.empty else 0.0
    total_postponed = float(postponed["amount"].sum()) if not postponed.empty else 0.0
   
    next_due = None
    if not scheduled.empty:
        next_due = scheduled.sort_values("pay_on").iloc[0]["pay_on"]

    # Find first day below buffer (if any)
    first_below = None
    if not forecast_df.empty:
        below = forecast_df[forecast_df["balance"] < min_buffer]
        if not below.empty:
            first_below = below.iloc[0]["date"]
    
    # ---- Build message
    lines: List[str] = []

    lines.append("=== Student Cashflow Summary (no-AI) === ")
    lines.append(f"Start balance: {_fmt_money(start_balance)} | Min buffer: {_fmt_money(min_buffer)}")

    if end_balance is not None:
        lines.append(f"Projected balance at horizon end: {_fmt_money(end_balance)}")
    if min_balance is not None:
        lines.append(f"Lowest projected balance: {_fmt_money(min_balance)}")
    
    if first_below:
        lines.append(f"⚠ Buffer risk: balance dips below buffer on {first_below}")
    
    lines.append("")
    lines.append("--- Payments plan ---")
    lines.append(f"Scheduled payments total: {_fmt_money(total_scheduled)}")
    if next_due:
        lines.append(f"Next scheduled payment date: {next_due}")
    
    if not postponed.empty:
        lines.append(f"Postponed (can't fit buffer): {_fmt_money(total_postponed)}")
        # show top 3 postponed items
        preview = postponed.sort_values(["priority", "pay_on"]).head(3)
        for _, r in preview.iterrows():
            lines.append(f"  - {r['pay_on']} • {r['description']} ({_fmt_money(float(r['amount']))})")
    
    # Simple suggesntion rule of thumb
    lines.append("")
    suggestion = "Suggestion: keep subscriptions last; prioritize rent/tuition/utilities first."
    if first_below:
        suggestion = "Suggestion: increase start balance or postpone low-priority items to stay above buffer."
    lines.append(suggestion)

    return "\n".join(lines)