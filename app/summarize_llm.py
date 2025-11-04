"""
summarize_llm.py
----------------
LLM-powered natural language summary based on:
- forecast_df (daily balance projection)
- plan_df (scheduled/postponed/applied items)

Now using OpenRouter + Anthropic Claude 3.5 Sonnet for financial summaries.
If the LLM call fails, it falls back to summarize.py
"""

from __future__ import annotations
from typing import Optional
import os
import pandas as pd
from openai import OpenAI
from app.summarize import generate_summary as generate_deterministic_summary


def _fmt_mxn(x: float) -> str:
    return f"${x:,.2f} MXN"


def _compute_stats(
    forecast_df: pd.DataFrame, plan_df: pd.DataFrame, start_balance: float, min_buffer: float
) -> dict:
    end_balance = float(forecast_df.iloc[-1]["balance"]) if not forecast_df.empty else start_balance
    min_balance = float(forecast_df["balance"].min()) if not forecast_df.empty else start_balance

    scheduled = plan_df[plan_df["status"] == "scheduled"] if not plan_df.empty else pd.DataFrame
    postponed = plan_df[plan_df["status"] == "postponed"] if not plan_df.empty else pd.DataFrame

    total_scheduled = float(scheduled["amount"].sum()) if not scheduled.empty else 0.0
    total_postponed = float(postponed["amount"].sum()) if not postponed.empty else 0.0

    next_due: Optional[str] = None
    if not scheduled.empty:
        next_due = str(scheduled.sort_values("pay_on").iloc[0]["pay_on"])

    first_below: Optional[str] = None
    if not forecast_df.empty:
        below = forecast_df[forecast_df["balance"] < min_buffer]
        if not below.empty:
            first_below = str(below.iloc[0]["date"])

    sch_preview = []
    if not scheduled.empty:
        for _, r in scheduled.sort_values(["priority", "pay_on"]).head(5).iterrows():
            sch_preview.append(f"{r['pay_on']} · {r['description']} ({_fmt_mxn(float(r['amount']))})")

    pos_preview = []
    if not postponed.empty:
        for _, r in postponed.sort_values(["priority", "pay_on"]).head(5).iterrows():
            pos_preview.append(f"{r['pay_on']} · {r['description']} ({_fmt_mxn(float(r['amount']))})")

    return {
        "start_balance": start_balance,
        "end_balance": end_balance,
        "min_balance": min_balance,
        "min_buffer": min_buffer,
        "first_below": first_below,
        "total_scheduled": total_scheduled,
        "total_postponed": total_postponed,
        "next_due": next_due,
        "scheduled_preview": sch_preview,
        "postponed_preview": pos_preview,
    }


def generate_ai_summary(
    forecast_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    start_balance: float,
    min_buffer: float,
    model: str = "anthropic/claude-3.5-sonnet",
) -> str:
    """
    Build a concise student-friendly summary via Claude 3.5 Sonnet on OpenRouter.
    If the LLM call fails, return the deterministic summary instead.
    """
    stats = _compute_stats(forecast_df, plan_df, start_balance, min_buffer)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️ No OPENROUTER_API_KEY found — using fallback summary.")
        return generate_deterministic_summary(forecast_df, plan_df, start_balance, min_buffer)

    # Build the context prompt
    prompt = f"""
You are a friendly and wise financial assistant for university students. 
Your task is to summarize their projected financial situation in clear, encouraging language.

Context (amounts in MXN):
- Start balance: {_fmt_mxn(stats['start_balance'])}
- Min buffer: {_fmt_mxn(stats['min_buffer'])}
- Projected end balance: {_fmt_mxn(stats['end_balance'])}
- Lowest projected balance: {_fmt_mxn(stats['min_balance'])}
- First day below buffer: {stats['first_below'] or 'none'}
- Scheduled payments total: {_fmt_mxn(stats['total_scheduled'])}
- Postponed payments total: {_fmt_mxn(stats['total_postponed'])}
- Next scheduled payment date: {stats['next_due'] or 'none'}

Top scheduled payments:
- {(chr(10) + '- ').join(stats['scheduled_preview']) if stats['scheduled_preview'] else 'none'}

Top postponed payments:
- {(chr(10) + '- ').join(stats['postponed_preview']) if stats['postponed_preview'] else 'none'}

Write a short, clear, and motivating summary. 
Offer practical, kind advice for managing money wisely.
Start your response with:
"Here’s your financial summary:"
"""

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You generate friendly, motivating, and practical financial summaries for students.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Error generating AI summary: {e}")
        return generate_deterministic_summary(forecast_df, plan_df, start_balance, min_buffer)
