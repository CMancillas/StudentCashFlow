"""
streamlit_app.py
----------------
Minimal UI for the Student Cashflow agent MVP.

Inputs:
- start balance
- horizon (days)
- min buffer

Buttons:
- Load & Normalize: ingest + classify -> preview table
- Plan: forecast + plan -> line chart + plan table
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

from app.ingest import load_all_transactions
from app.classify import classify_transactions
from app.planner import build_forecast_and_plan
from app.summarize_llm import generate_ai_summary
from app.summarize import generate_summary as generate_deterministic_summary
from app.payment_automation import automate_payments_with_model
from app.ingest_csv import load_transactions_from_csv

st.set_page_config(page_title="Student Cashflow Agent", layout="wide")

st.title("Student Cashflow Agent")

# ---- Sidebar inputs
st.sidebar.header("Inputs")
start_balance: float = st.sidebar.number_input("Start balance (MXN)", value=5000.0, step=100.0)
horizon_days: int = st.sidebar.number_input("Horizon (days)", value=14, min_value=1, max_value=60, step=1)
min_buffer: float = st.sidebar.number_input("Min buffer (MXN)", value=1000.0, step=100.0)
uploaded_csv = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
use_uploaded_csv = st.sidebar.checkbox("Use uploaded CSV (instead of auto-ingest)", value=False)

st.sidebar.markdown("---")
load_btn = st.sidebar.button("Load & Normalize")
plan_btn = st.sidebar.button("Plan")
payment_btn = st.sidebar.button("Perform Automatic Payments")
ai_btn = st.sidebar.button("AI Summary")

# ---- Placeholders
norm_ph = st.container()
plan_ph = st.container()

# ---- Load & Normalize
if load_btn:
    try:
        if use_uploaded_csv and uploaded_csv is not None:
             df_raw = load_transactions_from_csv(uploaded_csv)
        else:
            df_raw = load_all_transactions()

        #df_raw: pd.DataFrame = load_all_transactions()
        df_cls: pd.DataFrame = classify_transactions(df_raw)
        df_cls = df_cls.reset_index(drop=True)
        df_cls.insert(0, "id", range(1, len(df_cls) + 1))

        with norm_ph:
            st.subheader("Normalized transactions")
            st.caption("Merged from bank CSV + email-like txt. Includes category, periodicity and priority.")
            st.dataframe(df_cls, use_container_width=True, height=360)
    except Exception as e:
        st.error(f"Error while loading/normalizing data: {e}")

# ---- Plan (forecast + plan)
if plan_btn:
    try:
        # Reuse latest normalization if already done; otherwise compute quick
        try:
            df_cls # type: ignore [name-defined]
        except NameError:
            df_raw: pd.DataFrame = load_all_transactions()
            df_cls: pd.DataFrame = classify_transactions(df_raw)
            df_cls = df_cls.reset_index(drop=True)
            df_cls.insert(0, "id", range(1, len(df_cls) + 1))
        
        forecast_df, plan_df = build_forecast_and_plan(
            df_cls,
            start_balance=start_balance,
            horizon_days=horizon_days,
            min_buffer=min_buffer,
        )

        out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "forecast.csv").write_text(forecast_df.to_csv(index=False))
        (out_dir / "plan.csv").write_text(plan_df.to_csv(index=False))

        with plan_ph:
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.subheader("Balance forecast")
                st.line_chart(
                    forecast_df.set_index("date")[["balance"]],
                    use_container_width=True,
                )
                st.caption(
                    f"Start: {start_balance:,.2f} | "
                    f"End: {float(forecast_df.iloc[-1]['balance']):,.2f} | "
                    f"Min: {float(forecast_df['balance'].min()):,.2f}"
                )
            
            with col2:
                st.subheader("Payment plan")
                st.dataframe(
                    plan_df.sort_values(["status", "priority", "pay_on"]),
                    use_container_width=True, height=360
                )
                
            
                        
            st.success("Saved outputs/forecast.csv and outputs/plan.csv")
    
    except Exception as e:
        st.error(f"Error while planning: {e}")


# ---- Perform Automatic Payments
if payment_btn:
    try:
        # Ensure we have df_cls, forecast_df, plan_df
        try:
            df_cls  # Check if df_cls is defined
        except NameError:
            df_raw: pd.DataFrame = load_all_transactions()
            df_cls: pd.DataFrame = classify_transactions(df_raw)
            df_cls = df_cls.reset_index(drop=True)
            df_cls.insert(0, "id", range(1, len(df_cls) + 1))
        
        # Verifica que la columna is_recurring esté presente en df_cls
        if 'is_recurring' not in df_cls.columns:
            st.error("The column 'is_recurring' is missing in the classified transactions.")

        
        # Generate the payment plan (plan_df) using the classified data (df_cls)
        forecast_df, plan_df = build_forecast_and_plan(
            df_cls,
            start_balance=start_balance,
            horizon_days=horizon_days,
            min_buffer=min_buffer,
        )

        # Now we have the plan_df, we can proceed with automating payments
        updated_plan_df, updated_balance = automate_payments_with_model(
            plan_df,  # Plan generated in planner.py
            start_balance=start_balance,
            min_buffer=min_buffer
        )

        # Display the updated payment plan and available balance
        st.subheader("Updated Payment Plan")
        st.dataframe(updated_plan_df, use_container_width=True)
        st.subheader(f"New Available Balance: ${updated_balance:,.2f}")
        
    except Exception as e:
        st.error(f"Error while automating payments: {e}")




# ---- AI Summary
if ai_btn:
    try:
        # Ensure we have df_cls, forecast_df, plan_df
        try:
            df_cls # type: ignore[name-defined]
        except NameError:
            df_raw: pd.DataFrame = load_all_transactions()
            df_cls: pd.DataFrame = classify_transactions(df_raw)
            df_cls = df_cls.reset_index(drop=True)
        
        try:
            forecast_df # type: ignore[name-defined]
            plan_df # type: ignore[name-defined]
        except NameError:
            forecast_df, plan_df = build_forecast_and_plan(
                df_cls,
                start_balance=start_balance,
                horizon_days=horizon_days,
                min_buffer=min_buffer,
            )
        
        st.subheader("AI Summary")
        text = generate_ai_summary(
            forecast_df, plan_df, start_balance=start_balance, min_buffer=min_buffer
        )

        # Show the AI summary
        with st.expander("AI-generated summary (expand to view)"):
            st.write(text)

        # Also show deterministic as a collapsible fallback reference
        with st.expander("Deterministic summary (fallback)"):
            st.code(generate_deterministic_summary(forecast_df, plan_df, start_balance, min_buffer))
        
    except Exception as e:
        st.error(f"Error while generating AI summary: {e}")