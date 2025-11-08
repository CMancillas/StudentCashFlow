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
import altair as alt

from app.ingest import load_all_transactions
from app.classify import classify_transactions
from app.planner import build_forecast_and_plan, build_optimized_forecast_and_plan
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
use_llm_classification = st.sidebar.checkbox("Use AI classification (Claude 3.5 Sonnet)", value=False)


st.sidebar.markdown("---")
load_btn = st.sidebar.button("Load & Normalize")
plan_btn = st.sidebar.button("Plan")
ai_btn = st.sidebar.button("AI Summary")

# ---- Placeholders
norm_ph = st.container()
plan_ph = st.container()



# ---- Load & Normalize
if load_btn:
    try:
        if uploaded_csv is not None:
            df_raw = load_transactions_from_csv(uploaded_csv)
        else:
            df_raw = load_all_transactions()

    
        if use_llm_classification:
            from app.classify_llm import classify_with_llm
            df_cls = classify_with_llm(df_raw)
        else:
            df_cls = classify_transactions(df_raw)
        
        df_cls = df_cls.reset_index(drop=True)
        df_cls.insert(0, "id", range(1, len(df_cls) + 1))

        with norm_ph:
            st.subheader("Normalized transactions")
            st.caption("Loaded from the uploaded CSV.")
            st.dataframe(df_cls, use_container_width=True, height=360)
        
    except Exception as e:
        st.error(f"Error while loading/normalizing data: {e}")

# ---- Plan (forecast + plan)
if plan_btn:
    try:
        # Load the transactions directly from CSV
        if uploaded_csv is not None:
            df_cls = load_transactions_from_csv(uploaded_csv)
        else:
            st.error("Please upload a CSV file with columns: type, description, amount, date, priority")
            st.stop()

        # Ensure required columns exist
        required_cols = {"type", "description", "amount", "date", "priority"}
        if not required_cols.issubset(df_cls.columns):
            st.error(f"CSV must contain the following columns: {required_cols}")
            st.stop()

        # Cast columns to correct types
        df_cls["type"] = df_cls["type"].astype(str)
        df_cls["amount"] = df_cls["amount"].astype(float)
        df_cls["priority"] = df_cls["priority"].astype(int)
        df_cls["date"] = pd.to_datetime(df_cls["date"])

        # Compute optimized plan
        forecast_df, plan_df = build_optimized_forecast_and_plan(
            df_cls,
            start_balance=start_balance,
            horizon_days=horizon_days,
            min_buffer=min_buffer,
        )

        # Save results
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(out_dir / "forecast.csv", index=False)
        plan_df.to_csv(out_dir / "plan.csv", index=False)

        # ---- Display results
        with plan_ph:
            col1, col2 = st.columns([1, 1], gap="large")

            # === LEFT COLUMN ===
            with col1:
                st.subheader("📈 Balance forecast")
                st.line_chart(
                    forecast_df.set_index("date")[["balance"]],
                    use_container_width=True,
                )
                st.caption(
                    f"Start: {start_balance:,.2f} | "
                    f"End: {float(forecast_df.iloc[-1]['balance']):,.2f} | "
                    f"Min: {float(forecast_df['balance'].min()):,.2f}"
                )

                # ✅ NEW: Cumulative balance vs expenses
                st.subheader("💰 Cumulative balance vs expenses")
                combo_df = forecast_df.copy()
                combo_df["expenses"] = forecast_df["balance"].diff().fillna(0).clip(upper=0).abs()
                st.area_chart(
                    combo_df.set_index("date")[["balance", "expenses"]],
                    use_container_width=True,
                )

            # === RIGHT COLUMN ===
            with col2:
                st.subheader("🧾 Payment plan")
                st.dataframe(
                    plan_df.sort_values(["status", "priority", "pay_on"]),
                    use_container_width=True, height=360
                )

                # ✅ NEW: Optimization timeline
                st.subheader("⏱️ Optimization timeline")
                if "pay_on" in plan_df.columns and "status" in plan_df.columns:
                    timeline_chart = (
                        alt.Chart(plan_df)
                        .mark_circle(size=90)
                        .encode(
                            x="pay_on:T",
                            y="amount:Q",
                            color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                            tooltip=["description", "amount", "status", "priority", "pay_on"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(timeline_chart, use_container_width=True)

            # ✅ NEW: Priority distribution (below both)
            st.subheader("🎯 Priority distribution")
            priority_chart = (
                alt.Chart(plan_df)
                .mark_bar()
                .encode(
                    x=alt.X("priority:O", title="Priority"),
                    y=alt.Y("count():Q", title="Number of payments"),
                    color=alt.Color("status:N", legend=alt.Legend(title="Status")),
                    tooltip=["priority", "status", "count()"]
                )
                .properties(height=300)
            )
            st.altair_chart(priority_chart, use_container_width=True)

            st.success("✅ Saved outputs/forecast.csv and outputs/plan.csv")
           
    except Exception as e:
        st.error(f"Error while planning: {e}")



# ---- AI Summary
if ai_btn:
    try:
        # Ensure we have df_cls, forecast_df, plan_df
        try:
            df_cls # type: ignore[name-defined]

        except NameError:
            # Usar load_transactions_from_csv si estamos subiendo un archivo CSV
            if uploaded_csv is not None: 
                df_raw: pd.DataFrame = load_transactions_from_csv(uploaded_csv)  
            else:
                st.error("No CSV file uploaded.")
                st.stop()

            df_cls: pd.DataFrame = classify_transactions(df_raw)
            df_cls = df_cls.reset_index(drop=True)

        try:
            forecast_df  # type: ignore[name-defined]
            plan_df  # type: ignore[name-defined]
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

        with st.expander("AI-generated summary (expand to view)"):
            st.write(text)

        with st.expander("Deterministic summary (fallback)"):
            st.code(generate_deterministic_summary(forecast_df, plan_df, start_balance, min_buffer))

    except Exception as e:
        st.error(f"Error while generating AI summary: {e}")
