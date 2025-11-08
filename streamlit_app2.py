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
from app.planner import build_forecast_and_plan, build_optimized_forecast_and_plan, build_ml_forecast_and_plan
from app.summarize_llm import generate_ai_summary
from app.summarize import generate_summary as generate_deterministic_summary
from app.payment_automation import automate_payments_with_model
from app.ingest_csv import load_transactions_from_csv

st.set_page_config(page_title="Student Cashflow Agent", layout="wide")

st.title("Student Cashflow Agent")

if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
if "df_cls" not in st.session_state:
    st.session_state["df_cls"] = None
if "start_balance" not in st.session_state:
    st.session_state["start_balance"] = None
if "show_preview" not in st.session_state:
    st.session_state["show_preview"] = True 

# ---- Sidebar inputs
st.sidebar.header("Inputs")
horizon_days: int = st.sidebar.number_input("Horizon (days)", value=14, min_value=1, max_value=60, step=1)
#min_buffer: float = st.sidebar.number_input("Min buffer (MXN)", value=1000.0, step=100.0)
uploaded_csv = st.sidebar.file_uploader("Upload transactions CSV", type=["csv"])
use_llm_classification = st.sidebar.checkbox("Use AI classification (Claude 3.5 Sonnet)", value=False)

if "df_cls" not in st.session_state:
    st.session_state["df_cls"] = None
if "start_balance" not in st.session_state:
    st.session_state["start_balance"] = None

st.sidebar.markdown("---")
load_btn = st.sidebar.button("Load & Normalize")
plan_btn = st.sidebar.button("Plan")
ai_btn = st.sidebar.button("AI Summary")

# ---- Placeholders
norm_ph = st.container()
plan_ph = st.container()

# ---- Preview uploaded CSV immediately
if uploaded_csv is not None and st.session_state["show_preview"]:
    try:
        uploaded_csv.seek(0)  # aseguramos leer desde el inicio
        df_preview = load_transactions_from_csv(uploaded_csv)
        df_preview = df_preview.reset_index(drop=True)
        df_preview_with_id = df_preview.copy()
        df_preview_with_id.insert(0, "id", range(1, len(df_preview_with_id) + 1))

        st.session_state["uploaded_df"] = df_preview  # sin id

        st.subheader("📂 Uploaded CSV preview")
        st.caption("Raw transactions from your file. This is the data the agent will use.")
        st.dataframe(df_preview_with_id, use_container_width=True, height=260)
    except Exception as e:
        st.error(f"Error while reading uploaded CSV: {e}")

        
# ---- Load & Normalize
if load_btn:
    try:
        # 1. Origen de datos: PRIORIDAD al CSV actual
        if uploaded_csv is not None:
            uploaded_csv.seek(0)
            df_raw = load_transactions_from_csv(uploaded_csv)
            st.session_state["uploaded_df"] = df_raw.copy()
        elif st.session_state.get("uploaded_df") is not None:
            df_raw = st.session_state["uploaded_df"].copy()
        else:
            st.error("Please upload a CSV file with columns: type, description, amount, date, priority")
            st.stop()

        # 2. Clasificación
        if use_llm_classification:
            from app.classify_llm import classify_with_llm
            df_cls = classify_with_llm(df_raw)
        else:
            df_cls = classify_transactions(df_raw)

        # 3. Normalizar índice + id
        df_cls = df_cls.reset_index(drop=True)
        if "id" in df_cls.columns:
            df_cls = df_cls.drop(columns=["id"])
        df_cls.insert(0, "id", range(1, len(df_cls) + 1))

        # 4. Asegurar priority numérico
        df_cls["priority"] = pd.to_numeric(df_cls["priority"], errors="coerce")

        # 5. Guardar df normalizado y ocultar preview crudo
        st.session_state["df_cls"] = df_cls
        st.session_state["show_preview"] = False

        # 6. Start balance = suma de amount con priority == 0
        mask_p0 = df_cls["priority"] == 0
        start_balance = float(df_cls.loc[mask_p0, "amount"].sum())
        st.session_state["total_balance"] = start_balance

        # 7. UI
        with norm_ph:
            st.markdown(
                f"""
                <div style="
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-radius: 0.9rem;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                ">
                    <div style="font-size: 0.9rem; color: #6b7280;">
                        Start balance (sum of priority 0 rows)
                    </div>
                    <div style="
                        font-size: 2.4rem;
                        font-weight: 700;
                        color: {'#16a34a' if start_balance >= 0 else '#dc2626'};
                    ">
                        ${start_balance:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("Normalized transactions")
            st.caption("Loaded from the uploaded CSV.")
            st.dataframe(df_cls, use_container_width=True, height=360)

    except Exception as e:
        st.error(f"Error while loading/normalizing data: {e}")





# ---- Plan (Forecast + Plan)
if plan_btn:
    try:
        # --- Load Data ---
        if uploaded_csv is not None:
            uploaded_csv.seek(0)
            df_raw = load_transactions_from_csv(uploaded_csv)
            st.session_state["uploaded_df"] = df_raw.copy()
        elif st.session_state.get("df_cls") is not None:
            df_cls = st.session_state["df_cls"].copy()
        elif st.session_state.get("uploaded_df") is not None:
            df_raw = st.session_state["uploaded_df"].copy()
        else:
            st.error("Please upload a CSV and load transactions first.")
            st.stop()

        # --- Classification ---
        if "df_cls" not in locals():
            if use_llm_classification:
                from app.classify_llm import classify_with_llm
                df_cls = classify_with_llm(df_raw)
            else:
                df_cls = classify_transactions(df_raw)

            df_cls = df_cls.reset_index(drop=True)
            if "id" in df_cls.columns:
                df_cls = df_cls.drop(columns=["id"])
            df_cls.insert(0, "id", range(1, len(df_cls) + 1))

        # --- Prepare Data ---
        df_cls["type"] = df_cls["type"].astype(str)
        df_cls["amount"] = pd.to_numeric(df_cls["amount"], errors="coerce")
        df_cls["priority"] = pd.to_numeric(df_cls["priority"], errors="coerce")
        df_cls["date"] = pd.to_datetime(df_cls["date"])

        mask_p0 = df_cls["priority"] == 0
        start_balance = float(df_cls.loc[mask_p0, "amount"].sum())
        st.session_state["start_balance"] = start_balance
        min_buffer = start_balance * 0.30
        

        with plan_ph:
            # --- Display Starting Balance ---
            st.markdown(
                f"""
                <div style="
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                    border-radius: 0.9rem;
                    background-color: #f8fafc;
                    border: 1px solid #e2e8f0;
                ">
                    <div style="font-size: 0.9rem; color: #6b7280;">
                        Start balance (sum of priority 0 rows)
                    </div>
                    <div style="
                        font-size: 2.4rem;
                        font-weight: 700;
                        color: {'#16a34a' if start_balance >= 0 else '#dc2626'};
                    ">
                        ${start_balance:,.2f}
                        ${min_buffer:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        

            
            forecast_df, plan_df = build_optimized_forecast_and_plan(
                df_cls,
                start_balance,
                horizon_days,
                min_buffer
            )
            

            # --- Save Outputs ---
            out_dir = Path("outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            forecast_df.to_csv(out_dir / "forecast.csv", index=False)
            plan_df.to_csv(out_dir / "plan.csv", index=False)

            # --- Two-column Layout ---
            col1, col2 = st.columns([1, 1], gap="large")

            # LEFT COLUMN - Forecast Visualization
            with col1:
                st.subheader("📈 Balance Forecast")
                st.line_chart(
                    forecast_df.set_index("date")[["balance"]],
                    use_container_width=True,
                )
                st.caption(
                    f"Start: {start_balance:,.2f} | "
                    f"End: {float(forecast_df.iloc[-1]['balance']):,.2f} | "
                    f"Min: {float(forecast_df['balance'].min()):,.2f}"
                )

                st.subheader("⏱️ Optimization Timeline")
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
          

            # RIGHT COLUMN - Plan Visualization
            with col2:
                forecast_df, plan_df = build_ml_forecast_and_plan(
                    df_cls,
                    start_balance,
                    horizon_days,
                    min_buffer
                )

                st.subheader("📈 Balance Forecast with ML")
                st.line_chart(
                    forecast_df.set_index("date")[["balance"]],
                    use_container_width=True,
                )
                st.caption(
                    f"Start: {start_balance:,.2f} | "
                    f"End: {float(forecast_df.iloc[-1]['balance']):,.2f} | "
                    f"Min: {float(forecast_df['balance'].min()):,.2f}"
                )

                st.subheader("🧾 Payment Plan")
                st.dataframe(
                    plan_df.sort_values(["priority"]),
                    use_container_width=True, height=360
                )



            # --- Save state ---
            st.session_state["show_preview"] = False
            st.success("✅ Saved outputs/forecast.csv and outputs/plan.csv")

    except Exception as e:
        st.error(f"Error while planning: {e}")



# ---- AI Summary
if ai_btn:
    try:
        if st.session_state["uploaded_df"] is None:
            st.error("Please upload a CSV and load transactions first.")
            st.stop()

        df_cls = st.session_state["uploaded_df"].copy()

        start_balance = float(df_cls.loc[df_cls["priority"] == 0, "amount"].sum())
        min_buffer = start_balance * 0.30
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
