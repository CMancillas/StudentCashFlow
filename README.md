💸 Student Cashflow Agent

An AI-assisted financial planning tool for students

🧩 Overview

Student Cashflow Agent helps students forecast their finances, plan payments, and stay above a safety balance buffer.
It ingests bank statements in CSV format, classifies transactions by type and priority, generates forecasts (both deterministic and ML-based), and produces AI-generated summaries with actionable recommendations.

This prototype demonstrates how AI can enhance personal finance automation through data classification, balance forecasting, and natural-language explanations — all while keeping the logic transparent and user-controlled.

🚀 Features

💰 Transaction Ingestion

- Loads CSV files with columns like type, description, amount, date, and priority.
- Important: Any CSV files you want to link for testing must follow this structure exactly.
- Handles inconsistent formats safely with flexible parsing and error handling.

🧠 Classification

Uses deterministic rules to assign transaction priorities:

Priority	Meaning
0	Income (positive amounts)
1	Must pay — rent, tuition, essential bills
2	Basic services — internet, utilities
3	Medium importance — subscriptions, insurance
4–5	Optional or luxury

📈 Forecasting

Two complementary forecasting modes:

Deterministic: Rule-based balance projection using transaction priorities and buffers.

ML-based: Uses scikit-learn (RandomForestRegressor) trained on 10 months of real transaction CSVs with mixed income and expenses to predict daily deltas and project short-term balances.

🤖 AI Summaries

The AI summary is powered by OpenAI: GPT-3.5 Turbo Instruct, generating natural, encouraging recommendations.

Summaries include insights such as next due payments, lowest projected balance, and spending advice.

🧾 Payment Planning

Prioritizes payments automatically to prevent overdrafts.

Displays results as:

A balance forecast chart

A payment plan table sorted by balance impact and priority

🧪 How to Run
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Set your OpenRouter API key
How to Get an API Key

1.- Create an account at OpenRouter.ai.
2.- Go to Settings → Integrations and generate a personal API key.
3.- Copy the key (it will look like sk-or-...).

Configure the Key in Your Local Environment
For Linux / macOS (bash/zsh):
   export OPENROUTER_API_KEY="sk-or-your-key-here"

For Windows (PowerShell):
   $env:OPENROUTER_API_KEY = "sk-or-your-key-here"

# 3️⃣ Run the Streamlit app
streamlit run streamlit_app.py

🧩 Project Structure
app/
 ├── __init__.py
 ├── ingest.py                 # CSV ingestion and normalization
 ├── ingest_csv.py             # Alternative ingestion module for structured CSVs
 ├── classify.py               # Rule-based transaction classification
 ├── classifiy_llm.py          # AI-assisted classification (DeepSeek)
 ├── forecast_ml.py            # ML-based balance forecasting (RandomForest)
 ├── train_forecast_model.py   # Model training for ML forecasts
 ├── planner.py                # Payment planning logic
 ├── optimize_payments.py      # Optimization algorithm for payment scheduling
 ├── payment_automation.py     # Handles automated payment export/integration
 ├── summarize.py              # Deterministic summary of financial data
 ├── summarize_llm.py          # AI-generated summary and recommendations
 ├── config.py                 # Global configuration (currently optional)

streamlit_app.py               # Streamlit web interface
requirements.txt               # Project dependencies
README.md                      # Documentation


🔍 Future Improvements
- Integration with Gmail and Google Calendar for payment reminders.
- Sync with GnuCash or Google Sheets for long-term budgeting.
- AI-based anomaly detection for unusual spending patterns.
- Support for multi-currency accounts and expense tagging.
- Automated payment execution through secure integrations (e.g., banking APIs or payment gateways).

🧑‍💻 Author
Carlos Mancillas
Computer Science Student, University of Sonora
AI/ML Internship Candidate @ Paystand