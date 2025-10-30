# Student Cashflow Agent (MVP)

This project is a small prototype of an AI-friendly financial assistant for students.  
Goal: **show that the agent can read transactions, detect upcoming payments, project the cash balance, and suggest what to pay or postpone.**  
It is intentionally simple so it can be demoed in ≤5 minutes.

## What it does (MVP)

1. Reads bank-like CSV files from `data/raw/`.
2. Reads email-like payment notices from `data/sample_emails/`.
3. Normalizes everything into a single transactions table.
4. Runs a simple planner:
   - projects balance for the next N days,
   - respects a minimum buffer,
   - prioritizes rent/tuition/utilities over subscriptions.
5. Outputs:
   - `outputs/normalized_transactions.csv`
   - `outputs/forecast.csv`
   - `outputs/plan.csv`
6. (Optional) shows the plan in a small Streamlit UI.

## Project structure

```text
student-cashflow-agent/
├── README.md
├── requirements.txt
├── .env.example
├── cli.py
├── streamlit_app.py
├── data/
│   ├── raw/             # sample CSV inputs
│   └── sample_emails/   # sample "emails" as .txt
├── outputs/             # generated files
└── app/
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── classify.py
    ├── planner.py
    ├── summarize.py
    ├── exporters.py
    └── utils.py
