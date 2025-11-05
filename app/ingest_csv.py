# app/ingest_csv.py
import pandas as pd

def load_transactions_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load user-provided CSV with transactions (incomes and expenses).
    Expected columns: type, description, amount, date, priority
    - type: 'income' or 'expense'
    - priority: integer or category (1 = must pay, etc.)
    """
    df = pd.read_csv(file_path, parse_dates=["date"])

    # Ensure columns exist
    expected_cols = ["type", "description", "amount", "date", "priority"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Normalize types and add missing columns used elsewhere
    df["source"] = "user_csv"
    df["due_date"] = df["date"]

    # Assign status based on type
    df["status"] = df["type"].apply(lambda t: "applied" if t == "income" else "scheduled")

    # Sort for consistency
    df = df.sort_values("date").reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = load_transactions_from_csv("data/transactions.csv")
    print(df.head(20))
