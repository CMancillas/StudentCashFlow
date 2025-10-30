"""
ingest.py
---------
This file is in charge of the FIRST step of the project: getting our data
into one single shape.

Right now we have two kinds of inputs:
1) CSV files that look like bank statements -> they live in `data/raw/`
2) Fake email notifications written as .txt files -> they live in `data/sample_emails/`

The goal here is simple:
- read everything
- make the columns look the same,
- and give back ONE pandas DataFrame with:
    date, description, amount, source, due_date

Later we will add classification, planning, AI, etc. But this file is just:
"take messy inputs -> give me a clean table."
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re

import pandas as pd
from dateutil import parser

def load_bank_csv(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load all CSVs from data/raw/ and normalize basic columns."""
    raw_path: Path = Path(raw_dir)
    frames: List[pd.DataFrame] = []

    for csv_file in raw_path.glob("*.csv"):
        df: pd.DataFrame = pd.read_csv(csv_file)
        df = df.rename(
            columns={
                "Date": "date",
                "Description": "description",
                "Amount": "amount",
            }
        )
        df["source"] = f"bank_csv:{csv_file.name}"
        df["due_date"] = pd.NaT
        df = df[["date", "description", "amount", "source", "due_date"]]
        frames.append(df)
    
    if not frames:
        return pd.DataFrame(
            columns=["date", "description", "amount", "source", "due_date"]
        )
    
    all_banks: pd.DataFrame = pd.concat(frames, ignore_index=True)
    all_banks["date"] = pd.to_datetime(all_banks["date"])
    all_banks["amount"] = pd.to_numeric(all_banks["amount"])
    return all_banks                                             

EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"Amount:\s*([0-9.,]+).*?Due date:\s*([0-9-]+).*?Concept:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)

def load_email_samples(email_dir: str = "data/sample_emails") -> pd.DataFrame:
    """Load .txt email-like files and turn them into payment rows"""
    email_path: Path = Path(email_dir)
    rows: List[Dict[str, object]] = []

    for txt_file in email_path.glob("*.txt"):
        content: str = txt_file.read_text(encoding="utf-8")
        match: re.Match[str] | None = EMAIL_PATTERN.search(content)
        if match is None:
            continue

        amount_str: str = match.group(1).replace(",", "")
        due_str: str = match.group(2).strip()
        concept: str = match.group(3).strip()

        amount: float = -abs(float(amount_str))
        due_date = parser.parse(due_str)

        rows.append(
            {
                "date": due_date,
                "description": concept,
                "amount": amount,
                "source": f"email:{txt_file.name}",
                "due_date": due_date,
            }
        )
    
    if not rows:
        return pd.DataFrame(
            columns=["date", "description", "amount", "source", "due_date"]
        )
    
    return pd.DataFrame(rows)

def load_all_transactions(raw_dir: str = "data/raw", email_dir: str = "data/sample_emails") -> pd.DataFrame:
    """Glue: CSV rows + email rows -> one sorted DataFrame."""
    bank_df: pd.DataFrame = load_bank_csv(raw_dir)
    email_df: pd.DataFrame = load_email_samples(email_dir)

    if bank_df.empty and email_df.empty:
        return pd.DataFrame(
            columns=["date", "description", "amount", "source", "due_date"]
        )
    
    all_tx: pd.DataFrame = pd.concat([bank_df, email_df], ignore_index=True)
    return all_tx.sort_values("date").reset_index(drop=True)

if __name__ == "__main__":
    df: pd.DataFrame = load_all_transactions()
    print(df)