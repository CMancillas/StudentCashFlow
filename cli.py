"""
cli.py
------
Tiny command-line entrypoint for the project.

Right now it only supports:
    python cli.py ingest

What it does:
1. loads all transactions (CSV + email) from app.ingest
2. classifies them (app.classify)
3. adds an incremental id
4. saves to outputs/normalized_transactions.csv
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

from app.ingest import load_all_transactions
from app.classify import classify_transactions

def cmd_ingest() -> None:
    """Load, classify and save the normalized transactions."""
    # 1) load raw + emails
    df_raw: pd.DataFrame = load_all_transactions()

    # 2) classify
    df_cls: pd.DataFrame = classify_transactions(df_raw)

    # 3) add id column (1, 2, 3, ...)
    df_cls.insert(0, "id", range(1, len(df_cls) + 1))

    # 4) ensure output dir exists
    out_dir: Path = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path: Path = out_dir / "normalized_transactions.csv"

    # 5) save to CSV
    df_cls.to_csv(out_path, index=False)

    print(f"[ok] wrote {len(df_cls)} rows to {out_path}")

def main() -> None:
    """Very small CLI dispatcher."""
    if len(sys.argv) < 2:
        print("Usage: python cli.py <command>")
        print("Commands:")
        print(" ingest  Load + classify + save to outputs/")
        sys.exit(1)
    
    cmd: str = sys.argv[1]

    if cmd == "ingest":
        cmd_ingest()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    

if __name__ == "__main__":
    main()