"""
cli.py
------
Simple CLI for the Student Cashflow Agent.

Commands:
    python cli.py ingest
        -> load (ingest) + classify + save to outputs/normalized_transactions.csv

    python cli.py plan --start-balance 5000 --horizon 14 --min-buffer 1000
        -> load + classify + run planner + save:
            outputs/forecast.csv
            outputs/plan.csv
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List

import pandas as pd

from app.ingest import load_all_transactions
from app.classify import classify_transactions
from app.planner import build_forecast_and_plan
from app.summarize import generate_summary
from app.summarize_llm import generate_ai_summary

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

def cmd_plan(args: List[str]) -> None:
    """
    Run forecast + payment plan and save both CSVs.
    args is the list that comes after 'plan'
    e.g. ['--start-balance', '5000', '--horizon', '14', '--min-buffer', '1000']
    """
   
    # default values
    start_balance: float = 5000.0
    horizon: int = 14
    min_buffer: float = 1000.0

    # very small arg parser (no need for argparse yet)
    i = 0
    while i < len(args):
        if args[i] == "--start-balance" and i + 1 < len(args):
            start_balance = float(args[i + 1])
            i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon = int(args[i + 1])
            i += 2
        elif args[i] == "--min-buffer" and i + 1 < len(args):
            min_buffer = float(args[i + 1])
            i += 2
        else:
            print(f"Unknown option: {args[i]}")
            sys.exit(1)
    

    # 1) load + classify
    df_raw: pd.DataFrame = load_all_transactions()
    df_cls: pd.DataFrame = classify_transactions(df_raw)

    # 2) build forecast + plan
    forecast_df, plan_df = build_forecast_and_plan(
        df_cls,
        start_balance=start_balance,
        horizon_days=horizon,
        min_buffer=min_buffer,
    )

    # 3) save
    out_dir: Path = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    forecast_path: Path = out_dir / "forecast.csv"
    plan_path: Path = out_dir / "plan.csv"

    forecast_df.to_csv(forecast_path, index=False)
    plan_df.to_csv(plan_path, index=False)

    print(f"[ok] forecast saved to {forecast_path}")
    print(f"[ok] plan saved to {plan_path}")

def cmd_summarize(args: list[str]) -> None:
    # defaults 
    start_balance: float = 5000.0
    horizon: int = 14
    min_buffer: float = 1000.0

    # tiny args parser
    i = 0
    while i < len(args):
        if args[i] == "--start-balance" and i + 1 < len(args):
            start_balance = float(args[i + 1]); i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon = int(args[i + 1]); i += 2
        elif args[i] == "--min-buffer" and i + 1 < len(args):
            min_buffer = float(args[i + 1]); i += 2
        else:
            print(f"Unknown option: {args[i]}"); sys.exit(1)
        
    # load + classifiy
    df_raw: pd.DataFrame = load_all_transactions()
    df_cls: pd.DataFrame = classify_transactions(df_raw)

    # forecast & plan
    forecast_df, plan_df = build_forecast_and_plan(
        df_cls,
        start_balance=start_balance,
        horizon_days=horizon,
        min_buffer=min_buffer,
    )

    # summary
    summary_text: str = generate_summary(
        forecast_df, plan_df, start_balance=start_balance, min_buffer=min_buffer
    )

    # save + print
    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\n[ok] summary saved to {summary_path}")

def cmd_summarize_llm(args: list[str]) -> None:
    # defaults
    start_balance: float = 5000.0
    horizon: int = 14
    min_buffer: float = 1000.0

    # tiny arg parser
    i = 0 
    while i < len(args):
        if args[i] == "--start-balance" and i + 1 < len(args):
            start_balance = float(args[i + 1]); i += 2
        elif args[i] == "--horizon" and i + 1 < len(args):
            horizon = int(args[i + 1]); i += 2
        elif args[i] == "--min-buffer" and i + 1 < len(args):
            min_buffer = float(args[i + 1]); i += 2
        else:
            print(f"Unknown option: {args[i]}"); sys.exit(1)

    # load + classify
    df_raw: pd.DataFrame = load_all_transactions()
    df_cls: pd.DataFrame = classify_transactions(df_raw)

    # forecast & plan 
    forecast_df, plan_df = build_forecast_and_plan(
        df_cls,
        start_balance=start_balance,
        horizon_days=horizon,
        min_buffer=min_buffer,
    )

    # AI summary (with fallback)
    text = generate_ai_summary(
        forecast_df, plan_df, start_balance=start_balance, min_buffer=min_buffer
    )

    out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/ "summary_ai.txt"
    out_path.write_text(text, encoding="utf-8")

    print(text)
    print(f"\n[ok] AI summary saved to {out_path}")


def main() -> None:
    """Very small CLI dispatcher."""
    if len(sys.argv) < 2:
        print("Usage: python cli.py <command> [options]")
        print("Commands:")
        print(" ingest  Load + classify + save to outputs/")
        print("  plan --start-balance 5000 --horizon 14 --min-buffer 1000")
        print("  summarize --start-balance 5000 --horizon 14 --min-buffer 1000")
        print("  summarize-llm --start-balance 5000 --horizon 14 --min-buffer 1000")
        sys.exit(1)
    
    cmd: str = sys.argv[1]
    args: List[str] = sys.argv[2:]

    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "plan":
        cmd_plan(args)
    elif cmd == "summarize":
        cmd_summarize(args)
    elif cmd == "summarize-llm":
        cmd_summarize_llm(args)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    

if __name__ == "__main__":
    main()