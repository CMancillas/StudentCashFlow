import pandas as pd

def optimize_payments(df: pd.DataFrame, start_balance: float, min_buffer: float) -> pd.DataFrame:
    """
    Optimize which payments to apply or postpone based on priority and available balance.
    Always applies incomes and high-priority expenses (1, 2),
    then uses a knapsack-like approach for optional payments.
    """

    # --- Preparation ---
    df = df.copy()
    df["status"] = "postponed"

    # Rename date column if necessary
    if "date" not in df.columns and "pay_on" in df.columns:
        df = df.rename(columns={"pay_on": "date"})

    # Ensure correct types
    df["amount"] = df["amount"].astype(float)
    df["priority"] = df["priority"].astype(int)
    df["date"] = pd.to_datetime(df["date"])

    # --- Always apply incomes ---
    incomes = df[df["type"] == "income"].copy()
    expenses = df[df["type"] == "expense"].copy()

    df.loc[incomes.index, "status"] = "applied"
    start_balance += incomes["amount"].sum()

    available_balance = start_balance - min_buffer
    print(f"💰 Start balance: {start_balance:.2f}, min buffer: {min_buffer:.2f}, available: {available_balance:.2f}")

    # --- Apply mandatory payments (priority 1 & 2) ---
    df_sorted = expenses.sort_values(["priority", "date"]).reset_index(drop=True)
    must_pay = df_sorted["priority"].isin([1, 2])
    total_must_pay = abs(df_sorted.loc[must_pay, "amount"]).sum()

    if total_must_pay > available_balance:
        print(f"⚠️ Priority 1 & 2 payments exceed available balance ({total_must_pay} > {available_balance})")
        df.loc[df_sorted.loc[must_pay].index, "status"] = "postponed"
        return df

    df.loc[df_sorted.loc[must_pay].index, "status"] = "applied"
    available_balance -= total_must_pay

    print(f"Priority 1 & 2 payments applied (${total_must_pay} total). Remaining balance: ${available_balance}")

    # --- Knapsack optimization for optional payments ---
    optional_df = df_sorted[~must_pay].reset_index(drop=True)
    n = len(optional_df)

    dp = [0] * (int(available_balance) + 1)
    decision = [-1] * (int(available_balance) + 1)

    # Nonlinear weighting to favor higher-priority optional payments
    max_priority = optional_df["priority"].max()
    values = (max_priority - optional_df["priority"] + 1) ** 3

    for i in range(n):
        amount = int(abs(optional_df.loc[i, "amount"]))
        value = values[i]
        if amount <= available_balance:
            for balance in range(int(available_balance), amount - 1, -1):
                if dp[balance - amount] + value > dp[balance]:
                    dp[balance] = dp[balance - amount] + value
                    decision[balance] = i

    # --- Reconstruct decisions ---
    remaining_balance = int(available_balance)
    applied_indices = []

    while remaining_balance > 0 and decision[remaining_balance] != -1:
        idx = decision[remaining_balance]
        applied_indices.append(idx)
        remaining_balance -= int(abs(optional_df.loc[idx, "amount"]))

    print("\nOptional payments selected:")
    for i in applied_indices:
        desc = optional_df.loc[i, "description"]
        amt = optional_df.loc[i, "amount"]
        pr = optional_df.loc[i, "priority"]
        print(f"  - {desc} (${amt}) [priority {pr}]")

    # Mark selected optional payments as applied
    df.loc[optional_df.loc[applied_indices].index, "status"] = "applied"

    print(f"\nFinal remaining balance: ${remaining_balance:.2f}")

    # --- Final result ---
    df = df.sort_values("date").reset_index(drop=True)
    return df
