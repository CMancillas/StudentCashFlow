import pandas as pd

def optimize_payments(df: pd.DataFrame, start_balance: float, min_buffer: float) -> pd.DataFrame:
    total_balance = start_balance
    available_balance = int(total_balance - min_buffer)

    # Sort payments by priority (lower = more important) and due date
    df_sorted = df.sort_values(['priority', 'pay_on']).reset_index(drop=True)
    df_sorted['status'] = 'postponed'

    # 1️⃣ Always pay priority 1 and 2 items first
    must_pay = df_sorted['priority'].isin([1, 2])
    total_must_pay = df_sorted.loc[must_pay, 'amount'].sum()

    if total_must_pay > available_balance:
        print(f"⚠️ Priority 1 & 2 payments exceed available balance ({total_must_pay} > {available_balance})")
        df_sorted.loc[must_pay, 'status'] = 'postponed'
        return df_sorted

    df_sorted.loc[must_pay, 'status'] = 'applied'
    available_balance -= total_must_pay

    print(f"Priority 1 & 2 payments applied (${total_must_pay} total). Remaining balance: ${available_balance}")

    # 2️⃣ Apply knapsack optimization for remaining payments (priority >= 3)
    optional_df = df_sorted[~must_pay].reset_index(drop=True)
    n = len(optional_df)

    dp = [0] * (available_balance + 1)
    decision = [-1] * (available_balance + 1)

    max_priority = optional_df['priority'].max()
    # Nonlinear weighting to favor higher-priority optional items
    values = (max_priority - optional_df['priority'] + 1) ** 3

    for i in range(n):
        amount = int(optional_df.loc[i, 'amount'])
        value = values[i]
        if amount <= available_balance:
            for balance in range(available_balance, amount - 1, -1):
                if dp[balance - amount] + value > dp[balance]:
                    dp[balance] = dp[balance - amount] + value
                    decision[balance] = i

    # 3️⃣ Reconstruct chosen payments
    remaining_balance = available_balance
    applied_indices = []

    while remaining_balance > 0 and decision[remaining_balance] != -1:
        idx = decision[remaining_balance]
        desc = optional_df.loc[idx, 'description']
        amt = optional_df.loc[idx, 'amount']
        applied_indices.append(idx)
        remaining_balance -= amt

    print("\nOptimized optional payments applied:")
    for i in applied_indices:
        desc = optional_df.loc[i, 'description']
        amt = optional_df.loc[i, 'amount']
        pr = optional_df.loc[i, 'priority']
        print(f"  - {desc} (${amt}) [priority {pr}]")

    print(f"\nFinal remaining balance: ${remaining_balance}")

    # Mark selected payments as applied
    optional_df.loc[applied_indices, 'status'] = 'applied'

    # 4️⃣ Update main DataFrame with optimized results
    df_sorted.update(optional_df)

    return df_sorted


def main():
# Example dataset — includes more cases to exceed available money and test optimization
    data = {
        "pay_on": [
            "2025-11-01", "2025-11-03", "2025-11-05", "2025-11-07", "2025-11-10",
            "2025-11-13", "2025-11-14", "2025-11-15", "2025-11-17", "2025-11-18",
            "2025-11-19", "2025-11-20"
        ],
        "description": [
            "Rent", "Tuition", "Netflix", "Gym", "Spotify",
            "Internet", "Steam", "Amazon Prime", "Groceries", "Dining Out",
            "Electricity", "Phone Plan"
        ],
        "amount": [1000, 2000, 500, 150, 200, 300, 250, 120, 400, 180, 350, 250],
        "priority": [1, 2, 4, 3, 4, 3, 5, 5, 2, 4, 1, 3],
        "is_recurring": [True, True, True, True, True, True, False, True, False, False, True, True],
    }

    df = pd.DataFrame(data)

    # Available money and minimum buffer
    available_balance = 6000
    min_buffer = 1000
    usable_balance = available_balance - min_buffer

    # Separate mandatory and optional payments
    mandatory_payments = df[df["priority"].isin([1, 2])].copy()
    optional_payments = df[~df["priority"].isin([1, 2])].copy()

    # Calculate mandatory total and deduct from available funds
    mandatory_total = mandatory_payments["amount"].sum()
    usable_balance -= mandatory_total

    print(f"Priority 1 & 2 payments applied (${mandatory_total} total). Remaining balance: ${usable_balance}")

    # Sort optional payments by priority (lower number = higher importance)
    optional_payments = optional_payments.sort_values("priority")

    # ---- Knapsack Optimization ----
    # Goal: maximize number of payments under usable_balance
    chosen = []
    remaining_budget = usable_balance

    for _, row in optional_payments.iterrows():
        if row["amount"] <= remaining_budget:
            chosen.append(row)
            remaining_budget -= row["amount"]

    # Convert chosen list to DataFrame
    chosen_payments = pd.DataFrame(chosen)

    # ---- Display results ----
    if not chosen_payments.empty:
        print("\nOptimized optional payments applied:")
        for _, row in chosen_payments.iterrows():
            print(f"  - {row['description']} (${row['amount']}) [priority {row['priority']}]")

    print(f"\nFinal remaining balance: ${remaining_budget}")

    # ---- Final Output ----
    # Combine all applied payments (mandatory + chosen)
    applied_df = pd.concat([mandatory_payments, chosen_payments], ignore_index=True)
    applied_df["status"] = "applied"

    # Identify postponed payments using both description and pay_on (avoids duplicates)
    mask = df.apply(lambda x: ((x["description"], x["pay_on"]) in 
                            list(zip(applied_df["description"], applied_df["pay_on"]))), axis=1)
    remaining_df = df[~mask].copy()
    remaining_df["status"] = "postponed"

    # Combine final results and sort by date
    final_df = pd.concat([applied_df, remaining_df], ignore_index=True)
    final_df = final_df.sort_values("pay_on").reset_index(drop=True)

    print("\n=== Final Result ===")
    print(final_df)


if __name__ == "__main__":
    main()
