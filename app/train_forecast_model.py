import os
import pandas as pd
from forecast_ml import build_daily_net_series, train_delta_model, forecast_balance_ml

# 1️⃣ Directory where your monthly CSV files are stored
csv_dir = "data/training"  # adjust this path if needed
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

# 2️⃣ Load and combine all CSVs into one DataFrame
df_list = []
for file in csv_files:
    path = os.path.join(csv_dir, file)
    df = pd.read_csv(path)
    df_list.append(df)

tx_history = pd.concat(df_list, ignore_index=True)

# Check combined data
print(f"Combined transaction history: {len(tx_history)} records")
print(tx_history.head())

# 3️⃣ Build daily net delta series and train the model
daily_series = build_daily_net_series(tx_history)
model = train_delta_model(daily_series, model_type="rf")

if model is None:
    print("⚠️ Could not train the model (maybe not enough samples).")
else:
    print("✅ Model trained successfully.")

# 4️⃣ Generate forecast for the next 14 days
forecast_df = forecast_balance_ml(
    tx_history=tx_history,
    start_balance=5000.0,     # example starting balance
    horizon_days=14,          # forecast horizon (days ahead)
    model_type="rf"
)

print(forecast_df)

# 5️⃣ Save forecast to CSV
forecast_df.to_csv("forecast_output.csv", index=False)
print("📈 Forecast saved as forecast_output.csv")

def load_transaction_history(csv_dir: str) -> pd.DataFrame:
    """
    Loads and combines all CSV files in the specified directory into one DataFrame.
    """
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"⚠️ No CSV files found in directory: {csv_dir}")
        return pd.DataFrame()

    df_list = []
    for file in csv_files:
        path = os.path.join(csv_dir, file)
        print(f"📂 Loading: {file}")
        df = pd.read_csv(path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df["date"] = pd.to_datetime(combined_df["date"])
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    print(f"✅ Combined transaction history: {len(combined_df)} records\n")
    return combined_df


def train_and_forecast(tx_history: pd.DataFrame):
    """
    Builds the daily delta series, trains the model, and generates a forecast.
    """
    # Build daily net delta series
    daily_series = build_daily_net_series(tx_history)

    # Train model
    model = train_delta_model(daily_series, model_type="rf")
    if model is None:
        print("⚠️ Could not train the model (maybe not enough samples).")
        return None
    else:
        print("✅ Model trained successfully.\n")

    # Generate forecast
    forecast_df = forecast_balance_ml(
        tx_history=tx_history,
        start_balance=5000.0,  # Example starting balance
        horizon_days=14,       # Forecast for the next 14 days
        model_type="rf"
    )

    print("📊 Forecast results:\n")
    print(forecast_df)

    # Save forecast to CSV
    forecast_path = "forecast_output.csv"
    forecast_df.to_csv(forecast_path, index=False)
    print(f"\n💾 Forecast saved as {forecast_path}")
    return forecast_df


if __name__ == "__main__":
    """
    Main entry point. When executed directly, the script will:
    1. Load transaction data
    2. Train the model
    3. Generate and save a forecast
    """
    print("🚀 Starting forecast training and prediction...\n")

    # Directory containing your training CSV files
    csv_dir = "data/training"  # change if needed
    tx_history = load_transaction_history(csv_dir)

    if tx_history.empty:
        print("❌ No transaction data available. Exiting.")
    else:
        train_and_forecast(tx_history)

