from sklearn.linear_model import LinearRegression
import pandas as pd

def train_regression_model(df: pd.DataFrame):
    """Train a linear regression model to predict whether a payment can be applied based on historical data."""
    # Make sure 'pay_on' is a datetime column before using .dt accessor
    df['pay_on'] = pd.to_datetime(df['pay_on'], errors='coerce')

    # Feature engineering (e.g., priority, amount, due_date)
    df['day_of_month'] = df['pay_on'].dt.day  # Extract the day of the month
    df['is_recurring'] = df['is_recurring'].astype(int)  # Convert boolean to integer

    # Prepare features (X) and target (y)
    X = df[['priority', 'amount', 'day_of_month', 'is_recurring']]  # Example features
    y = (df['status'] == 'pending').astype(int)  # Target: Predict if a payment can be applied

    model = LinearRegression()
    model.fit(X, y)  # Train the model with features and target

    return model

def automate_payments_with_model(df: pd.DataFrame, start_balance: float, min_buffer: float) -> pd.DataFrame:
    """Automate payments based on regression model predictions."""
    total_balance = start_balance

    # Make sure the 'pay_on' column is in datetime format
    df['pay_on'] = pd.to_datetime(df['pay_on'], errors='coerce')

    # Debug: Print columns to check for 'is_recurring'
    print("Columns in the DataFrame:", df.columns)

    # Sort payments by priority first (lower number = higher priority), then by pay_on date
    df_sorted = df.sort_values(['priority', 'pay_on'])

    # Train a model (or use a pre-trained model)
    model = train_regression_model(df)
    
    # Process payments
    for idx, row in df_sorted.iterrows():
        if row['status'] == 'pending':  # Only process pending payments
            # Prepare features for prediction
            features = [[row['priority'], row['amount'], row['pay_on'].day, row['is_recurring']]]
            prediction = model.predict(features)[0]  # Predict if the payment can be applied

            # If the model predicts the payment can be made and we have enough balance, pay the payment
            if prediction > 0.5 and total_balance - row['amount'] >= min_buffer:
                df.at[idx, 'status'] = 'applied'  # Mark as applied
                total_balance -= row['amount']  # Subtract the amount from the balance
            else:
                df.at[idx, 'status'] = 'postponed'  # Mark as postponed

    return df, total_balance

def main():
    """Test the automate_payments_with_model function."""
    # Sample test data (You can replace this with the actual data or load from CSV)
    data = {
        'pay_on': ['2025-11-01', '2025-11-03', '2025-11-05'],
        'description': ['Rent', 'Tuition', 'Netflix'],
        'amount': [1000, 2000, 500],
        'status': ['pending', 'pending', 'pending'],
        'priority': [1, 1, 2],
        'is_recurring': [True, True, True]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure 'pay_on' is in datetime format
    df['pay_on'] = pd.to_datetime(df['pay_on'])

    print("Original DataFrame:")
    print(df)

    # Call automate_payments_with_model to process payments
    start_balance = 5000.0  # Example balance
    min_buffer = 1000.0  # Minimum buffer to keep
    updated_df, updated_balance = automate_payments_with_model(df, start_balance, min_buffer)

    print("\nUpdated DataFrame after automating payments:")
    print(updated_df)
    print(f"\nUpdated Balance: ${updated_balance:,.2f}")

if __name__ == "__main__":
    main()
