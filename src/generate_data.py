import os
import pandas as pd
import numpy as np

def generate_and_save_data(file_path="data/customers.csv", n_customers=200):
    """Generates a realistic synthetic customer dataset and saves it to the data folder."""
    print(f"🎨 Generating {n_customers} synthetic customer records...")
    np.random.seed(42)
    
    account_age = np.random.randint(1, 72, n_customers)
    monthly_charges = np.round(np.random.uniform(20.0, 180.0, n_customers), 2)
    total_tickets = np.random.randint(0, 15, n_customers)
    membership_type = np.random.randint(0, 3, n_customers) # 0: Basic, 1: Standard, 2: Premium

    # Mathematical rule to determine churn risk variables
    churn_chance = (total_tickets * 0.15) + (monthly_charges / 300) - (account_age / 150)
    churned = (churn_chance > 0.4).astype(int)

    df = pd.DataFrame({
        "account_age_months": account_age,
        "monthly_charges": monthly_charges,
        "total_tickets": total_tickets,
        "membership_type": membership_type,
        "churned": churned
    })
    
    # Create the data/ directory if it doesn't exist yet
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the file into the data folder
    df.to_csv(file_path, index=False)
    print(f"✅ Success! Generated data saved directly into: {file_path}")

if __name__ == "__main__":
    generate_and_save_data()
