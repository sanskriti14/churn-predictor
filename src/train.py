import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_pipeline():
    print("🚀 Starting Machine Learning Training Pipeline...")
    
    # 1. Load Data
    data_path = os.path.join("data", "customers.csv")
    df = pd.read_csv(data_path)
    
    # 2. Feature Engineering (Mapping text categories to structural numbers)
    # Mapping: Basic=1, Standard=2, Premium=3
    membership_mapping = {"Basic": 1, "Standard": 2, "Premium": 3}
    df['membership_encoded'] = df['membership_type'].map(membership_mapping)
    
    # Define Inputs (X) and Target Output (y)
    features = ['account_age_months', 'monthly_charges', 'total_tickets', 'membership_encoded']
    X = df[features]
    y = df['churned']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. Initialize and Fit the Model
    print("🏋️ Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Performance
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ Training Complete. Model Test Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Save (Pickle) the Trained Model Artifact
    os.makedirs("models", exist_ok=True)
    model_save_path = os.path.join("models", "churn_model.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model artifact successfully saved to {model_save_path}")

if __name__ == "__main__":
    train_pipeline()
