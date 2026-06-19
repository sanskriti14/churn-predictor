import streamlit as st
import pandas as pd
import pickle
import os

# Set page configurations
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮")

st.title("🔮 Customer Churn Prediction Dashboard")
st.write("Input a customer's usage metrics below to predict if they are likely to cancel their subscription.")

# 1. Load the pre-trained model artifact
model_path = os.path.join("models", "churn_model.pkl")

if not os.path.exists(model_path):
    st.error("⚠️ Model artifact not found! Please run 'python src/train.py' in your terminal first to train the model.")
else:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Create UI Interactive Input Elements
    st.header("👤 Customer Profile Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        account_age = st.slider("Account Age (Months)", min_value=1, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=10.0, max_value=250.0, value=75.0)
        
    with col2:
        total_tickets = st.number_input("Support Tickets Opened", min_value=0, max_value=20, value=1)
        membership = st.selectbox("Membership Type", options=["Basic", "Standard", "Premium"])

    # Map the UI text input back to the model's structural expected integers
    membership_mapping = {"Basic": 1, "Standard": 2, "Premium": 3}
    membership_encoded = membership_mapping[membership]

    # 3. Predict Button Logic
    st.markdown("---")
    if st.button("Analyze Risk of Churn"):
        # Format inputs exactly like the training DataFrame structure
        input_data = pd.DataFrame([{
            'account_age_months': account_age,
            'monthly_charges': monthly_charges,
            'total_tickets': total_tickets,
            'membership_encoded': membership_encoded
        }])
        
        # Run inference
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1] # Probability of churning

        # 4. Display Results Visually
        st.header("📊 Prediction Output")
        if prediction == 1:
            st.error(f"🚨 **High Risk Alert:** This customer is likely to CHURN. (Probability: {prediction_proba * 100:.1f}%)")
        else:
            st.success(f"💚 **Low Risk:** This customer is likely to STAY. (Probability of churn: {prediction_proba * 100:.1f}%)")
