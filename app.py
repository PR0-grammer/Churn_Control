import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ======================================================
# LOAD MODEL
# ======================================================
lgb_clf = lgb.Booster(model_file="lgbm_churn_model.txt")

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ======================================================
# CUSTOM PAGE CSS
# ======================================================
st.markdown("""
    <style>

        /* ===============================
           KPI CARD STYLING
           =============================== */
        .stat-card {
            border-radius: 15px;
            padding: 25px;
            border: 1px solid #B7E8E3;
            box-shadow: 0px 4px 12px #00000010;
        }

    </style>
""", unsafe_allow_html=True)


# ======================================================
# HEADER (LOGO + TITLE)
# ======================================================
header_logo, header_title = st.columns([1, 4])

with header_logo:
    st.image("logo.png", width=350)  

with header_title:
    st.markdown("""
        <h1 style='font-size:90px; margin-top:10px; color:#46a1d7;'>
            Customer Churn Prediction
        </h1>
        <h4 style='margin-top:-15px; color:#265177;'>
            Predicts one customer at a time & explain results with SHAP
        </h4>
    """, unsafe_allow_html=True)
    st.markdown("Created by AAA")

# ======================================================
# KPI CARDS (CAN BE REPLACED WITH IMAGES)
# ======================================================
st.markdown("### Dataset At a Glance")
k1, k2, k3 = st.columns(3)

with k1:
    st.markdown("""
        <div class="stat-card">
            <h3 style="color:#077A9B;">Total Customers</h3>
            <h2>115,000+</h2>
        </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
        <div class="stat-card">
            <h3 style="color:#077A9B;">Churn Rate</h3>
            <h2>12.19%</h2>
        </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
        <div class="stat-card">
            <h3 style="color:#077A9B;">Accuracy</h3>
            <h2>95%</h2>
        </div>
    """, unsafe_allow_html=True)


# ======================================================
# INPUT FORM
# ======================================================
st.markdown("---")
st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    # =====INPUTS =====
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
        num_complaints = st.number_input("Number of Complaints", min_value=0, max_value=50, value=0)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    with col2:
        balance = st.number_input("Balance (SAR)", min_value=0.0, value=20000.0, step=1000.0)
        income = st.number_input("Income (SAR)", min_value=0.0, value=50000.0, step=1000.0)
        outstanding_loans = st.number_input("Outstanding Loans (SAR)", min_value=0.0, value=0.0, step=1000.0)
        tenure = st.number_input("Customer Tenure (months)", min_value=0, max_value=600, value=24)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        customer_segment = st.selectbox("Customer Segment", ["Retail", "SME", "Corporate"])

    submit_button = st.form_submit_button("Predict Churn ✔️")



# ======================================================
# RUN MODEL INFERENCE
# ======================================================
if submit_button:

    input_df = pd.DataFrame([{
        "Age": age,
        "Number of Dependents": num_dependents,
        "Income": income,
        "Customer Tenure": tenure,
        "Outstanding Loans": outstanding_loans,
        "Balance": balance,
        "NumOfProducts": num_products,
        "NumComplaints": num_complaints,
        "Gender": gender,
        "Marital Status": marital_status,
        "Education Level": education,
        "Customer Segment": customer_segment
    }])

    # Convert categorical columns
    categorical_features = [
    "Gender",
    "Marital Status",
    "Education Level",
    "Customer Segment"
    ]
    
    for col in categorical_features:
        input_df[col] = input_df[col].astype("category")

    # Predict
    pred = lgb_clf.predict(input_df)[0]
    prob = lgb_clf.predict_proba(input_df)[0, 1]

    st.markdown("## Prediction Result")
    st.success(f"Churn Prediction: **{'YES' if pred == 1 else 'NO'}**")
    st.info(f"Churn Probability: **{prob * 100:.2f}%**")


    # ======================================================
    # SHAP INTERPRETATION
    # ======================================================
    st.subheader("SHAP Feature Explanation")

    explainer = shap.TreeExplainer(lgb_clf)
    shap_values = explainer.shap_values(input_df)

    shap_vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value,
        shap_vals[0],
        feature_names=input_df.columns,
        show=False
    )

    st.pyplot(fig)
    plt.close(fig)


