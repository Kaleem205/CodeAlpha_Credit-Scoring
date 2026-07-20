import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- 1. System Configuration ---
st.set_page_config(
    page_title="Credit Intelligence Engine",
    layout="wide"
)

# --- 2. Safe Modern CSS Injection ---
st.markdown("""
    <style>
    /* Gradient Text for Main Title - Enlarged */
    .title-glow {
        font-size: 18.5rem; 
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        padding-bottom: 0rem;
        line-height: 1.2;
    }
    
    /* Sleek, Modern Button */
    div.stButton > button {
        background: linear-gradient(135deg, #1e293b, #3b82f6);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        height: 3.5rem;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.3);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Artifact Loading ---
@st.cache_resource
def load_engine_assets():
    model_path = '../models/credit_model.pkl'
    scaler_path = '../models/scaler.pkl'
    
    if not os.path.exists(model_path):
        model_path = 'models/credit_model.pkl'
        scaler_path = 'models/scaler.pkl'
        
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("System Error: Serialization artifacts missing from /models directory.")
        st.stop()
        
    return joblib.load(model_path), joblib.load(scaler_path)

model, scaler = load_engine_assets()

# --- 4. Main Interface Header ---
st.markdown('<p class="title-glow">Credit Intelligence Engine</p>', unsafe_allow_html=True)
st.markdown("Real-time applicant risk profiling and automated underwriting.")

# Engine Controls hidden in a clean expander
with st.expander("⚙️ Engine Calibration & Thresholds"):
    st.caption("Adjust the classification boundary for risk tolerance. Higher thresholds require the model to be more confident before approving.")
    approval_threshold = st.slider(
        "Approval Probability Threshold", 
        min_value=0.10, max_value=0.90, value=0.50, step=0.05
    )

st.write("")

# --- 5. Application Data Entry ---
with st.container(border=True):
    st.subheader("Applicant Financial Profile")
    st.write("")
    
    col_fin, col_loan, col_demo = st.columns(3)
    
    with col_fin:
        account_status = st.selectbox(
            "Account Liquidity Tier",
            options=[1, 2, 3, 4],
            format_func=lambda x: [
                "1: Unrecorded / No Account", 
                "2: Deficit / Debit Status", 
                "3: Low Balance (< 200 DM)", 
                "4: High Balance (>= 200 DM)"
            ][x-1]
        )
        
    with col_loan:
        amount = st.number_input("Requested Principal ($)", min_value=100, max_value=20000, value=1500, step=100)
        duration = st.number_input("Term Length (Months)", min_value=1, max_value=72, value=12)
        
    with col_demo:
        age = st.number_input("Applicant Age", min_value=18, max_value=100, value=28)
        
    st.write("")
    submit_application = st.button("Execute Inference Protocol", use_container_width=True)

# --- 6. Execution & Visualization ---
if submit_application:
    # Construct baseline payload
    payload = {
        'Account_Balance': int(account_status), 'Duration_of_Credit_monthly': int(duration),
        'Payment_Status_of_Previous_Credit': 2, 'Purpose': 9, 'Credit_Amount': int(amount),
        'Value_Savings_Stocks': 2, 'Length_of_current_employment': 4, 'Instalment_per_cent': 2,
        'Sex_Marital_Status': 2, 'Guarantors': 1, 'Duration_in_Current_address': 4,
        'Most_valuable_available_asset': 1, 'Age_years': int(age), 'Concurrent_Credits': 3,
        'Type_of_apartment': 1, 'No_of_Credits_at_this_Bank': 1, 'Occupation': 2,
        'No_of_dependents': 1, 'Telephone': 1, 'Foreign_Worker': 1
    }
    
    # Preprocess
    df_input = pd.DataFrame([payload])[scaler.feature_names_in_]
    scaled_input = pd.DataFrame(scaler.transform(df_input), columns=scaler.feature_names_in_)
    
    # Inference
    probabilities = model.predict_proba(scaled_input)[0]
    prob_default = probabilities[0]
    prob_approve = probabilities[1]
    
    # Apply custom threshold
    is_approved = prob_approve >= approval_threshold
    
    st.write("")
    st.subheader("Inference Results")
    
    # Layout for results
    res_col_left, res_col_right = st.columns([1.2, 2])
    
    with res_col_left:
        with st.container(border=True):
            if is_approved:
                st.success("### ✅ APPROVED")
                st.metric(label="Approval Confidence", value=f"{prob_approve * 100:.1f}%")
                st.caption("Applicant metrics align with acceptable underwriting bounds.")
            else:
                st.error("### ❌ REJECTED")
                st.metric(label="Default Risk Assessment", value=f"{prob_default * 100:.1f}%")
                st.caption("Applicant metrics exhibit high probability of default.")
            
    with res_col_right:
        with st.container(border=True):
            st.markdown("**Probability Distribution**")
            
            # Generate a clean, highly professional horizontal bar chart
            fig, ax = plt.subplots(figsize=(7, 2.5))
            
            # Transparent background for the plot to blend into Streamlit seamlessly
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            
            classes = ['High Risk (Default)', 'Low Risk (Approve)']
            probs = [prob_default, prob_approve]
            colors = ['#ef4444', '#22c55e'] 
            
            bars = ax.barh(classes, probs, color=colors, height=0.5, edgecolor='none')
            ax.set_xlim(0, 1.0)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='gray')
            
            # Formatting the y-axis text
            ax.tick_params(axis='y', colors='gray', length=0)
            ax.tick_params(axis='x', colors='gray')
            
            # Add threshold line
            ax.axvline(x=approval_threshold, color='#94a3b8', linestyle='--', linewidth=1.5)
            ax.text(approval_threshold + 0.02, 0.5, 'Decision Threshold', color='#94a3b8', va='center', rotation=90)
            
            # Remove borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            
            st.pyplot(fig)