import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Telco Churn Intelligence Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. PORTFOLIO-GRADE CUSTOM CSS STYLING
st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #e0e6ed;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.03);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3c72;
        margin-top: 0.25rem;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(30, 60, 114, 0.4) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD SAVED ARTIFACTS
@st.cache_resource
def load_artifacts():
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('churn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('churn_feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# 4. SIDEBAR NAVIGATION
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Telco Analytics Engine")
    st.markdown("---")
    if artifacts_loaded:
        st.success("🟢 **Model Engine Online**")
        st.caption("Logistic Regression v1.0")
    else:
        st.error("🔴 **Artifact Error**")
    st.markdown("---")
    st.markdown("### Portfolio Overview")
    st.write("This enterprise intelligence application computes predictive customer churn risk scores utilizing historical usage, contractual obligations, and billing telemetry.")
    st.caption("Built with Python, Scikit-Learn, Pandas, & Streamlit.")
    st.markdown("---")

# 5. HEADER BANNER
st.markdown("""
    <div class="header-container">
        <div class="header-title">📡 Telecom Customer Churn Intelligence Hub</div>
        <div class="header-subtitle">Real-time machine learning inference for subscriber retention and customer lifetime value (CLV) optimization.</div>
    </div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"⚠️ **Error Loading Model Artifacts:** {load_error}")
    st.info("Please ensure `churn_model.pkl`, `churn_scaler.pkl`, and `churn_feature_names.pkl` are generated and placed in the project folder.")
    st.stop()

# 6. MODEL PERFORMANCE METRICS
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.markdown('<div class="metric-card"><div class="metric-label">Model Accuracy</div><div class="metric-value">80.21%</div></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-card"><div class="metric-label">Precision Score</div><div class="metric-value">65.99%</div></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-card"><div class="metric-label">Recall Rate</div><div class="metric-value">52.15%</div></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown('<div class="metric-card"><div class="metric-label">Algorithm</div><div class="metric-value">Logistic Reg.</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 7. TABBED USER INPUTS
st.subheader("📋 Customer Telemetry & Account Inputs")

tab1, tab2, tab3 = st.tabs([
    "👤 Demographics & Contract", 
    "🌐 Network & Value-Added Services", 
    "💳 Billing & Payment Architecture"
])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen Status", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
    with c2:
        tenure = st.slider("Tenure Duration (Months)", 0, 72, 12, help="Number of active subscription months.")
        contract = st.selectbox("Contract Terms", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing Preferred", ["No", "Yes"])

with tab2:
    c3, c4 = st.columns(2)
    with c3:
        phone_service = st.selectbox("Voice Line Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines Active", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Broadband Technology", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Cybersecurity Add-on", ["No", "Yes", "No internet service"])
    with c4:
        online_backup = st.selectbox("Cloud Backup Storage", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Hardware Protection Plan", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Premium Technical Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("IPTV Streaming Service", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Movie Streaming Service", ["No", "Yes", "No internet service"])

with tab3:
    c5, c6 = st.columns(2)
    with c5:
        payment_method = st.selectbox("Primary Payment Gateway", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with c6:
        monthly_charges = st.number_input("Monthly Recurring Charges ($)", 0.0, 200.0, 65.0, step=1.0)
        total_charges = st.number_input("Cumulative Total Charges ($)", 0.0, 10000.0, 780.0, step=10.0)

st.markdown("<br>", unsafe_allow_html=True)

# 8. PREDICTION & INFERENCE ENGINE
if st.button("🚀 RUN CHURN PREDICTION INFERENCE", use_container_width=True):
    raw_input = {
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Pipeline Processing
    input_df = pd.DataFrame([raw_input])
    cat_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True).astype(int)
    
    # Feature Alignment
    input_reindexed = input_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Scaling (Passing full feature DataFrame to match fitted StandardScaler)
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaled_full = pd.DataFrame(scaler.transform(input_reindexed), columns=feature_names)
    input_reindexed[num_cols] = scaled_full[num_cols]

    # Model Evaluation
    pred_class = model.predict(input_reindexed)[0]
    churn_prob = model.predict_proba(input_reindexed)[0][1]

    st.markdown("---")
    st.subheader("🎯 Executive Risk Assessment & Diagnostics")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if pred_class == 1:
            st.error("🚨 **HIGH RISK OF CHURN DETECTED**")
            st.metric(label="Inferred Churn Probability", value=f"{churn_prob:.1%}")
        else:
            st.success("✅ **STABLE CUSTOMER (LOW CHURN RISK)**")
            st.metric(label="Inferred Churn Probability", value=f"{churn_prob:.1%}")

    with res_col2:
        st.write("**Churn Probability Gauge:**")
        st.progress(float(churn_prob))

        if pred_class == 1:
            st.warning("""
            • **Contract Conversion:** Trigger automated 15% discount offer for converting to a 12-month or 24-month contract.
            • **Service Stabilization:** Offer 3 months complimentary Premium Tech Support & Cybersecurity add-ons.
            • **Priority Outreach:** Escalate customer account to Tier-2 Customer Retention Specialists.
            """)
        else:
            st.info("""
            • **Cross-sell Potential:** Account is stable; evaluate eligibility for fiber upgrade packages.
            • **Loyalty Program:** Enroll subscriber into annual tenure reward benefits.
            """)