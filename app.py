import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Page configuration --------------------
st.set_page_config(
    page_title="Biomarker AI | Health Insights",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- Load model assets --------------------
@st.cache_resource
def load_assets():
    base_dir = Path(__file__).resolve().parent
    with (base_dir / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (base_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError as exc:
    st.error(f"Required model file not found: {exc.filename}. Keep model.pkl and scaler.pkl beside app.py.")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model files: {exc}")
    st.stop()

# -------------------- Styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#142b3b; --muted:#637889; --teal:#087f8c; --mint:#e9f7f5; --line:#dce8ec; }
.stApp { background: #f5f8fb; color: var(--ink); font-family: 'DM Sans', sans-serif; }
.block-container { max-width: 1200px; padding-top: 1.5rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
.hero { background: linear-gradient(120deg,#102e43 0%,#126b78 100%); color:white; padding:2rem 2.2rem; border-radius:22px; margin-bottom:1.25rem; }
.hero-tag { color:#aee9df; text-transform:uppercase; letter-spacing:2px; font-size:.76rem; font-weight:700; }
.hero h1 { color:white; font-family:Manrope,sans-serif; font-size:clamp(1.8rem,3vw,2.65rem); margin:.45rem 0 .5rem; }
.hero p { color:#d7e9ef; font-size:1rem; max-width:740px; margin:0; }
.section-card { background:white; border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.35rem; margin-bottom:1rem; }
.section-kicker { color:var(--teal); text-transform:uppercase; letter-spacing:1.5px; font-size:.72rem; font-weight:800; }
.section-title { font-family:Manrope,sans-serif; font-size:1.25rem; font-weight:800; color:var(--ink); margin:.25rem 0 .4rem; }
.helper { color:var(--muted); font-size:.9rem; }
.metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; min-height:105px; }
.metric-label { color:var(--muted); font-size:.82rem; margin-bottom:.35rem; }
.metric-value { color:var(--ink); font-family:Manrope,sans-serif; font-size:1.55rem; font-weight:800; }
.result-control { background:#e9f7f0; border:1px solid #bde5d0; border-radius:16px; padding:1.2rem; }
.result-patient { background:#fff2ee; border:1px solid #ffd1c5; border-radius:16px; padding:1.2rem; }
.result-title { font-family:Manrope,sans-serif; font-size:1.35rem; font-weight:800; margin:.25rem 0; }
div.stButton > button { background:#087f8c; color:white; border:0; border-radius:11px; min-height:3rem; font-weight:700; width:100%; }
div.stButton > button:hover { background:#066570; color:white; border:0; }
div[data-testid="stNumberInput"] input { border-radius:10px; }
div[data-testid="stSlider"] { padding-top:.3rem; }
.disclaimer { color:#6c7f8c; font-size:.8rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:1.4rem; }
@media (max-width: 700px) { .block-container { padding:1rem .8rem 2rem; } .hero { padding:1.4rem; border-radius:16px; } .section-card { padding:1rem; } }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
  <div class="hero-tag">Clinical ML • Decision support demo</div>
  <h1>Biomarker Health Insights</h1>
  <p>Explore a K-Nearest Neighbors (KNN) model prediction from physical and blood-analysis biomarkers in a clean, easy-to-read workspace.</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Quick summary --------------------
st.markdown('<div class="section-card"><div class="section-kicker">Workspace</div><div class="section-title">Patient profile</div><div class="helper">Enter the available biomarker values below, then run the model to view its predicted class and confidence scores.</div></div>', unsafe_allow_html=True)

# -------------------- Input fields --------------------
with st.form("biomarker_form"):
    st.markdown('<div class="section-title">Biomarker inputs</div>', unsafe_allow_html=True)
    st.caption("Use the measurement units shown beside each field.")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age (years)", min_value=10, max_value=100, value=50)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1, format="%.1f")
        glucose = st.number_input("Glucose (mg/dL)", min_value=10.0, max_value=300.0, value=90.0, step=1.0)
    with col2:
        insulin = st.number_input("Insulin (µU/mL)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
        homa = st.number_input("HOMA", min_value=0.1, max_value=30.0, value=2.0, step=0.1, format="%.2f")
        leptin = st.number_input("Leptin (ng/mL)", min_value=1.0, max_value=120.0, value=20.0, step=0.5)
    with col3:
        adiponectin = st.number_input("Adiponectin (µg/mL)", min_value=1.0, max_value=40.0, value=10.0, step=0.5)
        resistin = st.number_input("Resistin (ng/mL)", min_value=1.0, max_value=100.0, value=10.0, step=0.5)
        mcp_1 = st.number_input("MCP-1 (pg/mL)", min_value=10.0, max_value=2000.0, value=400.0, step=10.0)
    submitted = st.form_submit_button("✨ Analyze biomarkers")

input_data = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "glucose": glucose,
    "insulin": insulin,
    "homa": homa,
    "leptin": leptin,
    "adiponectin": adiponectin,
    "resistin": resistin,
    "mcp_1": mcp_1,
}])

# Keep the same feature order as the original app.
if submitted:
    try:
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]

        # Resolve probabilities by class label, rather than assuming array order.
        prob_by_class = {label: float(prob) for label, prob in zip(classes, probabilities)}
        control_prob = prob_by_class.get(1, 0.0)
        patient_prob = prob_by_class.get(2, 0.0)

        st.markdown("---")
        st.markdown('<div class="section-kicker">Analysis complete</div><div class="section-title">Prediction overview</div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1.15, 1])
        with r1:
            if prediction == 1:
                st.markdown('<div class="result-control"><div>MODEL CLASSIFICATION</div><div class="result-title">Control Group</div><div>Predicted class: 1</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-patient"><div>MODEL CLASSIFICATION</div><div class="result-title">Patient Group</div><div>Predicted class: 2</div></div>', unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="metric-card"><div class="metric-label">Model confidence for predicted class</div><div class="metric-value">' + f'{float(np.max(probabilities)):.1%}' + '</div><div class="helper">Based on the model probability output</div></div>', unsafe_allow_html=True)

        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Control Group · Class 1**")
            st.progress(max(0.0, min(1.0, control_prob)))
            st.write(f"{control_prob:.2%}")
        with p2:
            st.markdown("**Patient Group · Class 2**")
            st.progress(max(0.0, min(1.0, patient_prob)))
            st.write(f"{patient_prob:.2%}")

        with st.expander("Review submitted biomarker values"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
else:
    st.info("Enter the biomarker values and select **Analyze biomarkers** to see the model output.")

with st.expander("About this demo"):
    st.write("This interface displays outputs from the supplied KNN model and scaler. It is an educational demonstration, not a medical diagnosis or a substitute for advice from a qualified healthcare professional.")

st.markdown('<div class="disclaimer">For educational use only. Model predictions may be incorrect and should not be used alone for clinical decisions.</div>', unsafe_allow_html=True)
