import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================================
# Konfigurasi Halaman
# =========================================================
st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --ink: #2B2A28;
    --paper: #FAF7F2;
    --panel: #FFFFFF;
    --border: #E4DED3;
    --teal: #2F6F62;
    --clay: #C1542D;
    --muted: #756F64;
}

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.stApp {
    background-color: var(--paper);
}

.eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--teal);
    margin-bottom: 0.3rem;
}

.page-title {
    font-family: 'Fraunces', serif;
    font-size: 2.3rem;
    font-weight: 600;
    color: var(--ink);
    margin: 0;
    line-height: 1.15;
}

.page-subtitle {
    font-family: 'Inter', sans-serif;
    color: var(--muted);
    font-size: 1rem;
    margin-top: 0.5rem;
    max-width: 640px;
}

hr {
    border-color: var(--border) !important;
    margin: 1.6rem 0 !important;
}

div[data-testid="stForm"] {
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 2rem 2rem 1.2rem 2rem;
}

div[data-testid="stForm"] label {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 0.92rem;
    color: var(--ink);
}

div[data-testid="stFormSubmitButton"] button {
    background-color: var(--teal);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 0.7rem 0;
    transition: background-color 0.15s ease;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #244F45;
    color: #FFFFFF;
}

.result-card {
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    background-color: var(--panel);
    border: 1px solid var(--border);
    border-left: 5px solid var(--accent-color);
    margin-top: 0.5rem;
}
.result-status {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--accent-color);
    margin: 0 0 0.3rem 0;
}
.result-explainer {
    color: var(--ink);
    font-size: 0.97rem;
    margin: 0 0 1.1rem 0;
}

.risk-track {
    position: relative;
    height: 10px;
    background-color: #EEEAE2;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}
.risk-fill {
    position: absolute;
    top: 0; left: 0; height: 100%;
    background-color: var(--accent-color);
    border-radius: 999px;
}
.risk-caption {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
}
.risk-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--ink);
}

:root, .stApp {
    --background-color: var(--paper) !important;
    --secondary-background-color: var(--panel) !important;
    --text-color: var(--ink) !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background-color: var(--panel) !important;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    background-color: var(--panel) !important;
    color: var(--ink) !important;
    font-weight: 500;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--teal) !important;
}
div[data-testid="stExpander"] summary svg {
    fill: var(--ink) !important;
}
div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background-color: var(--panel) !important;
}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] div {
    color: var(--ink) !important;
}

/* Slider thumb & track warna teal */
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--teal) !important;
    border-color: var(--teal) !important;
}
div[data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background-color: var(--teal) !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# Load Artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    model          = joblib.load('best_rf_model.joblib')
    scaler         = joblib.load('scaler.joblib')
    encoders       = joblib.load('encoders.joblib')
    top10_features = joblib.load('top_features.joblib')
    feature_stats  = joblib.load('feature_stats.joblib')   # ← baru: min/max tiap fitur
    return model, scaler, encoders, top10_features, feature_stats

model, scaler, encoders, top10_features, feature_stats = load_artifacts()

FITUR_MODEL = top10_features

# =========================================================
# Helper: render input yang tepat per fitur
# =========================================================
def render_input(col_name: str, feature_stats: dict, encoders: dict):
    """
    - Kolom kategorik  → st.selectbox  (pakai encoder.classes_)
    - Kolom int        → st.slider     (step=1, value=midpoint)
    - Kolom float      → st.slider     (step adaptif, value=midpoint)
    """
    label = col_name.replace('_', ' ').title()

    # Kategorik
    if col_name in encoders:
        return st.selectbox(label, options=encoders[col_name].classes_)

    # Numerik — ambil range dari feature_stats
    stats  = feature_stats.get(col_name, {})
    f_min  = stats.get('min', 0.0)
    f_max  = stats.get('max', 1.0)
    dtype  = stats.get('dtype', 'float64')

    # Jaga-jaga kalau min == max (edge case)
    if f_min == f_max:
        f_max = f_min + 1

    midpoint = (f_min + f_max) / 2

    if 'int' in dtype:
        return st.slider(
            label,
            min_value=int(f_min),
            max_value=int(f_max),
            value=int(midpoint),
            step=1
        )
    else:
        # Step adaptif: ~100 langkah dalam range, tapi minimal 0.01
        raw_step = (f_max - f_min) / 100
        step = round(raw_step, max(2, -int(np.floor(np.log10(abs(raw_step)))) + 1)) if raw_step > 0 else 0.01
        return st.slider(
            label,
            min_value=float(f_min),
            max_value=float(f_max),
            value=float(round(midpoint, 4)),
            step=float(step),
            format="%.4g"
        )

# =========================================================
# Header
# =========================================================
st.markdown('<div class="eyebrow">Alat bantu tim customer success</div>', unsafe_allow_html=True)
st.markdown('<p class="page-title">Seberapa besar risiko pelanggan ini churn?</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Isi data pelanggan di bawah, lalu sistem akan memperkirakan '
    'apakah pelanggan tersebut cenderung bertahan atau berisiko berhenti berlangganan.</p>',
    unsafe_allow_html=True
)

with st.expander("Data apa saja yang dipakai untuk memprediksi?"):
    st.write(
        "Model ini hanya melihat **10 hal yang paling berpengaruh** terhadap keputusan pelanggan "
        "untuk berhenti, supaya pengisian formnya cepat:"
    )
    st.markdown("\n".join([f"- {f.replace('_', ' ').title()}" for f in FITUR_MODEL]))

st.write("")

# =========================================================
# Form Input
# =========================================================
user_input = {}

with st.form("prediction_form"):
    cols = st.columns(2)

    for i, col_name in enumerate(FITUR_MODEL):
        with cols[i % 2]:
            user_input[col_name] = render_input(col_name, feature_stats, encoders)

    st.write("")
    submit_button = st.form_submit_button(label="Lihat prediksi", use_container_width=True)

# =========================================================
# Logika Prediksi
# =========================================================
if submit_button:
    input_data = pd.DataFrame([user_input], columns=FITUR_MODEL)

    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    input_scaled = scaler.transform(input_data)

    prediction       = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    churn_prob       = prediction_proba[0][1] * 100

    is_churn     = prediction[0] == 1
    accent       = "#C1542D" if is_churn else "#2F6F62"
    status_text  = "Berisiko churn" if is_churn else "Cenderung bertahan"
    explainer    = (
        "Pelanggan ini menunjukkan tanda-tanda akan berhenti berlangganan. "
        "Pertimbangkan untuk menjangkau mereka lebih dulu."
        if is_churn else
        "Pelanggan ini diperkirakan akan tetap bertahan dalam waktu dekat."
    )

    st.write("")
    st.markdown(f"""
    <div class="result-card" style="--accent-color: {accent};">
        <p class="result-status">{status_text}</p>
        <p class="result-explainer">{explainer}</p>
        <div class="risk-track">
            <div class="risk-fill" style="width: {churn_prob:.1f}%;"></div>
        </div>
        <div class="risk-caption">
            <span>Kemungkinan churn</span>
            <span class="risk-value">{churn_prob:.1f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
