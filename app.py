import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Prediksi Churn Pelanggan", layout="centered")

# =========================================================
# Load Artifacts
# =========================================================
@st.cache_resource
def load_artifacts():
    model          = joblib.load('best_rf_model.joblib')
    scaler         = joblib.load('scaler.joblib')
    encoders       = joblib.load('encoders.joblib')
    top10_features = joblib.load('top_features.joblib')
    feature_stats  = joblib.load('feature_stats.joblib')
    return model, scaler, encoders, top10_features, feature_stats

model, scaler, encoders, top10_features, feature_stats = load_artifacts()
FITUR_MODEL = top10_features

# =========================================================
# Helper: render input per fitur
# =========================================================
def render_input(col_name, feature_stats, encoders):
    label = col_name.replace('_', ' ').title()

    if col_name in encoders:
        return st.selectbox(label, options=encoders[col_name].classes_)

    stats = feature_stats.get(col_name, {})
    f_min = stats.get('min', 0.0)
    f_max = stats.get('max', 1.0)
    dtype = stats.get('dtype', 'float64')

    if f_min == f_max:
        f_max = f_min + 1

    midpoint = (f_min + f_max) / 2

    if 'int' in dtype:
        return st.number_input(label, min_value=int(f_min), max_value=int(f_max), value=int(midpoint), step=1)
    else:
        return st.number_input(label, min_value=float(f_min), max_value=float(f_max), value=float(round(midpoint, 4)), step=float(round((f_max - f_min) / 100, 4)))

# =========================================================
# Header
# =========================================================
st.title("Prediksi Churn Pelanggan")
st.caption("UAS Bengkel Koding · Azzam Izzuddin (A11.2023.14992) · UDINUS")
st.divider()

with st.expander("Fitur yang digunakan model"):
    for f in FITUR_MODEL:
        st.markdown(f"- `{f}`")

# =========================================================
# Form Input
# =========================================================
with st.form("prediction_form"):
    st.subheader("Data Pelanggan")
    col1, col2 = st.columns(2)
    user_input = {}

    for i, col_name in enumerate(FITUR_MODEL):
        with (col1 if i % 2 == 0 else col2):
            user_input[col_name] = render_input(col_name, feature_stats, encoders)

    st.divider()
    submitted = st.form_submit_button("Prediksi", use_container_width=True, type="primary")

# =========================================================
# Hasil Prediksi
# =========================================================
if submitted:
    input_data = pd.DataFrame([user_input], columns=FITUR_MODEL)

    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    churn_prob   = model.predict_proba(input_scaled)[0][1] * 100

    st.divider()
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ **Berisiko Churn** — Kemungkinan churn: **{churn_prob:.1f}%**")
        st.warning(
            "**Rekomendasi:**\n"
            "- Kirim penawaran retensi atau diskon khusus\n"
            "- Tindak lanjuti tiket support yang belum selesai\n"
            "- Hubungi pelanggan secara personal"
        )
    else:
        st.success(f"✅ **Cenderung Bertahan** — Kemungkinan churn: **{churn_prob:.1f}%**")
        st.info(
            "**Rekomendasi:**\n"
            "- Pertahankan dengan program loyalitas\n"
            "- Tawarkan upgrade ke paket Premium"
        )

    st.progress(int(churn_prob), text=f"Risiko Churn: {churn_prob:.1f}%")
