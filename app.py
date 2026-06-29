# ============================================================
# app.py — Customer Churn Prediction
# UAS Bengkel Koding - Universitas Dian Nuswantoro
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ============================================================
# MUAT MODEL
# ============================================================

@st.cache_resource
def load_model():
    model        = joblib.load("model/best_rf_model.joblib")
    scaler       = joblib.load("model/scaler.joblib")
    top_features = joblib.load("model/top_features.joblib")
    return model, scaler, top_features

model, scaler, top_features = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("Customer Churn Prediction")
st.caption("UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro")
st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")
st.divider()

# ============================================================
# ENCODING MAP (urutan alfabetis = urutan LabelEncoder sklearn)
# ============================================================

encoding_map = {
    "gender"              : {"Female": 0, "Male": 1},
    "country"             : {"Australia": 0, "Brazil": 1, "Canada": 2,
                              "France": 3, "Germany": 4, "India": 5,
                              "UK": 6, "USA": 7},
    "city"                : {"Berlin": 0, "London": 1, "Mumbai": 2,
                              "New York": 3, "Paris": 4, "Sydney": 5,
                              "Toronto": 6, "São Paulo": 7},
    "acquisition_channel" : {"Ads": 0, "Email": 1, "Organic": 2,
                              "Referral": 3, "Social Media": 4},
    "device_type"         : {"Desktop": 0, "Mobile": 1, "Tablet": 2},
    "payment_method"      : {"Bank Transfer": 0, "Credit Card": 1,
                              "Debit Card": 2, "PayPal": 3, "Wallet": 4},
    "coupon_code"         : {"SAVE10": 0, "SAVE20": 1, "SAVE30": 2,
                              "WELCOME": 3, "No Coupon": 4},
}

# ============================================================
# FORM INPUT — hanya 10 fitur yang dipakai model
# ============================================================

st.subheader("Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    satisfaction_score  = st.slider("Satisfaction Score (1-5)", 1.0, 5.0, 4.0, 0.1)
    total_spent         = st.number_input("Total Spent ($)", 0.0, 5000.0, 200.0)
    support_tickets     = st.number_input("Support Tickets", 0, 20, 2)
    device_type         = st.selectbox("Device Type",
                            ["Mobile", "Desktop", "Tablet"])
    coupon_code         = st.selectbox("Coupon Code",
                            ["No Coupon", "SAVE10", "SAVE20", "SAVE30", "WELCOME"])

with col2:
    payment_method      = st.selectbox("Payment Method",
                            ["Credit Card", "Debit Card", "PayPal",
                             "Bank Transfer", "Wallet"])
    gender              = st.selectbox("Gender", ["Male", "Female"])
    country             = st.selectbox("Country",
                            ["India", "USA", "Germany", "UK",
                             "France", "Canada", "Australia", "Brazil"])
    acquisition_channel = st.selectbox("Acquisition Channel",
                            ["Email", "Social Media", "Ads", "Referral", "Organic"])
    city                = st.selectbox("City",
                            ["Mumbai", "New York", "Berlin", "London",
                             "Paris", "Toronto", "Sydney", "São Paulo"])

st.divider()

# ============================================================
# TOMBOL PREDIKSI
# ============================================================

if st.button("Prediksi Churn", use_container_width=True, type="primary"):

    # Susun input sesuai urutan top_features
    input_data = {
        "satisfaction_score"  : satisfaction_score,
        "total_spent"         : total_spent,
        "support_tickets"     : support_tickets,
        "device_type"         : encoding_map["device_type"].get(device_type, 0),
        "coupon_code"         : encoding_map["coupon_code"].get(coupon_code, 4),
        "payment_method"      : encoding_map["payment_method"].get(payment_method, 0),
        "gender"              : encoding_map["gender"].get(gender, 0),
        "country"             : encoding_map["country"].get(country, 0),
        "acquisition_channel" : encoding_map["acquisition_channel"].get(acquisition_channel, 0),
        "city"                : encoding_map["city"].get(city, 0),
    }

    input_df   = pd.DataFrame([input_data])
    input_top10 = input_df[top_features]
    input_scaled = scaler.transform(input_top10)

    # Prediksi
    prediction       = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    churn_proba      = prediction_proba[1] * 100
    no_churn_proba   = prediction_proba[0] * 100

    # ============================================================
    # TAMPILAN HASIL
    # ============================================================

    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"Pelanggan ini diprediksi **CHURN** dengan probabilitas {churn_proba:.1f}%")
    else:
        st.success(f"Pelanggan ini diprediksi **TIDAK CHURN** dengan probabilitas {no_churn_proba:.1f}%")

    m1, m2 = st.columns(2)
    m1.metric("Probabilitas Churn",       f"{churn_proba:.2f}%")
    m2.metric("Probabilitas Tidak Churn", f"{no_churn_proba:.2f}%")

    # Bar chart probabilitas
    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(
        ["Tidak Churn", "Churn"],
        [no_churn_proba, churn_proba],
        color=["#2ECC71", "#E74C3C"],
        edgecolor="white", height=0.5
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilitas (%)", fontsize=9)
    ax.set_title("Distribusi Probabilitas Prediksi", fontsize=10, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, [no_churn_proba, churn_proba]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    # Tabel input
    st.subheader("Data yang Diinput")
    display_df = pd.DataFrame({
        "Fitur"  : list(input_data.keys()),
        "Nilai"  : list(input_data.values())
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # Penjelasan fitur penting
    st.subheader("Fitur Paling Berpengaruh terhadap Churn")
    st.markdown("""
| Fitur | Keterangan |
|---|---|
| satisfaction_score | Skor kepuasan pelanggan (1-5) — semakin rendah, semakin berisiko churn |
| total_spent | Total pengeluaran pelanggan — pelanggan dengan pengeluaran tinggi lebih berisiko |
| support_tickets | Jumlah tiket dukungan — semakin banyak, semakin tidak puas |
| device_type | Jenis perangkat yang digunakan |
| coupon_code | Kode kupon yang digunakan pelanggan |
    """)

    st.divider()
    st.caption("Model: Random Forest | Top-10 Features | UAS Bengkel Koding UDINUS")
