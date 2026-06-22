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
    model   = joblib.load("model/best_model.joblib")
    scaler  = joblib.load("model/scaler.joblib")
    columns = joblib.load("model/feature_columns.joblib")
    return model, scaler, columns

model, scaler, feature_columns = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("Customer Churn Prediction")
st.caption("UAS Bengkel Koding Data Science — Universitas Dian Nuswantoro")
st.markdown("Masukkan data pelanggan di bawah ini untuk memprediksi kemungkinan churn.")
st.divider()

# ============================================================
# FORM INPUT
# ============================================================

st.subheader("Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    age                 = st.number_input("Age", 18, 80, 35)
    gender              = st.selectbox("Gender", ["Male", "Female"])
    subscription_type   = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    is_premium_user     = st.selectbox("Premium User", [0, 1],
                            format_func=lambda x: "Ya" if x == 1 else "Tidak")
    device_type         = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
    acquisition_channel = st.selectbox("Acquisition Channel",
                            ["Email", "Social Media", "Ads", "Referral", "Organic"])
    payment_method      = st.selectbox("Payment Method",
                            ["Credit Card", "Debit Card", "PayPal",
                             "Bank Transfer", "Wallet"])
    country             = st.selectbox("Country",
                            ["India", "USA", "Germany", "UK",
                             "France", "Canada", "Australia", "Brazil"])
    city                = st.selectbox("City",
                            ["Mumbai", "New York", "Berlin", "London",
                             "Paris", "Toronto", "Sydney", "São Paulo"])

with col2:
    total_visits        = st.number_input("Total Visits", 0, 500, 50)
    avg_session_time    = st.number_input("Avg Session Time (menit)", 0.0, 120.0, 20.0)
    pages_per_session   = st.number_input("Pages per Session", 0.0, 30.0, 5.0)
    email_open_rate     = st.number_input("Email Open Rate", 0.0, 1.0, 0.3)
    email_click_rate    = st.number_input("Email Click Rate", 0.0, 1.0, 0.1)
    total_spent         = st.number_input("Total Spent ($)", 0.0, 5000.0, 200.0)
    avg_order_value     = st.number_input("Avg Order Value ($)", 0.0, 1000.0, 50.0)
    discount_used       = st.selectbox("Discount Used", [0, 1],
                            format_func=lambda x: "Ya" if x == 1 else "Tidak")

with col3:
    last_3_month_purchase_freq = st.number_input("Purchase Freq (3 Bulan)", 0, 30, 3)
    support_tickets     = st.number_input("Support Tickets", 0, 20, 2)
    refund_requested    = st.selectbox("Refund Requested", [0, 1],
                            format_func=lambda x: "Ya" if x == 1 else "Tidak")
    delivery_delay_days = st.number_input("Delivery Delay (Hari)", 0, 30, 2)
    satisfaction_score  = st.number_input("Satisfaction Score (1-5)", 1.0, 5.0, 4.0)
    nps_score           = st.number_input("NPS Score (0-10)", 0, 10, 7)
    marketing_spend_per_user = st.number_input("Marketing Spend ($)", 0.0, 500.0, 20.0)
    lifetime_value      = st.number_input("Lifetime Value ($)", 0.0, 10000.0, 500.0)

st.divider()

# ============================================================
# PREDIKSI
# ============================================================

if st.button("Prediksi Churn", use_container_width=True, type="primary"):

    # Susun input ke DataFrame
    input_data = {
        "gender"                    : gender,
        "age"                       : age,
        "country"                   : country,
        "city"                      : city,
        "acquisition_channel"       : acquisition_channel,
        "device_type"               : device_type,
        "subscription_type"         : subscription_type,
        "is_premium_user"           : is_premium_user,
        "total_visits"              : total_visits,
        "avg_session_time"          : avg_session_time,
        "pages_per_session"         : pages_per_session,
        "email_open_rate"           : email_open_rate,
        "email_click_rate"          : email_click_rate,
        "total_spent"               : total_spent,
        "avg_order_value"           : avg_order_value,
        "discount_used"             : discount_used,
        "support_tickets"           : support_tickets,
        "refund_requested"          : refund_requested,
        "delivery_delay_days"       : delivery_delay_days,
        "satisfaction_score"        : satisfaction_score,
        "nps_score"                 : nps_score,
        "marketing_spend_per_user"  : marketing_spend_per_user,
        "lifetime_value"            : lifetime_value,
        "last_3_month_purchase_freq": last_3_month_purchase_freq,
        "payment_method"            : payment_method,
    }

    input_df = pd.DataFrame([input_data])

    # Encoding kategorikal (urutan alfabetis mengikuti LabelEncoder sklearn)
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
        "subscription_type"   : {"Basic": 0, "Premium": 1, "Standard": 2},
        "payment_method"      : {"Bank Transfer": 0, "Credit Card": 1,
                                  "Debit Card": 2, "PayPal": 3, "Wallet": 4},
    }

    input_encoded = input_df.copy()
    for col, mapping in encoding_map.items():
        if col in input_encoded.columns:
            input_encoded[col] = input_encoded[col].map(mapping)

    # Urutkan kolom sesuai urutan saat training lalu scaling
    input_encoded = input_encoded[feature_columns]
    input_scaled  = scaler.transform(input_encoded)

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

    # Metric cards
    m1, m2 = st.columns(2)
    m1.metric("Probabilitas Churn",       f"{churn_proba:.2f}%")
    m2.metric("Probabilitas Tidak Churn", f"{no_churn_proba:.2f}%")

    # Bar chart probabilitas
    fig, ax = plt.subplots(figsize=(6, 2.5))
    bars = ax.barh(
        ["Tidak Churn", "Churn"],
        [no_churn_proba, churn_proba],
        color=["#2ECC71", "#E74C3C"],
        edgecolor="white",
        height=0.5
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

    # Tabel data input
    st.subheader("Data Pelanggan yang Diinput")
    st.dataframe(
        input_df.T.rename(columns={0: "Nilai"}),
        use_container_width=True
    )

    st.divider()

    # Info model
    st.caption("Model: Random Forest | F1-Score Churn: 0.67 | Recall: 0.98")