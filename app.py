"""
app.py — Customer Churn Prediction
UAS Bengkel Koding | Azzam Izzuddin | A11.2023.14992 | UDINUS
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# LOAD MODEL & PREPROCESSING
# ─────────────────────────────────────────────
MODEL_PATH    = "model/best_rf_model.joblib"
SCALER_PATH   = "model/scaler.joblib"
FEATURES_PATH = "model/top_features.joblib"

@st.cache_resource
def load_artifacts():
    model    = joblib.load(MODEL_PATH)
    scaler   = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]):
    st.error(
        "⚠️ File model tidak ditemukan. Pastikan folder `model/` berisi:\n"
        "- `best_rf_model.joblib`\n"
        "- `scaler.joblib`\n"
        "- `top_features.joblib`"
    )
    st.stop()

model, scaler, features = load_artifacts()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.caption("UAS Bengkel Koding · Azzam Izzuddin (A11.2023.14992) · UDINUS")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR — INFO MODEL
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Info Model")
    st.success("✅ Model berhasil dimuat")
    st.markdown(f"**Algoritma:** Random Forest (Tuned)")
    st.markdown(f"**Jumlah fitur:** {len(features)}")
    st.markdown(f"**Teknik balancing:** SMOTE + class_weight='balanced'")
    st.markdown("---")
    st.markdown("**Fitur yang digunakan:**")
    for f in features:
        st.markdown(f"- `{f}`")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Prediksi Manual", "📂 Prediksi Batch (CSV)"])

# ══════════════════════════════════════════════
# TAB 1 — PREDIKSI MANUAL
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Masukkan Data Pelanggan")
    st.info("Isi data pelanggan di bawah ini lalu klik **Prediksi**.")

    # Label Encoding mapping (sesuai urutan alphabetical LabelEncoder default)
    gender_map           = {"Female": 0, "Male": 1, "Unknown": 2}
    country_map          = {"Germany": 0, "India": 1, "USA": 2}
    subscription_map     = {"Basic": 0, "Premium": 1, "Standard": 2}
    device_map           = {"Desktop": 0, "Mobile": 1, "Tablet": 2}
    acquisition_map      = {"Email": 0, "Organic": 1, "Paid Ad": 2, "Referral": 3, "Social Media": 4}
    payment_map          = {"Bank Transfer": 0, "Credit Card": 1, "Debit Card": 2, "PayPal": 3}
    age_group_map        = {"Adult": 0, "Middle": 1, "Senior": 2, "Young": 3}

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demografi**")
        gender           = st.selectbox("Gender",            list(gender_map.keys()))
        age              = st.number_input("Usia (tahun)",   min_value=18, max_value=100, value=30)
        country          = st.selectbox("Negara",            list(country_map.keys()))
        subscription     = st.selectbox("Tipe Langganan",    list(subscription_map.keys()))
        device           = st.selectbox("Tipe Perangkat",    list(device_map.keys()))
        acquisition      = st.selectbox("Akuisisi Channel",  list(acquisition_map.keys()))
        payment          = st.selectbox("Metode Pembayaran", list(payment_map.keys()))
        age_group        = st.selectbox("Kelompok Usia",     list(age_group_map.keys()))

    with col2:
        st.markdown("**🛒 Perilaku Transaksi**")
        total_visits         = st.number_input("Total Kunjungan",             min_value=0,   value=20)
        avg_session_time     = st.number_input("Rata-rata Durasi Sesi (mnt)", min_value=0.0, value=15.0, step=0.5)
        total_spent          = st.number_input("Total Pengeluaran ($)",        min_value=0.0, value=300.0, step=10.0)
        avg_order_value      = st.number_input("Rata-rata Nilai Order ($)",    min_value=0.0, value=50.0, step=5.0)
        num_transactions     = st.number_input("Jumlah Transaksi",             min_value=0,   value=10)
        refund_requested     = st.number_input("Jumlah Refund Diminta",        min_value=0,   value=0)
        last_3m_purchase     = st.number_input("Frekuensi Beli 3 Bln Terakhir", min_value=0, value=2)

    with col3:
        st.markdown("**📬 Engagement & Layanan**")
        satisfaction_score   = st.slider("Skor Kepuasan (1–5)",          1.0, 5.0, 3.5, 0.1)
        nps_score            = st.slider("NPS Score (0–10)",              0.0, 10.0, 7.0, 0.5)
        email_open_rate      = st.slider("Email Open Rate (0–1)",         0.0, 1.0, 0.3, 0.01)
        email_click_rate     = st.slider("Email Click Rate (0–1)",        0.0, 1.0, 0.1, 0.01)
        support_tickets      = st.number_input("Jumlah Tiket Support",    min_value=0, value=1)
        loyalty_points       = st.number_input("Poin Loyalitas",          min_value=0, value=500)
        lifetime_value       = st.number_input("Lifetime Value ($)",      min_value=0.0, value=1000.0, step=50.0)

        st.markdown("**📅 Waktu**")
        tenure_days          = st.number_input("Lama Berlangganan (hari)",   min_value=0, value=365)
        recency_days         = st.number_input("Hari Sejak Pembelian Terakhir", min_value=0, value=30)

    # Fitur turunan (feature engineering dari notebook)
    email_engagement_rate = (email_open_rate + email_click_rate) / 2
    spend_per_visit       = total_spent / (total_visits + 1)
    refund_ticket_ratio   = refund_requested / (support_tickets + 1)
    is_recently_active    = int(last_3m_purchase > 0)

    st.markdown("---")

    if st.button("🔮 Prediksi Churn", type="primary", use_container_width=True):

        # Bangun input dict — urutan harus sesuai dengan features
        raw_input = {
            "age"                    : age,
            "gender"                 : gender_map.get(gender, 0),
            "country"                : country_map.get(country, 0),
            "subscription_type"      : subscription_map.get(subscription, 0),
            "device_type"            : device_map.get(device, 0),
            "acquisition_channel"    : acquisition_map.get(acquisition, 0),
            "payment_method"         : payment_map.get(payment, 0),
            "total_visits"           : total_visits,
            "avg_session_time"       : avg_session_time,
            "total_spent"            : total_spent,
            "avg_order_value"        : avg_order_value,
            "num_transactions"       : num_transactions,
            "refund_requested"       : refund_requested,
            "last_3_month_purchase_freq": last_3m_purchase,
            "satisfaction_score"     : satisfaction_score,
            "nps_score"              : nps_score,
            "email_open_rate"        : email_open_rate,
            "email_click_rate"       : email_click_rate,
            "support_tickets"        : support_tickets,
            "loyalty_points"         : loyalty_points,
            "lifetime_value"         : lifetime_value,
            "tenure_days"            : tenure_days,
            "recency_days"           : recency_days,
            "email_engagement_rate"  : email_engagement_rate,
            "spend_per_visit"        : spend_per_visit,
            "refund_ticket_ratio"    : refund_ticket_ratio,
            "is_recently_active"     : is_recently_active,
            "age_group"              : age_group_map.get(age_group, 0),
        }

        # Filter hanya fitur yang dipakai model
        try:
            X_input = pd.DataFrame([{k: raw_input[k] for k in features}])
        except KeyError as e:
            st.error(f"Fitur tidak ditemukan: {e}")
            st.stop()

        X_scaled = scaler.transform(X_input)
        pred     = model.predict(X_scaled)[0]
        proba    = model.predict_proba(X_scaled)[0]

        churn_prob    = proba[1] * 100
        no_churn_prob = proba[0] * 100

        st.markdown("---")
        st.subheader("📋 Hasil Prediksi")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            if pred == 1:
                st.error("⚠️ **CHURN**\nPelanggan ini diprediksi akan berhenti berlangganan.")
            else:
                st.success("✅ **TIDAK CHURN**\nPelanggan ini diprediksi tetap berlangganan.")
        with res_col2:
            st.metric("Probabilitas Churn",    f"{churn_prob:.1f}%")
        with res_col3:
            st.metric("Probabilitas Tidak Churn", f"{no_churn_prob:.1f}%")

        # Rekomendasi
        if pred == 1:
            st.warning(
                "**Rekomendasi Bisnis:**\n"
                "- 📧 Kirim email retensi dengan penawaran eksklusif\n"
                "- 🎁 Berikan diskon atau upgrade berlangganan\n"
                "- 📞 Hubungi pelanggan secara personal\n"
                "- 🔍 Tindak lanjuti tiket support yang belum terselesaikan"
            )
        else:
            st.info(
                "**Rekomendasi Bisnis:**\n"
                "- 🏆 Tingkatkan program loyalitas untuk mempertahankan pelanggan\n"
                "- 📊 Monitor engagement secara berkala\n"
                "- 🎯 Tawarkan upsell produk Premium"
            )

# ══════════════════════════════════════════════
# TAB 2 — PREDIKSI BATCH (CSV)
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Upload CSV untuk Prediksi Batch")
    st.markdown(
        "Upload file CSV yang sudah berisi kolom sesuai fitur model. "
        "Kolom kategorikal (`gender`, `country`, `subscription_type`, `device_type`, "
        "`acquisition_channel`, `payment_method`, `age_group`) **sudah di-encode** "
        "menggunakan LabelEncoder atau gunakan file yang sudah diproses."
    )

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.markdown(f"**Data yang diunggah:** {df_upload.shape[0]:,} baris × {df_upload.shape[1]} kolom")
        st.dataframe(df_upload.head(5), use_container_width=True)

        missing_cols = [f for f in features if f not in df_upload.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ada di CSV: {missing_cols}")
        else:
            if st.button("🚀 Jalankan Prediksi Batch", type="primary"):
                with st.spinner("Memproses prediksi..."):
                    X_batch  = df_upload[features].copy()
                    X_scaled = scaler.transform(X_batch)
                    preds    = model.predict(X_scaled)
                    probas   = model.predict_proba(X_scaled)[:, 1]

                df_result = df_upload.copy()
                df_result["churn_prediction"] = preds
                df_result["churn_probability"] = (probas * 100).round(2)
                df_result["status"] = df_result["churn_prediction"].map({0: "Tidak Churn", 1: "Churn"})

                st.success(f"✅ Prediksi selesai untuk {len(df_result):,} pelanggan.")

                total_churn = preds.sum()
                st.markdown(f"""
                | Metrik | Nilai |
                |--------|-------|
                | Total Pelanggan | {len(preds):,} |
                | Prediksi Churn  | {total_churn:,} ({total_churn/len(preds)*100:.1f}%) |
                | Prediksi Tidak Churn | {len(preds)-total_churn:,} ({(len(preds)-total_churn)/len(preds)*100:.1f}%) |
                """)

                st.dataframe(
                    df_result[["churn_prediction", "churn_probability", "status"]
                               + [c for c in df_result.columns if c not in ["churn_prediction","churn_probability","status"]]
                    ].head(20),
                    use_container_width=True
                )

                csv_out = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_churn.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("UAS Bengkel Koding · Azzam Izzuddin (A11.2023.14992) · Universitas Dian Nuswantoro · 2024")
