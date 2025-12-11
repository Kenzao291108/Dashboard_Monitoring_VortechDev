import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ---------------- Config ----------------
CSV_FILE = "inference_log.csv"
REFRESH_INTERVAL_MS = 2000  # refresh setiap 2 detik

# ---------------- Auto-refresh ----------------
count = st_autorefresh(interval=REFRESH_INTERVAL_MS, limit=None, key="datarefresh")

st.title("Dashboard Sensor Suhu & Kelembaban VortechDev")
st.markdown("Realtime monitoring data dari ESP32 dan prediksi model.")

# ---------------- Load Data ----------------
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    st.warning(f"File {CSV_FILE} belum ada. Pastikan script inference.py berjalan.")
    st.stop()

# ---------------- Table ----------------
st.subheader("Data Terbaru")
st.dataframe(df.tail(20))  # tampilkan 20 data terakhir

# ---------------- Grafik Suhu ----------------
st.subheader("Suhu (°C)")
fig_temp = px.line(
    df,
    x="timestamp",
    y="suhu",
    title="Suhu Real-Time",
    markers=True
)
st.plotly_chart(fig_temp, use_container_width=True, key="fig_temp")

# ---------------- Grafik Kelembaban ----------------
st.subheader("Kelembaban (%)")
fig_hum = px.line(
    df,
    x="timestamp",
    y="kelembaban",
    title="Kelembaban Real-Time",
    markers=True
)
st.plotly_chart(fig_hum, use_container_width=True, key="fig_hum")

# ---------------- Grafik Prediksi ----------------
st.subheader("Prediksi Status")
fig_pred = px.line(
    df,
    x="timestamp",
    y="pred_smooth",
    title="Prediksi Status (normal/warning/overheat)",
    markers=True
)
st.plotly_chart(fig_pred, use_container_width=True, key="fig_pred")

# ---------------- Filter Berdasarkan Status ----------------
st.subheader("Filter Berdasarkan Status")
status_filter = st.multiselect(
    "Pilih status untuk filter:",
    options=df["true_label"].unique(),
    default=df["true_label"].unique()
)

filtered_df = df[df["true_label"].isin(status_filter)]
st.dataframe(filtered_df.tail(20))
