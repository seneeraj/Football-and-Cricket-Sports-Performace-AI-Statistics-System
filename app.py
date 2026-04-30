import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Multi-Sport AI", layout="wide")

st.title("⚽🏏 Multi-Sport AI Performance Analyzer")

# Sidebar
st.sidebar.header("Navigation")
sport = st.sidebar.radio("Select Sport", ["Football (EPL)", "Cricket (IPL)"])

# ==============================
# ⚽ FOOTBALL SECTION
# ==============================
if sport == "Football (EPL)":

    st.header("⚽ Football Player Performance Predictor")

    model_f = pickle.load(open("models/football_model.pkl", "rb"))

    col1, col2, col3 = st.columns(3)

    with col1:
        goals = st.number_input("Goals Scored", 0, 50)
        assists = st.number_input("Assists", 0, 30)
        minutes = st.number_input("Minutes Played", 0, 4000)

    with col2:
        influence = st.number_input("Influence", 0.0, 1000.0)
        creativity = st.number_input("Creativity", 0.0, 1000.0)
        threat = st.number_input("Threat", 0.0, 1000.0)

    with col3:
        ict = st.number_input("ICT Index", 0.0, 1000.0)
        form = st.number_input("Form", 0.0, 20.0)
        bps = st.number_input("Bonus Points System (BPS)", 0, 1000)
        clean = st.number_input("Clean Sheets", 0, 50)
        conceded = st.number_input("Goals Conceded", 0, 100)

    if st.button("Predict Football Score"):
        features = np.array([[goals, assists, minutes, influence, creativity,
                              threat, ict, form, bps, clean, conceded]])

        prediction = model_f.predict(features)[0]

        st.success(f"Predicted Player Score: {round(prediction,2)}")

        st.subheader("📊 Feature Contribution")
        st.bar_chart(model_f.coef_)

# ==============================
# 🏏 CRICKET SECTION
# ==============================
elif sport == "Cricket (IPL)":

    st.header("🏏 Cricket Player Performance Predictor")

    model_c = pickle.load(open("models/cricket_model.pkl", "rb"))

    col1, col2 = st.columns(2)

    with col1:
        runs = st.number_input("Total Runs", 0, 5000)
        strike_rate = st.number_input("Strike Rate", 0.0, 300.0)

    with col2:
        wickets = st.number_input("Wickets", 0, 500)
        fours = st.number_input("Number of 4s", 0, 500)
        sixes = st.number_input("Number of 6s", 0, 300)

    if st.button("Predict Cricket Impact"):
        features = np.array([[runs, strike_rate, wickets, fours, sixes]])

        prediction = model_c.predict(features)[0]

        st.success(f"Predicted Impact Score: {round(prediction,2)}")

        st.subheader("📊 Feature Contribution")
        st.bar_chart(model_c.coef_)

# ==============================
# 📊 OPTIONAL: LEADERBOARD
# ==============================

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Upload Dataset (Optional)")

uploaded_file = st.sidebar.file_uploader("Upload CSV for analysis")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    if st.checkbox("Show Correlation Heatmap"):
        st.write(df.corr())