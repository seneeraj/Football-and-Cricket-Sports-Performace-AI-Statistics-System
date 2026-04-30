import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Multi-Sport AI Dashboard", layout="wide")

# ======================
# LOAD MODEL
# ======================
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        return None
    return pickle.load(open(path, "rb"))

model_f = load_model("models/football_model.pkl")
model_c = load_model("models/cricket_model.pkl")

# ======================
# HEADER
# ======================
st.title("⚽🏏 Multi-Sport AI Analytics Dashboard")
st.markdown("---")

sport = st.sidebar.selectbox("Select Sport", ["Football (EPL)", "Cricket (IPL)"])

# ======================
# RADAR CHART FUNCTION
# ======================
def radar_chart(player_vals, avg_vals, labels):
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    player_vals = np.append(player_vals, player_vals[0])
    avg_vals = np.append(avg_vals, avg_vals[0])
    angles = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.plot(angles, player_vals, label="Player")
    ax.fill(angles, player_vals, alpha=0.3)
    ax.plot(angles, avg_vals, label="Average")
    ax.legend(loc="upper right")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    return fig

# ======================
# FOOTBALL SECTION
# ======================
if sport == "Football (EPL)":

    st.header("⚽ Football Analytics")

    file_path = "data/football_players.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        player = st.selectbox("Select Player", df["name"])
        row = df[df["name"] == player].iloc[0]

        goals = row["goals"]
        assists = row["assists"]
        minutes = row["minutes"]
        influence = row["influence"]
        creativity = row["creativity"]
        threat = row["threat"]
        ict = row["ict"]
        form = row["form"]
        bps = row["bps"]
        clean = row["clean"]
        conceded = row["conceded"]

    else:
        st.warning("Football dataset not found")
        st.stop()

    # Prediction
    if st.button("Predict Football Score"):
        features = np.array([[goals, assists, minutes, influence, creativity,
                              threat, ict, form, bps, clean, conceded]])

        pred = model_f.predict(features)[0]
        st.success(f"Predicted Score: {round(pred,2)}")

    # ======================
    # VISUALS
    # ======================

    avg = df.mean(numeric_only=True)

    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [goals, assists, influence, creativity, threat],
        "Average": [
            avg["goals"], avg["assists"],
            avg["influence"], avg["creativity"], avg["threat"]
        ]
    }, index=["Goals","Assists","Influence","Creativity","Threat"])

    st.bar_chart(comp_df)

    # Radar
    st.subheader("🕸️ Radar Chart")
    labels = ["Goals","Assists","Influence","Creativity","Threat"]

    player_vals = np.array([
        goals, assists, influence, creativity, threat
    ])
    avg_vals = np.array([
        avg["goals"], avg["assists"],
        avg["influence"], avg["creativity"], avg["threat"]
    ])

    st.pyplot(radar_chart(player_vals, avg_vals, labels))

    # Leaderboard
    st.subheader("🏆 Top Players")
    st.dataframe(df.sort_values("goals", ascending=False).head(10))

    # Distribution
    st.subheader("📈 Goals Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["goals"], bins=20)
    ax.axvline(goals)
    st.pyplot(fig)

# ======================
# CRICKET SECTION
# ======================
elif sport == "Cricket (IPL)":

    st.header("🏏 Cricket Analytics")

    file_path = "data/player_stats.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        player = st.selectbox("Select Player", df["batsman"])
        row = df[df["batsman"] == player].iloc[0]

        runs = row["batsman_runs"]
        strike_rate = row["strike_rate"]
        wickets = row["wickets"]
        fours = row["is_four"]
        sixes = row["is_six"]

    else:
        st.warning("Cricket dataset not found")
        st.stop()

    # Prediction
    if st.button("Predict Cricket Impact"):
        features = np.array([[runs, strike_rate, wickets, fours, sixes]])
        pred = model_c.predict(features)[0]
        st.success(f"Predicted Impact Score: {round(pred,2)}")

# ======================
# FOOTBALL PERCENTILE + RANK
# ======================

st.subheader("📊 Player Ranking & Percentile")

# Percentile calculation
goals_pct = (df["goals"] < goals).mean() * 100
assists_pct = (df["assists"] < assists).mean() * 100
influence_pct = (df["influence"] < influence).mean() * 100

# Overall percentile (average)
overall_pct = np.mean([goals_pct, assists_pct, influence_pct])

# Rank based on goals
rank = df["goals"].rank(ascending=False, method="min")
player_rank = int(rank[df["name"] == player].values[0])
total_players = len(df)

# KPI Display
col1, col2, col3 = st.columns(3)

col1.metric("Overall Percentile", f"{round(overall_pct,1)}%")
col2.metric("Rank", f"{player_rank} / {total_players}")
col3.metric("Goals Percentile", f"{round(goals_pct,1)}%")

# Tier Classification
if overall_pct > 85:
    st.success("🔥 Elite Player")
elif overall_pct > 60:
    st.info("⭐ Good Player")
else:
    st.warning("📈 Developing Player")



    # ======================
# CRICKET PERCENTILE + RANK
# ======================

st.subheader("📊 Player Ranking & Percentile")

# Percentile
runs_pct = (df["batsman_runs"] < runs).mean() * 100
sr_pct = (df["strike_rate"] < strike_rate).mean() * 100
wicket_pct = (df["wickets"] < wickets).mean() * 100

overall_pct = np.mean([runs_pct, sr_pct, wicket_pct])

# Rank based on runs
rank = df["batsman_runs"].rank(ascending=False, method="min")
player_rank = int(rank[df["batsman"] == player].values[0])
total_players = len(df)

# KPI display
col1, col2, col3 = st.columns(3)

col1.metric("Overall Percentile", f"{round(overall_pct,1)}%")
col2.metric("Rank", f"{player_rank} / {total_players}")
col3.metric("Runs Percentile", f"{round(runs_pct,1)}%")

# Tier classification
if overall_pct > 85:
    st.success("🔥 Elite Player")
elif overall_pct > 60:
    st.info("⭐ Good Player")
else:
    st.warning("📈 Developing Player")

    



    

    # ======================
    # VISUALS
    # ======================

    avg = df.mean(numeric_only=True)

    st.subheader("📊 Player vs Average")
    comp_df = pd.DataFrame({
        "Player": [runs, strike_rate, wickets, fours, sixes],
        "Average": [
            avg["batsman_runs"], avg["strike_rate"],
            avg["wickets"], avg["is_four"], avg["is_six"]
        ]
    }, index=["Runs","Strike Rate","Wickets","Fours","Sixes"])

    st.bar_chart(comp_df)

    # Radar
    st.subheader("🕸️ Radar Chart")
    labels = ["Runs","Strike Rate","Wickets","Fours","Sixes"]

    player_vals = np.array([runs, strike_rate, wickets, fours, sixes])
    avg_vals = np.array([
        avg["batsman_runs"], avg["strike_rate"],
        avg["wickets"], avg["is_four"], avg["is_six"]
    ])

    st.pyplot(radar_chart(player_vals, avg_vals, labels))

    # Leaderboard
    st.subheader("🏆 Top Players")
    st.dataframe(df.sort_values("batsman_runs", ascending=False).head(10))

    # Distribution
    st.subheader("📈 Runs Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["batsman_runs"], bins=20)
    ax.axvline(runs)
    st.pyplot(fig)
