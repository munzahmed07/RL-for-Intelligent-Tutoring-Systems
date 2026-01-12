import streamlit as st
import time
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from tutoring_env import TutoringEnv

# Page Config
st.set_page_config(page_title="AI Tutor Demo", page_icon="🎓")

st.title("🎓 RL Intelligent Tutor: Live Simulation")
st.markdown(
    "Watch the AI agent adapt difficulty in real-time to maximize student mastery.")

# Sidebar controls
agent_type = st.sidebar.selectbox(
    "Choose Agent", ["PPO (Policy Gradient)", "DQN (Deep Q-Network)"])
speed = st.sidebar.slider("Simulation Speed", 0.05, 1.0, 0.2)

if st.button("Start Live Simulation"):
    # Load Model
    model_path = f"models/{agent_type.split()[0].lower()}_tutor"
    try:
        if "PPO" in agent_type:
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
    except:
        st.error("Model not found! Did you train the agents yet?")
        st.stop()

    # Init Env
    env = TutoringEnv()
    obs, _ = env.reset()

    # UI Elements
    col1, col2, col3 = st.columns(3)
    with col1:
        mastery_metric = st.empty()
    with col2:
        eng_metric = st.empty()
    with col3:
        diff_metric = st.empty()

    chart_placeholder = st.empty()
    log_placeholder = st.empty()

    # Data logging
    data = {"Step": [], "Mastery": [], "Engagement": [], "Difficulty": []}

    # Run Episode
    for step in range(50):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        # Decode action to text
        diff_map = {0: "Easy 🟢", 1: "Medium 🟡", 2: "Hard 🔴"}
        difficulty = diff_map[action.item()]

        # Update Data
        data["Step"].append(step)
        data["Mastery"].append(env.student_mastery)
        data["Engagement"].append(env.student_engagement)
        data["Difficulty"].append((action + 1) * 0.33)  # Scaled for graph

        # Update Metrics
        mastery_metric.metric(
            "Student Mastery", f"{env.student_mastery:.2f}", delta=None)
        eng_metric.metric("Engagement", f"{env.student_engagement:.2f}")
        diff_metric.metric("AI Decision", difficulty)

        # Update Chart
        df = pd.DataFrame(data)
        chart_placeholder.line_chart(
            df.set_index("Step")[["Mastery", "Engagement"]])

        # Simulated "Thinking" time
        time.sleep(speed)

    st.success("Simulation Complete! The student reached their potential.")
