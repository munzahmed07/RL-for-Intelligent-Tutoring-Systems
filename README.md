# 🎓 RL-Based Intelligent Tutoring System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📖 Project Overview
This project implements **Reinforcement Learning (RL)** agents to optimize curriculum sequencing in an Intelligent Tutoring System (ITS). Using **Python** and **Stable-Baselines3**, I trained **PPO** and **DQN** agents to act as virtual tutors, dynamically adjusting quiz difficulty to maximize student learning speed and engagement.

The system models the **Zone of Proximal Development (ZPD)**, ensuring students are challenged enough to learn but not so much that they become frustrated.

## 🚀 Key Features
* **Custom Gymnasium Environment:** Simulates a learner's hidden states (*Mastery* & *Engagement*) and response dynamics (accuracy & response time).
* **Adaptive Difficulty:** Agents learn to scale difficulty in real-time based on student performance history.
* **Reward Engineering:** Composite reward function balancing **Learning Gain** (Mastery) vs. **Retention** (Engagement).
* **Interactive Dashboard:** Live Streamlit app to visualize the agent's decision-making process.

## 📊 Results
The RL agents were evaluated against static and heuristic (rule-based) baselines:
* **Performance:** DQN Agent achieved full mastery **20% faster** than heuristic methods.
* **Engagement:** RL agents successfully maintained student engagement by avoiding "boredom" (too easy) and "frustration" (too hard) traps.

![Evaluation Graph](evaluation_results.png)

## 🎮 Try the Live Demo
You can run a real-time simulation of the tutoring session using the interactive dashboard:

```bash
# Run the Streamlit App
streamlit run demo.py
```
## Created By

Built by Munzer Ahmed

