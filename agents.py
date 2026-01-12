import gymnasium as gym
from stable_baselines3 import PPO, DQN
from tutoring_env import TutoringEnv
import os

# 1. Create directories to store the trained models and logs
models_dir = "models"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 2. Instantiate the Custom Environment
env = TutoringEnv()

# --- TRAINING PPO AGENT ---
print("--- Training PPO Agent ---")
# MlpPolicy = Multi-Layer Perceptron (Simple Neural Network)
# We use this because our data is just numbers (arrays), not images.
ppo_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train for 30,000 steps (enough for this simple problem)
ppo_model.learn(total_timesteps=30000)

# Save the brain
ppo_model.save(f"{models_dir}/ppo_tutor")
print("PPO Agent Saved!")

# --- TRAINING DQN AGENT ---
print("--- Training DQN Agent ---")
dqn_model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Train for 30,000 steps
dqn_model.learn(total_timesteps=30000)

# Save the brain
dqn_model.save(f"{models_dir}/dqn_tutor")
print("DQN Agent Saved!")

print("All training complete.")
