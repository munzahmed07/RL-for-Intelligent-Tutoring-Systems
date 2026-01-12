import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from tutoring_env import TutoringEnv
import os

# --- Define the Heuristic (Rule-based) Agent ---
# Simple logic: If correct, increase difficulty. If wrong, decrease.


def heuristic_agent(last_correct, current_action):
    if last_correct == 1:
        return min(2, current_action + 1)  # Go harder (max 2)
    else:
        return max(0, current_action - 1)  # Go easier (min 0)

# --- Evaluation Function ---


def evaluate_agent(agent_type, model=None, episodes=50):
    env = TutoringEnv()
    all_masteries = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        mastery_trajectory = []
        last_action = 1  # Start Medium

        while not done:
            if agent_type == "PPO":
                action, _ = model.predict(obs)
            elif agent_type == "DQN":
                action, _ = model.predict(obs)
            elif agent_type == "Heuristic":
                # Get last accuracy from observation (4th index in history)
                last_acc = obs[4]
                action = heuristic_agent(last_acc, last_action)
                last_action = action
            elif agent_type == "Static":
                action = 1  # Always Medium

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Record the student's mastery level at this step
            mastery_trajectory.append(env.student_mastery)

        all_masteries.append(mastery_trajectory)

    # Calculate average mastery progression across all episodes
    avg_mastery = np.mean(all_masteries, axis=0)
    return avg_mastery


# --- Main Execution ---
print("Loading Models...")
# Check if models exist before loading
if os.path.exists("models/ppo_tutor.zip") and os.path.exists("models/dqn_tutor.zip"):
    ppo_model = PPO.load("models/ppo_tutor")
    dqn_model = DQN.load("models/dqn_tutor")

    print("Running Evaluations (this takes ~10 seconds)...")
    ppo_res = evaluate_agent("PPO", model=ppo_model)
    dqn_res = evaluate_agent("DQN", model=dqn_model)
    heur_res = evaluate_agent("Heuristic")
    static_res = evaluate_agent("Static")

    # --- Plotting Results ---
    print("Generating Graph...")
    plt.figure(figsize=(10, 6))
    plt.plot(ppo_res, label='PPO Agent (RL)', linewidth=2, color='blue')
    plt.plot(dqn_res, label='DQN Agent (RL)', linewidth=2, color='green')
    plt.plot(heur_res, label='Heuristic (Rules)',
             linestyle='--', color='orange')
    plt.plot(static_res, label='Static (Fixed)', linestyle=':', color='gray')

    plt.title('Student Mastery Progression: RL vs Traditional Methods')
    plt.xlabel('Questions Answered (Time)')
    plt.ylabel('Student Mastery Level (0-1)')
    plt.legend()
    plt.grid(True)

    # Save the graph
    plt.savefig('evaluation_results.png')
    print("Success! Graph saved as 'evaluation_results.png'")
    # plt.show() # Uncomment if you want to see the window popup
else:
    print("Error: Model files not found. Did the training finish?")
