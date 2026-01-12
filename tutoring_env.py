import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TutoringEnv(gym.Env):
    """
    Custom Environment for an Intelligent Tutoring System.
    Simulates a student learning a topic.

    Action Space:
        0: Give Easy Question
        1: Give Medium Question
        2: Give Hard Question

    Observation Space (State Features):
        - Last 5 accuracies (0 or 1)
        - Mean response time of last 5 attempts
        - Current estimated difficulty level
    """

    def __init__(self):
        super(TutoringEnv, self).__init__()

        # Actions: 0=Easy, 1=Medium, 2=Hard
        self.action_space = spaces.Discrete(3)

        # Observations: [Acc_1, Acc_2, Acc_3, Acc_4, Acc_5, Mean_RT, Student_Mastery]
        # We allow the agent to see the "Student Mastery" to simulate a knowledge tracing model
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(7,), dtype=np.float32)

        # Simulation parameters
        self.student_mastery = 0.0  # Hidden state: 0.0 to 1.0 (Skill level)
        self.student_engagement = 1.0  # Hidden state: 0.0 to 1.0 (Interest)
        self.max_steps = 50
        self.current_step = 0

        # History buffers
        self.accuracy_history = [0] * 5
        self.rt_history = [0.0] * 5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomize initial student skill slightly
        self.student_mastery = np.random.uniform(0.1, 0.4)
        self.student_engagement = 1.0
        self.current_step = 0
        self.accuracy_history = [0] * 5
        self.rt_history = [0.0] * 5

        return self._get_obs(), {}

    def _get_obs(self):
        # Create state vector
        obs = np.array(self.accuracy_history +
                       [np.mean(self.rt_history), self.student_mastery], dtype=np.float32)
        return obs

    def step(self, action):
        difficulty = (action + 1) * 0.33  # Map 0,1,2 to 0.33, 0.66, 0.99

        # --- Simulating Student Response ---
        # Probability of success depends on (Skill - Difficulty)
        # If Skill > Difficulty, high prob. If Skill < Difficulty, low prob.
        success_prob = 1.0 / \
            (1.0 + np.exp(-10 * (self.student_mastery - difficulty + 0.1)))
        is_correct = 1 if np.random.random() < success_prob else 0

        # Simulate Response Time (RT): Harder questions take longer, but high skill reduces time
        base_rt = 5.0 + (difficulty * 10.0)
        actual_rt = base_rt - (self.student_mastery *
                               5.0) + np.random.normal(0, 1)
        actual_rt = max(1.0, actual_rt)  # Min 1 second

        # --- Update Student Internal State (Hidden Dynamics) ---
        learning_gain = 0.0

        # Mastery Gain: Highest when difficulty is slightly above current mastery (Zone of Proximal Development)
        if is_correct:
            # More gain for harder questions
            learning_gain = 0.02 * (1 + difficulty)
        else:
            # Small gain even from failure (learning from mistakes)
            learning_gain = 0.005

        self.student_mastery = min(1.0, self.student_mastery + learning_gain)

        # Engagement Update:
        # Drops if too easy (Boredom) or too hard (Frustration)
        gap = abs(self.student_mastery - difficulty)
        if gap > 0.4:
            self.student_engagement -= 0.05  # Engagement drop
        else:
            self.student_engagement = min(
                1.0, self.student_engagement + 0.02)  # Engagement boost

        # --- Update History for Observation ---
        self.accuracy_history.pop(0)
        self.accuracy_history.append(is_correct)
        self.rt_history.pop(0)
        self.rt_history.append(actual_rt)

        # --- Calculate Reward ---
        # Reward = Balance of Mastery Gain + Engagement Maintenance
        reward = (learning_gain * 100) + (self.student_engagement * 10)

        if is_correct:
            reward += 1  # Small bonus for accuracy

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}
