import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# Q Learning Parameters
# -----------------------------
alpha = 0.1
gamma = 0.9
epsilon = 0.1

actions = [0,1,2,3]  # 4 lanes
q_table = {}

# -----------------------------
# Environment
# -----------------------------
class TrafficEnv:
    def __init__(self):
        self.state = np.random.randint(0,10,4)

    def reset(self):
        self.state = np.random.randint(0,10,4)
        return tuple(self.state)

    def step(self, action):
        reward = -sum(self.state)

        # Cars pass in selected lane
        self.state[action] = max(0, self.state[action] - 5)

        # New cars arrive
        self.state += np.random.randint(0,3,4)

        return tuple(self.state), reward

# -----------------------------
# Choose Action
# -----------------------------
def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.choice(actions)
    else:
        if state not in q_table:
            q_table[state] = [0,0,0,0]
        return np.argmax(q_table[state])

# -----------------------------
# Train AI
# -----------------------------
def train(episodes=500):
    env = TrafficEnv()

    for _ in range(episodes):
        state = env.reset()

        for _ in range(50):
            if state not in q_table:
                q_table[state] = [0,0,0,0]

            action = choose_action(state)
            next_state, reward = env.step(action)

            if next_state not in q_table:
                q_table[next_state] = [0,0,0,0]

            q_table[state][action] += alpha * (
                reward + gamma * max(q_table[next_state]) - q_table[state][action]
            )

            state = next_state

# -----------------------------
# Fixed Timer Simulation
# -----------------------------
def fixed_timer():
    env = TrafficEnv()
    total_wait = 0
    state = env.reset()

    for i in range(50):
        action = i % 4
        state, reward = env.step(action)
        total_wait += -reward

    return total_wait

# -----------------------------
# AI Simulation
# -----------------------------
def ai_simulation():
    env = TrafficEnv()
    total_wait = 0
    state = env.reset()

    for _ in range(50):
        if state not in q_table:
            q_table[state] = [0,0,0,0]

        action = np.argmax(q_table[state])
        state, reward = env.step(action)
        total_wait += -reward

    return total_wait

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚦 Smart Traffic Signal using Reinforcement Learning")

st.write("AI learns optimal signal switching to reduce traffic waiting time.")

if st.button("Train AI"):
    train(episodes=800)
    st.success("AI Training Completed!")

if st.button("Compare Fixed Timer vs AI"):
    fixed = fixed_timer()
    ai = ai_simulation()

    st.write("### Results")
    st.write(f"Fixed Timer Total Waiting Time: {fixed}")
    st.write(f"AI Total Waiting Time: {ai}")

    fig, ax = plt.subplots()
    ax.bar(["Fixed Timer", "AI"], [fixed, ai])
    ax.set_ylabel("Total Waiting Time")
    st.pyplot(fig)
