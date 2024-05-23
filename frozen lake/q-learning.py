import gymnasium as gym
import numpy as np

# Create the Frozen Lake environment
env = gym.make("FrozenLake-v1")

# Initialize the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
num_episodes = 10000
gamma = 0.95
alpha = 0.7
epsilon = 0.5

# Q-Learning algorithm
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()[0]
    done = False
    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit

        # Take the action and observe the next state and reward
        next_state, reward, done, truncated, info = env.step(action)

        # Update the Q-table
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        # Move to the next state
        state = next_state

print("Q-table:")
print(Q)
