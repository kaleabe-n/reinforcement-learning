import gymnasium as gym
import numpy as np

# Create the Frozen Lake environment
env = gym.make("FrozenLake-v1")

# Initialize the Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
num_episodes = 10000
gamma = 0.95
alpha = 0.1
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995

# Q-Learning algorithm
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()[0]

    done = False
    while not done:
        # Epsilon-greedy action selection with decay
        epsilon = epsilon_start * (epsilon_decay**episode)
        epsilon = max(epsilon, epsilon_end)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit

        # Take the action and observe the next state and reward
        next_state, reward, done, _, __ = env.step(action)

        # Update the Q-table
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        # Move to the next state
        state = next_state

print("Q-table:")
print(Q)
