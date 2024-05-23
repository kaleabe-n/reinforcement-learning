import gymnasium as gym
import numpy as np

# Create the Frozen Lake environment
env = gym.make("FrozenLake-v1")

# Initialize the Q-table and other variables
Q = np.zeros((env.observation_space.n, env.action_space.n))
visits = np.zeros((env.observation_space.n, env.action_space.n))
num_episodes = 10000
gamma = 0.95

# UCB algorithm
for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()[0]
    done = False

    while not done:
        # Calculate the UCB value for each action
        ucb_values = [
            Q[state, a]
            + np.sqrt(
                2 * np.log(np.sum(visits[state, :]) + 1) / (visits[state, a] + 1e-5)
            )
            for a in range(env.action_space.n)
        ]

        # Choose the action with the highest UCB value
        action = np.argmax(ucb_values)

        # Take the action and observe the next state and reward
        next_state, reward, done, truncated, info = env.step(action)

        # Update the Q-table and visit counts
        Q[state, action] = Q[state, action] + (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        ) / (visits[state, action] + 1)
        visits[state, action] += 1

        # Move to the next state
        state = next_state

print("Final Q-table:")
print(Q)
