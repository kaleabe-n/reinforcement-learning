import numpy as np


# Define the multi-armed bandit problem
class MultiArmedBandit:
    def __init__(self, num_arms, rewards):
        self.num_arms = num_arms
        self.rewards = rewards
        self.q_values = np.zeros(num_arms)
        self.pulls = np.zeros(num_arms)

    def pull(self, arm):
        reward = np.random.normal(self.rewards[arm], 1.0)
        self.pulls[arm] += 1
        return reward


# Epsilon-greedy algorithm
def epsilon_greedy(bandit, num_iterations, epsilon):
    num_arms = bandit.num_arms

    for _ in range(num_iterations):
        # Explore with probability epsilon
        if np.random.rand() < epsilon:
            arm = np.random.randint(0, num_arms)
        # Exploit the arm with the highest Q-value
        else:
            arm = np.argmax(bandit.q_values)

        # Pull the arm and observe the reward
        reward = bandit.pull(arm)

        # Update the Q-value for the chosen arm
        bandit.q_values[arm] = bandit.q_values[arm] + (
            reward - bandit.q_values[arm]
        ) / (bandit.pulls[arm] + 1)

    return bandit.q_values


# Example usage
num_arms = 5
rewards = [5, 10, 2, 8, 6]
bandit = MultiArmedBandit(num_arms, rewards)
final_q_values = epsilon_greedy(bandit, 10000, 0.1)
print("Final Q-values:")
print(final_q_values)
