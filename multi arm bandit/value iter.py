import numpy as np


# Define the multi-armed bandit problem
class MultiArmedBandit:
    def __init__(self, num_arms, rewards):
        self.num_arms = num_arms
        self.rewards = rewards
        self.values = np.zeros(num_arms)
        self.pulls = np.zeros(num_arms)

    def pull(self, arm):
        reward = np.random.normal(self.rewards[arm], 1.0)
        self.pulls[arm] += 1
        return reward


# Value iteration algorithm
def value_iteration(bandit, num_iterations):
    num_arms = bandit.num_arms

    for _ in range(num_iterations):
        # Update the values for each arm
        for arm in range(num_arms):
            bandit.pulls[arm] += 1
            bandit.values[arm] = bandit.values[arm] + (1 / bandit.pulls[arm]) * (
                bandit.pull(arm) - bandit.values[arm]
            )

    return bandit.values


# Example usage
num_arms = 5
rewards = [5, 10, 2, 8, 6]
bandit = MultiArmedBandit(num_arms, rewards)
final_values = value_iteration(bandit, 1000)
print("Final values:")
print(final_values)
