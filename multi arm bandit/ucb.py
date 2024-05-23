import numpy as np


# Define the multi-armed bandit problem
class MultiArmedBandit:
    def __init__(self, num_arms, rewards):
        self.num_arms = num_arms
        self.rewards = rewards
        self.pulls = np.zeros(num_arms)
        self.rewards_sum = np.zeros(num_arms)

    def pull(self, arm):
        reward = np.random.normal(self.rewards[arm], 1.0)
        self.pulls[arm] += 1
        self.rewards_sum[arm] += reward
        return reward


# UCB algorithm
def ucb(bandit, num_steps):
    num_arms = bandit.num_arms
    total_reward = 0

    for step in range(num_steps):
        # Calculate the UCB value for each arm
        ucb_values = [
            bandit.rewards_sum[arm] / (bandit.pulls[arm] + 1e-5)
            + np.sqrt(2 * np.log(step + 1) / (bandit.pulls[arm] + 1e-5))
            for arm in range(num_arms)
        ]

        # Choose the arm with the highest UCB value
        chosen_arm = np.argmax(ucb_values)

        # Pull the chosen arm and update the reward
        reward = bandit.pull(chosen_arm)
        total_reward += reward

    return total_reward


# Example usage
num_arms = 5
rewards = [5, 10, 2, 8, 6]
bandit = MultiArmedBandit(num_arms, rewards)
total_reward = ucb(bandit, 1000)
print(f"Total reward: {total_reward}")
