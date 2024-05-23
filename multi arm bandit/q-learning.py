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


# Q-learning algorithm
def q_learning(bandit, num_iterations, alpha, gamma):
    num_arms = bandit.num_arms

    for _ in range(num_iterations):
        # Choose an arm to pull
        arm = np.argmax(bandit.q_values)

        # Pull the arm and observe the reward
        reward = bandit.pull(arm)

        # Update the Q-value for the chosen arm
        bandit.q_values[arm] = bandit.q_values[arm] + alpha * (
            reward - bandit.q_values[arm]
        )

        # Explore other arms with a small probability
        if np.random.rand() < 0.1:
            arm = np.random.randint(0, num_arms)
            reward = bandit.pull(arm)
            bandit.q_values[arm] = bandit.q_values[arm] + alpha * (
                reward - bandit.q_values[arm]
            )

    return bandit.q_values


# Example usage
num_arms = 5
rewards = [5, 10, 2, 8, 6]
bandit = MultiArmedBandit(num_arms, rewards)
final_q_values = q_learning(bandit, 10000, 0.1, 0.9)
print("Final Q-values:")
print(final_q_values)
