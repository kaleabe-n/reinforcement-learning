import numpy as np


# Define the multi-armed bandit problem
class MultiArmedBandit:
    def __init__(self, num_arms, rewards):
        self.num_arms = num_arms
        self.rewards = rewards
        self.values = np.zeros(num_arms)
        self.pulls = np.zeros(num_arms)
        self.policy = np.zeros(num_arms, dtype=int)

    def pull(self, arm):
        reward = np.random.normal(self.rewards[arm], 1.0)
        self.pulls[arm] += 1
        return reward


# Policy iteration algorithm
def policy_iteration(bandit, num_iterations):
    num_arms = bandit.num_arms

    # Initialize the policy randomly
    bandit.policy = np.random.randint(0, num_arms, size=num_arms)

    for _ in range(num_iterations):
        # Policy evaluation
        for arm in range(num_arms):
            bandit.pulls[arm] += 1
            bandit.values[arm] = bandit.values[arm] + (1 / bandit.pulls[arm]) * (
                bandit.pull(bandit.policy[arm]) - bandit.values[arm]
            )

        # Policy improvement
        for arm in range(num_arms):
            best_value = float("-inf")
            best_action = None
            for action in range(num_arms):
                bandit.pulls[action] += 1
                value = bandit.values[action] + (1 / bandit.pulls[action]) * (
                    bandit.pull(action) - bandit.values[action]
                )
                if value > best_value:
                    best_value = value
                    best_action = action
            bandit.policy[arm] = best_action

    return bandit.policy


# Example usage
num_arms = 5
rewards = [5, 10, 2, 8, 6]
bandit = MultiArmedBandit(num_arms, rewards)
final_policy = policy_iteration(bandit, 1000)
print("Final policy:")
print(final_policy)
