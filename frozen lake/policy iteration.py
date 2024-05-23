import gymnasium as gym
import numpy as np
from tqdm import tqdm


def policy_iteration(env, gamma, max_iteration, tolerance):

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function
    V = np.array([0] * num_states, dtype=float)
    P = np.array([0] * num_states, dtype=float)

    def helper(state, depth):
        curr_value = float("-inf")
        curr_policy = 0
        # iterate through each action
        for a in range(num_actions):
            for transition in env.get_wrapper_attr("P")[state][a]:
                p, s_prime, r, _ = transition
                # reward for reaching goal as s'
                if s_prime == 15:
                    prime_value = 10 * p
                elif env.unwrapped.desc[s_prime // 4][s_prime % 4] == (b"H"):
                    prime_value = -10 * p
                else:
                    # avoid unecessary iterations for performance
                    if depth < 9:
                        # make recursive call
                        prime_value = p * (gamma * helper(s_prime, depth + 1) - 1)
                    else:
                        prime_value = float("-inf")
                if prime_value > curr_value:
                    curr_value = prime_value
                    curr_policy = a
        # update the value if the new value is better
        if curr_value > V[state]:
            V[state] = curr_value
            P[state] = curr_policy
        return V[state]

    for _ in range(max_iteration):
        V_prev = V.copy()
        depth = 0
        helper(0, depth)
        if np.max(np.abs(V - V_prev)) < tolerance:
            break
    return P


env: gym.Env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="rgb_array",
    desc=["SFFF", "FHFH", "FFFH", "HFFG"],
)

temp = policy_iteration(env, 0.9, 1, 0.001)
print(temp)
temp = temp.reshape((4, 4))
d = {3: "u", 1: "d", 2: "r", 0: "l"}
for row in temp:
    print([d[i] for i in row])
