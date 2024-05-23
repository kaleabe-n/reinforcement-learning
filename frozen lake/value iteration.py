import gymnasium as gym
import numpy as np


def value_iteration(env, gamma, max_iteration, tolerance):

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the value function
    V = np.array([0] * num_states,dtype=float)

    def helper(state, depth):
        curr_value = float("-inf")
        #iterate through all the actions
        for a in range(num_actions):
            #iterate through all the trantions based on the action for example if slippery environement
            for transition in env.get_wrapper_attr("P")[state][a]:
                p, s_prime, r, _ = transition
                if s_prime == 15:
                    #reward for reaching goal if s' is the goal state
                    prime_value = 10 * p
                elif env.unwrapped.desc[s_prime // 4][s_prime % 4] == (b"H"):
                    prime_value = -10 * p
                else:
                    if depth < 10:
                        #make recursive call
                        prime_value = p * (gamma * helper(s_prime, depth + 1) - 1)
                    else:
                        prime_value = float("-inf")
                curr_value = max(curr_value, prime_value)
        #update the value if the current value is greater than the previous value
        V[state] = max(V[state], curr_value)
        return V[state]

    for _ in range(max_iteration):
        V_prev = V.copy()
        depth = 0
        helper(0, depth)
        if np.max(np.abs(V - V_prev)) < tolerance:
            break
    return V


env: gym.Env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="rgb_array",
    desc=["SFFF", 
          "FHFH", 
          "FFFH", 
          "HFFG"],
)

print("traiing...")
print(value_iteration(env, 0.9, 1, 0.001).reshape((4, 4)))
