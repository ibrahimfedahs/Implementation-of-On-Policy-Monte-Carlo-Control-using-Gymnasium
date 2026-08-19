# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description







## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm



## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Parameters
num_episodes = 10000
gamma = 0.99
alpha = 0.1
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995

# Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))

episode_rewards = []


# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# Monte Carlo Control
for episode in range(num_episodes):

    state, info = env.reset()

    episode_data = []
    done = False

    # Generate complete episode
    while not done:

        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode_data.append((state, action, reward))

        state = next_state
        done = terminated or truncated

    # Calculate returns and update Q-values
    G = 0

    for state, action, reward in reversed(episode_data):

        G = reward + gamma * G

        # Incremental Monte Carlo update
        Q[state, action] = Q[state, action] + alpha * (
            G - Q[state, action]
        )

    # Store episode reward
    episode_rewards.append(sum(x[2] for x in episode_data))

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# Display Q-table
print("\nFinal Q-table:")
print(np.round(Q, 3))


# Estimated State-Value Function
V = np.max(Q, axis=1)

print("\nEstimated State-Value Function:")
print(np.round(V, 3))


# Learned Policy
policy = np.argmax(Q, axis=1)

print("\nLearned Policy:")

action_symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}

for state in range(env.observation_space.n):
    print(f"State {state}: {action_symbols[policy[state]]}")


# Average reward over last 1000 episodes
average_reward = np.mean(episode_rewards[-1000:])

print("\nAverage reward over last 1000 episodes:",
      round(average_reward, 3))


# Learning curve
window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid()
plt.show()

env.close()



```

---

## Output

```text
Final Q-table:



Estimated State-Value Function:







Learned Policy:





Average reward over last 1000 episodes: 
```


---

## Result
```text



```
---

## Inference
```text



```





---

