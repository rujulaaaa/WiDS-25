import math
import random
from collections import deque, namedtuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -------------------------
# Environment
# -------------------------
env = gym.make("CartPole-v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Replay Memory
# -------------------------
Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward"))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# -------------------------
# Q Network
# -------------------------
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# Hyperparameters
# -------------------------
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# -------------------------
# Init networks
# -------------------------
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done = 0

# -------------------------
# Action selection
# -------------------------
def select_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]],
                             device=device, dtype=torch.long)

# -------------------------
# Optimize model
# -------------------------
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device, dtype=torch.bool
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = \
            target_net(non_final_next_states).max(1)[0]

    expected_q_values = reward_batch + GAMMA * next_state_values

    loss = nn.SmoothL1Loss()(state_action_values.squeeze(),
                             expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# -------------------------
# Training Loop
# -------------------------
num_episodes = 600
episode_durations = []

for i_episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)

    for t in range(1, 10000):
        action = select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        done = terminated or truncated

        if done:
            next_state = None
        else:
            next_state = torch.tensor(obs, dtype=torch.float32,
                                      device=device).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state

        optimize_model()

        # Soft update target network
        target_net_state = target_net.state_dict()
        policy_net_state = policy_net.state_dict()
        for key in policy_net_state:
            target_net_state[key] = \
                policy_net_state[key] * TAU + target_net_state[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state)

        if done:
            episode_durations.append(t)
            print(f"Episode {i_episode} finished after {t} steps")
            break

env.close()
print("Training complete")
