import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame  # để bắt phím ESC

# ---------------------------
# Q-Network
# ---------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ---------------------------
# DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, eps_min=0.01, eps_decay=0.995):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()
        self.action_dim = action_dim

    def act(self, state, exploit=False):
        if (not exploit) and (random.random() < self.epsilon):
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_net(state)
        return q_values.argmax().item()

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


# ---------------------------
# Train + Run loop
# ---------------------------
env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)

episodes = 500
target_update = 10
threshold_reward = 450   # reward ngưỡng để chuyển sang "play mode"
threshold_episode = 0.1  # episode ngưỡng để chuyển sang "play mode"

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state

        agent.update()
        total_reward += reward

        env.render()

    agent.epsilon = max(agent.eps_min, agent.epsilon * agent.eps_decay)

    if ep % target_update == 0:
        agent.update_target()

    print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    # Nếu agent đủ giỏi thì vào chế độ chơi liên tục
    if total_reward >= threshold_reward and ep >= threshold_episode:
        print("Agent đã đạt ngưỡng, vào chế độ chơi liên tục. Nhấn ESC để thoát.")
        pygame.init()
        running = True
        while running:
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.act(state, exploit=True)  # luôn chọn hành động tốt nhất
                state, reward, done, truncated, _ = env.step(action)
                env.render()
                
                if reward < 200:
                    running = False
                    done = True
                    break   
                
                # kiểm tra phím ESC
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        running = False
                        done = True
                        break
        break  # thoát training loop

env.close()
