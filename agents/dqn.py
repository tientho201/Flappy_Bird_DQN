import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
import random
from training.replay_buffer import ReplayBuffer
class DQN(nn.Module):
    def __init__(self, state_dim , action_dim, hidden = 128):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden, hidden),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr = 1e-3, gamma = 0.99, epsilon = 1.0, eps_min = 0.01, eps_decay = 0.995 , device = None):
        self.gamma = gamma # discount factor (Hệ số chiết khấu cho phần thưởng tương lai )
        self.epsilon = epsilon # epsilon-greedy policy (Tỷ lệ khám phá ban đầu)
        self.eps_min = eps_min # minimum epsilon (Tỷ lệ khám phá tối thiểu)
        self.eps_decay = eps_decay # epsilon decay (Tốc độ giảm epsilon)
        self.q_net = DQN(state_dim, action_dim) # Q network
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()
        self.action_dim = action_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0 
    
    # Choose action with epsilon-greedy policy
    def act(self, state, epsilon = 0.0 ):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(q_values.argmax().cpu().numpy())


    
    def update(self, buffer, batch_size = 64):
        if len(buffer) < batch_size:
            return 0.0
        # Sample a batch from the buffer
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        
        q_values = self.q_net(states).gather(1, actions).squeeze(1)
        
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            # Calculate targets (Công thức Bellman)
            targets = rewards + self.gamma * next_q * (1 - dones.float()) # (1 - dones) is for terminal state
        
        loss = self.loss_fn(q_values, targets) # Calculate loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step += 1
        return loss.item()
        

    def update_target(self):
        # Update target network
        self.target_net.load_state_dict(self.q_net.state_dict()) # Load Q-network parameters to target network

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)


    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target()
