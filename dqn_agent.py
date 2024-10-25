import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size=9, hidden_size=256, output_size=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # Added extra hidden layer
        self.fc4 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # Activation for the new layer
        x = self.fc4(x)
        return x

class ReplayMemory:
    def __init__(self, capacity=200000):  # Adjusted capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=100000, replay_memory_capacity=200000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(capacity=replay_memory_capacity)
        self.gamma = gamma
        
        self.steps_done = 0
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def select_action(self, state, available_actions):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < epsilon:
            return random.choice(available_actions)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                q_values = q_values.cpu().numpy()[0]
                # Mask invalid actions using NumPy
                q_values_invalid = -np.inf * np.ones(len(q_values))
                for action in available_actions:
                    q_values_invalid[action] = q_values[action]
                return int(np.argmax(q_values_invalid))
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
    
    def optimize_model(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        transitions = self.memory.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        batch_state = torch.FloatTensor(batch_state).to(self.device)
        batch_action = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)
        
        # Compute Q(s_t, a) using policy network
        current_q = self.policy_net(batch_state).gather(1, batch_action)
        
        # Compute argmax_a' Q(s_{t+1}, a') using policy network
        next_actions = self.policy_net(batch_next_state).argmax(1, keepdim=True)
        
        # Compute V(s_{t+1}) using target network
        next_q = self.target_net(batch_next_state).gather(1, next_actions).detach()
        
        expected_q = batch_reward + (self.gamma * next_q * (1 - batch_done))
        
        # Compute loss
        loss = F.mse_loss(current_q, expected_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())