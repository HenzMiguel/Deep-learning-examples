import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from collections import deque
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math
import numpy as np



class ReplayBuffer():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        experience = (state, action, next_state, reward, done)
        self.memory.append(experience)

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, dones = (zip(*batch))

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype = torch.long).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)

        return states, actions, next_states, rewards, dones

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_layers, buffer_size):
        super().__init__()
        
        self.fc1 = nn.Linear(num_inputs, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc3 = nn.Linear(hidden_layers, num_actions)

        self.memory = ReplayBuffer(buffer_size)
        self.epsilon = 0
        self.t_step = 0


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(F.relu(x))
        x = self.fc3(F.relu(x))
        return x

    def select_action(self, q_values, start, end, decay):
        if self.epsilon == 0:
            self.epsilon = start

        self.epsilon =  max(end, self.epsilon * decay)

        if random.random() < self.epsilon:
            return random.choice(range(len(q_values)))
        
        return q_values.argmax(dim=0).item()
    
    def calculate_loss(self, target_network: nn.Module, states, actions, next_states, rewards, dones):

        # Estimativa do modelo
        q_values = self.forward(states.to(device)).gather(1, actions.to(device)).squeeze(1)


        with torch.no_grad():

            next_state_q_value = target_network(next_states.to(device)).amax(1)
            
            gamma = 0.99

            # TD targets
            targets = rewards.to(device) + gamma * next_state_q_value * (1 - dones.to(device))

        # diferença entre a estimativa e valor real
        loss = nn.MSELoss()(q_values, targets)

        return loss

    def update_network(self, online_network: nn.Module, tau: float):
        online_dic = online_network.state_dict()
        target_dic = self.state_dict()

        for key in online_dic:
            target_dic[key] = (online_dic[key] * tau + target_dic[key] * (1 - tau))

        self.load_state_dict(target_dic)
    

    def describe_episode(self, episode, reward, episode_reward, step):
        print(f"episode: {episode}| duration: {step}| returns: {episode_reward}| {"crashed" if reward <= 0 else "landed"} |")

# Initialise the environment
env = gym.make("LunarLander-v3")

if isinstance(env.action_space, Discrete):
    actions = env.action_space.n
if isinstance(env.observation_space, Box):
    features = env.observation_space.shape[0]


device = "cuda" if torch.cuda.is_available() else "cpu"
hidden_layers = 64
buffer_size = int(1e5)


dqn = DQN(features, actions, hidden_layers, buffer_size=buffer_size).to(device)
target_dqn = DQN(features, actions, hidden_layers, buffer_size=buffer_size).to(device)

optimizer = optim.Adam(dqn.parameters(), lr=5e-4)
batch_size = 100
gamma = 0.99

successful_conclusions = 0
successful = False
avg_ep_reward = 0

for episode in range(2000):
    
    state, info = env.reset()
    done = False
    steps = 0
    episode_reward = 0

    while not done:
        steps += 1

        dqn.eval()
        
        with torch.no_grad():
            q_values = dqn(torch.tensor(state).to(device))
        
        dqn.train()

        action = dqn.select_action(q_values, start=1.0, end=0.01, decay=0.995)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        dqn.memory.push(state, action, next_state, reward, done)

        dqn.t_step = (dqn.t_step + 1) % 4 
        if dqn.t_step == 0 and len(dqn.memory) >= batch_size:
            states, actions, next_states, rewards, dones = dqn.memory.sample(batch_size)
            loss = dqn.calculate_loss(target_dqn, states, actions, next_states, rewards, dones)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            target_dqn.update_network(dqn, tau=0.001)

        episode_reward += float(reward)
        state = next_state

    if episode_reward >= 200:
        successful_conclusions+=1
    else:
        successful_conclusions = 0
    
    if successful_conclusions >= 5:
        torch.save(dqn.state_dict(), "dqn_lunar_lander.pt")
        successful = True

    avg_ep_reward += episode_reward
    if episode % 10 == 0:
        avg_ep_reward /= 10
        print(f"Episode: {episode}, reward: {avg_ep_reward}")
        avg_ep_reward = 0
if not successful:
    torch.save(dqn.state_dict(), "dqn_lunar_lander.pt")
env.close()