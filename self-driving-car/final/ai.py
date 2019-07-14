# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    
    def forward(self, state):
        # print("inside model & state is..")
        # print(state)
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        # print("q_values")
        # print(q_values)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = list(zip(*random.sample(self.memory, batch_size)))
        # print("after zip... ")
        # print("lenght")
        # print(len(samples))
        # print(samples)
        samples = list(map(lambda x: Variable(torch.cat(x, 0)), samples))
        # print(samples)
        return samples

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        # print("select action...")
        # print("State")
        # print(state)
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        # print("probs")
        # print(probs)
        action = probs.multinomial(num_samples=1)
        # print("actiondata[0,0]")
        # print(action.data[0,0])
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # print("3.in learn...run the model...")
        outputs = self.model(batch_state)
        # print("outputs")
        # print(outputs)
        # print("batch_action")
        # print(batch_action)
        # print("batch_action.unsqueeze(1)")
        # print(batch_action.unsqueeze(1))
        outputs = outputs.gather(1, batch_action.unsqueeze(1))
        # print("gather")
        # print(outputs)
        # print("squeeze")
        outputs = outputs.squeeze(1)
        # print("4.outputs...")
        # print(outputs)
        next_outputs = self.model(batch_next_state)
        # print("5.next outputs...")
        # print(next_outputs)
        next_outputs: torch.Tensor = next_outputs.detach().max(1)[0]
        # print("6..detach().max(1)[0]")
        # print(next_outputs)
        target = self.gamma*next_outputs + batch_reward
        # print("target")
        # print(target)
        td_loss = F.smooth_l1_loss(outputs, target)

        print(td_loss)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        # print("\n\n\n\n")
        # print("new_signal")
        # print(new_signal)
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # print("new_state")
        # print(new_state)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            # print("1.sampling...")
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            # print("2.learning...")
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
