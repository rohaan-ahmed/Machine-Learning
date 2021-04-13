# AI for Self Driving Car
# Deep Q Learning

# Importing the libraries

import numpy as np
import random
import os # To save and load model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

Temperature = 7
Hidden_Layer_Size = 100
Memory_Samples = 100
Reward_Window_Size = 100
Learning_Rate = 0.01
Capacity = 100000

# Class defining Architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size    # Size of input neuron layer
        self.nb_action = nb_action      # Size of output neuron layer
        self.fc1 = nn.Linear(in_features = input_size, out_features = Hidden_Layer_Size)   # Input/Hidden Layer
        self.fc2 = nn.Linear(in_features = Hidden_Layer_Size, out_features = nb_action)    # Output Layer
        
    def forward(self, state):           # State is the current state vector: 3 signals, Orientation, and Negative Orientation
        x = F.relu(self.fc1(state))     # Activating Hidden layer neurons
        q_values = self.fc2(x)          # Activating Output layer neurons
        return q_values
    
# Experience Replay Class

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity    # Maximum states saved in memory at a time
        self.memory = []            # Empty list of previous states
        
    def push(self, event):  # Function to save event to memory
        self.memory.append(event)   # Event: Current State, Last State, Last Action, Last Reward
        if len(self.memory) > self.capacity:
            del self.memory[0]      # delete oldest state in memory
            
    def sample(self, batch_size):   # Random Sampling function
        samples = zip(*random.sample(self.memory, batch_size))      # Randomly sample memory for batch_size elements
        return map(lambda x: Variable(torch.cat(x, 0)), samples)    # Restructure samples for pytorch
            
# Deep Q Learning Class
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma                                                      # Discount rate
        self.reward_window = []                                                 # Sliding window of rewards
        self.model = Network(input_size = input_size, nb_action = nb_action)    # Declaring new Neural Network
        self.memory = ReplayMemory(capacity = Capacity)                           # Declaring new Replay Memory object
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = Learning_Rate) # Declaring Optimizer to use for Learning
        self.last_state = torch.Tensor(input_size).unsqueeze(0)                  # Declaring and restructing last_state
        self.last_action = 0                                                    # Declaring last_action
        self.last_reward = 0.0                                                  # Declaring last_reward
    
    def select_action(self, state): # State is the current state vector: 3 signals, Orientation, and Negative Orientation
        probs = F.softmax(self.model(Variable(state, volatile = True))*Temperature)       # Probability distribution of actions with Temperature
        action = probs.multinomial(1)       # Take random draw from the probability distribution
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # Learning function
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0] 
        target = self.gamma * next_outputs + batch_reward
        td_loss =  F.smooth_l1_loss(outputs, target)    # Compute TD Loss
        self.optimizer.zero_grad()              # Initialize the Optimizer
        td_loss.backward(retain_graph = True)   # Perform back propagation
        self.optimizer.step()                   # Update weights
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)   # New state is current signals and orientation
        # Save new state to memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state) # select next action to take
        if len(self.memory.memory) > Memory_Samples:       # Begin learning after 100 states have been transitioned
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(Memory_Samples)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action               # Update all current state variables
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > Reward_Window_Size:      # Keep only 1000 rewards in the sliding reward window
            del self.reward_window[0]
        return action
    
    def score(self):    # Returns average reward of the reward window
        rew = sum(self.reward_window)/(len(self.reward_window) + 1.0)
        #print('Avg Reward ' + str(rew), 'Last Reward: ' + str(self.reward_window[-1]))
        return rew
        
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),},
                    'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> Loading Model...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Load Completed')
        else:
            print('No file named last_brain.pth found')
                
        
    

    









    
            
            
            
        