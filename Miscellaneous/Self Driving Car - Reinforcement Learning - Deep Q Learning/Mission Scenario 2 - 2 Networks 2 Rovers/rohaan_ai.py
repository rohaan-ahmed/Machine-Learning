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
Learning_Rate2 = 0.0001
Capacity = 100000

# Class defining Architecture of the Car Neural Network

class Car_Network(nn.Module):
    
    best_score = 0
    best_network_config = 0
    
    def __init__(self, input_size, nb_action):
        super(Car_Network, self).__init__()
        self.input_size = input_size    # Size of input neuron layer
        self.nb_action = nb_action      # Size of output neuron layer
        self.fc1 = nn.Linear(in_features = input_size, out_features = Hidden_Layer_Size)   # Input/Hidden Layer
        self.fc2 = nn.Linear(in_features = Hidden_Layer_Size, out_features = nb_action)    # Output Layer
        
    def forward(self, state):           # State is the current state vector: 3 signals, Orientation, and Negative Orientation
        x = F.relu(self.fc1(state))     # Activating Hidden layer neurons
        q_values = self.fc2(x)          # Activating Output layer neurons
        return q_values

# Class defining Architecture of the Car Neural Network

class Cluser_Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Cluser_Network, self).__init__()
        self.input_size = input_size    # Size of input neuron layer
        self.nb_action = nb_action      # Size of output neuron layer
        
        self.q_network = nn.Sequential(
                            nn.Linear(in_features = input_size, out_features = int(0.25* Hidden_Layer_Size)),
                            nn.Linear(in_features = int(0.25* Hidden_Layer_Size), out_features = int(0.5* Hidden_Layer_Size)),
                            nn.Linear(in_features = int(0.5* Hidden_Layer_Size), out_features = int(0.25* Hidden_Layer_Size)),
                            nn.Linear(in_features = int(0.25 * Hidden_Layer_Size), out_features = nb_action)
                            )
        
        self.best_network_config = self.q_network.state_dict()
        
    def forward(self, state):           # State is the current state vector: 3 signals, Orientation, and Negative Orientation
        return self.q_network(state)
    
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
        samples = random.sample(self.memory, batch_size) # Randomly sample memory for batch_size elements
        # samples_copy = samples
            
        # print('---samples copy orig---')
        # print(list(samples_copy))
        
        samples = zip(*samples)
        samples = map(lambda x: Variable(torch.cat(x, 0)), samples)
        
        # print('---samples copy---')
        # print(samples_copy)
        # samples_copy = zip(*samples_copy)      
        # samples_copy = map(lambda x: Variable(torch.cat(x, 0)), samples_copy)
        # print('---samples copy list---')
        # print(list(samples_copy))
        
        return samples
            
# Deep Q Learning Class
class DQN_car():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma                                                      # Discount rate
        self.reward_window = []                                                 # Sliding window of rewards
        self.model = Car_Network(input_size = input_size, nb_action = nb_action)    # Declaring new Neural Car_Network
        self.memory = ReplayMemory(capacity = Capacity)                           # Declaring new Replay Memory object
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = Learning_Rate) # Declaring Optimizer to use for Learning
        self.last_state = torch.Tensor(input_size).unsqueeze(0)                  # Declaring and restructing last_state
        self.last_action = 0                                                    # Declaring last_action
        self.last_reward = 0.0                                                  # Declaring last_reward
        
        
    def select_action(self, state): # State is the current state vector: 3 signals, Orientation, and Negative Orientation
        probs = F.softmax(self.model(Variable(state, volatile = True))*Temperature)       # Probability distribution of actions with Temperature
        action = probs.multinomial(1)       # Take random draw from the probability distribution
        # print('action')
        # print(action)
        # print('action.data[0, 0]')
        # print(action.data[0, 0])
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # Learning function 
        outputs = self.model(batch_state)
        # print('batch_action car')
        # print(batch_action)
        # print('outputs car')
        # print(outputs) 
        # print('batch_state car')
        # print(batch_state) 
        # print('batch_next_state car')
        # print(batch_next_state) 
        # print('batch_reward car')
        # print(batch_reward)
        outputs = outputs.gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # print('outputs car gathered')
        # print(outputs)
        next_outputs = self.model(batch_next_state).detach().max(1)[0] 
        # print('next_outputs car')
        # print(next_outputs)
        target = self.gamma * next_outputs + batch_reward
        td_loss =  F.smooth_l1_loss(outputs, target)    # Compute TD Loss
        self.optimizer.zero_grad()              # Initialize the Optimizer
        td_loss.backward(retain_graph = True)   # Perform back propagation
        self.optimizer.step()                   # Update weights
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)   # New state is current signals and orientation
        # Save new state to memory
        # print('self.last_action car')
        # # print(self.last_action)
        # print(torch.LongTensor([int(self.last_action)]))
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        temp = self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])
        action = self.select_action(new_state) # select next action to take
        if len(self.memory.memory) > Memory_Samples:       # Begin learning after 100 states have been transitioned
            temp = self.memory.sample(Memory_Samples)
            batch_state, batch_next_state, batch_action, batch_reward = temp
            # print('batch_action car pre')
            # print(batch_action)
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
        
    
    def save_load_best_model(self, score):    # Returns average reward of the reward window
        print('Car_Network.best_score {0}'.format(Car_Network.best_score))
        print('Current score {0}'.format(score))
        if Car_Network.best_score == 0 and score == 0:
            for layer in self.model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            print('Rover Model Reset')
        elif score > Car_Network.best_score:
            Car_Network.best_network_config = self.model.state_dict()
            torch.save(Car_Network.best_network_config, 'best_rover_ai_network')
            Car_Network.best_score = score
            print('Best Rover Model Saved')
        else:
            self.model.load_state_dict(Car_Network.best_network_config)
            print('Best Rover Model Loaded')
        return
    
    def plot(self):
        pass

class DQN_car_cluster():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma                                                      # Discount rate
        self.reward_window = []                                                 # Sliding window of rewards
        self.model = Cluser_Network(input_size = input_size, nb_action = nb_action)    # Declaring new Neural Car_Network
        self.memory = ReplayMemory(capacity = Capacity)                           # Declaring new Replay Memory object
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = Learning_Rate2) # Declaring Optimizer to use for Learning
        self.last_state = torch.Tensor(input_size).unsqueeze(0)                  # Declaring and restructing last_state
        self.last_action = torch.Tensor(nb_action)                  # Declaring last_action
        self.last_reward = 0.0                                                  # Declaring last_reward
        self.best_score = 0
        self.best_network_config = self.model.state_dict()
    
    def select_action(self, state): # State is the current state vector: 3 signals, Orientation, and Negative Orientation
        raw_values = self.model(Variable(state, volatile = True))
        actions = (raw_values - torch.min(raw_values)) / (torch.max(raw_values) - torch.min(raw_values)) * 50 - 20 # Normalize actions to between 0 and 100
        actions = actions.data[0]
        # actions = raw_values.data[0]
        # print('actions cluster')
        # print(actions)
        return actions
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action): # Learning function
        outputs = self.model(batch_state)
        # print('batch_action cluster')
        # print(batch_action)
        # print('outputs cluster')
        # print(outputs) 
        # print('batch_state cluster')
        # print(batch_state) 
        # print('batch_next_state cluster')
        # print(batch_next_state) 
        # print('batch_reward cluster')
        # print(batch_reward)
        
        # print(outputs) 
        # outputs = outputs.gather(1, batch_action.unsqueeze(1)).squeeze(1)

        next_outputs = self.model(batch_next_state)
        
        # print('next_outputs cluster')
        # print(next_outputs) 
        
        # target = self.gamma * next_outputs + batch_reward
        target = []
        for i in range (0,len(batch_reward)):
            target.append(self.gamma * next_outputs[i,:] + batch_reward[i])
        # print('target cluster')
        # print(target[0])
        td_loss =  F.smooth_l1_loss(outputs, target[0])    # Compute TD Loss
        self.optimizer.zero_grad()              # Initialize the Optimizer
        td_loss.backward(retain_graph = True)   # Perform back propagation
        self.optimizer.step()                   # Update weights
        
    def update(self, reward, map_state):
        new_state = torch.Tensor(map_state).float().unsqueeze(0)   # New state is current signals and orientation
        self.last_action = torch.unsqueeze(self.last_action,0)
        # Save new state to memory
        # print('self.last_action cluster')
        # print(self.last_action)
        self.memory.push((self.last_state, new_state, self.last_action, torch.Tensor([self.last_reward])))
        action = self.select_action(new_state) # select next action to take
        if len(self.memory.memory) > Memory_Samples:       # Begin learning after 100 states have been transitioned
            temp = self.memory.sample(Memory_Samples)
            batch_state, batch_next_state, batch_action, batch_reward = temp
            # print('batch_action cluster pre')
            # print(batch_action)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action               # Update all current state variables
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > Reward_Window_Size:      # Keep only 1000 rewards in the sliding reward window
            del self.reward_window[0]
        return action
                
    def save_load_best_model(self, score):    # Returns average reward of the reward window
        if score > self.best_score:
            self.best_network_config = self.model.state_dict()
            torch.save(self.best_network_config, 'best_cluster_ai_network')
            self.best_score = score
            print('Best Model Saved')
        else:
            self.model.load_state_dict(self.best_network_config)
            print('Best Model Loaded')
        return
    

    









    
            
            
            
        