import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from utils.general import get_logger
from utils.test_env import EnvTest
from q2_schedule import LinearExploration, LinearSchedule
from q3_linear_torch import Linear
import copy

from configs.q4_nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    Model configuration can be found in the Methods section of the above paper.
    """

    def initialize_models(self):
        """Creates the 2 separate networks (Q network and Target network). The input
        to these models will be an img_height * img_width image
        with channels = n_channels * self.config.state_history
        MAKE URE YOU USE THESE VARIABLES FOR INPUT SIZE

        Each network has the following architecture (see th nature paper for more details):
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
            - Conv2d with 32 8x8 filters and stride 4 + ReLU activation
            - Conv2d with 64 4x4 filters and stride 2 + ReLU activation
            - Conv2d with 64 3x3 filters and stride 1 + ReLU activation
            - Flatten
            - Linear with output 512. What is the size of the input?
                you need to calculate this img_height, img_width, and number of filter.
            - Relu
            - Linear with 512 input and num_actions outputs

        1. Set self.q_network to be a model with num_actions as the output size
        2. Set self.target_network to be the same configuration self.q_network but initialized from scratch
        3. What is the input size of the model?

        To simplify, we specify the paddings as:
            ((stride - 1) * img_height - stride + filter_size) // 2
        Make sure you follow use this padding for every layer

        Hints:
            1. Simply setting self.target_network = self.q_network is incorrect.
            2. The following functions might be useful
                - nn.Sequential
                - nn.Conv2d
                - nn.ReLU
                - nn.Flatten
                - nn.Linear
            3. If you use OrderedDict, make sure the keys for the the layers are:
                - "0", "2", "4" for three Conv2d layers
                - "7" for the first Linear layer
                - "9" for the final Linear layer
        """
        state_shape = list(self.env.observation_space.shape)
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.action_space.n
        strides = np.array([4, 2, 1])  # The stride size for every conv2d layer
        filter_sizes = np.array([8, 4, 3])  # The filter size for every conv2d layer
        numb_filters = np.array([32, 64, 64])  # number of filters for every conv2d layer
        ##############################################################
        ################ YOUR CODE HERE - 25-30 lines lines ################
        
        # print('state_shape = {0}'.format(state_shape))
        # input_shape = img_height * img_width * n_channels
        # print('input_shape = {0}'.format(input_shape))
        paddings = ( (strides - 1) * img_height - strides + filter_sizes) // 2
        # print('paddings = {0}'.format(paddings))
        
        self.q_network = nn.Sequential(
                            nn.Conv2d(state_shape[0], numb_filters[0], kernel_size=filter_sizes[0], stride=strides[0], padding=paddings[0]),
                            nn.Conv2d(numb_filters[0], numb_filters[1], kernel_size=filter_sizes[1], stride=strides[1], padding=paddings[1]),
                            nn.Conv2d(numb_filters[1], numb_filters[2], kernel_size=filter_sizes[2], stride=strides[2], padding=paddings[2]),
                            nn.Flatten(),
                            nn.Linear(5120, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_actions)
                            )
        print('----------------------------')
        print('Q-Network Architecture:')
        print(self.q_network)
        
        self.target_network = nn.Sequential(
                            nn.Conv2d(state_shape[0], numb_filters[0], kernel_size=filter_sizes[0], stride=strides[0], padding=paddings[0]),
                            nn.Conv2d(numb_filters[0], numb_filters[1], kernel_size=filter_sizes[1], stride=strides[1], padding=paddings[1]),
                            nn.Conv2d(numb_filters[1], numb_filters[2], kernel_size=filter_sizes[2], stride=strides[2], padding=paddings[2]),
                            nn.Flatten(),
                            nn.Linear(5120, 512),
                            nn.ReLU(),
                            nn.Linear(512, num_actions)
                            )
        print('Target Network Architecture:')
        print(self.target_network)
        print('----------------------------')
        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values(self, state, network):
        """
        Returns Q values for all actions

        Args:
            state: (torch tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            network: (str)
                The name of the network, either "q_network" or "target_network"

        Returns:
            out: (torch tensor) of shape = (batch_size, num_actions)

        Hint:
            1. What are the input shapes to the network as compared to the "state" argument?
            2. You can forward a tensor through a network by simply calling it (i.e. network(tensor))
        """
        out = None
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines lines ################
        
        out = torch.empty(state.shape[0], self.env.action_space.n)
        
        # print('state = {0}'.format(state.shape))
        # print('out = {0}'.format(out.shape))
        
        if (network =='q_network'):
            out = self.q_network(state)
            # for i in range (0,state.shape[0]):
                # out[i] = self.q_network(torch.flatten(state[i]))
                # out[i] = self.q_network(state[i])
            
        if (network =='target_network'):
            out = self.target_network(state)
            # for i in range (0,state.shape[0]):
                # out[i] = self.target_network(torch.flatten(state[i]))
                # out[i] = self.target_network(state[i])
        
        ##############################################################
        ######################## END YOUR CODE #######################
        return out


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((8, 8, 6))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
