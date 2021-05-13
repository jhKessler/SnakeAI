from collections import namedtuple, deque
from random import sample

import torch
                        
class ReplayMemory(object):
    """Class that saves multiple Transitions"""

    def __init__(self, gamma: float):
        self.gamma = gamma
        self.memory = []

    def push(self, data: tuple):
        """Save a transition"""
        self.memory.append(data)

    def reset(self):
        """Deletes the data"""
        self.memory = []
    
    def calculate_q_values(self) -> tuple:
        """Calculates q values for all state/action pairs"""
        states, actions, rewards = zip(*self.memory)
        rewards = torch.FloatTensor(rewards)

        # values for calculating q values in form of gamma**i for steps the reward is away
        q_values_coeffs = torch.FloatTensor([self.gamma ** i for i in range(len(rewards))])
        q_values = []

        for i in range(len(states)):
            future_rewards = rewards[i:]
            needed_q_coeffs = q_values_coeffs[:-i] if i != 0 else q_values_coeffs
            q_values.append(torch.matmul(future_rewards, needed_q_coeffs))

        self.reset()
        return states, actions, q_values

    def __len__(self):
        return len(self.memory)
