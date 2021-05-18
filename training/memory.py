from collections import namedtuple, deque
from random import sample

import torch
                        
class ReplayMemory(object):
    """Class that saves multiple Transitions"""


    @staticmethod
    def stackList(list) -> torch.tensor:
        return torch.stack(list)


    def __init__(self, gamma: float, device):
        self.memory = []
        self.device = device


    def push(self, data: tuple):
        """Saves a transition"""
        self.memory.append(data)


    def reset(self):
        """Deletes the data"""
        self.memory = []
    

    def getMemory(self) -> tuple:
        states, actions, rewards, next_states = zip(*self.memory)
        states = ReplayMemory.stackList(states).to(self.device)
        actions = ReplayMemory.stackList(actions).to(self.device)
        next_states = ReplayMemory.stackList(next_states).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        self.reset()
        return states, actions, rewards, next_states


    def __len__(self) -> int:
        return len(self.memory)
        