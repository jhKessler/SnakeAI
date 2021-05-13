from training.dataclasses import ReplayMemory
from training.network import QNetwork
from training.utils import *
from training.enviroment import SnakeEnv

from itertools import chain
from random import random, choice

import torch

BATCH_SIZE = 128
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY_STEPS = 30
GAMMA = 0.99
N_ACTIONS = 4
ACTIONS = ["left", "right", "up", "down"]
GAMESIZE = 20
TRAINING_EPISODES = 100
LR = 0.001

# dict for converting argmax to action and vice versa
ARGMAX_ACTION_CONVERT = {
    k: value for k, value in zip(chain(ACTIONS, range(len(ACTIONS))),
                                 chain(range(len(ACTIONS)), ACTIONS))
}

env = SnakeEnv(gamesize=GAMESIZE)
net = QNetwork(inp_dim=GAMESIZE, outp_dim=N_ACTIONS)
net.apply(weights_init)
optimizer = torch.optim.Adam(net.parameters(), LR)
episode_memory = ReplayMemory(GAMMA)

# training loop
for episode in range(TRAINING_EPISODES):
    
    net.eval()

    episode_length = 0
    episode_over = False
    current_state = env.getState()

    # play episode of game
    while not episode_over:

        # determine if action is chosen randomly or if network is allowed to determine it
        epsilon = calculate_eps(EPS_START, EPS_END, EPS_DECAY_STEPS, episode_length)
        take_random_action = random() < epsilon

        # take action
        if take_random_action:
            action = choice(ACTIONS)
            random_argmax = ARGMAX_ACTION_CONVERT[action]
            action_vector = to_onehot(num=random_argmax, size=N_ACTIONS)
        else:
            action_vector = net(current_state)
            net_argmax = int(torch.argmax(action_vector))
            action = ARGMAX_ACTION_CONVERT[net_argmax]

        previous_state, current_state, reward, episode_over = env.step(action)
        episode_memory.push((previous_state, action_vector, reward))
    
    net.train()
    # train on episode memory
    episode_data = episode_memory.calculate_q_values()
