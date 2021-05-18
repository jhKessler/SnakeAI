from training import *

from itertools import chain
import random

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPS_START = 0.8
EPS_END = 0.2
EPS_DECAY_STEPS = 10
GAMMA = 0.1
N_ACTIONS = 4
ACTIONS = ["left", "right", "up", "down"]
GAMESIZE = 20
TRAINING_EPISODES = 1000
LR = 0.001

# dict for converting argmax to action and vice versa
ARGMAX_ACTION_CONVERT = {
    key: value for key, value in zip(chain(ACTIONS, range(len(ACTIONS))),
                                 chain(range(len(ACTIONS)), ACTIONS))
}


env = SnakeEnv(gamesize=GAMESIZE)
policy_net = QNetwork(inp_dim=GAMESIZE, outp_dim=N_ACTIONS, nodes_per_layer=1024).to(DEVICE)
#policy_net.apply(weights_init)

target_net = QNetwork(inp_dim=GAMESIZE, outp_dim=N_ACTIONS, nodes_per_layer=1024).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = torch.optim.Adam(policy_net.parameters(), LR)
episode_memory = ReplayMemory(GAMMA, device=DEVICE)
scores = []
loss_fn = nn.MSELoss()

# training loop
for episode in range(TRAINING_EPISODES):

    episode_length = 0
    episode_over = False
    current_state = env.getState()
    episode_score = 1

    # play episode of game
    while not episode_over:
        
        # determine if action is chosen randomly or if network is allowed to determine it
        epsilon = calculate_eps(EPS_START, EPS_END, EPS_DECAY_STEPS, episode_length)
        take_random_action = random.random() < epsilon

        # take action
        if take_random_action:
            action_vector = torch.rand(N_ACTIONS).to(DEVICE)
            net_argmax = int(torch.argmax(action_vector))
            action = ARGMAX_ACTION_CONVERT[net_argmax]

        else:
            policy_net.eval()
            action_vector = policy_net(current_state.to(DEVICE)).view(-1)
            net_argmax = int(torch.argmax(action_vector))
            action = ARGMAX_ACTION_CONVERT[net_argmax]
            policy_net.train()

        # observe actions impact on enviroment
        previous_state, current_state, reward, episode_over = env.step(action)
        episode_score = max(episode_score, env.getScore())
        if current_state is not None:
            episode_memory.push((previous_state, action_vector, reward, current_state))

        episode_length += 1

    scores.append(episode_score)

    policy_net.train()
    
    states, actions, rewards, next_states = episode_memory.getMemory()

    actions = torch.argmax(actions, dim=1)
    state_values = policy_net(states)
    predicted_values = state_values.gather(1, actions.unsqueeze(-1)).view(-1)
    next_state_value = (target_net(next_states).max() * GAMMA) + rewards
    loss = loss_fn(predicted_values, next_state_value)

    policy_net.zero_grad()
    loss.backward()
    optimizer.step()

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    print(episode_length, episode_score)
    
    


    
