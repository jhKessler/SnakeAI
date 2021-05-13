import torch
import torch.nn as nn

def calculate_eps(start: float, end:float, decay: int, steps: int):
    """Calculates current eps value"""
    diff = (start - end) / decay
    return start - (diff * steps)

def weights_init(layer):
    """Initializes weights of Network"""
    if type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight)

def to_onehot(num: int, size: int) -> torch.tensor:
    onehot = torch.zeros(size)
    onehot[num] = 1
    return onehot