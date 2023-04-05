import numpy as np
import torch
import random


def seed_everything(seed):
    """
    Set seed to random, numpy, torch, gym environment
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)