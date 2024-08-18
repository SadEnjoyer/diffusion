import torch
import random
import numpy as np


def set_seed(seed, use_deterministic_algorithms=False):
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)