from __future__ import annotations
import numpy as np
import torch
import random
from typing import NamedTuple

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class Generators(NamedTuple):
    np_gen: np.random.RandomState
    torch_gen: torch.Generator

def create_generators(seed: int, device: torch.device = torch.device('cpu')) -> Generators:
    numpy_gen = np.random.RandomState(seed)
    torch_gen = torch.Generator(device).manual_seed(seed)
    return Generators(numpy_gen, torch_gen)
