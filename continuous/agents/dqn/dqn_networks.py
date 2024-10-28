import torch
import torch.nn as nn
from typing import Tuple
from agents.base_networks import BaseNetwork
from agents.utils import weight_init
from functools import partial
import numpy as np
class QNetwork(BaseNetwork):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, device: torch.device, generator: torch.Generator):
        super().__init__(device, generator)
    
        # for states actions come in the beginning
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                stddev = 1 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev, generator=generator)
                torch.nn.init.zeros_(m.bias.data)

        self.to(device).apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        inpt = obs
        q = self.net(inpt)
        return q
