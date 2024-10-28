import torch
import torch.nn as nn
from agents.base_networks import BaseNetwork, SequentialBaseNetwork
from agents.utils import weight_init
from functools import partial
import numpy as np

class RND(BaseNetwork):
    def __init__(self,
                 obs_dim: int,
                 hidden_dim: int,
                 rnd_rep_dim: int,
                 device: torch.device,
                 generator: torch.Generator):
        super().__init__(device, generator)
        self.predictor = SequentialBaseNetwork([nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim)], device, generator)
        self.target = SequentialBaseNetwork([nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim)], device, generator)

        self.target.freeze()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                stddev = 1 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev, generator=generator)
                torch.nn.init.zeros_(m.bias.data)

        self.to(device).apply(init_weights)#partial(weight_init, generator=generator))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error
