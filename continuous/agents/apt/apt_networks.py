import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base_networks import BaseNetwork, SequentialBaseNetwork
from agents.utils import weight_init
from typing import Tuple
from functools import partial

class APTNetwork(BaseNetwork):
    """
    ICM with a trunk to save memory for KNN
    """
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int, rep_dim: int, device: torch.device, generator: torch.Generator):
        super().__init__(device, generator)
        self.trunk = nn.Sequential(nn.Linear(state_dim, rep_dim),
                                   nn.ReLU())

        self.forward_net = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim * num_actions))

        self.backward_net = nn.Sequential(
            nn.Linear(2 * rep_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_actions), nn.Softmax(dim=-1))

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.rep_dim = rep_dim
        self.to(device).apply(partial(weight_init, generator=generator))

    def forward(self,
                obs: torch.Tensor, 
                action: torch.Tensor,
                next_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        obs = self.trunk.forward(obs)
        next_obs = self.trunk.forward(next_obs)
        next_obs_hat = self.forward_net.forward(obs).reshape(
            -1, self.rep_dim, self.num_actions)
        action_hat = self.backward_net.forward(torch.cat([obs, next_obs], dim=-1))

        action_rep = action[:, None].repeat(1, obs.shape[1])
        next_obs_hat = next_obs_hat.gather(-1, action_rep.unsqueeze(-1)).squeeze(-1)
        
        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        
        backward_error = F.cross_entropy(action_hat, action)
        return forward_error, backward_error

    def get_rep(self, obs: torch.Tensor) -> torch.Tensor:
        rep = self.trunk.forward(obs)
        return rep

