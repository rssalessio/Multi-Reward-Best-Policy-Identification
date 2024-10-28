import torch
import torch.nn as nn
from agents.base_networks import BaseNetwork, SequentialBaseNetwork
from agents.utils import weight_init
from functools import partial
class DisagreementNetwork(BaseNetwork):
    
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int, ensemble_size: int, device: torch.device, generator: torch.Generator):
        super().__init__(device, generator)
        self.ensemble = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, hidden_dim),
                          nn.ReLU(), nn.Linear(hidden_dim, state_dim * num_actions))
            for _ in range(ensemble_size)
        ])
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.to(device).apply(partial(weight_init, generator=generator))

    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor):
        #import ipdb; ipdb.set_trace()
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        errors = []
        action = action[:, None].repeat(1, obs.shape[1])
        for model in self.ensemble:
            next_obs_hat: torch.Tensor = model.forward(obs).reshape(-1, self.state_dim, self.num_actions).gather(-1, action.unsqueeze(-1)).squeeze(-1)
            model_error = torch.norm(next_obs - next_obs_hat,
                                     dim=-1,
                                     p=2,
                                     keepdim=True)
            errors.append(model_error)

        return torch.cat(errors, dim=1)

    def get_disagreement(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        preds = []
        action = action[:, None].repeat(1, obs.shape[1])
        for model in self.ensemble:
            next_obs_hat: torch.Tensor = model.forward(obs).reshape(-1, self.state_dim, self.num_actions).gather(-1, action.unsqueeze(-1)).squeeze(-1)
            preds.append(next_obs_hat)
        preds = torch.stack(preds, dim=0)
        return torch.var(preds, dim=0).mean(dim=-1)