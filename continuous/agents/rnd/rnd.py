from __future__ import annotations
import torch
from agents.dqn.dqn_agent import DQNAgent
from agents.rnd.rnd_networks import RND
from agents.rnd.rnd_config import RNDConfig
from typing import Optional
from agents.utils import TorchTransitions, RMS, compute_model_parameters
from utils.random import Generators

class RNDAgent(DQNAgent):
    """ Random network distillation agent
        @see https://arxiv.org/pdf/1810.12894.pdf
    """
    NAME = 'RND'
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: RNDConfig,
            device: torch.device,
            generators: Generators):
        super().__init__(state_dim, num_actions, num_rewards, config.dqn_config, device, generators)
        self.rnd_scale = config.rnd_scale

        self.rnd = RND(self.state_dim, 
                       self._cfg.hidden_layer_size,
                       config.rnd_rep_dim, device, generators.torch_gen).to(device)
        self.intrinsic_reward_rms = RMS(device=device)

        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=config.lr)
        self.rnd.train()

    def update_rnd(self, obs: torch.Tensor):
        prediction_error = self.rnd.forward(obs)
        loss = prediction_error.mean()
        self.rnd_opt.zero_grad()
        loss.backward()
        self.rnd_opt.step()

    def compute_intrinsic_reward(self, obs: torch.Tensor, eps: float = 1e-8):
        prediction_error = self.rnd.forward(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + eps)
        return reward

    def _update(self) -> float | None:
        if self._replay.size < self._cfg.min_replay_size:
            return None

        if self._gradient_steps % self._cfg.sgd_period != 0:
            return None
        
        minibatch = self._replay.sample(self._cfg.batch_size)

        batch = TorchTransitions.from_minibatch(minibatch, device=self._device, num_rewards=self.num_rewards)

        self.update_rnd(batch.o_t)

        with torch.no_grad():
            intr_reward = self.compute_intrinsic_reward(batch.o_t)
            reward = intr_reward

        batch_with_rew = TorchTransitions(batch.o_tm1, batch.a_tm1, reward,
                                          batch.d_t, batch.o_t)

        # update critic
        loss = self._update_critic(batch_with_rew)

        return loss

    @staticmethod
    def make_default_agent(state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: RNDConfig,
            device: torch.device,
            generators: Generators) -> RNDAgent:
        return RNDAgent(state_dim, num_actions, num_rewards, config, device, generators)