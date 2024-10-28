from __future__ import annotations
import torch
from agents.dqn.dqn_agent import DQNAgent
from agents.apt.apt_networks import APTNetwork
from agents.apt.apt_config import APTConfig
from typing import Optional
from agents.utils import TorchTransitions, RMS, PBE
from utils.random import Generators



class APTAgent(DQNAgent):
    """ APT Agent
        @see https://arxiv.org/pdf/2103.04551.pdf
    """
    NAME = 'APT'
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: APTConfig,
            device: torch.device,
            generators: Generators):
        super().__init__(state_dim, num_actions, num_rewards, config.dqn_config, device, generators)

        self.apt = APTNetwork(state_dim, num_actions,
                              config.hidden_layer_size,
                              config.rep_dim, device, generators.torch_gen)

        # optimizers
        self.apt_opt = torch.optim.Adam(self.apt.parameters(), lr=config.lr)
        self.apt.train()

        # particle-based entropy
        self.rms = RMS(device)
        self.pbe = PBE(self.rms, config.knn_clip, config.knn_k,
                       config.knn_avg, config.knn_rms, device)


    def update_apt(self, batch: TorchTransitions) -> float:
        forward_error, backward_error = self.apt.forward(
            batch.o_tm1, batch.a_tm1, batch.o_t)

        loss = forward_error.mean() + backward_error.mean()

        self.apt_opt.zero_grad()
        loss.backward()
        self.apt_opt.step()

        return loss.item()

    def compute_intrinsic_reward(self, batch: TorchTransitions) -> torch.Tensor:
        rep = self.apt.get_rep(batch.o_tm1)
        reward = self.pbe(rep)
        reward = reward.reshape(-1, 1)
        return reward

    def _update(self) -> float | None:
        if self._replay.size < self._cfg.min_replay_size:
            return None

        if self._gradient_steps % self._cfg.sgd_period != 0:
            return None
        
        minibatch = self._replay.sample(self._cfg.batch_size)

        batch = TorchTransitions.from_minibatch(minibatch, device=self._device, num_rewards=self.num_rewards)

        self.update_apt(batch)

        with torch.no_grad():
            intr_reward = self.compute_intrinsic_reward(batch)
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
            config: APTConfig,
            device: torch.device,
            generators: Generators) -> APTAgent:
        return APTAgent(state_dim, num_actions, num_rewards, config, device, generators)