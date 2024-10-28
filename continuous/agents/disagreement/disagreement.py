from __future__ import annotations
import torch
from agents.dqn.dqn_agent import DQNAgent
from agents.disagreement.disagreement_networks import DisagreementNetwork
from agents.disagreement.disagreement_config import DisagreementConfig
from typing import Optional
from agents.utils import TorchTransitions
from utils.random import Generators



class DisagreementAgent(DQNAgent):
    """ Disagreement agent
        @see https://arxiv.org/pdf/1906.04161.pdf
    """
    NAME = 'Disagreement Agent'
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: DisagreementConfig,
            device: torch.device,
            generators: Generators):
        super().__init__(state_dim, num_actions, num_rewards, config.dqn_config, device, generators)
        self.disagreement = DisagreementNetwork(state_dim, num_actions, config.hidden_layer_size, config.ensemble_size, device, generators.torch_gen)

        # optimizers
        self.disagreement_opt = torch.optim.Adam(self.disagreement.parameters(), lr=config.lr)
        self.disagreement.train()

    def update_disagreement(self, batch: TorchTransitions) -> float:
        error = self.disagreement.forward(batch.o_tm1, batch.a_tm1, batch.o_t)
        loss = error.mean()
        self.disagreement_opt.zero_grad()
        loss.backward()
        self.disagreement_opt.step()

        return loss.item()

    def compute_intrinsic_reward(self, batch: TorchTransitions) -> torch.Tensor:
        reward = self.disagreement.get_disagreement(batch.o_tm1, batch.a_tm1,
                                                    batch.o_t).unsqueeze(1)
        return reward

    def _update(self) -> float | None:
        if self._replay.size < self._cfg.min_replay_size:
            return None

        if self._gradient_steps % self._cfg.sgd_period != 0:
            return None
        
        minibatch = self._replay.sample(self._cfg.batch_size)

        batch = TorchTransitions.from_minibatch(minibatch, device=self._device, num_rewards=self.num_rewards)

        self.update_disagreement(batch)

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
            config: DisagreementConfig,
            device: torch.device,
            generators: Generators) -> DisagreementAgent:
        return DisagreementAgent(state_dim, num_actions, num_rewards, config, device, generators)