import abc
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import lzma
import os
from agents.agent import Agent
from agents.utils import TimeStep, NumpyObservation, TorchTransitions
from agents.dqn.dqn_config import DQNConfig
from agents.dqn.dqn_networks import QNetwork
from typing import Union
from utils.random import Generators


class DQNAgent(Agent):
    NAME = 'DQN'
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: DQNConfig,
            device: torch.device,
            generators: Generators):
        super().__init__(config.epsilon_0, config.replay_capacity, device, generators)

        self._cfg = config
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.q_net = QNetwork(state_dim, num_actions, config.hidden_layer_size, device, generators.torch_gen)
        self.tgt_q_net = QNetwork(state_dim, num_actions, config.hidden_layer_size, device, generators.torch_gen).clone(self.q_net).freeze()
        
        self._overall_steps = 0
        self._gradient_steps = 0

        # optimizers
        self.critic_opt = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.tgt_q_net.freeze()
        

    def select_action(self, observation: NumpyObservation, step: int) -> int:
        if self._np_rng.uniform() < self.eps_fn(step):
            return np.random.choice(self.num_actions)

        return self.select_greedy_action(observation)
    
    def select_greedy_action(self, observation: NumpyObservation) -> int:
        obs = torch.as_tensor(observation, device=self._device).unsqueeze(0)
        q = self.q_net.forward(obs)
        return q[0].argmax(-1).item()

    def _update_critic(self, batch: TorchTransitions) -> float:

        with torch.no_grad():
            q = self.q_net.forward(batch.o_t)
            next_actions = q.max(-1)[1]
            q_target = self.tgt_q_net.forward(batch.o_t).gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target_y = batch.r_t.squeeze(-1) +  self._cfg.discount * (1-batch.d_t.squeeze(-1)) * q_target
                

        q = self.q_net.forward(batch.o_tm1).gather(-1, batch.a_tm1[:, None]).squeeze(-1)
        critic_loss = F.huber_loss(q, target_y.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self._gradient_steps += 1

        self.tgt_q_net.soft_update(self.q_net, self._cfg.target_soft_update)
        
        return critic_loss.item()

    def update(self, timestep: TimeStep) -> Union[float, None]:
        self._overall_steps += 1
        self._replay.add(timestep.to_float32())

        return self._update()

    @abc.abstractclassmethod
    def _update(self) -> Union[float, None]:
        raise NotImplementedError('_update() not implemented!')
    
    def save_model(self, path: str, seed: int, step: int):
        model_path = f"{path}/models"
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        file_path = f"{model_path}/{self.NAME}_networks_{seed}_{step}.pkl.lzma"
        with lzma.open(file_path, 'wb') as f:
            model = {
                'network': self.q_net.state_dict(),
                'target_network': self.tgt_q_net.state_dict()
            }
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def make_default_agent(state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: DQNConfig,
            device: torch.device,
            generators: Generators) -> Agent:
        return DQNAgent(state_dim, num_actions, num_rewards, config, device, generators)