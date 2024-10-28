from __future__ import annotations
import abc
import numpy as np
import torch
from agents.utils import NumpyObservation, TimeStep
from agents.replay_buffer import ReplayBuffer
from datetime import datetime
from utils.dataset import Dataset
from configs.config import EnvConfig
from utils.random import Generators
from agents.replay_buffer import ReplayBuffer
class Agent(abc.ABC):
    """
    Base class of an exploration agent
    """
    NAME = 'AbstractAgent'
    def __init__(self, epsilon: float, capacity: int, device: torch.device, generators: Generators):
        self._epsilon = epsilon
        self._replay =  ReplayBuffer(capacity=capacity, rng=generators.np_gen)
        self._np_rng = generators.np_gen
        self._torch_rng = generators.torch_gen
        self._device = device

    def eps_fn(self, steps: int) -> float:
        """ Epsilon-based exploration 
            Computes epsilon =  x0/(x0+t)
        """
        x0 = self._epsilon
        return min(1, x0 / max(1, (steps + x0)))
    
    @property
    def buffer(self) -> ReplayBuffer:
        return self._replay

    @abc.abstractmethod
    def select_action(self, observation: NumpyObservation, step: int) -> int:
        pass
    
    @abc.abstractmethod
    def select_greedy_action(self, observation: NumpyObservation) -> int:
        pass

    @abc.abstractmethod
    def update(self, timestep: TimeStep) -> None:
        pass
    
    @abc.abstractstaticmethod
    def make_default_agent(
            state_dim: int,
            num_actions: int,
            num_rewards: int,
            config: any,
            device: torch.device,
            generators: Generators) -> Agent:
        raise NotImplementedError('make_default_agent not implemented')
    
    def dump_buffer(self, path: str, env_config: EnvConfig, seed: int):
        file_path = f"{path}/{self.NAME}_{seed}.pkl"
        Dataset(self.buffer, self.NAME, env_config).dump(file_path)

    @abc.abstractmethod
    def save_model(self, path: str, seed: int, step: int):
        # file_path = f"{path}/{self.NAME}_model_{seed}_{step}.pkl"
        raise Exception('Not implemented')