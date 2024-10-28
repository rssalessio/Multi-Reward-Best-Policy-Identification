from __future__ import annotations
import os
import random
import lzma
import pickle
from typing import NamedTuple
from agents.replay_buffer import ReplayBuffer
from configs.config import EnvConfig

class Dataset(NamedTuple):
    """ Dat object to save/load the results """
    buffer: ReplayBuffer
    agent_name: str
    env_config: EnvConfig
    
    def dump(self, path: str):
        with lzma.open(f'{path}.lzma', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> Dataset:
        if not os.path.exists(path):
            raise Exception(f'Path {path} does not exist!')

        with lzma.open(path, 'rb') as f:
            buff = pickle.load(f)
        return buff
