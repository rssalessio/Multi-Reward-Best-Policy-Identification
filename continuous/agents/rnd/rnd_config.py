from typing import NamedTuple
from agents.dqn.dqn_config import DQNConfig

class RNDConfig(NamedTuple):
    dqn_config: DQNConfig
    rnd_rep_dim: int
    lr: float
    rnd_scale: float = 1.
    