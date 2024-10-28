from typing import NamedTuple
from agents.dqn.dqn_config import DQNConfig

class DisagreementConfig(NamedTuple):
    dqn_config: DQNConfig
    ensemble_size: int = 5
    lr: float = 1e-4
    hidden_layer_size: int = 32
    