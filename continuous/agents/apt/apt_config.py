from typing import NamedTuple
from agents.dqn.dqn_config import DQNConfig

class APTConfig(NamedTuple):
    dqn_config: DQNConfig
    rep_dim: int
    lr: float
    hidden_layer_size: int
    knn_clip: float =0.0
    knn_k: int = 12
    knn_avg: bool = True
    knn_rms: bool = False
