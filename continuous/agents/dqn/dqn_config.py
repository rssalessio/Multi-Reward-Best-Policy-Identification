from typing import NamedTuple


class DQNConfig(NamedTuple):
    hidden_layer_size: int = 32
    batch_size: int = 128
    discount: float = 0.99
    replay_capacity: int = 100000
    min_replay_size: int = 128
    sgd_period: int = 1
    target_soft_update: float = 1e-2
    lr: float = 5e-4
    epsilon_0: float = 10
