from typing import NamedTuple, Tuple


class DBMRBPIConfig(NamedTuple):
    ensemble_size: int = 5
    hidden_layer_size: int = 32
    batch_size: int = 128
    discount: float = 0.99
    replay_capacity: int = 100000
    min_replay_size: int = 128
    sgd_period: int = 1
    target_soft_update: float = 1e-2
    lr_Q: float = 1e-4
    lr_M: float = 1e-5
    mask_prob: float = 0.7
    noise_scale: float = 0
    delta_min: float = 1e-9
    kbar: float = 1.
    prior_scale: float = 5
    epsilon_0: float = 10
    exploration_prob: float = 0.
    enable_mix: bool = True
    per_step_randomization: bool = True
    lr_delta_min: Tuple[float, float] = (-6, -3)
