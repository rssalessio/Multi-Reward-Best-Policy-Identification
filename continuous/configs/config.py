from typing import NamedTuple, Dict, List, Optional, Union
from envs.cartpole_swingup import CartpoleSwingupConfig
from envs.deepsea import DeepSeaConfig

EnvConfig = Union[DeepSeaConfig, CartpoleSwingupConfig]
class EnvSimulationConfig(NamedTuple):
    seeds: List[int]
    agents_config: Dict[str, NamedTuple]
    env_config: EnvConfig
    collection_steps: int
    model_checkpoint_frequency: Optional[int] = None