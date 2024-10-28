from typing import NamedTuple, List, Tuple
from enum import Enum
from tabular.envs.riverswim import RiverSwim
from tabular.envs.doublechain import DoubleChain
from tabular.envs.forked_riverswim import ForkedRiverSwim
from tabular.envs.n_arms import NArms

class EnvType(Enum):
    RIVERSWIM = 'Riverswim'
    FORKED_RIVERSWIM = 'ForkedRiverswim'
    DOUBLE_CHAIN = 'DoubleChain'
    N_ARMS = 'NArms'

class RiverSwimParameters(NamedTuple):
    num_states:int = 5

class DoubleChainParameters(NamedTuple):
    length:int = 5
    p: float = 0.7

class NArmsParameters(NamedTuple):
    num_arms: int = 6
    p0: float = 1.0

class ForkedRiverSwimParameters(NamedTuple):
    river_length: int = 5

class EnvParameters(NamedTuple):
    env_type: EnvType
    parameters: RiverSwimParameters | DoubleChainParameters | NArmsParameters | ForkedRiverSwimParameters
    horizon: int

class SimulationGeneralParameters(NamedTuple):
    num_sims: int
    num_rewards: int
    freq_eval: int
    discount_factor: float
    delta: float

class SimulationParameters(NamedTuple):
    env_parameters: EnvParameters
    sim_parameters: SimulationGeneralParameters

class SimulationConfiguration(NamedTuple):
    sim_parameters: SimulationGeneralParameters
    envs: List[Tuple[EnvParameters, List[NamedTuple]]]

def make_env(env: EnvParameters):
    match env.env_type:
        case EnvType.RIVERSWIM:
            return RiverSwim(num_states=env.parameters.num_states)
        case EnvType.DOUBLE_CHAIN:
            return DoubleChain(length = env.parameters.length, p = env.parameters.p)
        case EnvType.N_ARMS:
            return NArms(num_arms = env.parameters.num_arms, p0 = env.parameters.p0)
        case EnvType.FORKED_RIVERSWIM:
            return ForkedRiverSwim(river_length = env.parameters.river_length)
        case _:
            raise NotImplementedError(f'Type {env.env_type.value} not found.')