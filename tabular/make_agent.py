from tabular.agents.agent import Agent, AgentParameters
from tabular.agents.random import RandomAgent, RandomAgentParameters
from tabular.agents.rf_ucrl import RFUCRL, RFUCRLParameters
from tabular.agents.mr_psrl import MRPSRL, MRPSRLparameters
from tabular.agents.ide3al import IDE3AL, IDE3ALParameters
from tabular.agents.mr_nas import MRNaS, MRNaSParameters
from enum import Enum

class AgentType(Enum):
    RANDOM_AGENT = 'Random Agent'
    RF_UCRL = 'RF-UCRL'
    MR_PSRL = 'MR-PSRL'
    IDE3AL = 'IDE3AL'
    MR_NAS = 'MR-NaS'

def make_agent(agent_parameters: AgentParameters, **kwargs) -> Agent:
    if isinstance(agent_parameters, RandomAgentParameters):
        return RandomAgent(agent_parameters)
    elif isinstance(agent_parameters, RFUCRLParameters):
        return RFUCRL(agent_parameters)
    elif isinstance(agent_parameters, MRNaSParameters):
        return MRNaS(agent_parameters)
    elif isinstance(agent_parameters, IDE3ALParameters):
        return IDE3AL(agent_parameters)
    elif isinstance(agent_parameters, MRPSRLparameters):
        return MRPSRL(agent_parameters)
    else:
        raise NotImplementedError(f'Type {agent_parameters.__str__} not found.')
