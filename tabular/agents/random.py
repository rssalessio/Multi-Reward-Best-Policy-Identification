import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
from enum import Enum
from tabular.envs.mdp import MDP

class RandomAgentParameters(NamedTuple):
    agent_parameters: AgentParameters
    type: str = 'Random agent'

class RandomAgent(Agent):
    """ RandomAgent Algorithm """

    def __init__(self, parameters: RandomAgentParameters):
        super().__init__(parameters.agent_parameters)
        self.parameters = parameters

    def forward(self, state: int, step: int) -> int:
        act =  np.random.choice(self.na)
        return act
    
    @property
    def mdp(self) -> MDP:
        return MDP(P = self.empirical_transition)
    
    def process_experience(self, experience: Experience, step: int) -> None:
        pass

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1.