import numpy as np
from .agent import Agent, Experience, AgentParameters
from tabular.utils.posterior import PosteriorProbabilisties
from tabular.utils.utils import policy_iteration
from typing import NamedTuple

class MRPSRLparameters(NamedTuple):
    agent_parameters: AgentParameters
    type: str = 'MR-PSRL'


class MRPSRL(Agent):
    """ MR-PSRL Algorithm """

    def __init__(self,  parameters: MRPSRLparameters):
        super().__init__(parameters.agent_parameters)
        self.posterior = PosteriorProbabilisties(self.ns, self.na, prior_p=1)
        self.greedy_policy = np.zeros(self.ns, dtype=int)
        self.state_action_visits_copy = self.state_action_visits.copy()
        self.H = int(np.ceil(1 / (1 - self.discount_factor)))

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1

    def forward(self, state: int, step: int) -> int:
        return self.greedy_policy[state]
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, sp = experience.s_t, experience.a_t, experience.s_tp1
        self.posterior.update(s, a, sp)

        if step % self.H == 0:
            P = self.posterior.sample_posterior()
            # Random sampling of the reward
            R = np.random.dirichlet(np.ones(self.ns*self.na), size=1)
            R = R.reshape(self.ns, self.na, -1)

            _, pi, _ = policy_iteration(self.discount_factor, P, R)
            self.greedy_policy = pi[np.random.choice(pi.shape[0])]

            


    