import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
from enum import Enum
from numpy.typing import NDArray
from tabular.envs.mdp import MDP
import warnings

# Ignore specific RuntimeWarning
warnings.filterwarnings("ignore", message="overflow encountered in multiply")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

class RFUCRLParameters(NamedTuple):
    agent_parameters: AgentParameters
    type: str = 'RF-UCRL'

class RFUCRL(Agent):
    """ RF-UCRL Algorithm
        @see also http://proceedings.mlr.press/v132/kaufmann21a/kaufmann21a.pdf
    """

    def __init__(self, parameters: RFUCRLParameters):
        super().__init__(parameters.agent_parameters)
        self.parameters = parameters
        self.H = int(np.ceil(1/(1-self.discount_factor)))
        self.E = np.zeros((self.H + 1, self.ns, self.na))
        self.pi = np.zeros((self.H, self.ns), dtype=np.int64)
        self.sigma = (self.discount_factor ** np.arange(self.H)).cumsum()
        self.current_h = 0

    def _beta_ucrl(self, n: float):
        c1 = np.log(2 * self.ns * self.na / self.delta)
        c2 = np.log(np.e * (1 + n/(self.ns - 1)))
        return c1 + (self.ns - 1) * c2

    def _update_E(self):
        ones = np.ones((self.ns, self.na))
        P = self._empirical_transition()
        for h in reversed(range(self.H)):
            c = self.discount_factor * self.sigma[self.H - (h+1)]
            r = 2 * self._beta_ucrl(1 + self.state_action_visits) / (1  + self.state_action_visits)
            E_hp1 = self.E[h+1].max(-1)[None, None :]
            E_h = c * np.sqrt(r) + self.discount_factor * (P * E_hp1).sum(-1)
            self.E[h] = np.minimum(c * ones, E_h)# Worse performance with the minimum
    
    def _update_pi(self):
        self.pi = self.E.argmax(-1)

    def forward(self, state: int, step: int) -> int:
        return self.pi[self.current_h,state]
    
    @property
    def mdp(self) -> MDP:
        return MDP(P = self.empirical_transition())
    
    def process_experience(self, experience: Experience, step: int) -> None:
        self.current_h += 1

        if self.current_h >= self.H:
            self.current_h = 0
            self._update_E()
            self._update_pi()

    def _empirical_transition(self, prior_p: float = 0.5) -> NDArray[np.float64]:
        prior_transition = prior_p * np.ones((self.ns, self.na, self.ns))
        posterior_transition = prior_transition + self.exp_visits

        P = posterior_transition / posterior_transition.sum(-1, keepdims=True)
        return P
    
    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1.