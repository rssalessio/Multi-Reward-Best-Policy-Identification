import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
from enum import Enum
from numpy.typing import NDArray
from tabular.envs.mdp import MDP
import warnings
import cvxpy as cp


class IDE3ALParameters(NamedTuple):
    agent_parameters: AgentParameters
    xi: float
    type: str = 'ID3AL'

class IDE3AL(Agent):
    """ IDE3AL Algorithm

        @See also http://amsdottorato.unibo.it/10588/1/mutti_mirco_tesi.pdf 
    """

    def __init__(self, parameters: IDE3ALParameters):
        super().__init__(parameters.agent_parameters)
        self.parameters = parameters
        self.H = int(np.ceil(1/(1-self.discount_factor)))
        self.xi = parameters.xi if parameters.xi is not None else 0.3 / self.na

        assert 0 <= self.xi <= 1/self.na, 'xi needs to be in [0,1/|A|]'

        self.pi = np.ones((self.ns, self.na), dtype=np.float64) / self.na

        problem, variables = self._build_problem()
        self._pi_problem = {
            'variables': variables,
            'problem': problem
        }

    def _update_pi(self):
        self._pi_problem['variables']['P'].value = self.empirical_transition(prior_p=1).reshape((self.ns*self.na, self.ns))
        sol = self._pi_problem['problem'].solve(solver=cp.CLARABEL)
        self.pi = self._pi_problem['variables']['pi'].value

    def _build_problem(self):
        v = cp.Variable(self.ns, nonneg=True)
        pi = cp.Variable((self.ns, self.na), nonneg=True)
        P = cp.Parameter((self.ns * self.na, self.ns), nonneg=True)

        constraints = [pi >= self.xi]
        for s in range(self.ns):
            constraints.append(pi[s].sum() == 1)
            constraints.append(v[s] >= 1 - cp.multiply(P[:,s], pi.flatten()).sum())
            constraints.append(v[s] >= cp.multiply(P[:,s], pi.flatten()).sum() - 1)

        objective = cp.Minimize(v.sum())
        return cp.Problem(objective, constraints), {'v': v, 'pi': pi, 'P': P}

    def forward(self, state: int, step: int) -> int:
        return np.random.choice(self.na, p=self.pi[state])
    
    @property
    def mdp(self) -> MDP:
        return MDP(P = self.empirical_transition)
    
    def process_experience(self, experience: Experience, step: int) -> None:
        if step % self.H == 0:
            self._update_pi()

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1.