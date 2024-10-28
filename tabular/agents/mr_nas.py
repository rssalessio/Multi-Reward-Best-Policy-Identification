import numpy as np
from tabular.envs.mdp import MDP
from tabular.agents.agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
from tabular.utils.bound import BoundResult, solve_bound
from tabular.utils.period import Period, ConstantPeriod
from enum import Enum

class MRNaSParameters(NamedTuple):
    agent_parameters: AgentParameters
    period_computation_omega: Period
    enable_averaging: bool
    alpha: float = 0.99
    beta: float = 0.01
    discount_averaging: Optional[float] = None
    type: str = 'MR-NaS'

class MRNaS(Agent):
    """ MR-NaS Algorithm """

    def __init__(self, parameters: MRNaSParameters):
        self.alpha_exp = parameters.alpha
        self.beta_exp = parameters.beta
        super().__init__(parameters.agent_parameters)
        self.parameters = parameters
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        if self.parameters.period_computation_omega:
            self.period_omega = self.parameters.period_computation_omega
        else:
            self.period_omega = ConstantPeriod(int(np.ceil(1 / (1 - self.discount_factor))))
        self.updates = 1
        assert 0 < self.alpha_exp + self.beta_exp <=1, "alpha+beta must be in (0,1]"
    
    @property
    def mdp(self) -> MDP:
        return MDP(P = self.empirical_transition())

    def suggested_exploration_parameter(self, dim_state: int, dim_action: int) -> float:
        return self.alpha_exp
    
    def compute_exp(self, state: int, step:int) -> float:
        Ns = max(1, self.total_state_visits[state])
        Nsa = self.state_action_visits[state]
        beta = Ns * self.beta_exp * np.log(Ns)
        beta =  beta / max(1,np.max(Nsa - Nsa.min()))
        
        # step 1
        log_numerators = -beta * (Nsa / Ns)
        num_min = np.max(log_numerators)
        log_denominator = num_min + np.log(np.sum(np.exp(log_numerators - num_min)))
        log_expression = log_numerators - log_denominator

        return np.exp(log_expression)

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step, minimum_exploration=1e-3)
        exp_policy =  self.compute_exp(state, step) #self.uniform_policy #
        omega = (1-epsilon) * self.omega + epsilon * exp_policy
        omega = omega[state] / omega[state].sum()
        try:
            act =  np.random.choice(self.na, p=omega)
        except:
            import pdb
            pdb.set_trace()

        return act
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, sp = experience.s_t, experience.a_t,  experience.s_tp1

        if step % self.period_omega.get() == 0:
            self.prev_omega = self.omega.copy()
            try:
                results = solve_bound(self.mdp, self.discount_factor)
            except:
                self.updates += 1
                return
    
            omega = results.w
            
            if self.parameters.enable_averaging:
                if self.parameters.discount_averaging is None:
                    omega = ( self.updates * self.omega + omega) / (self.updates + 1)
                else:
                    k = self.parameters.discount_averaging
                    omega = k * self.omega + (1 - k) * omega

            self.omega = omega
            self.updates += 1

            self.period_omega.update(step)
