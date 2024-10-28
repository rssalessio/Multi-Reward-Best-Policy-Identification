from __future__ import annotations
import numpy as np
import numpy.typing as npt
from tabular.utils.utils import policy_iteration, generate_optimal_policies
from itertools import product
from typing import Tuple, List, NamedTuple

class MDPStatistics(NamedTuple):
    V: npt.NDArray[np.float64]
    Q: npt.NDArray[np.float64]
    pi: npt.NDArray[np.int64]
    idxs_subopt_actions: npt.NDArray[np.bool_]
    Delta: npt.NDArray[np.float64]
    Delta_sq: npt.NDArray[np.float64]
    avg_V_greedy: npt.NDArray[np.float64]
    var_V_greedy: npt.NDArray[np.float64]
    var_V_greedy_max: npt.NDArray[np.float64]
    span_V_greedy: npt.NDArray[np.float64]
    span_V_greedy_max: npt.NDArray[np.float64]

class MDP(object):
    """Class used to store information about an MDP
    """  

    P: npt.NDArray[np.float64]
    """ Transition function """
    abs_tol: float
    """ Absolute value error, used to stop the policy iteration procedure """

    
    def __init__(self, P: npt.NDArray[np.float64], abs_tol: float = 1e-6):
        """Initialize the MDP and compute quantities of interest

        Parameters
        ----------
        P : npt.NDArray[np.float64]
            Transition function, of shape |S|x|A|x|S|
        abs_tol : float, optional
            Absolute tolerance for policy iteration, by default 1e-6
        """        
        self.P = P
        self.abs_tol = abs_tol

        self._boundary_rewards = self.generate_boundary_rewards()

    @property
    def dim_state(self) -> int:
        """Number of states"""        
        return self.P.shape[0]
    
    @property
    def dim_action(self) -> int:
        """Number of actions"""
        return self.P.shape[1]
    
    @staticmethod
    def generate_random_mdp(ns: int, na: int) -> MDP:
        """ Return a randomly generated MDP """
        P = np.random.dirichlet(np.ones(ns), size=(ns, na))
        return MDP(P)
    
    def build_stationary_matrix(self, pi: npt.NDArray[np.float64],gamma: float) -> npt.NDArray[np.float64]:
        ns,na = self.dim_state, self.dim_action
        P = self.P[...,None] * pi[None,None,...]
        P = P.reshape((ns*na, ns*na))
        M = (np.eye(ns*na) - gamma * P)
        return np.linalg.inv(M)

    
    def get_mdp_statistics(self, R: npt.NDArray[np.float64], discount_factor: float, eps: float = 1e-16):
        V, policies,Q = self.policy_iteration(R, discount_factor)
        gaps = Q.max(-1, keepdims=True) - Q
        gaps_sq = np.clip(np.square(gaps), a_min=eps, a_max=np.infty)
        idxs_subopt = np.array([
            [False if np.any(policies[:, s] == a) else True for a in range(self.dim_action)] for s in range(self.dim_state)])
        
        avg_V_greedy = self.P @ V
        var_V_greedy =  self.P @ (V ** 2) - (avg_V_greedy) ** 2
        var_V_greedy_max = np.max(var_V_greedy[~idxs_subopt])

        span_V_greedy = np.maximum(np.max(V) - avg_V_greedy, avg_V_greedy- np.min(V))
        span_V_greedy_max = np.max(span_V_greedy[~idxs_subopt])
        return MDPStatistics(V=V, Q=Q, pi=policies, idxs_subopt_actions=idxs_subopt,
                            Delta =gaps,
                             Delta_sq=gaps_sq, avg_V_greedy=avg_V_greedy,
                             var_V_greedy=var_V_greedy, var_V_greedy_max=var_V_greedy_max,
                             span_V_greedy=span_V_greedy, span_V_greedy_max=span_V_greedy_max)

    
    def policy_iteration(self, R: npt.NDArray[np.float64], discount_factor: float):
        return policy_iteration(gamma=discount_factor, P=self.P, R=R)
    
    def value_iteration(self, R: npt.NDArray[np.float64], discount_factor: float):
        return self.policy_iteration(R=R, discount_factor=discount_factor)[-1]
    
    def generate_boundary_rewards(self) -> npt.NDArray[np.float64]:
        rewards = np.eye(self.dim_state * self.dim_action)
        return rewards.reshape(-1, self.dim_state, self.dim_action)

    def generate_random_rewards(self, N: int, alpha: float = 1) -> npt.NDArray[np.float64]:
        alpha = np.ones(self.dim_state * self.dim_action) * alpha
        return np.random.dirichlet(alpha, size=N).reshape(N, self.dim_state, self.dim_action)
    
    def eval_transition(self, Phat: npt.NDArray[np.float64], R: npt.NDArray[np.float64], discount_factor: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        N = R.shape[0]

        V_res, pi_res, Q_res = np.zeros(N), np.zeros(N), np.zeros(N)

        for i in range(N):
            V_true, pi_true, Q_true = self.policy_iteration(R=R[i], discount_factor=discount_factor)
            V_hat, pi_hat, Q_hat = policy_iteration(gamma=discount_factor, P = Phat, R = R[i])
            V_res[i] = np.linalg.norm(V_true-V_hat, ord=1) / self.dim_state
            Q_res[i] = np.linalg.norm((Q_true-Q_hat).flatten(), ord=1) / (self.dim_state * self.dim_action)

            X_set = {tuple(row) for row in pi_true}
            Y_set = {tuple(row) for row in pi_hat}
            sym_diff = X_set ^ Y_set # Symmetric difference
            pi_res[i] = len(sym_diff) / len(X_set.union(Y_set))

        return V_res, pi_res, Q_res
