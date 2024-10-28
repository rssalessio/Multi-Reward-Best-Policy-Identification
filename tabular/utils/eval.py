import numpy as np
from numpy.typing import NDArray
from tabular.envs.mdp import MDP
from tabular.utils.utils import policy_iteration, policy_evaluation
from typing import Callable, Tuple

def eval_transition(mdp: MDP, Phat: NDArray, discount_factor: float) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    boundary_rewards = mdp.generate_boundary_rewards()
    N = boundary_rewards.shape[0]
    V_res, pi_res, Q_res = np.zeros(N), np.zeros(N), np.zeros(N)

    for i in range(N):
        V_true, pi_true, Q_true = mdp.policy_iteration(R=boundary_rewards[i], discount_factor=discount_factor)
        V_hat, pi_hat, Q_hat = policy_iteration(gamma=discount_factor, P = Phat, R = boundary_rewards[i])
        V_res[i] = np.linalg.norm(V_true-V_hat, ord=np.infty)
        Q_res[i] = np.linalg.norm((Q_true-Q_hat).flatten(), ord=np.infty)
        print(f'{pi_true.shape} - {pi_hat.shape}')
        pi_res[i] = np.mean(pi_true != pi_hat)

    return V_res, pi_res, Q_res


