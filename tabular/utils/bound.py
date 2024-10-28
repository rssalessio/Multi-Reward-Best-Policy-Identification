import cvxpy as cp
from numpy.typing import NDArray
from tabular.envs.mdp import MDP
from tabular.utils.utils import *
from typing import Callable, NamedTuple, List

class BoundResult(NamedTuple):
    value: float
    w: NDArray[np.float64]

class BoundProblem(NamedTuple):
    class _Variables(NamedTuple):
        w: cp.Variable
    class _Parameters(NamedTuple):
        P: cp.Parameter
        opt_actions: List[cp.Parameter]
        sub_opt_actions: List[cp.Parameter]
        gaps_sq_inv: List[cp.Parameter]
        gaps_sq_min_inv: List[cp.Parameter]
    
    problem: cp.Problem
    discount_factor: float
    C: float
    variables: _Variables
    parameters: _Parameters

def solve_parametrized_bound(mdp: MDP, bound: BoundProblem) -> BoundResult:
    bound.parameters.P.value = mdp.P.reshape((mdp.dim_state * mdp.dim_action, mdp.dim_state))
    basis = np.eye(mdp.dim_state * mdp.dim_action)

    for i in range(mdp.dim_action *  mdp.dim_state):
        stats = mdp.get_mdp_statistics(R = basis[:,i].reshape(mdp.dim_state, mdp.dim_action)[..., np.newaxis], discount_factor=bound.discount_factor)
        bound.parameters.opt_actions[i].value = 1-(stats.idxs_subopt_actions).astype(np.float64)
        bound.parameters.sub_opt_actions[i].value = stats.idxs_subopt_actions.astype(np.float64)
        bound.parameters.gaps_sq_inv[i].value = 1 / stats.Delta_sq
        bound.parameters.gaps_sq_min_inv[i].value = 1 / stats.Delta_sq[stats.Delta_sq > 0].min()
    
    sol = bound.problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=100)
    bound.variables.w.value = np.random.uniform(size=(mdp.dim_state, mdp.dim_action))
    return BoundResult(sol / (bound.C*(1- bound.discount_factor) ** 2), bound.variables.w.value)
    

def evaluate_sampling(
        w: NDArray[np.float64],
        mdp: MDP,
        discount_factor: float) -> float:
    ns, na = mdp.dim_state, mdp.dim_action
    basis = np.eye(ns * na)
    

    M = basis.shape[1]
    obj = np.zeros(M)

    mdp_stats = [
        mdp.get_mdp_statistics(
            basis[:,i].reshape(ns, na)[..., np.newaxis],
            discount_factor)
        for i in range(M)
    ]
    for i in range(M):
        stats = mdp_stats[i]
        gaps_sq = stats.Delta_sq
        idxs_subopt = stats.idxs_subopt_actions

        gap_sq_min = gaps_sq[idxs_subopt].min() 
        
        # First term
        H_rsa = 2 * ((discount_factor * stats.span_V_greedy) ** 2) / gaps_sq

        # Second term
        H_r1 = 139 * ((1 + discount_factor) ** 2) / (1 -  discount_factor) ** 3
        H_r2 = ((4 * stats.var_V_greedy[~idxs_subopt].max() * discount_factor) / (1 -  discount_factor)) ** 2
        H_r3 = 6 * (discount_factor * stats.span_V_greedy[~idxs_subopt].max() * (1 + discount_factor) / (1 -  discount_factor)) ** (4/3)

        H_r =  min(H_r1, max(H_r2, H_r3)) / gap_sq_min

        T1 = np.max(H_rsa[idxs_subopt]/ w[idxs_subopt])
        T2 = np.max(H_r / w[~idxs_subopt])
        obj[i]= T1+T2
    
    
    return obj.max()



def solve_bound(mdp: MDP, discount_factor: float, navigation_constraints: bool = True, eps: float = 1e-9) -> BoundResult:
    ns, na = mdp.dim_state, mdp.dim_action
    basis = np.eye(ns * na)
    w = cp.Variable((ns, na), name='omega', nonneg=True)
    sigma = cp.Variable(1, nonneg=True)

    w.value = np.random.uniform(size=(ns, na))

    C = (1 - discount_factor) ** 4 * 1e-6 / (mdp.dim_state * mdp.dim_action)

    constraints = [cp.sum(w) == 1, w >= sigma, sigma >= 1e-16]
    
    if navigation_constraints:
        constraints.extend(
            [cp.sum(w[s]) == cp.sum(cp.multiply(mdp.P[:,:,s], w)) for s in range(ns)])

    obj = []


    M = basis.shape[1]

    mdp_stats = [
        mdp.get_mdp_statistics(
            basis[:,i].reshape(ns, na)[..., np.newaxis],
            discount_factor)
        for i in range(M)
    ]

    
    for i in range(M):
        stats = mdp_stats[i]
        gaps_sq = stats.Delta_sq
        idxs_subopt = stats.idxs_subopt_actions

        gap_sq_min = gaps_sq[idxs_subopt].min() 
        
        # First term
        H_rsa = C * 2 * ((discount_factor * stats.span_V_greedy) ** 2) / gaps_sq

        # Second term
        H_r1 = 139 * ((1 + discount_factor) ** 2) / (1 -  discount_factor) ** 3
        H_r2 = ((4 * stats.var_V_greedy[~idxs_subopt].max() * discount_factor) / (1 -  discount_factor)) ** 2
        H_r3 = 6 * (discount_factor * stats.span_V_greedy[~idxs_subopt].max() * (1 + discount_factor) / (1 -  discount_factor)) ** (4/3)

        H_r = C * min(H_r1, max(H_r2, H_r3)) / gap_sq_min

        T1 = cp.max(cp.multiply(H_rsa[idxs_subopt], cp.inv_pos(w[idxs_subopt])))
        T2 = cp.max(H_r * cp.inv_pos(w[~idxs_subopt]))
        obj.append(T1+T2)
    
    
    obj = cp.max(cp.hstack(obj))
    problem = cp.Problem(cp.Minimize(obj), constraints)
    value = problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=100)
    return BoundResult(value / C, w.value)


def build_problem(ns: int, na: int, discount_factor: float, navigation_constraints: bool = True, eps: float = 1e-12):
    w = cp.Variable((ns, na), name='omega', nonneg=True)
    sigma = cp.Variable(1, nonneg=True)
    P = cp.Parameter((ns * na, ns), name='P', nonneg=True)
    opt_actions = [cp.Parameter((ns, na), nonneg=True) for i in range(ns * na)]
    sub_opt_actions = [cp.Parameter((ns, na), nonneg=True) for i in range(ns * na)]
    gaps_sq_inv = [cp.Parameter((ns, na), nonneg=True) for i in range(ns * na)]
    gaps_sq_min_inv = [cp.Parameter(nonneg=True) for i in range(ns * na)]

    y_H_rsa = [cp.Variable((ns, na), nonneg=True) for i in range(ns * na)]
    y_H_r = [cp.Variable(nonneg=True) for i in range(ns * na)]

    C = (1 - discount_factor) ** 4 * 1e-6 / (ns * na)

    constraints = [cp.sum(w) == 1, w >= sigma, sigma >= eps]
    constraints.extend([
        y_H_rsa[i] == C * 2 * (discount_factor ** 2) * gaps_sq_inv[i] for i in range(ns * na)
    ])
    constraints.extend([
        y_H_r[i] == C* min(139, 16 * (discount_factor ** 2) / (1-discount_factor))* (1 + discount_factor) **  2 / (gaps_sq_min_inv[i] * (1-discount_factor)) 
            for i in range(ns * na)
    ])
    
    if navigation_constraints:
        constraints.extend(
            [cp.sum(w[s]) == cp.sum(cp.multiply(P[:,s], w.flatten())) for s in range(ns)])

    obj = []

    for i in range(ns * na):
        H_rsa = cp.multiply(sub_opt_actions[i], y_H_rsa[i])
        T1 = cp.max(cp.multiply(H_rsa, cp.inv_pos(w)))

        H_r = opt_actions[i] * y_H_r[i]
        T2 = cp.max(cp.multiply(H_r, cp.inv_pos(w)))

        print(f'T1: {T1.is_dcp(dpp=True)} - T2: {T2.is_dcp(dpp=True)}')

        obj.append(T1+T2)
    
    
    obj = cp.max(cp.hstack(obj))
    problem = cp.Problem(cp.Minimize(obj), constraints)
    return BoundProblem(
        problem=problem,
        discount_factor=discount_factor,
        C=C,
        variables= BoundProblem._Variables(w=w),
        parameters= BoundProblem._Parameters(P = P, opt_actions=opt_actions, 
                                             sub_opt_actions=sub_opt_actions, gaps_sq_inv=gaps_sq_inv,
                                             gaps_sq_min_inv=gaps_sq_min_inv)
    )

def solve_bound_new(mdp: MDP, discount_factor: float, navigation_constraints: bool = True, eps: float = 1e-9) -> BoundResult:
    ns, na = mdp.dim_state, mdp.dim_action
    w = cp.Variable((ns, na), name='omega', nonneg=True)
    sigma = cp.Variable(1, nonneg=True)

    w.value = np.random.uniform(size=(ns, na))

    C = (1 - discount_factor) ** 4 * 1e-6 / (mdp.dim_state * mdp.dim_action)
    MIN_GAP_SQ = np.infty
    constraints = [cp.sum(w) == 1, w >= sigma, sigma >= 1e-12]
    
    if navigation_constraints:
        constraints.extend(
            [cp.sum(w[s]) == cp.sum(cp.multiply(mdp.P[:,:,s], w)) for s in range(ns)])

    obj = []

    reward = np.ones((ns,na))[..., np.newaxis]

    mdp_stats = mdp.get_mdp_statistics(reward, discount_factor)

    var_V_greedy = np.zeros((ns, na))
    span_V_greedy = np.zeros((ns, na))

    var_V_greedy = np.maximum(var_V_greedy, mdp_stats.var_V_greedy)
    span_V_greedy = np.maximum(span_V_greedy, mdp_stats.span_V_greedy)
    
    stats = mdp_stats
    gaps_sq = stats.Delta_sq
    idxs_subopt = stats.idxs_subopt_actions

    T2 = cp.max(1 * cp.inv_pos(w[~idxs_subopt]))

    
    obj = cp.max(T2)
    problem = cp.Problem(cp.Minimize(obj), constraints)
    value = problem.solve(solver=cp.CLARABEL, verbose=False, max_iter=100)
    return BoundResult(value / (C*(1- discount_factor) ** 2), w.value)