# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.
#

#! /usr/bin/env python
# -*- coding: utf-8 -*-

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# Correct the imports
import numpy as np
from itertools import product
cimport numpy as cnp


# Correct the type annotations in the policy_evaluation function
cpdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] policy_evaluation(double gamma, cnp.ndarray[cnp.float64_t, ndim=3] P, cnp.ndarray[cnp.float64_t, ndim=3] R, cnp.ndarray[cnp.int64_t, ndim=1, mode='c'] pi, double atol=1e-6, long max_iter=1000):
    cdef long NS = P.shape[0]
    cdef long s = 0
    cdef long iter = 0
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] V = np.zeros(NS, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] V_next = np.zeros(NS, dtype=np.float64)
    cdef double Delta = 0

    for iter in range(max_iter):
        Delta = 0
        for s in range(NS):
            # Ensure proper use of numpy operations for matrix-vector multiplication
            V_next[s] = np.dot(P[s, pi[s], :], R[s, pi[s]] + gamma * V)

        Delta = max(Delta, np.max(np.abs(V_next - V)))
        V[:] = V_next  # Avoid creating a new array; just update the contents
        
        if Delta < atol:
            break
    return V

cpdef find_optimal_actions(cnp.ndarray[cnp.float64_t, ndim=2] Q):
    cdef long num_states = Q.shape[0]
    cdef list optimal_actions_per_state = []
    cdef int s
    cdef cnp.ndarray[cnp.int64_t, ndim=1, mode="c"] max_actions

    for s in range(num_states):
        max_actions = np.argwhere(np.isclose(Q[s], np.amax(Q[s]))).flatten()
        optimal_actions_per_state.append(max_actions)

    return optimal_actions_per_state

def generate_optimal_policies(Q):
    optimal_actions_per_state = find_optimal_actions(Q)
    all_possible_policies = list(product(*optimal_actions_per_state))

    # Convert to NumPy array
    policies_array = np.array(all_possible_policies, dtype=np.int64)
    return policies_array

# Correct the policy_iteration function's call to policy_evaluation and its types
cpdef policy_iteration(double gamma, cnp.ndarray[cnp.float64_t, ndim=3] P, cnp.ndarray[cnp.float64_t, ndim=3] R, double atol=1e-6, long max_iter = 1000):
    cdef long NS = P.shape[0]
    cdef long NA = P.shape[1]
    cdef long s = 0
    cdef long a = 0
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] pi = np.zeros((1,NS), dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=2, mode="c"] next_pi = np.zeros((1,NS), dtype=np.int64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] V = np.zeros(NS, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] Q = np.zeros((NS, NA), dtype=np.float64)
    cdef double Delta = 0
    cdef long policy_stable = 0
    cdef long iter = 0

    while policy_stable == 0 and iter < max_iter:
        policy_stable = 1
        V = policy_evaluation(gamma, P, R, pi[0], atol, max_iter)
        for s in range(NS):
            for a in range(NA):
                # Ensure proper use of numpy operations for matrix-vector multiplication
                Q[s, a] = np.dot(P[s, a, :], R[s, a] + gamma * V)
        next_pi = generate_optimal_policies(Q)

        if not np.array_equal(next_pi, pi):
            policy_stable = 0
        pi = next_pi  # Update pi without creating a new array
        iter += 1

    return V,pi,Q
