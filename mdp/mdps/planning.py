"""
Passive MDP solving algorithms
"""

import numpy as np

def value_iteration(problem, maxiters, gamma, delta):
    """
    Performs the value iteration algorithm for a specific environment.
    :param problem: problem
    :param maxiters: max iterations allowed
    :param gamma: gamma value
    :param delta: delta value
    :return: policy
    """

    q = np.zeros(problem.observation_space.n, dtype="int8")  # Initial policy
    v = np.zeros(problem.observation_space.n)  # Initial vector of length |S|
    v_iter = 0

    while True:
        v_iter = v_iter + 1
        v_backup = v.copy()

        q = (problem.T * (problem.R + gamma * v_backup)).sum(axis=2)
        v = np.max(q, axis=1)

        if np.max((abs(v - v_backup))) < delta or v_iter == maxiters:
            p = np.argmax(q, axis=1)
            return np.asarray(p)


def policy_iteration(problem, pmaxiters, vmaxiters, gamma, delta):
    """
    Performs the policy iteration algorithm for a specific environment.
    :param problem: problem
    :param pmaxiters: max iterations allowed for the policy improvement
    :param vmaxiters: max iterations allowed for the policy evaluation
    :param gamma: gamma value
    :param delta: delta value
    :return: policy
    """
    p = np.zeros(problem.observation_space.n, dtype="int8")  # Initial policy
    v = np.zeros(problem.observation_space.n, dtype="int8")  # Initial vector of length |S|
    r_i = np.arange(problem.observation_space.n)
    p_iter = 0

    while True:
        p_backup = p.copy()
        p_iter = p_iter + 1
        v_iter = 0

        while True:
            v_backup = v.copy()
            v_iter = v_iter + 1

            q = np.sum(problem.T * (problem.R + gamma * v_backup), axis=2)
            v = q[r_i, p]

            if max(abs(v - v_backup)) < delta or v_iter == vmaxiters:  # end second while
                break

        q_i = np.sum(problem.T * (problem.R + gamma * v), axis=2)
        p = np.argmax(q_i, axis=1)

        if np.array_equal(p, p_backup) or p_iter == pmaxiters:  # end first while
            return p
