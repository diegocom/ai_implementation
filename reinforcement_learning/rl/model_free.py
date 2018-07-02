"""
Model free algorithms for solving MDPs
"""


import numpy as np
from collections import defaultdict
import math


def epsilon_greedy(Q, nA, epsilon):
    """
    Epsilon-greedy action selection function
    :param q: q table
    :param state: agent's current state
    :param epsilon: epsilon parameter
    :return: action
    """

    def policy_fn(state):
        A = np.ones(nA, dtype=float) * epsilon / (nA-1)
        best_action = np.argmax(Q[state])
        A[best_action] = (1.0 - epsilon)
        return A

    return policy_fn


def softmax(Q, state, nA, t):
    """
    Softmax action selection function
    :param q: q table
    :param state: agent's current state
    :param t: t parameter (temperature)
    :return: action
    """
    probabilities = np.zeros(nA)

    for a in range(nA):
        nom = math.exp(Q[state][a] / t)
        denom = sum(math.exp(val / t) for val in Q[state][:])
        probabilities[a] = float(nom / denom)
    action = np.random.choice(range(nA), p=probabilities)

    return action


def q_learning(problem, episodes, alpha, gamma, expl_func, expl_param):
    """
    Performs the Q-Learning algorithm for a specific environment
    :param problem: problem
    :param episodes: number of episodes for training
    :param alpha: alpha parameter
    :param gamma: gamma parameter
    :param expl_func: exploration function (epsilon_greedy, softmax)
    :param expl_param: exploration parameter (epsilon, T)
    :return: (policy, rews, ep_lengths): final policy, rewards for each episode [array], length of each episode [array]
    """

    num_states = problem.observation_space.n
    num_action = problem.action_space.n

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(num_action))

    # The policy we're following
    policy = expl_func(Q, num_action, expl_param)

    # array di lunghezza episodes
    rewards = np.zeros(episodes, dtype="int16")
    lengths = np.zeros(episodes, dtype="int16")

    for episodio in range(episodes):

        stato_corrente = problem.reset()  # Reset the environment
        el = 0
        rew = 0

        # inzio REPEAT...UNTIL
        while True:

            # scelgo l'azione da fare
            action_probs = policy(stato_corrente)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # eseguo l'azione scelta
            stato_prossimo, r, done, ep_step = problem.step(action)
            el += 1
            rew += r

            # TD Update
            best_next_action = np.argmax(Q[stato_prossimo])
            td_target = r + gamma * Q[stato_prossimo][best_next_action]
            td_delta = td_target - Q[stato_corrente][action]
            Q[stato_corrente][action] += alpha * td_delta

            if done:
                break

            stato_corrente = stato_prossimo
        # fine REPEAT...UNTIL

        # aggiorno lengths
        lengths[episodio] = el

        # aggiorno rewards
        rewards[episodio] = rew

    # fine ciclo for

    # inizializzo la policy p a 0
    p = np.zeros(num_states, dtype="int8")  # Initial policy

    for stato in range(num_states):
        p[stato] = Q[stato].argmax()

    return p, rewards, lengths




def sarsa(problem, episodes, alpha, gamma, expl_func, expl_param):
    """
    Performs the SARSA algorithm for a specific environment
    :param problem: problem
    :param episodes: number of episodes for training
    :param alpha: alpha parameter
    :param gamma: gamma parameter
    :param expl_func: exploration function (epsilon_greedy, softmax)
    :param expl_param: exploration parameter (epsilon, T)
    :return: (policy, rews, ep_lengths): final policy, rewards for each episode [array], length of each episode [array]
    """
    num_states = problem.observation_space.n
    num_action = problem.action_space.n

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(num_action))

    # The policy we're following
    policy = expl_func(Q, num_action, expl_param)

    # array di lunghezza episodes
    rewards = np.zeros(episodes, dtype="int16")
    lengths = np.zeros(episodes, dtype="int16")

    for episodio in range(episodes):

        stato_corrente = problem.reset()  # Reset the environment
        el = 0
        rew = 0

        # scelgo l'azione da fare
        action_probs = policy(stato_corrente)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        # inzio REPEAT...UNTIL
        while True:

            # eseguo l'azione scelta
            stato_prossimo, r, done, ep_step = problem.step(action)
            el += 1
            rew += r

            next_action_probs = policy(stato_prossimo)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)

            # TD Update
            td_target = r + gamma * Q[stato_prossimo][next_action]
            td_delta = td_target - Q[stato_corrente][action]
            Q[stato_corrente][action] += alpha * td_delta

            if done:
                break

            stato_corrente = stato_prossimo
            action = next_action

        # fine REPEAT...UNTIL

        # aggiorno lengths
        lengths[episodio] = el

        # aggiorno rewards
        rewards[episodio] = rew

    # fine ciclo for

    # inizializzo la policy p a 0
    p = np.zeros(num_states, dtype="int8")  # Initial policy

    for stato in range(num_states):
        p[stato] = Q[stato].argmax()

    return p, rewards, lengths
    # return np.asarray(p), np.asarray(rewards), np.asarray(lengths)

