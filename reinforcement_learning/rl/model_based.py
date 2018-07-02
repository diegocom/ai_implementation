"""
Model free algorithm for solving MDPs
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


def model_based(problem, episodes, ep_limit, vmaxiters, gamma, delta):
    """
    Performs the model-based algorithm for a specific environment
    :param problem: problem
    :param episodes: number of episodes for training
    :param ep_limit: limit to episode length
    :param vmaxiters: max iterations allowed for VI
    :param gamma: gamma value
    :param delta: delta value
    :return: (policy, rews, ep_lengths): final policy, rewards for each episode [array], length of each episode [array]
    """

    N_STATI = problem.observation_space.n
    N_AZIONI = problem.action_space.n
    rewards= np.zeros(episodes)
    lengths = np.zeros(episodes)
    T1 = np.zeros((N_STATI,N_AZIONI,N_STATI))
    R1 = np.zeros((N_STATI,N_AZIONI,N_STATI))
    p1 = np.random.choice(N_AZIONI, N_STATI)

    #print("Creo policy random: ", p)
    for e in range(episodes):
        s = problem.reset()
        el = 0
        rew = 0
        lista_tuple = []
        
        for _ in range(ep_limit):
            #print("Stato: ", s)
            #print("Eseguo azione p[s]", p[s])

            sp, r, d, _ = problem.step(p1[s])  # Execute a step
   
            R1[s, p1[s], sp] = r

            rew += r
            el += 1
            lista_tuple.append((s,p1[s],sp,r))
            if d or el == ep_limit:  # If d == True, the episode has reached a terminal state
                break
            s = sp
   

        countNT = np.zeros((N_STATI,N_AZIONI,N_STATI))
        countDT = np.zeros((N_STATI, N_AZIONI))

        for tupla in lista_tuple:
           s = tupla[0]
           a = tupla[1]
           s1 = tupla[2]
           r = tupla[3]
           countNT[s, a, s1] = countNT[s, a, s1] + 1
           countDT[s, a] = countDT[s, a] + 1
           if(countDT[s, a] != 0):
               T1[s, a, s1] = countNT[s, a, s1] / countDT[s, a]
        
       
        rewards[e] = rew
        lengths[e] = el
        problem.T = T1
        problem.R = R1
        p1 = value_iteration(problem, vmaxiters, gamma, delta)



    return p1, rewards, lengths
