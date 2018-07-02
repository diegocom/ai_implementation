"""
Search algorithms: IDS, BFS, UC, GREEDY, A*
"""

from timeit import default_timer as timer
from datastructures.fringe import *
import numpy as np

expc = 0
def ids(problem, stype):
    """
    Iterative deepening depth-first search
    :param problem: problem
    :param stype: type of search: graph or tree (dls_gs or dls_ts)
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (time, npexp, maxdepth): elapsed time, number of expansions, max depth reached
    """
    cutoff = True
    depth  = 0
    time   = 0.0
    globals()["expc"] = 0
    while True:
        solution, cutoff_result, stats = stype(problem, depth)
        depth += 1
        time  += stats[0]
        if cutoff_result != cutoff:
            return solution, (time, stats[1], stats[2])


def dls_gs(problem, limit):
    """
    Depth-limited search (graph search)
    :param problem: problem
    :param limit: depth limit budget
    :return: (path, cutoff, stats): solution as a path, cutoff flag and stats
    The stats are a tuple of (time, npexp, maxdepth): elapsed time, number of expansions, max depth reached
    """
    t = timer()
    closed = []
    path, cutoff, expc, maxdepth = rdls_gs(problem, FringeNode(problem.startstate, 0, 0, None), limit, closed)
    return path, cutoff, (timer() - t, expc, maxdepth)


def dls_ts(problem, limit):
    """
    Depth-limited search (tree search)
    :param problem: problem
    :param limit: depth limit budget
    :return: (path, cutoff, stats): solution as a path, cutoff flag and stats
    The stats are a tuple of (time, npexp, maxdepth): elapsed time, number of expansions, max depth reached
    """
    t = timer()
    path, cutoff, expc, maxdepth = rdls_ts(problem, FringeNode(problem.startstate, 0, 0, None), limit)
    return path, cutoff, (timer() - t, expc + maxdepth, maxdepth)


def rdls_gs(problem, node, limit, closed):
    """
    Recursive depth-limited search (graph search version)
    :param problem: problem
    :param node: node to expand
    :param limit: depth limit budget
    :param closed: completely explored nodes
    :return: (path, cutoff, expc, maxdepth): path, cutoff flag, expanded nodes, max depth reached
    """
    global expc
    maxdepth = 1
    cutoff = False
    
    if problem.goalstate == node.state:
        path = build_path(node)
        return path, cutoff, expc, maxdepth

    if limit == 0:
        path = build_path(node)
        return path, True, expc, maxdepth

    if node.state in closed:
        # path = build_path(node)
        # print("FAILURE")
        return None, False, expc, maxdepth

    closed.append(node.state)

    cutoff_occurred = False
    expc += 1

    for action in range(problem.action_space.n):
        child = FringeNode(problem.sample(node.state, action), 1, 0, node)
        path, cutoff_result, expc, maxdepth = rdls_gs(problem, child, limit-1, closed)
        
        maxdepth += 1
        
        if cutoff_result:
            cutoff_occurred = True
        elif path != None and cutoff_result == False:
            return path, cutoff, expc, maxdepth

    if cutoff_occurred:
        return path, True, expc, maxdepth
        
    # print("FAILURE")
    return None, False, expc, maxdepth


def rdls_ts(problem, node, limit):
    """
    Recursive depth-limited search (tree search version)
    :param problem: problem
    :param node: node to expand
    :param limit: depth limit budget
    :return: (path, cutoff, expc, maxdepth): path, cutoff flag, expanded nodes, max depth reached
    """
    global expc
    maxdepth = 1
    cutoff = False
    
    if problem.goalstate == node.state:
        path = build_path(node)
        return path, cutoff, expc, maxdepth

    if limit == 0:
        path = build_path(node)
        return path, True, expc, maxdepth
    
    cutoff_occurred = False
    for action in range(problem.action_space.n):
        child = FringeNode(problem.sample(node.state, action), 1, 0, node)
        path, cutoff_result, expc, maxdepth = rdls_ts(problem, child, limit-1)
        
        expc += 1
        maxdepth += 1
        
        if cutoff_result:
            cutoff_occurred = True
        elif path != None and cutoff_result == False:
            return path, cutoff, expc, maxdepth

    if cutoff_occurred:
        return path, True, expc, maxdepth
        
    # print("FAILURE")
    return None, False, expc, maxdepth


def bfs(problem, stype):
    """
    Breadth-first search
    :param problem: problem
    :param stype: type of search: graph or tree (graph_search or tree_search)
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (time, expc, maxstates): elapsed time, number of expansions, max states in memory
    """
    t = timer()
    path, stats = stype(problem, QueueFringe())
    return path, (timer() - t, stats[0], stats[1])


def ucs(problem, stype):
    """
    Uniform-cost search
    :param problem: problem
    :param stype: type of search: graph or tree (graph_search or tree_search)
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (time, expc, maxstates): elapsed time, number of expansions, max states in memory
    """
    def g(n, c):
        """
        Path cost function
        :param n: node
        :param c: child state of 'n'
        :return: path cost from root to 'c'
        """
        if n is None:
            return 0
        return n.pathcost + 1

    t = timer()
    path, stats = stype(problem, PriorityFringe(), g)
    return path, (timer() - t, stats[0], stats[1])


def greedy(problem, stype):
    """
    Greedy best-first search
    :param problem: problem
    :param stype: type of search: graph or tree (graph_search or tree_search)
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (time, expc, maxstates): elapsed time, number of expansions, max states in memory
    """
    def f(p1, p2):
        """
        Computes the L1 norm distance between two n-dimensional points
        :param p1: first point
        :param p2: second point
        :return: L1 norm distance value
        """
        return l1_norm(problem.state_to_pos(p2.state), problem.state_to_pos(problem.goalstate))

    t = timer()
    path, stats = stype(problem, PriorityFringe(), f)
    return path, (timer() - t, stats[0], stats[1])


def astar(problem, stype):
    """
    A* best-first search
    :param problem: problem
    :param stype: type of search: graph or tree (graph_search or tree_search)
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (time, expc, maxstates): elapsed time, number of expansions, max states in memory
    """
    
    def g(n, c):
        """
        Path cost function
        :param n: node
        :param c: child state of 'n'
        :return: path cost from root to 'c'
        """
        if n is None:
            return 0
        return n.pathcost + 1

    def f(p1, p2):
        """
        Computes the L1 norm distance between two n-dimensional points
        :param p1: first point
        :param p2: second point
        :return: L1 norm distance value
        """
        if p1 is None:
            return l1_norm(problem.state_to_pos(p2.state), problem.state_to_pos(problem.goalstate))
        return l1_norm(problem.state_to_pos(p2.state), problem.state_to_pos(problem.goalstate)) + g(p1, 0)

    t = timer()
    path, stats = stype(problem, PriorityFringe(), f)
    return path, (timer() - t, stats[0], stats[1])


def graph_search(problem, fringe, f=lambda n, c: 0):
    """
    Graph search
    :param problem: problem
    :param fringe: fringe data structure
    :param f: node evaluation function
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (expc, maxstates): number of expansions, max states in memory
    """
    node = FringeNode(problem.startstate, 0, 0, None)
    fringe.add(node)
    closed = []

    expc = 0
    maxdepth = 1

    while True:
        
        if fringe.is_empty(): # print("FAILURE")
            return None, (expc, maxdepth)
        
        node = fringe.remove()
        expc += 1
        
        if problem.goalstate == node.state: # print("GOAL")
            path = build_path(node)
            return path, (expc, maxdepth)
              
        closed.append(node.state)

        for action in range(problem.action_space.n):
            child = FringeNode(problem.sample(node.state, action), node.pathcost + 1, 0, node)
            child.value = f(node, child)

            if (child.state not in fringe) and (child.state not in closed):
                fringe.add(child)
                maxdepth += 1
            elif child.state in fringe and child.value < fringe[child.state].value:
                fringe.replace(child)
                maxdepth += 1


def tree_search(problem, fringe, f=lambda n, c: 0):
    """
    Tree search
    :param problem: problem
    :param fringe: fringe data structure
    :param f: node evaluation function
    :return: (path, stats): solution as a path and stats
    The stats are a tuple of (expc, maxstates): number of expansions, max states in memory
    """
    node = FringeNode(problem.startstate, 0, 0, None)
    node.value = f(None, node)
    fringe.add(node)

    expc = 0
    maxdepth = 2

    while True:
        
        if fringe.is_empty(): # print("FAILURE")
            return None, (expc, maxdepth)
        
        node = fringe.remove()
        expc += 1

        if problem.goalstate == node.state: # print("GOAL")
            path = build_path(node)
            maxdepth = maxdepth - expc
            return path, (expc, maxdepth)
              
        for action in range(problem.action_space.n):
            child = FringeNode(problem.sample(node.state, action), node.pathcost + 1, 0, node)
            child.value = f(node, child)
            fringe.add(child)
            maxdepth += 1


def build_path(node):
    """
    Builds a path going backward from a node
    :param node: node to start from
    :return: path from root to 'node'
    """
    path = []
    while node.parent is not None:
        path.append(node.state)
        node = node.parent
    return tuple(reversed(path))


def l1_norm(p1, p2):
    """
    Computes the L1 norm distance between two n-dimensional points
    :param p1: first point
    :param p2: second point
    :return: L1 norm distance value
    """
    return np.sum(np.abs(np.asarray(p1) - np.asarray(p2)))


def l2_norm(p1, p2):
    """
    Computes the L1 norm distance between two n-dimensional points
    :param p1: first point
    :param p2: second point
    :return: L1 norm distance value
    """
    return np.linalg.norm((np.asarray(p1), np.asarray(p2)))


def chebyshev(p1, p2):
    """
    Computes the Chebyshev distance, between two n-dimensional points
    :param p1: first point
    :param p2: second point
    :return: Chebyshev distance value
    """
    return np.max(np.abs(np.asarray(p1) - np.asarray(p2)))