import gym
import gym_ai_lab
import search.algorithms as search

# ESERCIZIO 1
envname = "SmallMaze-v0"

print("\n----------------------------------------------------------------")
print("\tTREE SEARCH")
print("\tEnvironment: ", envname)
print("----------------------------------------------------------------\n")

env = gym.make(envname)
env.render()

# IDS
solution, stats = search.ids(env, search.dls_ts)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nIDS:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# BFS
solution, stats = search.bfs(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nBFS:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))


# UCS
solution, stats = search.ucs(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nUC:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# A*
solution, stats = search.astar(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nA*:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))




# ESERCIZIO 2
envname = "GrdMaze-v0"

print("\n----------------------------------------------------------------")
print("\tTREE SEARCH")
print("\tEnvironment: ", envname)
print("----------------------------------------------------------------\n")

env = gym.make(envname)
env.render()

# IDS
solution, stats = search.ids(env, search.dls_ts)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nIDS:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# BFS
solution, stats = search.bfs(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nBFS:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# UCS
solution, stats = search.ucs(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nUC:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# A*
solution, stats = search.astar(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nA*:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))




# ESERCIZIO 3
envname = "BlockedMaze-v0"

print("\n----------------------------------------------------------------")
print("\tTREE SEARCH")
print("\tEnvironment: ", envname)
print("----------------------------------------------------------------\n")

env = gym.make(envname)
env.render()
"""
# IDS
solution, stats = search.ids(env, search.dls_ts)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nIDS:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# BFS
solution, stats = search.bfs(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nBFS:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# UCS
solution, stats = search.ucs(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nUC:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))

# A*
solution, stats = search.astar(env, search.tree_search)
if solution is not None:
    solution = [env.state_to_pos(s) for s in solution]
print("\n\nA*:\n----------------------------------------------------------------"
      "\nExecution time: {0}s\nN° of states expanded: {1}\nMax n° of states in memory: {2}\nSolution: {3}".format(
        round(stats[0], 4), stats[1], stats[2], solution))
"""