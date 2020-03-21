#!/usr/bin/env python
import numpy as np
from collections import deque

# Values for node status
VIRGIN = 0
ACTIVE = 1
DEAD = 2


def dijkstra(graph, start, goal):
    """Plan a path from start to goal using Dijkstra's algorithm.

    Args:
        graph (ndarray): An adjacency matrix, size (n,n)
        start (int): The index of the start node
        goal (int): The index of the goal node

    Returns:
        deque: Indices of nodes along the shortest path
    """
    # TODO
    return deque()


def astar(graph, start, goal, heuristic):
    """Plan a path from start to goal using A* algorithm.

    Args:
        graph (ndarray): An adjacency matrix, size (n,n)
        start (int): The index of the start node
        goal (int): The index of the goal node
        heuristic (ndarray): The heuristic used for expanding the search

    Returns:
        deque: Indices of nodes along the shortest path
    """
    # TODO
    return deque()


def dynamic_programming(graph, start, goal):
    """Plan a path from start to goal using dynamic programming. The goal node
    and information about the shortest paths are saved as function attributes
    to avoid unnecessary recalculation.

    Args:
        graph (ndarray): An adjacency matrix, size (n,n)
        start (int): The index of the start node
        goal (int): The index of the goal node

    Returns:
        deque: Indices of nodes along the shortest path
    """
    distances = np.full((graph.shape[0],1), fill_value=float("Inf"))
    predcs = np.zeros((graph.shape[0],1))
    distances[start] = 0
    for i in range(len(graph)): # performing 18 times
        for row in range(len(graph)-1):
            successors = np.nonzero(graph[row,:])[0]
            for node in successors:#get_successors(row, ad_matrix):
                #breakpoint()
                if distances[row] != float("Inf") and distances[row] + graph[row, node] < distances[node]:
                    distances[node] = distances[row] + graph[row,node]
                    predcs[node] = row
    path = [goal]
    while goal != start:
        #breakpoint()
        van = predcs[int(goal)][0]
        path.append(int(van))
        goal = van
    path = np.array(path)
    return path[::-1]
    #return deque()
