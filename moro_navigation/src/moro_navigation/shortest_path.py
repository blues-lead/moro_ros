#!/usr/bin/env python
import numpy as np
from collections import deque
from MY_UTILS import PriorityQueue, Graph

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
    graph = Graph(graph)
    frontier = PriorityQueue()
    frontier.push(start,0)
    cur_node = start
    parents = {}
    cost_so_far = {}
    cost_so_far[start] = 0
    path = {}
    path[start] = None
    while not frontier.isempty():
        cur_id = frontier.pop()
        if graph.isGoalState(cur_id,goal):
            break
        for node in graph.get_successors(cur_id[0]):
            new_cost = cost_so_far[cur_id[0]] + node[1]
            if node[0] not in cost_so_far or new_cost < cost_so_far[node[0]]:
                cost_so_far[node[0]] = new_cost
                frontier.push(node[0],new_cost)
                path[node[0]] = cur_id[0]
    sequence = []
    while goal in path.keys():
        sequence.append(goal)
        goal = path[goal]
    return sequence[::-1]
    #return deque()


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
    graph = Graph(graph)
    frontier = PriorityQueue()
    frontier.push(start,0)
    cost_so_far = {}
    cost_so_far[start] = 0
    path = {}
    path[start] = None
    while not frontier.isempty():
        cur_id = frontier.pop()
        if graph.isGoalState(cur_id,goal):
            break
        for node in graph.get_successors(cur_id[0]):
            new_cost = cost_so_far[cur_id[0]] + node[1]
            if node[0] not in cost_so_far or new_cost < cost_so_far[node[0]]:
                cost_so_far[node[0]] = new_cost
                priority = new_cost + heuristic(node[0], goal)
                frontier.push(node[0],priority)
                path[node[0]] = cur_id[0]
    sequence = []
    while goal in path.keys():
        sequence.append(goal)
        goal = path[goal]
    return sequence[::-1]
    #return deque()


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
    # TODO IF COSTS ARE ALREADY CALCULATED THEN RETURN THEM
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
