#!/usr/bin/env python
import numpy as np
from itertools import permutations


def time_expand(graph, nodes, obstruction):
    """Create a time-expanded graph taking into account the given
    dynamic obstruction.

    Args:
        graph (ndarray): The original graph adjacency matrix, size (n,n)
        nodes (ndarray): Node coordinates in the graph, size (n,2)
        obstruction (array-like): Indeces of obstructed nodes, length t

    Returns:
        tuple: The time-expanded graph, size (tn,tn), and node
            coordinates in the new graph, size(tn,2)
    """
    time_steps = len(obstruction)
    exp_size = len(nodes)
    new_size = (time_steps+1)*exp_size
    expanded_graph = np.zeros((new_size,new_size))
    
    for i in range(time_steps):
        new_graph = np.copy(graph)
        tc = i*exp_size
        bc = tc + exp_size
        node_at_time_step = obstruction[i]
        new_graph[:,node_at_time_step] = 0 # remove "to"-columns from the graph
        new_graph[node_at_time_step,:] = 0 # remove "from"-rows from the graph
        expanded_graph[tc:bc, bc:bc+exp_size] = new_graph # Fill upper subdiagonal with A:s
    # Fill bottom-right corner of the expanded graph with the A
    # here the obstruction stops moving
    expanded_graph[new_size - exp_size:new_size,new_size - exp_size:new_size] = new_graph
    new_nodes = np.tile(nodes, ((time_steps+1),1))
    return expanded_graph, new_nodes


def joint_graph(graph, nodes):
    """Create a joint graph for two agents based on the given single
    agent graph

    Args:
        graph (ndarray): The single agent graph adjacency matrix, size (n,n)
        nodes (ndarray): Node coordinates in the graph, size (n,2)

    Returns:
        tuple: The joint graph, size (n^2-n,n^2-n), and node coordinates in the
            joint graph, size (n^2-n,2,2), where the second axis is the two
            agents and the third axis is coordinates
    """
    # TODO
    joint_graph = nodes = None

    return joint_graph, nodes
