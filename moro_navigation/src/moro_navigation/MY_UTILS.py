#!/usr/bin/env python
import numpy as np

class PriorityQueue:
    def __init__(self):
        self._queue = []
        
    def isempty(self):
        if len(self._queue) == 0:
            return True
        else:
            return False
        
    def pop(self):
        if self.isempty():
            return False
        min_val = float("Inf")
        min_ind = 0
        for i in range(len(self._queue)):
            if self._queue[i][1] < min_val:
                min_ind = i
                min_val = self._queue[i][1]
        to_ret = self._queue[min_ind]
        del self._queue[min_ind]
        return to_ret
    
    def push(self,data, priority):
        temp = tuple((data,priority))
        self._queue.append(temp)
    
    def __str__(self):
        return str(self._queue)
    
    def exists(self,val1):
        if val1 in self._queue:
            return True
        else:
            return False
    
class Graph:
    def __init__(self,adj_matrix):
        self.graph = adj_matrix
        
    def get_successors(self, node): # Now searching by row in matrix
        #breakpoint()
        #arr = [tuple((tuple((node,i)),self.graph[node,i])) for i in range(len(self.graph[node,:])) \
        #       if self.graph[node,i] != 0]
        arr = [tuple((i,self.graph[node,i])) for i in range(len(self.graph[node,:])) \
               if self.graph[node,i] != 0]
        return arr
    
    def isGoalState(self,node, goal):
        if node[0] == goal:
            return True
        else:
            return False
        
    def getCost(self, node):
        return node(1)