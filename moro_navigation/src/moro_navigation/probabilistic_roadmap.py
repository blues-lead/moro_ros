# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 20:35:20 2020

@author: Anton
"""
from utils import DotDict
import yaml
#import skimage
from skimage import io
from skimage.draw import line, circle
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils import distance

# The values in the occupancy grid map
OCCUPIED = 100
FREE = 0
UNKNOWN = -1


class ProbabilisticRoadmap(object):
    """Provides probabilistic roadmapping in a given map.

    Attributes:
        graph (ndarray): An adjacency matrix of size (num_nodes,num_nodes)
            consisting of edge costs between nodes.
        nodes (ndarray): Node coordinates of size (num_nodes,2)
    """

    def __init__(self, og, inflation_radius=0.25):
        """Initialize the ProbabilisticRoadmap.

        Args:
            og (nav_msgs/OccupancyGrid): The map to use for roadmapping.
            inflation_radius (float, optional): How much obstacles are inflated
                in the map in meters. Default: 0.25
        """
        super(ProbabilisticRoadmap, self).__init__()

        self.nodes = None
        self.graph = None

        # Unpack the data from the occupancy grid
        self._resolution = og.info.resolution
        self._origin = np.array([og.info.origin.position.x,
                                 og.info.origin.position.y])
        # xmin, ymin = 0,0
        # xmax, ymax = 10,10
        self._xmin = self._origin[0]
        self._xmax = self._origin[0] + og.info.width * self._resolution
        self._ymin = self._origin[1]
        self._ymax = self._origin[1] + og.info.height * self._resolution
        # _og_map 200x200
        self._og_map = np.array(og.data).reshape((og.info.height,
                                                  og.info.width))

        # Inflate the obstacles in the map by inflation_radius
        self._map = self._inflate_map(self._og_map, inflation_radius)

        # Create the graph. This fills out self.nodes and self.graph
        self.create_graph()
        np.random.seed()

    def heuristic_function(self, node1, node2):
        return np.linalg.norm(self.nodes[node1] - self.nodes[node2])

    def _figure_coordinates(self, position):
        """Get map figure coordinates for a position.

        Args:
            position (ndarray): Array of coordinates size (2,) or (n,2).
                For a single position also list or tuple of length 2.

        Returns:
            ndarray: Coordinates of position in map figure. Same size
                as position.
        """
        position = np.array(position)
        
        scaled = np.atleast_2d((position - self._origin) / self._resolution)
        # flip array in left-right direction
        return np.fliplr(scaled).astype(np.uint16).reshape(position.shape)

    def _is_free(self, position):
        """Check whether a position is free in the map.

        Args:
            position (ndarray): A single position to check, size (2,).
                Alternatively list or tuple of length 2.

        Returns:
            bool: Returns True if the position is free.
        """
        index = self._figure_coordinates(position)
        return self._map[tuple(index)] == FREE

    def _inflate_map(self, og, radius):
        """Inflate the obstacles in map by a given radius.

        Args:
            og (ndarray): Array representing an occupancy grid
            radius (double): Inflation radius in meters

        Returns:
            ndarray: Inflated occupancy grid. Same size as og.
        """
        new_map = copy(og)
        shape = og.shape
        new_radius = radius / self._resolution
        
        obstacles = np.nonzero(og == OCCUPIED)
        for i in range(np.size(obstacles[0])):
            x = obstacles[0][i]
            y = obstacles[1][i]
            rr,cc = circle(x,y,new_radius, shape)
            new_map[rr,cc] = OCCUPIED
        return new_map

    def _draw_sample(self):
        """Draw a random sample from the configuration space

        Returns:
            ndarray: Sampled coordinates, size (2,).
        """
        sample = np.random.random_sample(2)*10
        return sample

    def can_connect(self, p1, p2):
        """Check whether the connecting line segment between two points
        is unobstructed.

        Args:
            a (ndarray): Coordinates for first point, size (2,)
            b (ndarray): Coordinates for second point, size (2,)

        Returns:
            bool: Returns True if there are no obstacles between the points.
        """
        dxy = p2 - p1
        if np.isclose(dxy[0],0): # if kx+b doesn't perform
            x = p2[0]
            points_to_check = np.zeros((int(dxy[1]*10+2), 2))
            points_to_check[:,1] = np.linspace(p1[1], p2[1], int(dxy[1]*10)+2)
            points_to_check[:,0] = x
        else:
            rate = dxy[1]/dxy[0]
            b = p1[1] - rate*p1[0]
            rng = np.linalg.norm(dxy)
            x = np.linspace(p1[0], p2[0], int(rng*10)+2)
            y = rate*x + b
            points_to_check = np.zeros((int(rng*10)+2, 2))
            points_to_check[:,0] = x
            points_to_check[:,1] = y
        for point in points_to_check:
            if self._is_free(point) == False:
                return False
        return True


    def create_graph(self):
        """Create the nodes and connections in the graph. Fills out the class
        attributes self.nodes and self.graph with the corresponding values.
        """
        # FIXME do something with seed
        np.random.seed(4)
        amount = 80
        closeness_threshold = 0.8
        i = 0
        self.nodes = np.zeros((amount, 2))
        self.graph = np.zeros((amount, amount))
        while i < amount:
            sample = self._draw_sample()
            if self._is_close(sample, closeness_threshold) == True or not self._is_free(sample):
                continue
            else:
                self.nodes[i,:] = sample.T
                i += 1
        for i in range(self.nodes.shape[0]):
            for j in range(self.nodes.shape[0]):
                node1,node2 = self.nodes[i], self.nodes[j]
                if self.can_connect(node1,node2):
                    if i==j:
                        self.graph[i,j] = 0.1
                    else:
                        if sum(self.graph[i] > 4):
                            continue
                        length = np.linalg.norm(node2-node1)
                        self.graph[i,j] = length
                        self.graph[j,i] = length

        
    def _is_close(self,p, threshold):
       test = sum([np.linalg.norm(p - t) <= threshold for t in self.nodes])
       #print(test)
       if test == 0:
           return False
       else:
           return True
       

    def plot(self, path=[]): # path = None
        """Plot the map, nodes and connections of the ProbabilisticRoadmap

        Args:
            path (list, optional): Highlight the nodes that make up the path
        """
        ax = plt.gca()
        extent = (self._xmin, self._xmax, self._ymin, self._ymax)
        ax.imshow(self._og_map, cmap='Greys', origin='lower', extent=extent)
        ax.imshow(self._map, cmap='Reds', origin='lower',
                  extent=extent, alpha=0.3)
        ax.plot(self.nodes[:, 0], self.nodes[:, 1], 'bo')
        

        source, sink = np.nonzero(self.graph)
        source = self.nodes[source]
        sink = self.nodes[sink]
        lc = LineCollection(np.stack((source, sink), axis=1),
                            linewidths=[1], colors=[(0, 0.75, 1, 1)])
        ax.add_collection(lc)

        for i in range(len(self.nodes)):
            x,y = self.nodes[i,0], self.nodes[i,1]
            plt.text(x,y,str(i))

        ax.set_xlim((self._xmin, self._xmax))
        ax.set_ylim((self._ymin, self._ymax))
#
        if len(path) > 0: # if path:
            path = self.nodes[path]
            ax.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2)