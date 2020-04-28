#!/usr/bin/env python
import numpy as np
from utils import decompose, distance, wrap_to_pi
from path_smoothing import smooth_path

class PathFollower(object):
    """Implements path following using feedback control.
    """
    def __init__(self, pos, vel, time, goal_threshold=0.2):
        """Initialize the PathFollower.

        Args:
            pos (ndarray): Positions size (n,2)
            vel (ndarray): Velocities size (n,2)
            time (ndarray): Time since start size (n,)
            goal_threshold (float): Threshold in meters for when the goal is
                deemed reached. Default 0.2.
        """
        super(PathFollower, self).__init__()
        self._goal_reached = False
        self._pos = pos
        self._vel = vel
        self._time = time
        self._goal_threshold = goal_threshold
        self._curvature = map(self._get_curvature, range(len(time)))
        # Start indexing from 1, because velocity at index 0 is 0
        self._idx = 1

    def _get_curvature(self, idx):
        """Get the curvature of the path at the given index.

        Args:
            idx (int): Index of point in path at which to get curvature

        Returns:
            float: Curvature at the given index
        """
        try:
            # TODO
            x = self._vel[idx][0]
            y = self._vel[idx][1]
            a = np.sqrt(x**2+y**2)
            d = np.array((x/a, y/a)).T
            n = np.array((-y/a, x/a)).T

            x_n = self._vel[idx+1][0]
            y_n = self._vel[idx+1][1]
            a_n = np.sqrt(x_n**2+y_n**2).T
            d_n = np.array((x_n/a_n, y_n/a_n)).T

            time = self._time[idx]
            if  time == 0:
                time = 0.001

            d_dot = (d_n-d)/time

            curvature = np.dot(d_dot,n)/a

            return curvature
        except IndexError:
            return 0.

    def _get_nearest(self, pose, window=2):
        """Get the index of the path point closest to the given pose. Updates
        the current running index and whether the goal has been reached.

        Args:
            pose (ndarray): Pose with size (3,) or a position with size (2,)
            window (int, optional): Window size for how many indeces are
                considered ahead the current running index. Default 2.

        Returns:
            int: The nearest index
        """
        min_idx = self._idx
        max_idx = np.minimum(self._pos.shape[0], self._idx + window)
        idx = range(min_idx, max_idx)
        nearest = idx[np.argmin(distance(pose[:2], self._pos[idx]))]
        self._idx = nearest

        # Check if goal has been reached
        if distance(pose[:2], self._pos[-1]) <= self._goal_threshold:
            self._goal_reached = True

        return nearest

    def _get_desired_pose(self, idx):
        """Get the desired pose at the given path index. The desired heading
        is calculated from the velocity.

        Args:
            idx (int): Index of point in path

        Returns:
            ndarray: The desired pose at the given index
        """
        # TODO
        x,y = self._pos[idx]
        vx,vy = self._vel[idx]
        o = np.arctan(vy/vx)

        return np.array((x,y,o))

    def _get_transform(self, idx):
        """Get a matrix to transform the error into the coordinate system of
        the desired pose.

        Args:
            idx (int): The index of the desired point in path

        Returns:
            ndarray: The transformation matrix, size (3,3)
        """
        heading = np.arctan2(self._vel[idx, 1], self._vel[idx, 0])
        sin_h = np.sin(heading)
        cos_h = np.cos(heading)
        return np.array([[cos_h, sin_h, 0],
                         [-sin_h, cos_h, 0],
                         [0, 0, 1]])

    @property
    def goal_reached(self):
        """Determine whether the goal has been reached.

        Returns:
            bool: Returns True if the goal has been reached
        """
        return self._goal_reached

    def get_control(self, pose):
        """Calculate the values for the feedback control.

        Args:
            pose (ndarray): The current pose of the mobile base

        Returns:
            tuple: Linear and angular velocity
        """
        # TODO
        # linear = angular = None
        idx = self._get_nearest(pose)
        vel = np.linalg.norm(self._vel[idx])

        a = -2
        b = -2
        k2 = a*b
        k3 = -(a*b)
        ss = np.array(((0,vel),(-k2,-k3)))

        _, y_h, o_h = self._get_desired_pose(idx)
        xe = 0
        ye = pose[1] - y_h
        oe = pose[2] - o_h
        e = np.matmul(self._get_transform(idx), np.array((xe,ye,oe)))[1:3]

        u = np.matmul(ss,e)
        linear = u[0]
        angular = u[1]

        return linear, angular

if __name__ == "__main__":
    p = np.array([[4.26271167, 1.90129996],
        [6.97728825, 2.16089496],
        [5.94148132, 0.3774518],
        [4.35734741, 0.57725099],
        [2.79453604, 1.22789253],
        [3.28322137, 3.22574976],
        [4.28552018, 2.78609033],
        [7.33380168, 4.0844386],
        [9.07464646, 1.85209234],
        [9.4966894, 7.2826944],
        [9.35290857, 9.19987348],
        [5.27908823, 9.37571584],
        [7.32593449, 8.94651292]])

    sp = smooth_path(p)

    Pf = PathFollower(sp[0],sp[1],sp[4],0.2)
    Pf.get_control((0,0,0))
