import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats
import rospy
from tf.transformations import euler_from_quaternion
import pickle
import signal

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PF:
    def __init__(self, n, range_x, range_y):
        self.N = n
        self.rangeX = range_x
        self.rangeY = range_y
        self.particles = np.empty((self.N, 3))
        self.weights = np.empty((self.N))
        self.create_particles()
        self.prev_time_stamp = 0
        self.initialized = False
        self.dt = 0
        self.control = np.empty((2,1))
        self.beacons = {1:[7.3, 3.0], 2:[1,1],3:[9,9],4:[1,8],5:[5.8,8]}
        self.R = 0.1
        self.GT_POS = []
        self.mu = np.empty((1,3))

        from nav_msgs.msg import Odometry
        self.gt = rospy.Subscriber('base_pose_ground_truth', Odometry, self.initialize)
        self.check = rospy.Subscriber('base_pose_ground_truth', Odometry, self.get_gt)

        self.state_data_history = []
        self.ground_truth_state_history = []
        self.odometry_history = []
        signal.signal(signal.SIGINT, self.save_before_close)
        signal.signal(signal.SIGTERM, self.save_before_close)

    def save_before_close(self,signum, free):
        #pass
        with open('ground_truth_pf.pickle', 'wb') as file:
            pickle.dump(self.ground_truth_state_history,file)
        with open('states_pf.pickle','wb') as file:
            pickle.dump(self.state_data_history,file)   

    def initialize(self, msg):
        self.prev_time_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs*(10**-9)
        self.gt.unregister() # unregister subscriber. Function is implemented only once.
        self.create_particles()
        self.initialized = True
        #self.gt.unregister()
    
    def get_gt(self, msg):
        # Get ground truth for debugging purposes
        theta = msg.pose.pose.orientation
        theta = euler_from_quaternion([theta.x,theta.y,theta.z, theta.w])[2]
        self.GT_POS = [msg.pose.pose.position.x, msg.pose.pose.position.y, theta]


    def create_particles(self):
        self.particles[:,0] = uniform(0,self.rangeX,size=self.N)
        self.particles[:,1] = uniform(0,self.rangeY,size=self.N)
        self.particles[:,2] = uniform(0, 2*np.pi, size=self.N)
        self.weights.fill(1./self.N)

    def predict(self, odometry):
        if self.initialized == False:
            return
        w = odometry.twist.twist.angular.z
        v = odometry.twist.twist.linear.x
        #print("Odometry (v,w):", v, w)
        self.dt = (odometry.header.stamp.secs + odometry.header.stamp.nsecs*(10**-9))-self.prev_time_stamp # timestamps
        self.prev_time_stamp = odometry.header.stamp.secs + odometry.header.stamp.nsecs*(10**-9) # timestamps
        self.control[0] = v
        self.control[1] = w
        self.q = np.array(([0.4**2, 0],[0,.001**2]))
        self.propagate_state()
        #print(np.mean(self.particles[:,0]), np.mean(self.particles[:,1]))

    def propagate_state(self):
        std = [0.4**2, 0.001**2]
        N = len(self.particles)

        # move in the (noisy) commanded direction
        distL = (self.control[0] * self.dt) + (np.random.randn(N) * std[0]) # 0,05 linear command noise
        distA = (self.control[1] * self.dt) + (np.random.randn(N) * std[1])
        if self.control[1] != 0:
            term = (self.control[0] + np.random.randn(1)[0]*std[0])/(self.control[1] + np.random.randn(1)[0]*std[1])
            self.particles[:,0] = self.particles[:,0] - term*np.sin(self.particles[:,2])+ term*np.sin(self.particles[:,2]+(self.dt * distA))
            self.particles[:,1] = self.particles[:,1] + term*np.cos(self.particles[:,2])- term*np.cos(self.particles[:,2]+(self.dt * distA))
            self.particles[:, 2] += self.control[1] + (np.random.randn(N) * std[1])*self.dt # 0,04 angular command noise
            self.particles[:, 2] %= 2 * np.pi

        else:
            term = self.control[0] + np.random.randn(1)[0]*std[0]
            self.particles[:,0] = self.particles[:,0] + self.control[0]*np.cos(self.particles[:,2])*self.dt + (np.random.randn(N) * std[0]) #self.control[0]*self.dt
            self.particles[:,1] = self.particles[:,1] + self.control[0]*np.sin(self.particles[:,2])*self.dt + (np.random.randn(N) * std[0]) #self.control[0]*self.dt
            #theta = self.wrap_to_pi(self.state_vector[2])
            
        #self.state_vector = np.array([x,y,theta])
    def update(self,msg):
        cur_beacons = {}
        if self.initialized == False:
            return
        for z in msg:
            cur_beacons[z.ids[0]] = [z.pose.position.x, z.pose.position.y] # beacons now seen by the robot
        self.weights.fill(1.)

        for id in cur_beacons.keys():
            #print(id, cur_beacons[id])
            measurement = np.linalg.norm(cur_beacons[id])
            distance = np.linalg.norm(self.particles[:,0:2] - self.beacons[id], axis=1)+np.random.randn(1)[0]
            self.weights *= scipy.stats.norm(distance, self.R).pdf(measurement)
        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize
        if self.neff < 0.3:
            self.resample()
        self.estimate()
    
    def estimate(self):
        pos = self.particles[:,0:2]
        self.mu = np.average(pos, weights=self.weights, axis=0)
        print('Estimate:',self.mu[1])

    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random(self.N))
        
        # resample according to indexes
        self.particles = self.particles[indexes]
        self.weights = self.weights[indexes]
        self.weights /= np.sum(self.weights) # normalize

    def neff(self):
        return 1. / np.sum(np.square(self.weights))

    def wrap_to_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def save_data_for_analysis(self, msg):
        self.estimate()
        gtx = msg.pose.pose.position.x
        gty = msg.pose.pose.position.y
        gt_theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2])
        #
        ptx = self.mu[0]
        pty = self.mu[1]
        #pt_theta = self.mu[2]
        #
        self.state_data_history.append([ptx,pty])
        self.ground_truth_state_history.append([gtx,gty,gt_theta])

    


    


