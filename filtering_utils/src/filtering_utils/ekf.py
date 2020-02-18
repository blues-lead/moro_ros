#!/usr/bin/env python
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Odometry
import rospy
import pickle
import signal
#import pdb
#import warnings
#warnings.filterwarnings('error')

# Hello testing testing -Roope
# Testing continues -Roope
# Testing -Roope

class EKF:
    def __init__(self, state_vector_size, control_size, measurement_size):
        self.state_vector = np.zeros((state_vector_size, 1))
        self.cov_matrix = 1000. * np.identity(state_vector_size)
        self.q = np.zeros((control_size, control_size))
        self.R = np.zeros((measurement_size, measurement_size))
        self.motion_j_state = np.zeros((state_vector_size, state_vector_size))
        self.motion_j_noise = np.zeros((state_vector_size, control_size))
        self.obs_j_state = np.zeros((measurement_size, state_vector_size))
        self.Q = np.zeros((state_vector_size, state_vector_size))
        self.beacons = {1:[7.3, 3.0], 2:[1,1],3:[9,9],4:[1,8],5:[5.8,8]}
        #
        self.control = np.zeros((2,1))
        self.Z = np.zeros((2,1))
        #self.v_sigma = [] # for sampling sigma
        #self.w_sigma = [] # for sampling sigma
        self.prev_time_stamp = 0 # keeping the last time stamp
        #self.prev_state = np.array((3,1))
        from nav_msgs.msg import Odometry
        self.gt = rospy.Subscriber('base_pose_ground_truth', Odometry, self.initialize_state_vector) # Initializing state_vector with ground truth
        #
        self.state_data_history = []
        self.ground_truth_state_history = []
        self.odometry_history = []
        self.count = 400
        self.saved = False
        self.cov_parameters_history = []
        signal.signal(signal.SIGINT, self.save_before_close)
        signal.signal(signal.SIGTERM, self.save_before_close)

    def save_before_close(self,signum, free):
        with open('ground_truth.pickle', 'wb') as file:
            pickle.dump(self.ground_truth_state_history,file)
        with open('states.pickle','wb') as file:
            pickle.dump(self.state_data_history,file)
        with open('cov_params.pickle','wb') as file:
            pickle.dump(self.cov_parameters_history,file)

    def initialize_state_vector(self, msg): # Function for initializing state_vector
        #print("initialize state", self.state_vector.shape)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = euler_from_quaternion([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])[2]
        self.state_vector[0] = x
        self.state_vector[1] = y
        self.state_vector[2] = self.wrap_to_pi(theta)
        self.prev_time_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs*(10**-9)
        self.gt.unregister() # unregister subscriber. Function is implemented only once.

    def predict(self, odometry): # odometry added by me
        #TODO determine q-matrix
        #print('predict state', self.state_vector.shape)

        # Get v,w from odometry msg
        w = odometry.twist.twist.angular.z
        v = odometry.twist.twist.linear.x
        self.dt = (odometry.header.stamp.secs + odometry.header.stamp.nsecs*(10**-9))-self.prev_time_stamp
        #print(self.dt)
        #pdb.set_trace()
        #
        # get timestamp
        self.prev_time_stamp = odometry.header.stamp.secs + odometry.header.stamp.nsecs*(10**-9)
        #print('Seconds gone is', self.dt)
        #
        # form internal control vector
        self.control = np.array(([v,w]))
        #
        # determine q-matrix aka process noise
        self.q = np.array(([0.1, 0],[0,.1])) #FIXME FOR TEST PURPOSES [0.04, 0],[0,0.001]
        #
        self.propagate_state()
        self.calculate_cov()

    def update(self, msg): #

        self.cur_id = self.beacons[msg.ids[0]] # coordinates of current transmitter
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2])
        #print(quaternion_from_euler(0,0,theta))
        self.observation_jacobian_state_vector()
        
        floor = self.cov_matrix.dot(self.obs_j_state.transpose()).astype(np.float32)
        
        bottom = (self.obs_j_state.dot(self.cov_matrix).dot(self.obs_j_state.transpose()) + np.eye(2)*0.01).astype(np.float32)

        self.K = floor.dot(np.linalg.inv(bottom)) # K is 3x2
        expected_meas = self.measurement_model(self.state_vector)
        new_meas = self.measurement_model([pos_x, pos_y, theta])
        
        tempterm = np.array(([new_meas[0] - expected_meas[0], [new_meas[1] - expected_meas[1]]]))
       
        self.state_vector = self.state_vector + self.K.dot(tempterm)
        self.cov_matrix = (np.eye(3) - self.K.dot(self.obs_j_state)).dot(self.cov_matrix)
        print(self.state_vector)




    def propagate_state(self):
        if self.control[1] != 0:
            term = self.control[0]/self.control[1]
            x = self.state_vector[0] - term*np.sin(self.state_vector[2])+ term*np.sin(self.state_vector[2]+self.control[1]*self.dt)
            y = self.state_vector[1] + term*np.cos(self.state_vector[2])- term*np.cos(self.state_vector[2]+self.control[1]*self.dt)
            theta = self.state_vector[2] + self.control[1]*self.dt
            theta = self.wrap_to_pi(theta)

        else:
            term = self.control[0]
            x = self.state_vector[0] + self.control[0]*np.cos(self.state_vector[2])*self.dt #self.control[0]*self.dt
            y = self.state_vector[1] + self.control[0]*np.sin(self.state_vector[2])*self.dt #self.control[0]*self.dt
            theta = self.state_vector[2]
            
        self.state_vector = np.array([x,y,theta])
        


    def measurement_model(self,state):
        x = state[0]
        y = state[1]
        theta = state[2]
        px = self.cur_id[0]
        py = self.cur_id[1]

        r = np.sqrt((px-x)**2 + (py-y)**2)      #Distance
        #phi = np.arctan2(py-y, px-x) - theta    #Bearing
        phi = np.arctan((py - y)/(px - x)) - theta 

        self.Z[0] = r
        self.Z[1] = phi 
        return self.Z



    def calculate_cov(self):
        self.Q = self.motion_j_noise.dot(self.q).dot(self.motion_j_noise.transpose())
        self.cov_matrix = self.motion_j_state.dot(self.cov_matrix).dot(self.motion_j_state.transpose()) + self.Q

        self.motion_jacobian_state_vector()
        self.motion_jacobian_noise_components()

    def motion_jacobian_state_vector(self):
        if self.control[1] != 0:
            x = self.state_vector[0]
            y = self.state_vector[1]
            theta = self.state_vector[2]
            v = self.control[0]
            w = self.control[1]
            dt = self.dt
            dv = 0
            dw = 0
            row1term1 = 1
            row1term2 = 0
            row1term3 = row1term3 = (np.cos(theta + dt*(dw + w))*(dv+v))/(dw + w) \
                - (np.cos(theta)*(dv+v))/(dw + w)

            row2term1 = 0
            row2term2 = 1
            row2term3 = (np.sin(theta + dt*(dw + w))*(dv+v))/(dw + w) \
                - (np.sin(theta)*(dv+v))/(dw + w)

            row3term1 = 0
            row3term2 = 0
            row3term3 = 1

        else:
            row1term3 = 0
            row2term3 = 0
        self.motion_j_state = np.array(([1,0,row1term3],[0,1,row2term3],[0,0,1]))

    def motion_jacobian_noise_components(self):
        if self.control[1] != 0: # if angular velocity is not zero
            x = self.state_vector[0]
            y = self.state_vector[1]
            theta = self.state_vector[2]
            v = self.control[0]
            w = self.control[1]
            dt = self.dt
            dv = 0
            dw = 0

            dvpv = (dv + v)
            dwpw = (dw + w)

            row1term1 = (np.sin(theta + dt*dwpw)/dwpw) - (np.sin(theta)/dwpw)
            row2term1 = (np.cos(theta)/dwpw) - (np.cos(theta + dt*dwpw)/dwpw)
            row3term1 = 0

            row1term2 = (np.sin(theta)*dvpv)/(dwpw**2) - ((np.sin(theta + dt*dwpw)*dvpv)/(dwpw**2)) + \
                ((dt*np.cos(theta + dt*dwpw)*dvpv)/dwpw)
            row2term2 = ((np.cos(theta + dt*dwpw)*dvpv)/(dwpw**2)) - ((np.cos(theta)*dvpv)/(dwpw**2)) + \
                ((dt*np.sin(theta + dt*dwpw)*dvpv)/dwpw)
            row3term2 = dt


        else:
            row1term1 = self.dt
            row1term2 = 0
            row2term1 = self.dt
            row2term2 = 0
            row3term1 = 0
            row3term2 = 0
        self.motion_j_noise = np.array(([row1term1, row1term2],[row2term1,row2term2],[row3term1,row3term2]))

    def observation_jacobian_state_vector(self):
        row1term1 = (self.state_vector[0] - self.cur_id[0])/np.sqrt((self.state_vector[0] - self.cur_id[0])**2 + (self.state_vector[1] - self.cur_id[1])**2) #checked
        row1term2 = (self.state_vector[1] - self.cur_id[1])/np.sqrt((self.state_vector[0] - self.cur_id[0])**2 + (self.state_vector[1] - self.cur_id[1])**2) #checked
        row1term3 = 0
        row2term1 = (self.cur_id[1] - self.state_vector[1]) / ((self.cur_id[0] - self.state_vector[0])**2 + (self.cur_id[1] - self.state_vector[1])**2) #checked
        row2term2 = -1/((((self.cur_id[1]-self.state_vector[1])**2)/(self.cur_id[0]-self.state_vector[0]))+(self.cur_id[0]- self.state_vector[0])) #checked
        row2term3 = -1
        self.obs_j_state = np.array(([row1term1, row1term2, row1term3],[row2term1,row2term2,row2term3]))

    def print_initials(self):
        pass

    def wrap_to_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def save_data_for_analysis(self, msg):
        gtx = msg.pose.pose.position.x
        gty = msg.pose.pose.position.y
        gt_theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2])
        #
        ptx = self.state_vector[0][0]
        pty = self.state_vector[1][0]
        pt_theta = self.state_vector[2][0]
        #
        covx = self.cov_matrix[0][0]
        covy = self.cov_matrix[1][1]
        cov_theta = self.cov_matrix[2][2]
        self.state_data_history.append([ptx,pty,pt_theta])
        self.ground_truth_state_history.append([gtx,gty,gt_theta])
        self.cov_parameters_history.append([covx,covy,cov_theta])
        '''elif len(self.ground_truth_state_history) == self.count and self.saved == False:
            with open('ground_truth.pickle', 'wb') as file:
                pickle.dump(self.ground_truth_state_history,file)
            with open('states.pickle','wb') as file:
                pickle.dump(self.state_data_history,file)
            self.saved = True
            print("All data saved")'''

