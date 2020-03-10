#!/usr/bin/env python
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Odometry
import rospy
import pickle
import signal
import pdb
#import warnings
#warnings.filterwarnings('error')

class EKF:
    def __init__(self, state_vector_size, control_size, measurement_size):
        self.state_vector = np.zeros((state_vector_size, 1))
        self.cov_matrix = 1000. * np.identity(state_vector_size)
        self.q = np.zeros((control_size, control_size))
        self.R = np.zeros((measurement_size, measurement_size))
        self.R = np.diag(([0.1, 0.01]))
        self.motion_j_state = np.zeros((state_vector_size, state_vector_size))
        self.motion_j_noise = np.zeros((state_vector_size, control_size))
        self.obs_j_state = np.zeros((measurement_size, state_vector_size))
        self.Q = np.zeros((state_vector_size, state_vector_size))
        self.beacons = {1:[7.3, 3.0], 2:[1,1],3:[9,9],4:[1,8],5:[5.8,8]}
        self.new_meas = np.empty((1,3))
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
        #self.saved = False
        self.initialized = False
        self.cov_parameters_history = []
        signal.signal(signal.SIGINT, self.save_before_close)
        signal.signal(signal.SIGTERM, self.save_before_close)

    def save_before_close(self,signum, free):
        pass
        # with open('ground_truth.pickle', 'wb') as file:
        #     pickle.dump(self.ground_truth_state_history,file)
        # with open('states.pickle','wb') as file:
        #     pickle.dump(self.state_data_history,file)
        # with open('cov_params.pickle','wb') as file:
        #     pickle.dump(self.cov_parameters_history,file)

    def initialize_state_vector(self, msg): # Function for initializing state_vector
        #print("initialize state", self.state_vector.shape)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = euler_from_quaternion([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w])[2]
        self.state_vector[0] = x
        self.state_vector[1] = y
        self.state_vector[2] = self.wrap_to_pi(theta)
        #print("Inittial state",self.state_vector[0],self.state_vector[1], self.state_vector[2])
        self.prev_time_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs*(10**-9)
        self.gt.unregister() # unregister subscriber. Function is implemented only once.
        self.initialized = True

    def predict(self, odometry): # odometry added by me

        # Get v,w from odometry msg
        w = odometry.twist.twist.angular.z
        # = odometry.twist.twist.linear.x
        vx = odometry.twist.twist.linear.x
        vy = odometry.twist.twist.linear.y
        v = np.sqrt(vx**2 + vy**2)
        if vx < 0:
            v = -v
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
        self.q = np.array(([0.4**2, 0],[0,.001**2])) #FIXME FOR TEST PURPOSES [0.04, 0],[0,0.001]
        #
        self.propagate_state()
        self.calculate_cov()
        #print(self.state_vector)

    def update(self, msg): #
        self.cur_id = self.beacons[msg.ids[0]] # coordinates of current transmitter
        
        # landmark position in robot frame
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        #rng = np.sqrt(pos_x**2 + pos_y**2) # WORKING

        #bearing
        theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2])
        theta = self.process_angle(pos_x, pos_y, theta)
        rng = np.sqrt(self.new_meas[0]**2 + self.new_meas[1]**2) # TESTING
        self.observation_jacobian_state_vector()
        #nominator
        floor = self.cov_matrix.dot(self.obs_j_state.transpose()).astype(np.float32)
        
        #denominator
        # obs_j_state 2x3, cov_matrix 3x3, obs_j_state 3x2
        bottom = (self.obs_j_state.dot(self.cov_matrix).dot(self.obs_j_state.transpose()) + self.R).astype(np.float32) # palce self.R diag(0.1 0.01)

        self.K = floor.dot(np.linalg.inv(bottom)) # K is 3x2

        expected_meas = self.measurement_model(self.state_vector)
        new_meas = np.array(([rng, theta]))

        innovation = np.array(([new_meas[0] - expected_meas[0], new_meas[1] - expected_meas[1]]))

        self.state_vector = self.state_vector + self.K.dot(innovation)
        self.cov_matrix = (np.eye(3) - self.K.dot(self.obs_j_state)).dot(self.cov_matrix)
        print()
        print(self.state_vector)
        #print()
        #print(self.cov_matrix)

    def process_angle(self,x,y,a):
        rot_matrix = np.array(([np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]))
        xy = np.array([x, y]).T
        new_xy = rot_matrix.transpose().dot(xy)
        bearing = np.arctan2(new_xy[1], new_xy[0]) + a
        self.new_meas = np.array(([new_xy[0], new_xy[1], bearing]))
        return bearing

        

    def propagate_state(self):
        #NEW MODEL TEST
        theta =  self.wrap_to_pi(self.state_vector[2])
        w = self.control[1]
        v = self.control[0]
        if np.isclose(w, 0):
            self.state_vector[0] = self.state_vector[0] + (v * np.cos(theta + (w * self.dt) / 2) * self.dt)
            self.state_vector[1] = self.state_vector[1] + (v * np.sin(theta + (w * self.dt) / 2) * self.dt)
            self.state_vector[2] = self.wrap_to_pi((theta + (w * self.dt)))
        else:
            self.state_vector[0] = self.state_vector[0] + ((v / w) * (-np.sin(theta) + np.sin(theta + w * self.dt)))
            self.state_vector[1] = self.state_vector[1] + ((v / w) * (np.cos(theta) - np.cos(theta + w * self.dt)))
            self.state_vector[2] = self.wrap_to_pi((theta + (w * self.dt)))
        print(self.state_vector)

    def measurement_model(self,state):
        x = state[0]
        y = state[1]
        theta = state[2]
        px = self.cur_id[0]
        py = self.cur_id[1]

        r = np.sqrt((px-x)**2 + (py-y)**2)      #Distance
        #phi = np.arctan2(py-y, px-x) - theta    #Bearing
        phi = self.wrap_to_pi(np.arctan2(py-y, px-x)) - self.wrap_to_pi(theta) # TEST

        self.Z[0] = r
        self.Z[1] = self.wrap_to_pi(phi)
        return self.Z



    def calculate_cov(self):
        self.motion_jacobian_state_vector()
        self.motion_jacobian_noise_components()
        self.Q = self.motion_j_noise.dot(self.q).dot(self.motion_j_noise.transpose())
        self.cov_matrix = self.motion_j_state.dot(self.cov_matrix).dot(self.motion_j_state.transpose()) + self.Q

    def motion_jacobian_state_vector(self):
        # TESTING NEW MODEL
        v = self.control[0] 
        w = self.control[1]
        th = self.wrap_to_pi(self.state_vector[2])

        if np.isclose(w, 0):
            self.motion_j_state[0, :] = np.array([1, 0, -v * np.sin(th + (w * self.dt) / 2) * self.dt])
            self.motion_j_state[1, :] = np.array([0, 1, v * np.cos(th + (w * self.dt) / 2) * self.dt])
            self.motion_j_state[2, :] = np.array([0, 0, 1])
        else:
            self.motion_j_state[0, :] = np.array([1, 0, (v / w) * (-np.cos(th) + np.cos(th + w * self.dt))])
            self.motion_j_state[1, :] = np.array([0, 1, (v / w) * (-np.sin(th) + np.sin(th + w * self.dt))])
            self.motion_j_state[2, :] = np.array([0, 0, 1])

        '''if np.isclose(w, 0):
            row1term1 = 1
            row1term2 = 0
            row1term3 = -np.sin(theta + (w * self.dt) / 2) * self.dt*v
            
            row2term1 = 0
            row2term2 = 1
            row2term3 = np.cos(theta + (w * self.dt) / 2) * self.dt*v
            
            row3term1 = 0
            row3term2 = 0
            row3term3 = 1            
        else:
            row1term1 = 1
            row1term2 = 0
            row1term3 = (v / w) * (-np.cos(theta) + np.cos(theta + w * self.dt))
            
            row2term1 = 0
            row2term2 = 1
            row2term3 = (v / w) * (-np.sin(theta) + np.sin(theta + w * self.dt))
            
            row3term1 = 0
            row3term2 = 0
            row3term3 = 1
        self.motion_j_state = np.array(([row1term1,row1term2,row1term3],[row2term1,row2term2,row2term3],[row3term1,row3term2,row3term3]))'''

    def motion_jacobian_noise_components(self):
        #TESTING NEW MODEL
        th = self.wrap_to_pi(self.state_vector[2, 0])
        v = self.control[0]
        w = self.control[1]

        if np.isclose(w, 0):
            self.motion_j_noise[0, :] = np.array([self.dt * np.cos(th + (w * self.dt) / 2),
                                                  -(v / 2) * np.square(self.dt) * np.sin(th + (w * self.dt) / 2)])
            self.motion_j_noise[1, :] = np.array([self.dt * np.sin(th + (w * self.dt) / 2),
                                                  (v / 2) * np.square(self.dt) * np.cos(th + (w * self.dt) / 2)])
            self.motion_j_noise[2, :] = np.array([0, self.dt])
        else:
            sin_th_wt = np.sin(th+w*self.dt)    
            cos_th_wt = np.cos(th + w * self.dt)
            self.motion_j_noise[0, :] = np.array([(1 / w) * (sin_th_wt - np.sin(th)),
                                                  (v / np.square(w)) * (np.sin(th) - sin_th_wt) + (
                                                              (v * self.dt) / w) * cos_th_wt])
            self.motion_j_noise[1, :] = np.array([(1 / w) * (-cos_th_wt + np.cos(th)),
                                                  (v / np.square(w)) * (-np.cos(th) + cos_th_wt) + (
                                                              (v * self.dt) / w) * sin_th_wt])
            self.motion_j_noise[2, :] = np.array([0, self.dt])

        '''if np.isclose(w, 0):
            row1term1 = self.dt * np.cos(theta + (w * self.dt) / 2)
            row1term2 = -(v / 2) * np.square(self.dt) * np.sin(theta + (w * self.dt) / 2)
            
            row2term1 = self.dt * np.sin(theta + (w * self.dt) / 2)
            row2term2 = (v / 2) * np.square(self.dt) * np.cos(theta + (w * self.dt) / 2)
            
            row3term1 = 0
            row3term2 = self.dt
        else:
            stwdt = np.sin(theta+w*self.dt)
            ctwdt = np.cos(theta + w * self.dt)
            row1term1 = (1 / w) * (stwdt - np.sin(theta))
            row1term2 = (v / np.square(w)) * (np.sin(theta) - stwdt) + ((v * self.dt) / w) * ctwdt
            row2term1 = (1 / w) * (-ctwdt + np.cos(theta))
            row2term2 = (v / np.square(w)) * (-np.cos(theta) + ctwdt) + ((v * self.dt) / w) * stwdt
            row3term1 = 0
            row3term2 = self.dt
            self.motion_j_noise = np.array(([row1term1, row1term2],[row2term1, row2term2],[row3term1, row3term2]))'''
            

    def observation_jacobian_state_vector(self):
        row1term1 = (self.state_vector[0] - self.cur_id[0])/np.sqrt((self.state_vector[0] - self.cur_id[0])**2 + (self.state_vector[1] - self.cur_id[1])**2) #checked
        row1term2 = (self.state_vector[1] - self.cur_id[1])/np.sqrt((self.state_vector[0] - self.cur_id[0])**2 + (self.state_vector[1] - self.cur_id[1])**2) #checked
        row1term3 = 0
        row2term1 = (self.cur_id[1] - self.state_vector[1]) / ((self.cur_id[0] - self.state_vector[0])**2 + (self.cur_id[1] - self.state_vector[1])**2) #checked
        row2term2 = -1/((((self.cur_id[1]-self.state_vector[1])**2)/(self.cur_id[0]-self.state_vector[0]))+(self.cur_id[0]- self.state_vector[0])) #checked
        row2term3 = -1 # <=== "WORKING" implementation
        jacobian = [[row1term1, row1term2, row1term3],[row2term1, row2term2, row2term3]] #!
        self.obs_j_state = np.array(jacobian)

    def print_initials(self):
        pass

    def wrap_to_pi(self,angle):
        # return (angle + np.pi) % (2 * np.pi) - np.pi # TEST
        # https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap/15927914
        new_angle = angle
        while (new_angle > np.pi):
            new_angle -= - (np.pi*2)        
        while (new_angle < -np.pi):
            new_angle += (np.pi*2)
        return new_angle  


    def save_data_for_analysis(self, msg):
        gtx = msg.pose.pose.position.x
        gty = msg.pose.pose.position.y
        gt_theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])[2])
        #
        ptx = self.state_vector[0][0]
        pty = self.state_vector[1][0]
        pt_theta = self.state_vector[2][0]
        #
        covx =  np.sqrt(self.cov_matrix[0][0])
        covy = np.sqrt(self.cov_matrix[1][1])
        cov_theta = np.sqrt(self.cov_matrix[2][2])
        self.state_data_history.append([ptx,pty,pt_theta])
        self.ground_truth_state_history.append([gtx,gty,gt_theta])
        self.cov_parameters_history.append([covx,covy,cov_theta])
