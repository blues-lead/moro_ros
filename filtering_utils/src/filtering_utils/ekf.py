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
        self.state_vector = np.zeros((state_vector_size+10, 1))
        self.cov_matrix = 1000. * np.identity(state_vector_size+10)
        self.q = np.zeros((control_size, control_size))
        self.R = np.zeros((measurement_size, measurement_size))
        self.R = np.diag(([0.1, 0.01]))
        self.motion_j_state = np.zeros((state_vector_size+10, state_vector_size+10))
        self.motion_j_noise = np.zeros((state_vector_size, control_size))
        self.obs_j_state = np.zeros((measurement_size, state_vector_size+10))
        self.Q = np.zeros((state_vector_size, state_vector_size))
        self.beacons = {1:[7.3, 3.0], 2:[1,1],3:[9,9],4:[1,8],5:[5.8,8]}
        self.new_meas = np.empty((1,3))
        self.beacons_detected = [0]*5
        self.F = np.zeros((3,13))
        self.F[0,0]=1
        self.F[1,1]=1
        self.F[2,2]=1
        self.FT = self.F.transpose()
        #
        self.control = np.zeros((2,1))
        self.Z = np.zeros((2,1))
        self.prev_time_stamp = 0 # keeping the last time stamp
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
        #pass
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
        #print("Inittial state",self.state_vector[0],self.state_vector[1], self.state_vector[2])
        self.prev_time_stamp = msg.header.stamp.secs + msg.header.stamp.nsecs*(10**-9)
        self.gt.unregister() # unregister subscriber. Function is implemented only once.
        self.initialized = True

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
        self.q = np.array(([0.1**2, 0],[0,.001**2])) #FIXME FOR TEST PURPOSES [0.04, 0],[0,0.001]
        #
        self.propagate_state()
        self.calculate_cov()
        #print(self.state_vector)

    def update(self, msg): #
        self.cur_id = self.beacons[msg.ids[0]] # coordinates of current transmitter
        
        # landmark position in robot frame
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        # test
        #rng = np.sqrt(pos_x**2 + pos_y**2)
        rng = np.sqrt(pos_x**2 + pos_y**2)
        # test
        #bearing
        theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2])
        theta = self.process_angle(pos_x, pos_y, theta)
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
        #innovation = np.array(([abs(new_meas[0] - expected_meas[0]), abs(new_meas[1] - expected_meas[1])]))

        self.state_vector = self.state_vector + self.K.dot(innovation)
        self.cov_matrix = (np.eye(3) - self.K.dot(self.obs_j_state)).dot(self.cov_matrix)
        # if self.cov_matrix[0][0] > 50:
        #     pdb.set_trace()
        #self.cov_matrix = self.K*self.obs_j_state
        print(self.state_vector)
        print()
        #print(self.cov_matrix)

    def process_angle(self,x,y,a):
        rot_matrix = np.array(([np.cos(a), -np.sin(a)],[np.sin(a), np.cos(a)]))
        xy = np.array([x, y]).T
        new_xy = rot_matrix.transpose().dot(xy)
        bearing = np.arctan2(new_xy[1], new_xy[0]) + a
        self.new_meas = np.array(([new_xy[0], new_xy[1], bearing]))
        return bearing



    def propagate_state(self):
        temp_mat = np.zeros((3, 1))
        if self.control[1] != 0:
            term = self.control[0]/self.control[1]
            #x = self.state_vector[0] - term*np.sin(self.state_vector[2])+ term*np.sin(self.state_vector[2]+self.control[1]*self.dt)
            temp_mat[0] = self.state_vector[0] - term*np.sin(self.state_vector[2])+ term*np.sin(self.state_vector[2]+self.control[1]*self.dt)
            #y = self.state_vector[1] + term*np.cos(self.state_vector[2])- term*np.cos(self.state_vector[2]+self.control[1]*self.dt)
            temp_mat[1] = self.state_vector[1] + term*np.cos(self.state_vector[2])- term*np.cos(self.state_vector[2]+self.control[1]*self.dt)
            #theta = self.state_vector[2] + self.control[1]*self.dt
            temp_mat[2] = self.state_vector[2] + self.control[1]*self.dt
            #theta = self.wrap_to_pi(theta)
            temp_mat[2] = self.wrap_to_pi(self.state_vector[2])

        else:
            term = self.control[0]
            #x = self.state_vector[0] + self.control[0]*np.cos(self.state_vector[2])*self.dt #self.control[0]*self.dt
            self.state_vector[0] = self.state_vector[0] + self.control[0]*np.cos(self.state_vector[2])*self.dt #self.control[0]*self.dt
            #y = self.state_vector[1] + self.control[0]*np.sin(self.state_vector[2])*self.dt #self.control[0]*self.dt
            self.state_vector[1] = self.state_vector[1] + self.control[0]*np.sin(self.state_vector[2])*self.dt #self.control[0]*self.dt
            #theta = self.wrap_to_pi(self.state_vector[2])
            self.state_vector[2] = self.wrap_to_pi(self.state_vector[2])
            
        #self.state_vector = np.array([x,y,theta])
        

    def measurement_model(self,state):
        x = state[0]
        y = state[1]
        theta = state[2]
        px = self.cur_id[0]
        py = self.cur_id[1]

        r = np.sqrt((px-x)**2 + (py-y)**2)      #Distance
        phi = np.arctan2(py-y, px-x) - theta    #Bearing

        self.Z[0] = r
        self.Z[1] = self.wrap_to_pi(phi) #FIXME added for example
        return self.Z



    def calculate_cov(self):
        self.motion_jacobian_state_vector()
        self.motion_jacobian_noise_components()
        self.Q = self.motion_j_noise.dot(self.q).dot(self.motion_j_noise.transpose())
        self.cov_matrix = self.motion_j_state.dot(self.cov_matrix).dot(self.motion_j_state.transpose()) + self.Q

    def motion_jacobian_state_vector(self):
        v = self.control[0]
        w = self.control[1]
        theta = self.state_vector[2]
        term = v/w
        dt = self.dt
        temp_mat = np.eye(3)
        if np.isclose(w,0):
            # Linear motion model
            temp_mat[0,:] = np.array([1,0,-dt*v*np.sin(theta)])
            temp_mat[1,:] = np.array([0,1,dt*v*np.cos(theta)])
            temp_mat[2,:] = np.array([0,0,1])
        else:
            # Non-linear motion model
            temp_mat[0,:] = np.array([1,0,term*(np.cos(theta + dt*w) - np.cos(theta))])
            temp_mat[1,:] = np.array([0,1,term*(np.sin(theta + dt*w) - np.sin(theta))])
            temp_mat[2,:] = np.array([0,0,1])
        self.motion_j_state = np.eye(13)+np.matmul(np.matmul(np.transpose(self.F), temp_mat), self.F)


    def motion_jacobian_noise_components(self):
        #TODO check row1term2
        v = self.control[0]
        w = self.control[1]
        theta = self.state_vector[2][0]
        dt = self.dt
        if np.isclose(w,0):
            # linear motion model
            self.motion_j_noise[0,:] = np.array([np.cos(theta)*dt, 0])
            self.motion_j_noise[1,:] = np.array([np.sin(theta)*dt, 0])
            self.motion_j_noise[2,:] = np.array([0, 0])
        else:
            sigma1 = np.sin(theta + dt*w)
            sigma2 = np.cos(theta + dt*w)
            #term1 = (1/w)*(sigma1 - np.sin(theta))
            #term2 = (v/w**2)*(np.sin(theta) - sigma1) + (dt*v*sigma2)/w
            #print(term1, term2)
            self.motion_j_noise[0,:] = np.array([(1/w)*(sigma1 - np.sin(theta)), \
                (v/w**2)*(np.sin(theta) - sigma1) + (dt*v*sigma2)/w])
            #self.motion_j_noise[0,:] = np.array([term1, term2])
            self.motion_j_noise[1,:] = np.array([(1/w)*(np.cos(theta) - sigma2), (v/w**2)*(sigma2 - np.cos(theta)) + (dt*v*sigma1)/w])
            self.motion_j_noise[2,:] = np.array([0,dt])
            # non-linear motion model

    def observation_jacobian_state_vector(self):
        mx = self.cur_id[0]
        my = self.cur_id[1]
        x = self.state_vector[0][0]
        y = self.state_vector[1][0]
        a = (x - mx)**2 + (y - my)**2
        b = np.sqrt(a)
        self.obs_j_state[0,:] = np.array([(x-mx)/b, (y-my)/b, 0])
        self.obs_j_state[1,:] = np.array([(my-y)/a, (x-mx)/a, -1])

    def print_initials(self):
        pass

    def wrap_to_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
        # temp = angle
        # while (temp > np.pi):
        #     temp = temp - (np.pi*2)
        
        # while (temp < -np.pi):
        #     temp = temp + (np.pi*2)

        # return temp


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
        