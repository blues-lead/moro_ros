#!/usr/bin/env python
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_from_euler
from nav_msgs.msg import Odometry
import rospy
import pdb
import warnings
#warnings.filterwarnings('error')
import sympy

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
        self.v_sigma = [] # for sampling sigma
        self.w_sigma = [] # for sampling sigma
        self.prev_time_stamp = 0 # keeping the last time stamp
        self.prev_state = np.array((3,1))
        from nav_msgs.msg import Odometry
        self.gt = rospy.Subscriber('base_pose_ground_truth', Odometry, self.initialize_state_vector) # Initializing state_vector with ground truth
        #
        self.state_data_history = []
        self.ground_truth_state_history = []
        self.initialize_symbolic_models()
    
    def initialize_symbolic_models(self):
        x,y,theta, v, w, dv, dw, dt, mix, miy, dr, df = sympy.symbols('x,y,theta, v, w, dv, dw, dt, mix, miy, dr, df')
        # self.motion_model = sympy.Matrix([[x + ((v+dv)/(w+dw))*sympy.sin(theta + (w + dw)*dt)], \
        #     [y - ((v+dv)/(w+dw))*sympy.cos(theta + (w + dw)*dt)], \
        #     [theta + (w + dw)*dt]])
        self.motion_model = sympy.Matrix([[x - (v/w)*sympy.sin(theta) + (v/w)*sympy.sin(theta + w*dt)], \
            [y + (v/w)*sympy.cos(theta) - (v/w)*sympy.cos(theta + w*dt)], \
            [theta + w*dt]])
        self.motion_state = sympy.Matrix([x,y,theta])
        self.simple_motion_model = sympy.Matrix([[x + v*dt], \
            [y + v*dt], \
            [theta]]) # the second function was 'y + v*dt'
        self.measurement_model = sympy.Matrix([[sympy.sqrt((x - mix)**2 + (y - miy)**2) + dr] , \
            [sympy.atan((miy - y)/(mix - x)) - theta + df]])
        self.meas_noise_components = sympy.Matrix([dr, df])
        self.motion_noise_components = sympy.Matrix([dv,dw])
        self.F_jacobian = self.motion_model.jacobian(self.motion_state)
        self.F_simple_jacobian = self.simple_motion_model.jacobian(self.motion_state)
        self.G_jacobian = self.motion_model.jacobian(self.motion_noise_components)
        self.G_simple_jacobian = self.simple_motion_model.jacobian(self.motion_noise_components)
        self.H_jacobian = self.measurement_model.jacobian(self.motion_state)

    def substitute_values(self,model, state, control, params):
        """
        model - sympy model to substitute
        state - state_vector [x,y,theta]
        control - control_vector [v,w]
        params - parameters_to_model [dt,dv,dw,dr,df]
        """
        result = model.subs({'x':state[0],'y':state[1],'theta':state[2],'v':control[0], 'w':control[1], \
            'dt':params[0], 'dv':params[1], 'dw':params[2], 'dr':params[3], 'df':params[4]})
        return np.array(result)


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
        print("Initialized vector with goround truth")


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
        self.q = np.array(([0.04, 0],[0,0.001])) # FOR TEST PURPOSES
        #
        self.propagate_state()
        self.calculate_cov()

    def update(self, msg): #
        self.cur_id = self.beacons[msg.ids[0]] # coordinates of current transmitter
        pos_x = msg.pose.position.x
        pos_y = msg.pose.position.y
        theta = self.wrap_to_pi(euler_from_quaternion([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])[2])
        self.observation_jacobian_state_vector()
        
        floor = self.cov_matrix.dot(self.obs_j_state.transpose()).astype(np.float32)
        
        bottom = (self.obs_j_state.dot(self.cov_matrix).dot(self.obs_j_state.transpose()) + np.eye(2)*0.01).astype(np.float32)
        self.K = floor.dot(np.linalg.inv(bottom)) # K is 3x2
        expected_meas = self.measurement_model(self.state_vector)
        new_meas = self.measurement_model([pos_x, pos_y, theta])
        
        tempterm = np.array(([new_meas[0] - expected_meas[0], [new_meas[1] - expected_meas[1]]]))
       
        self.state_vector = self.state_vector + self.K.dot(tempterm) 
        try:
            self.cov_matrix = (np.eye(3) - self.K.dot(self.obs_j_state)).dot(self.cov_matrix)
        except RuntimeWarning:
            pdb.set_trace()
        
        self.prev_state = self.state_vector
        print(self.state_vector)
        #print(self.cov_matrix)



    def propagate_state(self):
        if self.control[1] != 0:
            test = self.substitute_values(self.motion_model, self.state_vector,self.control,[self.dt,0,0,0,0]).astype(np.float32)
        else:
            test = self.substitute_values(self.simple_motion_model,self.state_vector,self.control,[self.dt,0,0,0,0]).astype(np.float32)
            
        self.state_vector[0] = test[0]
        self.state_vector[1] = test[1]
        self.state_vector[2] = self.wrap_to_pi(test[2])
        


    def measurement_model(self,state):
        result = self.substitute_values(self.measurement_model, state, self.control, [self.dt,0,0,0,0])
        result[1] = self.wrap_to_pi(result[1])
        return np.array(result)

    def calculate_cov(self):
        self.Q = self.motion_j_noise.dot(self.q).dot(self.motion_j_noise.transpose())
        self.cov_matrix = self.motion_j_state.dot(self.cov_matrix).dot(self.motion_j_state.transpose()) + self.Q

        self.motion_jacobian_state_vector()
        self.motion_jacobian_noise_components()

    def motion_jacobian_state_vector(self):
        if self.control[1] != 0:
            result = self.substitute_values(self.F_jacobian, self.state_vector, self.control,[self.dt,0,0,0,0])
        else:
            result = self.substitute_values(self.F_simple_jacobian, self.state_vector, self.control,[self.dt,0,0,0,0])
        self.motion_j_state = result

    def motion_jacobian_noise_components(self):
        if self.control[1] != 0: # if angular velocity is not zero
            result = self.substitute_values(self.G_jacobian, self.state_vector, self.control, [self.dt,0,0,0,0])
        else:
            result = self.substitute_values(self.G_simple_jacobian, self.state_vector, self.control, [self.dt,0,0,0,0])
        self.motion_j_noise = result

    def observation_jacobian_state_vector(self):
        result = self.substitute_values(self.H_jacobian, self.state_vector, self.control, [self.dt,0,0,0,0])
        self.obs_j_state = result

    def print_initials(self):
        pass

    def wrap_to_pi(self,angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi