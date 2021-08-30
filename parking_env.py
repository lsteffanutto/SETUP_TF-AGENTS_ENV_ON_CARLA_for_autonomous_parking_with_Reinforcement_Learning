#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# [Lucas STEFFANUTTO]

# Creation of the Parking environment in Carla simulator to train it with tf-agents
# tf-agents tutorial here: https://www.tensorflow.org/agents/tutorials/2_environments_tutorial#wrapping_a_python_environment_in_tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import numpy as np
import abc
import tensorflow as tf

import math
import random
import time

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from utils_carla import *


#tf.compat.v1.enable_v2_behavior()

''' Why self in all the class ? => https://www.youtube.com/watch?v=pzxDxOD-gmw '''

# Choose your debug
v_infos = False
show_observations = False
show_debug_KARlab_physics = False
show_bounding_box = False
show_debug_collision_sensor = False
show_debug_obstacle_sensor = False
show_debug_radar = False
show_debug_get_radar_data = False

#Create a class of Environment
class CarlaParkingEnv(py_environment.PyEnvironment):
    
    def __init__(self):
        
        # Initialisation of the tf-agents model #
        
        # Describes a numpy array or scalar shape and dtype. An ArraySpec that specifies minimum and maximum values.
        # ACTIONS: 3 float => Throttle [0,1] ; Steer [-1 ; 1] ; Brake [0,1]
        self._action_spec = array_spec.BoundedArraySpec(shape=(3,), dtype=np.float32, minimum=[0.0, -1.0, 0.0], maximum=1.0, name='action')
        #print('\nACTIONS SPACE: ' + str(self._action_spec) + '\n')        

        # OBSERVATIONS: 33 float => Vehicle + radar
        self.min_obs = get_min_obs()        
        self.len_obs_state = 33    
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.len_obs_state,), dtype=np.float32, minimum=self.min_obs, maximum=1.0, name='observation')
        #print('\nSTATES SPACE: ' + str(self._observation_spec) + '\n')            
        
        self._state = np.zeros(33, dtype=np.float32)#.tolist()
        self._episode_ended = False
        
        #self.time_step_spec().step_type.dtype = ('np.float32')        #CANNOT SET THE ATTRIBUTE
        #print(str(self.time_step_spec().step_type) + 'ouiiiiii')
        
        #ArraySpec(shape=(), dtype=dtype('int32'), name='step_type')o
        #self.TimeStep().dtype='float32'
        
        if(show_observations):print('INIT OBSERVATIONS OF THE MODEL:\n' + str(self._state))            
    
    
        # Initialisation of the Carla environment #
        self.actor_list = []

        # Set a client and timeouts
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(200.0)

        # Get currently running world
        self.world = self.client.get_world()
        
        # Desactivate the graphic render (True)
        no_render = False
        if(no_render):
            self.settings = self.world.get_settings()
            self.settings.no_rendering_mode = True
            self.world.apply_settings(self.settings)

        # Get list of actor blueprints
        self.blueprint_library = self.world.get_blueprint_library()

        # Get a sort of KARlab blueprint
        self.bp_KARlab = self.blueprint_library.filter('vehicle.bmw.isetta')[0]
        self.bp_KARlab.set_attribute('color', '0,0,255')
        #print('\nInit OK, car blueprint créé\n')
        
    # Resets environment for new episode
    def _reset(self):
        '''Return the state of the environment and information that it is the first step of the simulation
        '''

        try:
            
            ############################################# RESET CARLA ENV ###############################################################
            
            self.actor_list = []
            
            # Spawn the vehicle
            self.location = carla.Vector3D(x=14, y=-24.4,z=0.2)
            self.transform = carla.Transform(self.location, carla.Rotation(yaw=180)) #[-180;180]
            self.vehicle = self.world.spawn_actor(self.bp_KARlab, self.transform)
            
            # Append actor to a list of spawned actors, we need to remove them later
            self.actor_list.append(self.vehicle)
            
            # To see the scene from a point of view
            self.spectator = self.world.get_spectator()
            #self.spectator.set_transform(carla.Transform(self.transform.location + carla.Location(x=-4,y=-10,z=10),carla.Rotation(pitch=-40,yaw=110))) # 3rd person view
            self.spectator.set_transform(carla.Transform(self.transform.location + carla.Location(x=7,z=3),carla.Rotation(yaw=180))) # RADAR VIEW
            #self.spectator.set_transform(carla.Transform(self.transform.location + carla.Location(x=-4,z=1),carla.Rotation(yaw=180))) # Vehicule view
            
            KARlab_physics(self.vehicle,weight_KARlab, max_steer_angle, radius_front_wheels, radius_rear_wheels, max_brake_torque, show_debug_KARlab_physics)
            #print('\nVehicule spawn with KARlab physics\n')
            
            #self.vehicle.apply_control(carla.VehicleControl(throttle=1.0))
            
            # RADAR 
            self.bp_radar_sensor = self.blueprint_library.find('sensor.other.radar')
            self.radar_offset = carla.Location(x=1.105,z=z_box_center_KARlab)        # offset relatively to the vehicle
            self.radar_sensor = radar_creation(self.world, self.actor_list, self.vehicle, self.bp_radar_sensor, self.radar_offset, horizontal_fov=115, vertical_fov=0.001, range=80, points_per_second = 120, sensor_tick = 0.1)
            # radar_data (4n values: each timestamp-frame, cloud of n points wich contains [velocity, azimuth, altitude, depth])

            self.size_obs_rad = (10,2)                                         # each 0.1 seconds, radar cloud of 10 points
            self.tab_obs_radar_norm = np.zeros(self.size_obs_rad, dtype=np.float32) # to well visualize the radar output
            self.radar_sensor.listen(lambda radar_data: rad_callback(self.world,radar_data,show_debug_radar,self.tab_obs_radar_norm)) # Radar is listening and return azimuth angle and distance for each point of the cloud generated         
            #print('\nRadar created is listening...\n')
            
            ############################################# RESET TF-AGENTS ENV ###############################################################
            
            # FIRST VEHICLE OBSERVATION #       
            # 13 Values => tab_obs_norm = [vehicle_pos_norm, cap_norm, velocity_norm, parking_location_norm, distance_goal_agent_norm])
            [self.tab_obs, self.tab_obs_norm] = get_observations(self.world, self.vehicle, v_infos, show_bounding_box, show_observations)   
            self.tab_obs_norm = np.array(self.tab_obs_norm, dtype=np.float32)            
            
            # FIRST RADAR OBSERVATION #
            time.sleep(0.1) # time to get a sensor_tick
            self.obs_radar_norm = np.concatenate(self.tab_obs_radar_norm)#.tolist() # what we will add to the observations input
            #self.obs_radar_norm = np.array(self.tab_obs_radar_norm, dtype=np.float32)#.tolist() # what we will add to the observations input
            
            # Debug
            if(show_debug_radar):draw_sensor_localisation(self.world, self.vehicle, self.radar_offset, 'radar') 
            if(show_debug_get_radar_data):
                print('Radar measure 1:\n' + str(self.tab_obs_radar_norm) + '\n')        
                time.sleep(1)
                print('\nRadar measure 2:\n' + str(self.tab_obs_radar_norm) + '\n\n' + 'Radar measure 2 vect: =' + str(np.concatenate(self.tab_obs_radar_norm).tolist()) + '\n\n' + 'Radar measure 2 => shape:' + str(len(self.obs_radar_norm)) +' is with type:' + str(type(self.obs_radar_norm)) + ' composed of type:' + str(type(self.obs_radar_norm[0])) + '\n')   

            ## FIRST OBSERVATION OF THE EPISODE = Kart observation + Radar observation
            self._state = np.concatenate([self.tab_obs_norm,self.obs_radar_norm])#.tolist() # Kart observation + Radar observation
            self._episode_ended = False    
            self.episode_start = time.time()
            
            #print('\n\n!!!observation match format !!!\n\n' + str(array_spec.check_array(self._state)))
            time.sleep(5)
            
            #print("\nVerifications of the environment setup:\n")
            #print('action_spec:', env_UC3.action_spec())
            #print('time_step_spec.observation: ', self.ts.observation)



            if(show_observations):
                
                round_state = [round(num, 2) for num in self._state]
                #print('FIRST OBSERVATIONS OF THE MODEL:\n' + str(self._state))            
                print(' ===> DEBUG:\nFIRST OBSERVATIONS OF THE MODEL:\n' + str(round_state))         
                   
                print('\nFirst Observation:\nShape:' + str(len(self._state)) +' is of type:' + str(type(self._state)) + ' composed of type ' + str(type(self._state[0])) + '\n')   
                for i in range(len(self.min_obs)):
                    print('obs N°' + str(i) + ' = '+ str(self._state[i]) + ' ; min_obs = ' + str(self.min_obs[i]))
                    if (self._state[i] > 1 or self._state[i] < self.min_obs[i]):
                        print('\n!!! ERREUR AU DESSUS observation n°' + str(i) + '!!!\n')
                print(' ===> END DEBUG')
                      
            print('\nRETURN by RESET() IS :\n' + str(ts.restart(np.array([self._state], dtype=np.float32))) + '\n')
            
            return ts.restart(np.array([self._state], dtype=np.float32))
        
    
        
    
        finally:
            
            print('destroying actors')
            for actor in self.actor_list:
                actor.destroy()
            print('done.') 
            
              
            
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _step(self, action):
        '''Function which takes action, applies it to the environment and returns the next step information:
        
            Return: "TimeStep(step_type, reward, discount, observation)"
            - The reward
            - The Observation of the next state
            - The step_type: FIRST, MID or LAST to indicate if it is the first, intermediate or last the sequence/episode
            - Discount: who ponderates the reward for the step after this next state
            
        '''
        
        self.remainging_distance = np.linalg.norm([self.tab_obs[10], self.tab_obs[11], self.tab_obs[12]])
        
        print("remaining distance to parking target = " + (str(self.remainging_distance)))
        
        # ACTION
        print("action choisie = " + str(action))
        
        if action[0] > 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=action[0]))
            print("on avance")    
            
        # CHECK EPISODE
        
        self.remainging_distance = np.linalg.norm([self.tab_obs[10], self.tab_obs[11], self.tab_obs[12]])
        print("remaining distance to parking target AFTER ACTION = " + (str(self.remainging_distance)))
        
        #lose
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            
            print('! END OF THE EPISODE !')
            
            print('destroying actors')
            for actor in self.actor_list:
                actor.destroy()
            print('done.') 
            
            reward = - 1.0
            return ts.termination(np.array([self._state], dtype=np.float32), reward)

        
        #win
        if self.remainging_distance < 0.5:
            print("objectif atteint")
            print('! END OF THE EPISODE !')
            
            self._episode_ended = True
            print('destroying actors')
            for actor in self.actor_list:
                actor.destroy()
            print('done.') 
            
            reward = 1.0
            return ts.termination(np.array([self._state], dtype=np.float32), reward)
            

