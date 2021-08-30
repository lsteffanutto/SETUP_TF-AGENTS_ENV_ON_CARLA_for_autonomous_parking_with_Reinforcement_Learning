#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# [Lucas STEFFANUTTO]

# Functions used in Carla simulator

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
  


import abc
import tensorflow as tf
import numpy as np
import math

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

# KARlab measurements
length_KARlab = 1.865
width_KARlab = 1.350
height_KARlab = 0.630
ground_clearance = 0.055
z_box_center_KARlab = (height_KARlab+ground_clearance)/2
KARlab_size = carla.Vector3D(length_KARlab,width_KARlab,height_KARlab+ground_clearance)
weight_KARlab = 233.5
max_steer_angle = 20.0
radius_front_wheels = 26.2
radius_rear_wheels = 29.1
max_brake_torque = 1000

debug=carla.DebugHelper

def print_element(dataset):
    for element in dataset:
        print(element)

# ==============================================================================
# -- VEHICLE ---------------------------------------------------------------
# ==============================================================================

def get_vehicle_infos(vehicle, parking_location, v_infos):
    """Return all the vehicle informations: position, yaw, velocity etc. Carla 0.9.8
    
    " Important: All the sensors use the UE coordinate system (x-forward, y-right, z-up), and return coordinates in local space"
    """
    
    vehicle_transform = vehicle.get_transform()
    #print('\nVehicule Position:' + str(vehicle_transform) + '\n')
        
    # Position and Velocity
    pos = vehicle_transform.location
    cap = vehicle_transform.rotation.yaw #yaw in [-180;180]
    velocity = vehicle.get_velocity()
    
    #Box Collider Dimensions and Points
    box_collider = vehicle.bounding_box
    col_size = vehicle.bounding_box.extent
    vehicle_bounding_box=bb_points(vehicle)

    #Display_Infos  
    if(v_infos):
        print('\n          Vehicule Infos:\n')  # Not normalized
        print('- Vehicule Position: ' + str(pos))
        print('- Vehicule Yaw: ' + str(cap))
        print('- Vehicule Velocity: ' + str(velocity))
        print('- Parking Target location: ' + str(parking_location))
        distance_goal_agent = parking_location - vehicle.get_location()
        print('- Remaining distance Parking-Vehicule: ' + str(distance_goal_agent))

        print('- Box Collider Center:\n' + str(box_collider))
        print('- Box Collider size: ' + str(col_size))
        print('- Box Collider Points from vehicle center:\n' + str(vehicle_bounding_box) + '\n')
    
    return pos, cap, velocity, col_size, box_collider

def bb_points(vehicle):
    """
    Returns the 3D points of the bounding box for a vehicle.
    """

    bb_4_points = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    bb_4_points[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    bb_4_points[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    bb_4_points[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    bb_4_points[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    bb_4_points[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    bb_4_points[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    bb_4_points[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    bb_4_points[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return bb_4_points

def draw_vehicle_collider(world,col_size,z_offset,color):
    """
    Draw the box collider of a vehicle
    """
    debug = world.debug
    world_snapshot = world.get_snapshot()
    
    for actor_snapshot in world_snapshot:
        actual_actor = world.get_actor(actor_snapshot.id)
        if actual_actor.type_id == 'vehicle.bmw.isetta':
            collider_location = actor_snapshot.get_transform().location + carla.Vector3D(0,0,z_offset)
            debug.draw_box(carla.BoundingBox(collider_location,carla.Vector3D(col_size.x,col_size.y,col_size.z)),actor_snapshot.get_transform().rotation, 0.05, color,0)        

def KARlab_physics(vehicle, weight_KARlab,  max_steer_angle, radius_front_wheels, radius_rear_wheels, max_brake_torque, show_debug_KARlab_physics):
    '''Change the physic of a Carla vehicle in order to put the KARlab Kart physics'''
    
    # Change Vehicle Physics Control parameters of the vehicle
    physics_control = vehicle.get_physics_control()
    
    # Set the physics as KARlab
    physics_control.mass = weight_KARlab
    physics_control.max_rpm = 11000
    
    wheels = KARlab_wheels_physics(max_steer_angle, radius_front_wheels, radius_rear_wheels, max_brake_torque)
    physics_control.wheels = wheels
    
    # Apply Vehicle Physics Control for the vehicle
    vehicle.apply_physics_control(physics_control)
    
    if(show_debug_KARlab_physics):
        print('\n          Vehicle KARlab physics')
        print('Vehicle Mass = ' + str(physics_control.mass) + ' kg')
        print('Left Front Wheels: ' + str(wheels[0]) + '\nRight front wheels: ' + str(wheels[1]))
        print('Left Rear Wheels: ' + str(wheels[2]) + '\nRight rear wheels: ' + str(wheels[3]))
        print('Vehicle Max Rpm = ' + str(physics_control.max_rpm) +'\n')
    
def KARlab_wheels_physics(max_steer_angle, radius_front_wheels, radius_rear_wheels, max_brake_torque):
    '''Return the wheels KARlab Kart physics'''

    # Create Wheels Physics Control for axle 
    front_left_wheel  = carla.WheelPhysicsControl(max_steer_angle=max_steer_angle, radius=radius_front_wheels, max_brake_torque=0)
    front_right_wheel = carla.WheelPhysicsControl(max_steer_angle=max_steer_angle, radius=radius_front_wheels, max_brake_torque=0)
    rear_left_wheel   = carla.WheelPhysicsControl(max_steer_angle=0, radius=radius_rear_wheels, max_brake_torque=max_brake_torque)
    rear_right_wheel  = carla.WheelPhysicsControl(max_steer_angle=0, radius=radius_rear_wheels, max_brake_torque=max_brake_torque)

    wheels = [front_left_wheel, front_right_wheel, rear_left_wheel, rear_right_wheel]

    return wheels

def normalize_vector(vect):
    """Return a normalized vector: 
    
    x_norm = x/|x|
    """
    length = math.sqrt(vect.x**2 + vect.y**2 + vect.z**2)     # !!!!!!! Maybe we have to divide by the max value i.e. the length of the map !!!!!!!!
    return vect/length 

def get_observations(world, vehicle, v_infos, show_bounding_box, show_observations):
    """Return the Agent observations and the afent normalized observation with 2 float tabs"""

    # Parking Target ( 3 values Vector3D)
    parking_location = carla.Vector3D(x=10.3, y=-24.4,z=0.1) 
    parking_location_norm = normalize_vector(parking_location) #parking_location_norm in [-1;1]
    world.debug.draw_string(parking_location, 'O   PARKING TARGET', draw_shadow=False,color=carla.Color(0,255,0,0), life_time=200,persistent_lines=True)
    
    # Vehicle Informations (Location, Velocity, Yaw = 3 values Vector3D + 1 value in ° + 3 values Vector3D in m/s)
    [pos, cap, velocity, col_size, box_collider]=get_vehicle_infos(vehicle,parking_location,v_infos)  
    vehicle_pos_norm = normalize_vector(pos)    #pos in [-1;1]
    cap_norm = cap/180                          #yaw in [-1;1]
    velocity_norm = velocity/36  #velocity in [-1;1] ; # max velocity = 130km/h = 36m/s
    
    # Remaining distance (3 values Vector3D)
    distance_goal_agent = parking_location - vehicle.get_location()
    #print('Remaining distance: ' +  str(distance_goal_agent) + 'm')
    distance_goal_agent_norm = normalize_vector(distance_goal_agent) #remaining_distance_norm in [-1;1]
    
    tab_obs = [pos.x, pos.y, pos.z, cap, velocity.x, velocity.y, velocity.z, parking_location.x, parking_location.y, parking_location.z,  distance_goal_agent.x, distance_goal_agent.y, distance_goal_agent.z]    
    tab_obs_norm = [vehicle_pos_norm.x, vehicle_pos_norm.y, vehicle_pos_norm.z, cap_norm, velocity_norm.x, velocity_norm.y, velocity_norm.z, parking_location_norm.x, parking_location_norm.y, parking_location_norm.z, distance_goal_agent_norm.x, distance_goal_agent_norm.y, distance_goal_agent_norm.z]    


    if (show_observations):
        
        print('\n          OBSERVATIONS:          \n ')
        print('- Vehicle position: ' +  str(pos) + '\n- Vehicle yaw: ' + str(cap) + '\n- Vehicle velocity: ' + str(velocity))
        print('- Parking location: ' +  str(parking_location))
        print('- Remaining distance: ' + str(distance_goal_agent)+'\n')
        
        print('\n          OBSERVATIONS NORM:          \n ')
        print('- Vehicle position norm: ' +  str(vehicle_pos_norm) + '\n- Vehicle yaw norm: ' + str(cap_norm) + '\n- Vehicle velocity: ' + str(velocity_norm))
        print('- Parking location norm: ' +  str(parking_location_norm))
        print('- Remaining distance norm: ' + str(distance_goal_agent_norm)+'\n')

        print('\n          TABS OBS:          \n ')
        print('TAB OBS:\n' + str(tab_obs) + '\n')
        print('TAB OBS NORM:\n' + str(tab_obs_norm))
        
        print('Shape:' + str(len(tab_obs_norm)) +' is of type: ' + str(type(tab_obs_norm)) + ' composed of type ' + str(type(tab_obs_norm[0])) + '\n')   
        


    return tab_obs, tab_obs_norm

# ==============================================================================
# -- SENSORS ---------------------------------------------------------------
# ==============================================================================

def radar_creation(world, actor_list, vehicle, bp_radar_sensor, radar_offset, horizontal_fov, vertical_fov, range, points_per_second, sensor_tick):
    """Create a radar sensor associated to a vehicule in the Carla world. The radar sensor is similar to the LIDAR. It creates a wall of points in front of the sensor with a distance, angle and velocity in relation to it.
    
    Args:
        vehicle: Vehicle to associate with the sensor
        bp_radar_sensor: Blueprint of the sensor
        radar_offset: Offset of the sensor from the center of the associated vehicle
        horizontal_fov: Horizontal field of view in degrees
        vertical_fov: Vertical field of view in degrees
        range: Maximum distance for the lasers to raycast.
        points_per_second: Points generated by all lasers per second.
        sensor_tick: Simulation seconds between sensor captures (ticks).

    """
    
    # Radar Features
    bp_radar_sensor.set_attribute('horizontal_fov', str(horizontal_fov))
    bp_radar_sensor.set_attribute('vertical_fov', str(vertical_fov))
    bp_radar_sensor.set_attribute('range', str(range))
    bp_radar_sensor.set_attribute('points_per_second', str(points_per_second))
    bp_radar_sensor.set_attribute('sensor_tick', str(sensor_tick))
     
     
    # Radar Spawn
    radar_sensor = world.spawn_actor(bp_radar_sensor,carla.Transform(radar_offset), attach_to=vehicle)
    actor_list.append(radar_sensor)
    print(('Created %s' % radar_sensor.type_id) + ' at position: ' + str(radar_offset) + ' from the vehicle')

    return radar_sensor
    
def rad_callback(world,radar_data,show_debug_radar,tab_obs_radar_norm):
    """Return a number of (sensor_tick*points_per_second) RadarDetection each sensor_tick time with a numpy array: "tab_obs_radar" and "tab_obs_radar_norm"
    
    Args:
        radar_data: Data detected when the radar is listening
        show_debug_radar: Debug to see the radar points in the Carla World
    Return:
    
        RadarMeasurement attributes (=> in "radar_data"): (frame=2193914, timestamp=9785.075751, point_count=6)
        frame: Frame count when the data was generated.
        timestamp: Simulation-time when the data was generated. (s)
        Simulation-time when the data was generated.
        
        AND
        
        RadarDetection attributes ( => in "detect"):  Each of these represents one of the points in the radar cloud generated by the radar who is hitting an obstacle
            altitude: Altitude angle in radians.
            azimuth: Azimuth angle in radians.
            depth: absolute distance in meters.
            velocity: Velocity towards the sensor.

        => !!! "points_per_second" radar points returned each sensor_tick time !!!
        
    Debug in Carla World:
    " Draw the points captured by the radar on the fly. The points will be colored depending on their velocity regarding the ego vehicle. "
        Blue for points approaching the vehicle.
        Red for points moving away from it.
        White for points static regarding the ego vehicle.
    """
    
    velocity_range = 7.5 # m/s
    current_rot = radar_data.transform.rotation
    
    # for all the detection of the points cloud generated by the radar
    if(show_debug_radar):print('\n\n          RADAR MEASURE\n')
    
    index=0
    size_obs_rad = (10,2)
    tab_obs_radar = np.zeros(size_obs_rad, dtype=float)
    
    num_radar_points_to_record = 10

    for detect in radar_data:
        
        rad_vel = detect.velocity
        azi = math.degrees(detect.azimuth)
        alt = math.degrees(detect.altitude)
        dist = detect.depth
        
        # The 0.25 adjusts a bit the distance so the dots can
        # be properly seen
        fw_vec = carla.Vector3D(x=detect.depth - 0.25)
        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=current_rot.yaw + azi,
                roll=current_rot.roll)).transform(fw_vec)

        def clamp(min_v, max_v, value):
            return max(min_v, min(value, max_v))
        # Color of the detected points debug in function of their relative speed to the car
        norm_velocity = detect.velocity / velocity_range # range [-1, 1]
        r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
        g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
        b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
        
        # RADAR OBSERVATIONS
        # Sometimes not enought radar point are created => we create more and take only the desired number
        if(index<num_radar_points_to_record):
            
            #tab_obs_radar[index,0] = detect.velocity           => adding in case we add moving obstacles
            tab_obs_radar[index,0] = azi
            tab_obs_radar[index,1] = detect.depth
            
            #tab_obs_radar_norm[index,0] = detect.velocity/36 # max velocity = 130km/h = 36m/s    => adding in case we add moving obstacles
            tab_obs_radar_norm[index,0] = azi/65             # max fov = + or - 60°, in [-1 ; 1]
            tab_obs_radar_norm[index,1] = detect.depth/80    # max radar range, in [0 ; 1]
            
            index+=1
            
            if(show_debug_radar):
                
                # Draw a point when the radar raycasts hit an obstacle
                world.debug.draw_point(radar_data.transform.location + fw_vec, size=0.075, life_time=0.06, persistent_lines=False, color=carla.Color(r, g, b))

                # Vector of the radar detection, his normalization and the ID of the detection (frame, timestamp and number of point in the radar raycasts cloud)
                # print detect by default to debug and see the ID of detections
                #if(show_debug_radar):print('- n°'+ str(index) + ' ' + str(detect) + '\nDetection_norm:' + str(normalize_radar(detect)) + ' ; ID = [frame:' + str(radar_data.frame) + ' | timestamp: ' + str(radar_data.timestamp) + ' | points:' + str(len(radar_data)) + ']\n') # + '   detected from ' + str(radar_data.transform) )
                # print detect in degrees to verify the values
                print('- n°'+ str(index) + ' ' + 'RadarDetection(velocity=' + str(rad_vel) + ', azimuth=' + str(azi) + ', altitude=' + str(alt) + ', depth=' + str(dist))
                
    index = 0 
    
    if(show_debug_radar):
        print('\n          OBSERVATIONS RADAR\n' + str(tab_obs_radar) + '\n')
        print('           OBSERVATIONS RADAR NORM\n' + str(tab_obs_radar_norm) + '\n')
    
def draw_sensor_localisation(world, vehicle, sensor_offset, sensor_name):
    """
    Draw the sensor position in the Carla World with the sensor name for debug
    """
    # Vehicule Transform, Location and Forward vector
    vehicle_trans = vehicle.get_transform()
    vehicle_forward = vehicle_trans.get_forward_vector().x
    vehicle_loc = vehicle.get_location()
    
    # Sensor Localisation 
    radar_location_real = vehicle_loc + carla.Location(sensor_offset.x*vehicle_forward,sensor_offset.y,sensor_offset.z)
    
    # Debug Drw in the world
    string_debug = 'X\n' + sensor_name.upper() 
    world.debug.draw_string(radar_location_real, string_debug, draw_shadow=False,color=carla.Color(255,0,0,0), life_time=200,persistent_lines=True)
    

# ==============================================================================
# -- RL ---------------------------------------------------------------
# ==============================================================================
def get_min_obs():
    
    min_obs = (np.ones((13,))*-1).tolist()   # angle, velocity and other distance => min=-1
    
    radar_min = np.empty((20,))
    radar_min[::2] = -1.0              
    radar_min[1::2] = 0.0
    min_obs_radar = radar_min.tolist() # Pairs number azimuth angle => min =-1 ; Impairs numbers distance => min = 0 (14th value of the observation vector = azimuth angle, 15th value of the observation vector = depth distance, and so on until 33)
    
    min_obs_tot = np.concatenate([min_obs,min_obs_radar])#.tolist()
    
    return min_obs_tot