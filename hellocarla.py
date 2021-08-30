#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# [Lucas STEFFANUTTO]

# First tests on Carla for data collections and creation of the Parking environment. It is a remake of Carla script "tutorial.py" with an adaptation to the KARlab UC3

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

import math
import random
import time

def main():
    
    actor_list = []
    collision_hist = []
    collision_hist_type = []
    obstacle_hist = []
    obstacle_hist_type = []

    try:
        debug=carla.DebugHelper
        
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(200.0)

        # Once we have a client we can retrieve the world that is currently
        # running. We load the map '/Game/Carla/Maps/Town05' for the parking zone
        world = client.get_world()
        
        # Choose your debug
        v_infos = True
        show_observations = True
        show_bounding_box = True
        show_debug_collision_sensor = False
        show_debug_obstacle_sensor = False
        show_debug_radar = True
        show_debug_KARlab_physics = True
        
        no_render = False
        if(no_render):
            settings = world.get_settings()
            settings.no_rendering_mode = True
            world.apply_settings(settings)
            
        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one that looks like the Kart of KARlab
        bp = blueprint_library.filter('vehicle.bmw.isetta')[0]

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's put the KARlab color.
        bp.set_attribute('color', '0,0,255')

        # Now we need to give an initial transform to the vehicle. We choose to spawn near the parking zone 
        location = carla.Vector3D(x=14, y=-24.4,z=0.2)
        #location = carla.Vector3D(x=12.3, y=-24.4,z=0.1)
        transform = carla.Transform(location, carla.Rotation(yaw=180))
        vehicle = world.spawn_actor(bp, transform)
        #print('location direct' + str(location))                
        
        # KARlab size
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
        
        # Vehicule Physics setting with KARlab features
        KARlab_physics(vehicle,weight_KARlab, max_steer_angle, radius_front_wheels, radius_rear_wheels, max_brake_torque, show_debug_KARlab_physics)
        
        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print(('created %s' % vehicle.type_id) + ' at position: ' + str(location))

        # Let's put the vehicle to drive around.
        #vehicle.set_autopilot(True)
        
        # Try to control the Car (here tensorflow will act) Throttle [0,1] ; Steer [-1 ; 1] ; brake [0,1]
        #vehicle.apply_control(carla.VehicleControl(throttle=1.0))

        # To see the scene from a point of view
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(x=-4,y=-8,z=10),carla.Rotation(pitch=-40,yaw=110))) # 3rd person view
        #spectator.set_transform(carla.Transform(transform.location + carla.Location(x=7,z=3),carla.Rotation(yaw=180))) # Radar view
        #spectator.set_transform(carla.Transform(transform.location + carla.Location(x=-4,z=1),carla.Rotation(yaw=180))) # Vehicule view

        ############################################   OBSERVATIONS   #############################################################################################
        
        # Parking Target ( 3 values Vector3D)
        parking_location = carla.Vector3D(x=10.3, y=-24.4,z=0.1)
        parking_location_norm = normalize_vector(parking_location)
        world.debug.draw_string(parking_location, 'O   PARKING TARGET', draw_shadow=False,color=carla.Color(0,255,0,0), life_time=200,persistent_lines=True)
        
        # Vehicle Informations (Location, Velocity, Yaw = 3 values Vector3D + 1 value in ° + 3 values Vector3D in m/s)
        [pos, cap, velocity, col_size, box_collider]=get_vehicle_infos(vehicle,parking_location,v_infos,show_bounding_box)  
        vehicle_pos_norm = velocity/36
        cap_norm = cap/360
        velocity_norm = normalize_vector(velocity)
        
        # Remaining distance (3 values Vector3D)
        distance_goal_agent = parking_location - vehicle.get_location()
        #print('Remaining distance: ' +  str(distance_goal_agent) + 'm')
        distance_goal_agent_norm = normalize_vector(distance_goal_agent)
        
        if (show_observations):
            print('\n          OBSERVATIONS:          \n ')
            print('- Vehicle position norm: ' +  str(vehicle_pos_norm) + '\n- Vehicle yaw norm: ' + str(cap_norm) + '\n- Vehicle velocity: ' + str(velocity_norm))
            print('- Parking location norm: ' +  str(parking_location_norm))
            print('- Remaining distance norm: ' + str(distance_goal_agent_norm)+'\n')

        time.sleep(1)
        
        ############################################   SENSORS   #############################################################################################

        # COLLISIONS SENSOR
        bp_collision_sensor = blueprint_library.find('sensor.other.collision')
        collision_sensor_offset = carla.Location(x=1.2,z=z_box_center_KARlab)
        collision_sensor = collision_sensor_creation(world, actor_list, vehicle, bp_collision_sensor, collision_sensor_offset)
        
        collision_sensor.listen(lambda collision_sensor_data: collision_detector(collision_sensor_data,collision_hist,show_debug_collision_sensor)) # Collision sensor is listening

        # OBSTACLE SENSOR
        bp_obstacle_sensor = blueprint_library.find('sensor.other.obstacle')
        obstacle_sensor_offset = carla.Location(x=2.1, z=z_box_center_KARlab)
        obstacle_sensor = obstacle_sensor_creation(world, actor_list, vehicle, bp_obstacle_sensor, obstacle_sensor_offset, distance=7, hit_radius=2.5)
        
        obstacle_sensor.listen(lambda obstacle_sensor_data: obstacle_detector(obstacle_sensor_data, obstacle_hist, show_debug_obstacle_sensor)) # Obstacle Sensor is listening

        # RADAR 
        bp_radar_sensor = blueprint_library.find('sensor.other.radar')
        radar_offset = carla.Location(x=1.105,z=z_box_center_KARlab)        # offset relatively to the vehicle
        radar_sensor = radar_creation(world, actor_list, vehicle, bp_radar_sensor, radar_offset, horizontal_fov=120, vertical_fov=0.001, range=80, points_per_second = 100, sensor_tick = 0.1)
        # radar_data (4n values: each timestamp-frame, cloud of n points wich contains [velocity, azimuth, altitude, depth])
        radar_sensor.listen(lambda radar_data: rad_callback(world,radar_data,show_debug_radar)) # Radar is listening
        
        # DEBUG
        print('\n          SENSORS OBSERVATIONS:          \n ')
        if(show_debug_collision_sensor):
            draw_sensor_localisation(world, vehicle, collision_sensor_offset, 'collision sensor') 
            show_detection(collision_hist, collision_hist_type, 'Collisions')
        if(show_debug_obstacle_sensor):
            draw_sensor_localisation(world, vehicle, obstacle_sensor_offset, 'obstacles sensor')
            show_detection(obstacle_hist, obstacle_hist_type, 'Obstacles')
        if(show_debug_radar):draw_sensor_localisation(world, vehicle, radar_offset, 'radar') 
        if(show_bounding_box):
            draw_vehicle_collider(world,col_size,box_collider.location.z,carla.Color(255,0,0,0))        # COLLIDERS Vehicule and KARlab
            draw_vehicle_collider(world,KARlab_size/2,z_box_center_KARlab,carla.Color(0,255,0,0))

        time.sleep(5)
    
    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')

        
##########################################################################   FUNCTIONS   ####################################################################
        
# ==============================================================================
# -- VEHICLE ---------------------------------------------------------------
# ==============================================================================

def get_vehicle_infos(vehicle,parking_location, v_infos, show_bounding_box):
    """Get all the vehicle informations: position, yaw, velocity etc. Carla 0.9.8
    
    " Important: All the sensors use the UE coordinate system (x-forward, y-right, z-up), and return coordinates in local space"
    """
    
    vehicle_transform = vehicle.get_transform()
    #print('\nVehicule Position:' + str(vehicle_transform) + '\n')
        
    # Position and Velocity
    pos = vehicle_transform.location
    cap = vehicle_transform.rotation.yaw
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
    
    if(show_bounding_box):
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

def normalize_radar(radar_vect):
    
    """Return a normalized vector: 
    
    x_norm = x/|x|
    """
    length = math.sqrt(radar_vect.velocity**2 + radar_vect.azimuth**2 + radar_vect.altitude**2 +  radar_vect.depth**2)     # !!! Maybe we have to divide y the max value i.e. the length of the map !!!
    radar_tab = [radar_vect.velocity, radar_vect.azimuth, radar_vect.altitude, radar_vect.depth]
    res_radar_norm = [x / length for x in radar_tab] #divide all the element of the tab by the norm of the initial vector
    return res_radar_norm  

# ==============================================================================
# -- SENSORS ---------------------------------------------------------------
# ==============================================================================

def show_detection(histogram, histogram_types, detection_name):
    """Get all the detections of a sensor and return the total number and the differents types of objects detected
    
    Args:
        histogram: All the detection of the sensor
        detection_name: Type of detection
    """
    num_detec = len(histogram)
    histogram_types = {x for x in histogram if x not in histogram_types}
    print('\n'+str(num_detec)+ ' ' + detection_name +' detected:\n'+ str(histogram_types)+'\n')
    
# COLLISIONS SENSOR
def collision_sensor_creation(world, actor_list, vehicle, bp_collision_sensor, collision_sensor_offset):
    """Create a collision sensor associated to a vehicule in the Carla world
    
    Args:
        vehicle: Vehicle to associate with the sensor
        bp_collision_sensor: Blueprint of the sensor
        collision_sensor_offset: Offset of the sensor from the center of the associated vehicle
    """
    # Collisions sensor Spawn
    collision_sensor = world.spawn_actor(bp_collision_sensor,carla.Transform(collision_sensor_offset), attach_to=vehicle)
    actor_list.append(collision_sensor)
    print(('Created %s' % collision_sensor.type_id) + ' at position: ' + str(collision_sensor_offset) + ' from the vehicle')
    
    return collision_sensor 

def collision_detector(event,collision_hist,show_debug_collision_sensor):
    """Get and record the collisions detected when the sensor is listening with debug logs
    
    Args:
        vehicle: Vehicle to associate with the sensor
        bp_collision_sensor: Blueprint of the sensor
        collision_sensor_offset: Offset of the sensor from the center of the associated vehicle
    """
    
    actor_we_collide_against = event.other_actor.type_id
    collision_hist.append(actor_we_collide_against)
    impulse = event.normal_impulse
    normal_impulse = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    
    
    if(normal_impulse>0 and show_debug_collision_sensor):
        print('\n!!! Collision with ' + str(actor_we_collide_against) + ' detected !!!\n')
        
    #if not (actor_we_collide_against in collision_hist):
    #collision_hist_type.append(actor_we_collide_against)

# OBSTACLES SENSOR
def obstacle_sensor_creation(world, actor_list, vehicle, bp_obstacle_sensor, obstacle_sensor_offset, distance, hit_radius):
    """Create an obstacles sensor associated to a vehicule in the Carla world
    
    Args:
        vehicle: Vehicle to associate with the sensor
        bp_obstacle_sensor: Blueprint of the sensor
        obstacle_sensor_offset: Offset of the sensor from the center of the associated vehicle
        distance: Distance to trace
        hit_radius: Radius of the trace.
    """
    # Obstacle sensor Features
    bp_obstacle_sensor.set_attribute('distance',str(distance))
    bp_obstacle_sensor.set_attribute('hit_radius',str(hit_radius))
    bp_obstacle_sensor.set_attribute('debug_linetrace','false')      # true = carla crash but we can see it
    
    # Obstacle sensor Spawn
    obstacle_sensor = world.spawn_actor(bp_obstacle_sensor,carla.Transform(obstacle_sensor_offset), attach_to=vehicle)
    actor_list.append(obstacle_sensor)
    print(('Created %s' % obstacle_sensor.type_id) + ' at position: ' + str(obstacle_sensor_offset) + ' from the vehicle')
    
    return obstacle_sensor  
    
def obstacle_detector(event, obstacle_hist, show_debug_obstacle_sensor):
    """Get and record the obstacles detected when the sensor is listening with debug logs
    
    Args:
        vehicle: Vehicle to associate with the sensor
        bp_obstacle_sensor: Blueprint of the sensor
        obstacle_sensor_offset: Offset of the sensor from the center of the associated vehicle
                distance: Distance to trace
        hit_radius: Radius of the trace.
    """
    
    actor_we_detect = event.other_actor.type_id
    obstacle_hist.append(actor_we_detect)
    distance_actor_we_detect = event.distance
    
    if(show_debug_obstacle_sensor):print('\n!!! Obstacle ' + str(actor_we_detect) +' at a distance of ' + str(round(distance_actor_we_detect,2)) +'m detected !!!\n')

# RADAR
def radar_creation(world, actor_list, vehicle, bp_radar_sensor, radar_offset, horizontal_fov, vertical_fov, range, points_per_second, sensor_tick):
    """Create a radar sensor associated to a vehicule in the Carla world. The radar sensor is similar to the LIDAR. It creates a wall of points in front of the sensor with a distance, angle and velocity in relation to it.
    
    Args:
        vehicle: Vehicle to associate with the sensor
        bp_radar_sensor: Blueprint of the sensor
        radar_offset: Offset of the sensor from the center of the associated vehicle
        horizontal_fov: Horizontal field of view in degrees
        vertical_fov: Vertical field of view in degrees
        range: Maximum distance for the lasers to raycast.
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
    
def rad_callback(world,radar_data,show_debug_radar):
    """The radar sensor is similar to de LIDAR. It creates a conic view, and shoots lasers inside to raycast their impacts.
    
    Args:
        radar_data: Data detected when the radar is listening
        show_debug_radar: Debug to see the radar points in the Carla World
    Return:
    
        RadarMeasurement attributes (=> in "radar_data"): (frame=2193914, timestamp=9785.075751, point_count=6)
        frame: Frame count when the data was generated.
        timestamp: Simulation-time when the data was generated. (s)
        Simulation-time when the data was generated.
        
        AND
        
        RadarDetection attributes ( => in "detect"):  Each of these represents one of the points in the cloud generated by the radar
            altitude: Altitude angle in radians.
            azimuth: Azimuth angle in radians.
            depth: Distance in meters.
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
    
    for detect in radar_data:
        
        index+=1
        
        azi = math.degrees(detect.azimuth)
        alt = math.degrees(detect.altitude)
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
        
        # Draw a point when the radar raycasts hit an obstacle
        if(show_debug_radar):world.debug.draw_point(radar_data.transform.location + fw_vec, size=0.075, life_time=0.06, persistent_lines=False, color=carla.Color(r, g, b))

        # Vector of the radar detection, his normalization and the ID of the detection (frame, timestamp and number of point in the radar raycasts cloud)
        if(show_debug_radar):print('- n°'+ str(index) + ' ' + str(detect) + '\nDetection_norm:' + str(normalize_radar(detect)) + ' ; ID = [frame:' + str(radar_data.frame) + ' | timestamp: ' + str(radar_data.timestamp) + ' | points:' + str(len(radar_data)) + ']\n') # + '   detected from ' + str(radar_data.transform) )
        
        
        
        
        #print('azimuth degree : ' + str(math.degrees(detect.azimuth)) + 'altitude degree : ' + str(math.degrees(detect.altitude)))
        #print('radar detection coordonnée: ' + str(position_detect_radar))
        
        
        #world.radar_sensor.stop()
    #print( str(len(radar_data)) + ' radar detections')
        #print('radar data:' + str(radar_data) + '\n')
    
    # To get a numpy [[vel, azimuth, altitude, depth],...[,,,]]:
        #print('altitude:'+str(radar_data.detect.raw_data.altitude))
    '''
    
    print('azimuth:'+str(radar_data.raw_data.azimuth))
    print('depth:'+str(radar_data.raw_data.depth))
    print('velocity:'+str(radar_data.raw_data.velocity))

    '''
    #radar_points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    #radar_points = np.reshape(radar_points, (len(radar_data), 4))
    #print('\nRadar points:\n'+str(radar_points))
    #print('\nCount mabite: '+str(radar_data.detect.get_detection_count()))
    
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


if __name__ == '__main__':

    main()
