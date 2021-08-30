#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# [Lucas STEFFANUTTO]

# RL loop and training of the agent with tf-agents

import glob
import os
import sys

from tf_agents.environments import utils

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import abc
#import tensorflow as tf
import carla
import numpy as np

import math
import random
import time
from parking_env import CarlaParkingEnv as env
import utils_carla


# Creation of a parking environment
env_UC3 = env()

# Episode reset
time_step = env_UC3._reset()
print('\nTime step return after reset in train.py:\n' + str(time_step))
#print('\n\ntime_step_spec.step_type:\n\n', str(type(env_UC3.time_step_spec().step_type)))

# Verifications of the RL spaces setup
'''
print("\nVerifications of the environment setup:\n")
print('action_spec:', env_UC3.action_spec())
print('time_step_spec.observation: ', env_UC3.time_step_spec().observation)
print('time_step_spec.step_type: ', env_UC3.time_step_spec().step_type)
print('time_step_spec.discount: ', env_UC3.time_step_spec().discount)
print('time_step_spec.reward:' , env_UC3.time_step_spec().reward)

print('dtypes reward: ' + str(env_UC3.time_step_spec().reward.dtype) + ' dtypes obs: ' + str(env_UC3.time_step_spec().observation.dtype) + ' dtypes ste_type: ' + str(env_UC3.time_step_spec().step_type.dtype) + ' dtypes discount: ' + str(env_UC3.time_step_spec().discount.dtype))
'''
# !!!! DEBUG THE FOLLOWING LINE !!!!!
# DEBUG TO SEE IF THE PYTHON ENVIRONMENT IS VALID FOR TENSORFLOW WRAPPING
print('\n\nY\'a un bug ICI\n' + (utils.validate_py_environment(env_UC3, episodes=1)))

# Test of the environment with a simple action for one step
'''
print("\nTest with same action: [Throttle=1.0, Steer=0.0, Brake=0.0]\n")
action = np.array([1.0, 0.0, 0.0])
print("Action is: " + str(action))
'''

# Test of the environment with the same simple action for one episode
'''            
while not env_UC3._episode_ended:
  time_step = env_UC3._step(action)
  print('time step during same action is: ' + str(time_step))
print('total time = ' + time.time() -  start_time) 
'''

# TensorFlow Wrapping of the environment for agent training
'''
tf_env = tf_py_environment.TFPyEnvironment(env_UC3)
'''


