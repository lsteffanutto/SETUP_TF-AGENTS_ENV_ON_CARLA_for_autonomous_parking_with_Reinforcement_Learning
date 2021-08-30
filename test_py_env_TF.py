#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# [Lucas STEFFANUTTO]

# First test and comprehension of the tf-agents library 
s
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

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

class CardGameEnv(py_environment.PyEnvironment):

  def __init__(self):
      
    # Describes a numpy array or scalar shape and dtype. An ArraySpec that specifies minimum and maximum values.
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
      

print('compatible: ' + str(tf.compat.v1.enable_v2_behavior()))
environment = CardGameEnv()
print(environment)
print(str(utils.validate_py_environment(environment, episodes=5)))
get_new_card_action = np.array(0, dtype=np.int32)
end_round_action = np.array(1, dtype=np.int32)

environment = CardGameEnv()
time_step = environment.reset()
print(time_step)
cumulative_reward = time_step.reward

for _ in range(3):
  time_step = environment.step(get_new_card_action)
  print(time_step)
  cumulative_reward += time_step.reward

time_step = environment.step(end_round_action)
print(time_step)
cumulative_reward += time_step.reward
print('Final Reward = ', cumulative_reward)

# Validates the environment follows the defined specs.
tf_env = tf_py_environment.TFPyEnvironment(environment)

print('\n\nWRAPP PYTHON ENVIRONMENT TO TENSORFLOW ENVIRONMENT DONE ?\n\n' + str(isinstance(tf_env, tf_environment.TFEnvironment)))
print("\nTimeStep Specs:", tf_env.time_step_spec())
print("\nAction Specs:", tf_env.action_spec())

print('\n\nRESET THE ENVIRONMENT AND TEST TO COMPUTE 3 STEPS\n\n')

# reset() creates the initial time_step after resetting the environment.
time_step = tf_env.reset()
num_steps = 3
transitions = []
reward = 0

for i in range(num_steps):
    
  action = tf.constant([i % 2])
  
  # applies the action and returns the new TimeStep.
  next_time_step = tf_env.step(action)
  transitions.append([time_step, action, next_time_step])
  reward += next_time_step.reward
  
  print('next step = ' + str(next_time_step) )
  print('reward after step n°' + str(i) + ' = ' + str(next_time_step.reward) )
  
  time_step = next_time_step

np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)

print('\n'.join(map(str, np_transitions)))
print('Total reward:', reward.numpy())

print('\n\nRESET THE ENVIRONMENT AND 5 episodes\n\n')

time_step = tf_env.reset()
rewards = []
steps = []
num_episodes = 5

for _ in range(num_episodes):
  
  print('\nepisode n°' + str(_) + '\n')
  
  episode_reward = 0
  episode_steps = 0
  
  while not time_step.is_last():
      
    action = tf.random.uniform([1], 0, 2, dtype=tf.int32)
    time_step = tf_env.step(action)
    episode_steps += 1
    episode_reward += time_step.reward.numpy()
    
    print('episode step:' + str(episode_steps))
    print('action:' + str(action))
    print('episode reward:' + str(episode_reward))

  print('episode total steps:' + str(episode_steps))
  print('episode total reward:' + str(episode_reward))  
  print('episode final time_step:' + str(time_step) +'\n')
    
  rewards.append(episode_reward) #tab of total reward for each episode 
  rewards.append(episode_reward) #tab of total reward for each episode 
  steps.append(episode_steps)    
  time_step = tf_env.reset()
  
  

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)

print('\nRESULTS:\n')
print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)


