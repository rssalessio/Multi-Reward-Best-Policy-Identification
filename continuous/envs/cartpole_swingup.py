# python3
# This file was modified from the BSuite repository, Copyright 2019
# DeepMind Technologies Limited. 
# 
# The file was originally Licensed under the Apache License, Version 2.0
# (the "License"), which  can be found in the root folder of this repository.
# Changes and additions are licensed under the MIT license, 
# Copyright (c) [2023] [NeurIPS authors, 11410] (see the LICENSE file
# in the project root for full information)
# ============================================================================
"""A swing up experiment in Cartpole."""
import numpy as np
from numpy.typing import NDArray
from .cartpole import  CartpoleState, CartpoleConfig, step_cartpole
from typing import NamedTuple, Dict, Tuple, Optional
from utils.random import Generators
from agents.utils import TimeStep


class CartpoleSwingupConfig(NamedTuple):
  height_threshold: float = 0.5
  theta_dot_threshold: float = 1.
  x_reward_threshold: float = 1.
  move_cost: float = 0.1
  x_threshold: float = 3.
  timescale: float = 0.01
  max_time: float = 10.
  init_range: float = 0.05
  num_base_rewards: int = 5
  num_random_rewards: int = 0
  random_rewards: Optional[NDArray[np.float64]] = None

  @property
  def num_rewards(self) -> int:
    return self.num_base_rewards + self.num_random_rewards

class CartpoleSwingup(object):
  NAME = 'CartpoleSwingup'
  """A difficult 'swing up' version of the classic Cart Pole task.

  In this version of the problem the pole begins downwards, and the agent must
  swing the pole up in order to see reward. Unlike the typical cartpole task
  the agent must pay a cost for moving, which aggravates the explore-exploit
  tradedoff. Algorithms without 'deep exploration' will simply remain still.
  """

  def __init__(self,
               config: CartpoleSwingupConfig,
               generators: Generators):
    # Setup.
    assert config.num_rewards > 0, 'Number of rewards needs to be positive'
    self._state: CartpoleState = CartpoleState(0, 0, 0, 0, 0)
    self._config = config
    self._rng = generators.np_gen
    self._init_fn = lambda: self._rng.uniform(low=-config.init_range, high=config.init_range)
    self.num_rewards = config.num_rewards
    self.num_base_rewards = config.num_base_rewards
    self.num_random_rewards = config.num_random_rewards
    

    self._x_base_rewards = np.linspace(-config.x_threshold/2, config.x_threshold/2, config.num_base_rewards)
    if config.num_random_rewards == 0:
      self._x_random_rewards = None
      self._x_rewards = self._x_base_rewards
    else:
      if config.random_rewards is None:
        self._x_random_rewards =  generators.np_gen.uniform(-config.x_threshold/2, config.x_threshold/2, size=config.num_random_rewards)
      else:
        assert config.random_rewards.shape == (config.num_random_rewards,), f'Incorrect shape of {config.random_rewards.shape}. Should be ({config.num_random_rewards})'
        self._x_random_rewards = config.random_rewards
      self._x_rewards = np.concatenate((self._x_base_rewards, self._x_random_rewards))
    

    # Reward/episode logic
    self._height_threshold = config.height_threshold
    self._theta_dot_threshold = config.theta_dot_threshold
    self._x_reward_threshold = config.x_reward_threshold
    self._move_cost = config.move_cost
    self._x_threshold = config.x_threshold
    self._timescale = config.timescale
    self._max_time = config.max_time

    # Problem config
    self._cartpole_config = CartpoleConfig(
        mass_cart=1.,
        mass_pole=0.1,
        length=0.5,
        force_mag=10.,
        gravity=9.8,
    )

  def reset(self) -> NDArray[np.float32]:
    self._reset_next_step = False
    self._state = CartpoleState(
        x=self._init_fn(),
        x_dot=self._init_fn(),
        theta=np.pi + self._init_fn(),
        theta_dot=self._init_fn(),
        time_elapsed=0.,
    )
    self._episode_return = 0.
    return self.observation

  def step(self, action: int) -> Tuple[TimeStep, Dict]:
    current_observation = self.observation.copy()
    if self._reset_next_step:
      raise ValueError('Error! Did you forget to reset the environment at the end of the episode?')

    self._state = step_cartpole(
        action=action,
        timescale=self._timescale,
        state=self._state,
        config=self._cartpole_config,
    )

    # Rewards only when the pole is central and balanced
    is_upright = (np.cos(self._state.theta) > self._height_threshold
                  and np.abs(self._state.theta_dot) < self._theta_dot_threshold
                  and np.abs(self._state.x) < self._x_reward_threshold)
    reward = -1. * np.abs(action - 1) * self._move_cost

    if is_upright:
      reward += 1.
    rewards = self.compute_rewards(action)
      
    #ewards = np.array([reward, 0.5*reward + 0.01*np.random.randn()])
    done = (self._state.time_elapsed > self._max_time or \
             np.abs(self._state.x) > self._x_threshold)

    self._reset_next_step = done
    return TimeStep(current_observation, action, rewards, done, self.observation), {'upright': is_upright}

  def compute_rewards(self, action: int) -> NDArray[np.float64]:
    rewards = np.zeros(self.num_rewards)
    c1 = float(np.cos(self._state.theta) > self._height_threshold)
    c2 = float(np.abs(self._state.theta_dot) < self._theta_dot_threshold)
    c3 = (np.abs(self._state.x - self._x_rewards) < self._x_reward_threshold).astype(np.float64)
    rewards = c1*c2*c3
    move_cost = -1. * np.abs(action - 1) * self._move_cost
    # rewards[0] = np.cos(self._state.theta)
    # rewards[1] = np.abs(self._state.theta_dot) * self._timescale
    rewards = rewards + move_cost
    #print(rewards)
    return rewards
  
  @property
  def config(self):
    config = self._config._replace(random_rewards = self._x_random_rewards)
    return config

  @property
  def num_actions(self) -> int:
    return 3
  
  @property
  def dim_state_space(self) -> int:
    return 8

  @property
  def observation(self) -> NDArray[np.float32]:
    """Approximately normalize output."""
    obs = np.zeros((8), dtype=np.float32)
    obs[0] = self._state.x / self._x_threshold
    obs[1] = self._state.x_dot / self._x_threshold
    obs[2] = np.sin(self._state.theta)
    obs[3] = np.cos(self._state.theta)
    obs[4] = self._state.theta_dot
    obs[5] = self._state.time_elapsed / self._max_time
    obs[6] = 1. if np.abs(self._state.x) < self._x_reward_threshold else -1.
    theta_dot = self._state.theta_dot
    obs[7] = 1. if np.abs(theta_dot) < self._theta_dot_threshold else -1.
    return obs

