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
"""The Cartpole reinforcement learning environment."""

import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray


class CartpoleState(NamedTuple):
  x: float
  x_dot: float
  theta: float
  theta_dot: float
  time_elapsed: float

class CartpoleConfig(NamedTuple):
  mass_cart: float
  mass_pole: float
  length: float
  force_mag: float
  gravity: float



def step_cartpole(action: int,
                  timescale: float,
                  state: CartpoleState,
                  config: CartpoleConfig) -> CartpoleState:
  """Helper function to step cartpole state under given config."""
  # Unpack variables into "short" names for mathematical equation
  force = (action - 1) * config.force_mag
  cos = np.cos(state.theta)
  sin = np.sin(state.theta)
  pl = config.mass_pole * config.length
  l = config.length
  m_pole = config.mass_pole
  m_total = config.mass_cart + config.mass_pole
  g = config.gravity

  # Compute the physical evolution
  temp = (force + pl * state.theta_dot**2 * sin) / m_total
  theta_acc = (g * sin - cos * temp) / (l * (4/3 - m_pole * cos**2 / m_total))
  x_acc = temp - pl * theta_acc * cos / m_total

  # Update states according to discrete dynamics
  x = state.x + timescale * state.x_dot
  x_dot = state.x_dot + timescale * x_acc
  theta = np.remainder(
      state.theta + timescale * state.theta_dot, 2 * np.pi)
  theta_dot = state.theta_dot + timescale * theta_acc
  time_elapsed = state.time_elapsed + timescale

  return CartpoleState(x, x_dot, theta, theta_dot, time_elapsed)

