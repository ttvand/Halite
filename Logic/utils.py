import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import base64 as b64
from kaggle_environments import make as make_environment
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import random
import tempfile
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
import yaml


###############################################################################
# Acting utilities                                                            #
###############################################################################

NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
CONVERT = "CONVERT"
SHIP_NONE = "SHIP_NONE"
GO_NEAREST_BASE = "GO_NEAREST_BASE"
SPAWN = "SPAWN"
BASE_NONE = "BASE_NONE"
ACTION_MAPPING = {
  # Ship actions
  0: NORTH,
  1: SOUTH,
  2: EAST,
  3: WEST,
  4: CONVERT,
  5: SHIP_NONE,
  6: GO_NEAREST_BASE,
  
  # Base (ship yard) actions. Separate since ships can be at bases and actions
  # can be selected independently.
  7: SPAWN,
  8: BASE_NONE,
  }
INVERSE_ACTION_MAPPING= {v: k for k, v in ACTION_MAPPING.items()}

def get_input_output_shapes(config):
  # Create a new environment, perform the preprocessing and record the shape
  env = make_environment('halite')
  env.reset(num_agents=config['num_agents_per_game'])
  env_configuration = env.configuration
  env_observation = env.state[0].observation
  obs_input = state_to_input(structured_env_obs(
    env_configuration, env_observation, active_id=0),
    num_mirror_dim=config['num_mirror_dim'])
  num_actions = len(ACTION_MAPPING)
  
  return obs_input.shape, num_actions, config['num_q_functions']

def get_action_costs():
  # Create a new environment, read the config and record the action costs
  env = make_environment('halite')
  action_costs = np.zeros((len(ACTION_MAPPING)))
  for k in ACTION_MAPPING:
    if ACTION_MAPPING[k] == CONVERT:
      action_costs[k] = env.configuration.convertCost
    elif ACTION_MAPPING[k] == SPAWN:
      action_costs[k] = env.configuration.spawnCost
    else:
      action_costs[k] = 0
      
  return action_costs

def add_action_costs_to_config(config):
  config['action_costs'] = get_action_costs()

def row_col_from_square_grid_pos(pos, size):
  col = pos % size
  row = pos // size
  
  return row, col

def get_base_pos(base_data, grid_size):
  base_pos = np.zeros((grid_size, grid_size), dtype=np.bool)
  for _, v in base_data.items():
    row, col = row_col_from_square_grid_pos(v, grid_size)
    base_pos[row, col] = 1
  
  return base_pos

def get_col_row(size, pos):
  return (pos % size, pos // size)

def move_ship_pos(pos, direction, size):
  col, row = get_col_row(size, pos)
  if direction == "NORTH":
    return pos - size if pos >= size else size ** 2 - size + col
  elif direction == "SOUTH":
    return col if pos + size >= size ** 2 else pos + size
  elif direction == "EAST":
    return pos + 1 if col < size - 1 else row * size
  elif direction == "WEST":
    return pos - 1 if col > 0 else (row + 1) * size - 1

def get_ship_halite_pos(ship_data, grid_size):
  ship_pos = np.zeros((grid_size, grid_size), dtype=np.bool)
  ship_halite = np.zeros((grid_size, grid_size), dtype=np.float32)
  for _, v in ship_data.items():
    row, col = row_col_from_square_grid_pos(v[0], grid_size)
    ship_pos[row, col] = 1
    ship_halite[row, col] = v[1]
  
  return ship_pos, ship_halite

def structured_env_obs(env_configuration, env_observation, active_id):
  grid_size = env_configuration.size
  halite = np.array(env_observation['halite']).reshape([
    grid_size, grid_size])
  
  num_episode_steps = env_configuration.episodeSteps
  step = env_observation.step
  relative_step = step/(num_episode_steps-2)
  
  num_agents = len(env_observation.players)
  rewards_bases_ships = []
  for i in range(num_agents):
    player_obs = env_observation.players[i]
    reward = player_obs[0]
    base_pos = get_base_pos(player_obs[1], grid_size)
    ship_pos, ship_halite = get_ship_halite_pos(player_obs[2], grid_size)
    rewards_bases_ships.append((reward, base_pos, ship_pos, ship_halite))
    
  # Move the agent's rewards_bases_ships to the front of the list
  agent_vals = rewards_bases_ships.pop(active_id)
  rewards_bases_ships = [agent_vals] + rewards_bases_ships
  
  return {
    'halite': halite,
    'relative_step': relative_step,
    'rewards_bases_ships': rewards_bases_ships,
    'step': step,
    }
  

def select_action_from_q(valid_actions, q_values, epsilon_greedy,
                         exploration_parameter, pick_first_on_tie):
  if valid_actions.dtype == np.bool:
    valid_actions = np.where(valid_actions)[0]
  
  if epsilon_greedy:
    # Select the best valid or a valid exploratory action
    best_q = q_values[valid_actions].max()
    best_a_ids = np.where(q_values[valid_actions] == best_q)[0]
    if pick_first_on_tie:
      best_a_ids = best_a_ids[:1]
    best_a = valid_actions[np.random.choice(best_a_ids)]
    exploratory_a = np.random.choice(valid_actions)
    explore = np.random.uniform() < exploration_parameter
    action = exploratory_a if explore else best_a
  else:
    # -500 to avoid overflow when exploration parameter is in [1e-3, 1]
    if exploration_parameter == 0:
      # Select the best valid action
      best_q = q_values[valid_actions].max()
      best_a_ids = np.where(q_values[valid_actions] == best_q)[0]
      if pick_first_on_tie:
        best_a_ids = best_a_ids[:1]
      action = valid_actions[np.random.choice(best_a_ids)]
    else:
      temp_valid_qs = np.array(
        [np.exp(q/exploration_parameter-500) for q in q_values[valid_actions]])
      sampled_q_value = np.random.choice(temp_valid_qs,
                                         p=temp_valid_qs/temp_valid_qs.sum())
      sampled_a_ids = np.where(temp_valid_qs == sampled_q_value)[0]
      if pick_first_on_tie:
        sampled_a_ids = sampled_a_ids[:1]
      action = valid_actions[np.random.choice(sampled_a_ids)]
    
  return action

# Predict network outputs where the number of inputs can be large. Use batching
# when there are more inputs than the max_batch_size
def my_keras_predict(model, inputs, max_batch_size=200):
  num_inputs = inputs[0].shape[0]
  num_batches = int(np.ceil(num_inputs/max_batch_size))
  
  outputs = []
  for i in range(num_batches):
    end_id = num_inputs if i == (num_batches-1) else (i+1)*max_batch_size
    batch_inputs = [input_mod[i*max_batch_size:end_id] for input_mod in inputs]
    batch_inputs = [b.astype(np.float32) for b in batch_inputs]
    if tf.__version__[0] == '2':
      # import pdb; pdb.set_trace()
      # import time; t=time.time(); model(batch_inputs); print(time.time()-t)
      batch_outputs = model(batch_inputs)
      if isinstance(batch_outputs, list):
        batch_outputs = [b.numpy() for b in batch_outputs]
      else:
        batch_outputs = [batch_outputs.numpy()]
    else:
      batch_outputs = [model.predict(batch_inputs)]
    outputs.append(batch_outputs)
    
  # Transpose the outputs
  outputs_transposed = list(map(list, zip(*outputs)))
  return [np.concatenate(o) for o in outputs_transposed]

def get_key_q_valid(q_values, player_obs, configuration, rewards_bases_ships):
  shipyard_keys = list(player_obs[1].keys())
  ship_keys = list(player_obs[2].keys())
  grid_size = configuration.size
  any_bases = np.stack([o[1] for o in rewards_bases_ships]).sum(0)
  
  key_q_valid = []
  
  num_actions = len(ACTION_MAPPING)
  
  for k in ship_keys:
    # Declaration of ship_valid in the loop to avoid overwriting with define
    # by reference (otherwise this breaks!)
    ship_valid = np.ones((num_actions), dtype=np.bool)
    ship_valid[INVERSE_ACTION_MAPPING[SPAWN]:] = 0
    ship_valid[INVERSE_ACTION_MAPPING[GO_NEAREST_BASE]] = bool(shipyard_keys)
    row, col = row_col_from_square_grid_pos(player_obs[2][k][0], grid_size)
    
    # No converting at the location of my or opponent bases
    ship_valid[INVERSE_ACTION_MAPPING[CONVERT]] = not any_bases[row, col]
    
    # Overwrite rule: if the ship has halite and is at one of its bases:
    # Don't move and deposit all halite cargo.
    if any_bases[row, col] and player_obs[2][k][1] > 0:
      ship_valid = np.zeros((num_actions), dtype=np.bool)
      ship_valid[INVERSE_ACTION_MAPPING[SHIP_NONE]] = 1
    
    key_q_valid.append((k, q_values[row, col], ship_valid, row, col, True))
  
  for k in shipyard_keys:
    # Declaration of base_valid in the loop to avoid overwriting with define
    # by reference (otherwise this breaks!)
    base_valid = np.ones((num_actions), dtype=np.bool)
    base_valid[:INVERSE_ACTION_MAPPING[SPAWN]] = 0
    base_valid[INVERSE_ACTION_MAPPING[GO_NEAREST_BASE]] = 0
    row, col = row_col_from_square_grid_pos(player_obs[1][k], grid_size)
    
    # Commented since it appears that ship moves are processed before detecting
    # collisions - filter out self destructive actions after sampling the ship
    # actions.
    # # No spawning when my agent currently has a ship at the base (clear waste)
    # base_valid[INVERSE_ACTION_MAPPING["SPAWN"]] = not rewards_bases_ships[
    #   0][2][row, col]
    key_q_valid.append((k, q_values[row, col], base_valid, row, col, False))
  
  return key_q_valid

def get_direction_nearest_base(observation, row, col, grid_size):
  num_bases = len(observation[1])
  horiz_distances = np.zeros((num_bases))
  vert_distances = np.zeros((num_bases))
  base_distances = np.zeros((num_bases))
  base_keys = list(observation[1].keys())
  for i, k in enumerate(base_keys):
    base_row, base_col = row_col_from_square_grid_pos(
      observation[1][k], grid_size)
    if base_row == row and base_col == col:
      return SHIP_NONE
    
    horiz_diff = base_col-col
    horiz_dist = min(np.abs(horiz_diff),
      min(np.abs(horiz_diff-grid_size), np.abs(horiz_diff+grid_size)))
    vert_diff = base_row-row
    vert_dist = min(np.abs(vert_diff),
      min(np.abs(vert_diff-grid_size), np.abs(vert_diff+grid_size)))
    horiz_distances[i] = horiz_dist
    vert_distances[i] = vert_dist
    base_distances[i] = horiz_dist + vert_dist
    
  shortest_distance = base_distances.min()
  shortest_ids = np.where(base_distances == shortest_distance)[0]
  half_grid = grid_size / 2
  shortest_directions = []
  for i in shortest_ids:
    base_row, base_col = row_col_from_square_grid_pos(
      observation[1][k], grid_size)
    if horiz_distances[i] > 0:
      if base_col > col:
        shortest_dir = EAST if (base_col - col) <= half_grid else WEST
      else:
        shortest_dir = WEST if (col - base_col) <= half_grid else EAST
      shortest_directions.append(shortest_dir)
    if vert_distances[i] > 0:
      if base_row > row:
        shortest_dir = SOUTH if (base_row - row) <= half_grid else NORTH
      else:
        shortest_dir = NORTH if (row - base_row) <= half_grid else SOUTH
      shortest_directions.append(shortest_dir)
      
  # Intentional: Do not only consider unique directions (
  # empowerment! Prefer the hill top to keep options open)
  return str(np.random.choice(shortest_directions))
  

# Q value based main function to act a single step
def get_agent_q_and_a(network, observation, player_obs, configuration,
                      epsilon_greedy, exploration_parameter, num_mirror_dim,
                      action_costs, pick_first_on_tie):
  # Preprocess the state so it can be fed in to the network
  obs_input = [np.expand_dims(state_to_input(
    observation, mirror_at_edges=False), 0)]
  obs_input_mirrored = [mirror_observation_edges(obs_input[0], num_mirror_dim)]
  
  # Obtain the q values
  q_values = my_keras_predict(network, obs_input_mirrored)[0][0]
  
  # Determine valid actions for each of the ships/shipyards
  all_key_q_valid = get_key_q_valid(
    q_values, player_obs, configuration, observation['rewards_bases_ships'])
  
  all_mapped_actions = {}
  grid_size = configuration.size
  num_actions = len(ACTION_MAPPING)
  actions = -1*np.ones((grid_size, grid_size, 3)).astype(np.int32)
  valid_actions = np.zeros((grid_size, grid_size, num_actions, 2),
                           dtype=np.bool)
  action_budget = player_obs[0]
  
  for i, (k, q_sub_values, valid_sub_actions, r, c, is_ship) in enumerate(
      all_key_q_valid):
    # Set actions we can't afford to invalid
    valid_sub_actions &= action_costs <= action_budget
    action_id = select_action_from_q(valid_sub_actions, q_sub_values,
                                     epsilon_greedy, exploration_parameter,
                                     pick_first_on_tie)
    valid_actions[r, c, :, int(is_ship)] = valid_sub_actions
    actions[r, c, int(is_ship)] = action_id
    action_budget -= action_costs[action_id]
    mapped_action = ACTION_MAPPING[action_id]
    if mapped_action == GO_NEAREST_BASE:
      mapped_action = get_direction_nearest_base(player_obs, r, c, grid_size)
      actions[r, c, 1] = INVERSE_ACTION_MAPPING[mapped_action]
      actions[r, c, 2] = action_id
    if not mapped_action in [SHIP_NONE, BASE_NONE]:
      all_mapped_actions[k] = mapped_action
      
  return (obs_input, q_values, actions, all_mapped_actions, valid_actions)


###############################################################################
# Learning utilities                                                          #
###############################################################################

# One step mixed q learning target computation
def one_step_mixed_q_targets(next_mixed_q_vals, experience):
  terminal_rewards = np.array([e.episode_reward for e in experience])
  last_episode_actions = np.array([e.last_episode_action for e in experience])
  target_qs = np.where(last_episode_actions, terminal_rewards,
                       next_mixed_q_vals).astype(np.float32)
  
  return target_qs

def halite_change_target_qs(experience, episode_ids, this_q_vals,
                            discount_factor, num_agents,
                            halite_normalizer=1e4):
  # Filter out next Q-values where the action is not valid. Filtered out since
  # every target computation performs a max operation.
  mean_best_valid_qs_filled, offsets, num_unit_actions = (
    get_mean_best_valid_qs_filled(
      experience, this_q_vals, episode_ids, num_agents))
  
  # Obtain the next q-values for all agents
  next_q_vals = np.concatenate([mean_best_valid_qs_filled[num_agents:],
                                np.full((num_agents), -999)])[offsets]
  next_q_vals = next_q_vals.astype(np.float32)
  
  rewards = np.array([e.halite_change for e in experience])
  rewards = np.sign(rewards)*np.sqrt(np.abs(rewards))/halite_normalizer
  last_episode_actions = np.array([e.last_episode_action for e in experience])
  discounts = np.ones_like(rewards)*discount_factor
  discounts[last_episode_actions] = 0
  target_qs = rewards + discounts*next_q_vals
  
  return target_qs
  
def get_mean_best_valid_qs_filled(experience, this_q_vals, episode_ids,
                                  num_agents):
  valid_actions = np.stack([e.valid_actions for e in experience])
  num_unit_actions = (valid_actions.sum(-2) > 0).sum((1, 2))
  best_q_sums = (valid_actions*np.expand_dims(this_q_vals, -1)).max(3).sum(
    (1, 2))
  mean_best_valid_qs = (best_q_sums.sum(-1)/num_unit_actions.sum(-1))
  
  # Compute the episode id offsets (agent and episode step offsets are easy).
  episode_steps = np.array([e.episode_step for e in experience])
  agent_ids = np.array([e.active_id for e in experience])
  episode_step_counts = np.bincount(episode_ids)
  last_step_ids = np.cumsum(episode_step_counts)-1
  episode_durations = episode_steps[last_step_ids]+1
  episode_offsets = np.zeros_like(episode_steps)
  start_id = 0
  for i in range(episode_durations.size):
    offset = np.sum(episode_durations[:i])*num_agents
    episode_offsets[start_id:(start_id + episode_step_counts[i])] = offset
    start_id += episode_step_counts[i]
  
  # Set Q values of inactive agents to zero and fill the other data
  # appropriately
  offsets = episode_offsets+episode_steps*num_agents+agent_ids
  mean_best_valid_qs_filled = np.zeros((episode_durations.sum()*num_agents))
  mean_best_valid_qs_filled[offsets] = mean_best_valid_qs
  
  return mean_best_valid_qs_filled, offsets, num_unit_actions

def aggregate_next_mixed_q_vals(
    experience, episode_ids, this_q_vals, num_agents):
  mean_best_valid_qs_filled, offsets, _ = get_mean_best_valid_qs_filled(
    experience, this_q_vals, episode_ids, num_agents)
  
  # Obtain the next q-values for all agents
  mean_best_valid_qs = mean_best_valid_qs_filled.reshape(
    (-1, num_agents))
  
  next_q_vals = np.zeros_like(mean_best_valid_qs_filled, dtype=np.float)
  
  # Assume that the terminal reward is 0.5-mean for now
  active_agents_count = (mean_best_valid_qs > 0).sum(1)
  q_sum_target_options = np.cumsum(np.flip(np.arange(num_agents)/(num_agents-1)))
  q_sum_targets = q_sum_target_options[active_agents_count-1]
  for i in range(num_agents):
    other_cols = np.arange(num_agents)
    other_cols = np.delete(other_cols, i)
    this_q_vals = mean_best_valid_qs[:, i]
    other_q_vals = mean_best_valid_qs[:, other_cols].sum(-1)
    
    next_q_vals[i::num_agents] = this_q_vals/(
      this_q_vals+other_q_vals)*q_sum_targets
    
  next_q_vals = np.concatenate([next_q_vals[num_agents:],
                                np.full((num_agents), np.inf)])[offsets]
  next_q_vals = next_q_vals.astype(np.float32)
  
  return next_q_vals

def terminal_target_qs(experience, episode_ids, this_q_vals,
                       num_agents_per_game):
  # Filter out next Q-values where the action is not valid. Filtered out since
  # every target computation performs a max operation.
  next_mixed_q_vals = aggregate_next_mixed_q_vals(
    experience, episode_ids, this_q_vals, num_agents_per_game)
    
  # Compute the target action values
  target_qs = one_step_mixed_q_targets(next_mixed_q_vals, experience)
  
  return target_qs
  
def get_all_target_qs(this_states, experience, target_qs, nan_coding_value,
                      this_q_vals, num_mirror_dim):
  # Set the target q values for the actions that were taken
  num_actions = len(ACTION_MAPPING)
  actions = np.stack([e.actions for e in experience])
  one_hot_base_actions = (
    np.arange(num_actions) == actions[..., 0, None]).astype(bool)
  one_hot_ship_actions = (
    np.arange(num_actions) == actions[..., 1, None]).astype(bool)
  
  # Address the go nearest base tied credit assignment
  one_hot_ship_actions[
    :, :, :, INVERSE_ACTION_MAPPING[GO_NEAREST_BASE]] = (
      actions[..., 2] == INVERSE_ACTION_MAPPING[GO_NEAREST_BASE])
      
  one_hot_actions = one_hot_base_actions + one_hot_ship_actions
  
  # Set to the nan coding when the action was not selected
  grid_size = this_q_vals.shape[1]
  all_target_qs = np.where(
    one_hot_actions,
    np.tile(target_qs.reshape((-1, 1, 1, 1)),
            [1, grid_size, grid_size, num_actions]),
    nan_coding_value*np.ones_like(this_q_vals))
  
  return all_target_qs

# Get the q learning observations, targets and observation weights
def q_learning(model, experience, episode_ids, nan_coding_value,
               symmetric_experience, num_agents_per_game, num_mirror_dim,
               reward_type, halite_change_discount):
  # Evaluate the q values of the current and next state for all observations
  this_states = np.stack([e.current_obs for e in experience])
  this_states = mirror_observation_edges(this_states, num_mirror_dim)
  this_q_vals = my_keras_predict(model, [this_states])[0]
  
  if reward_type == "Terminal":
    target_qs = terminal_target_qs(experience, episode_ids, this_q_vals,
                                   num_agents_per_game)
  elif reward_type == "Halite change":
    target_qs = halite_change_target_qs(
      experience, episode_ids, this_q_vals, halite_change_discount,
      num_agents_per_game)
  else:
    raise ValueError("Not implemented")
  
  all_target_qs = get_all_target_qs(this_states, experience, target_qs,
                                    nan_coding_value, this_q_vals,
                                    num_mirror_dim)
  
  if symmetric_experience:
    # TODO: Add symmetric observations and symmetric targets.
    # IMPORTANT: mind the violation of symmetry in the actions
    pass
    
  return this_states, all_target_qs

# Masked mse loss - values equal to mask_val are ignored in the loss
def masked_mse(y, p, mask_val):
  mask = K.cast(K.not_equal(y, mask_val), K.floatx())
  if tf.__version__[0] == '2':
    masked_loss = tf.losses.mse(y*mask, p*mask)
  else:
    mask = K.cast(mask, 'float32')
    masked_loss = K.mean(tf.math.square(p*mask - y*mask), axis=-1)
    # masked_loss = tf.compat.v1.losses.mean_squared_error(y*mask, p*mask)
    
  return masked_loss
    
# Make the masked mse loss
def make_masked_mse(nan_coding_value):
  def loss(y, p):
    return masked_mse(y, p, mask_val=nan_coding_value)
  
  return loss


###############################################################################
# Data utilities                                                              #
###############################################################################

def make_keras_picklable():
  def __getstate__(self):
    model_str = ""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as fd:
      tf.keras.models.save_model(self, fd.name, overwrite=True)
      model_str = fd.read()
    d = {'model_str': model_str }
    return d

  def __setstate__(self, state):
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as fd:
      fd.write(state['model_str'])
      fd.flush()
      model = tf.keras.models.load_model(fd.name)
    self.__dict__ = model.__dict__


  cls = tf.keras.models.Model
  cls.__getstate__ = __getstate__
  cls.__setstate__ = __setstate__

# Increment the iteration id of some .h5 model path
def increment_iteration_id(path, extension='.h5'):
  underscore_parts = path.split('_')
  id_ = int(underscore_parts[-1].split('.')[0])
  incremented_id = id_+1
  new_path = '_'.join(underscore_parts[:-1]) + '_' + str(
    incremented_id ) + extension
  
  return new_path

# Load all keras models for the given paths
def load_models(paths):
  models = []
  for p in paths:
    if tf.__version__[0] == '2':
      K.clear_session()
    models.append(load_model(p, custom_objects={'loss': make_masked_mse(0)}))
    
  return models

# Get the path in a folder with the highest iteration id
def get_most_recent_agent_extension(folder, extension='.h5'):
  ext_length = len(extension)
  agent_paths = [f for f in os.listdir(folder) if f[-ext_length:]==extension]
  
  # Order the agent models on their iteration ids
  agent_paths = decreasing_sort_on_iteration_id(agent_paths, extension)
  most_recent_agent_extension = agent_paths[0]
  
  return most_recent_agent_extension

# Order paths in a decreasing manner (sort on iteration id)
def decreasing_sort_on_iteration_id(paths):
  # Drop everything after the underscore and before the dot to get the ids
  ids = np.array([p.split('_')[1].split('.')[0] for p in paths]).astype(np.int)
  paths_np = np.array(paths)
  
  sort_ids = np.argsort(-ids)
  return paths_np[sort_ids].tolist()

def player_state_to_array(player_state, reward_divisor, ship_divisor):
  base_pos = np.expand_dims(player_state[1], -1)
  ship_pos = np.expand_dims(player_state[2], -1)
  ship_halite = np.expand_dims(np.sqrt(player_state[3]), -1)/ship_divisor
  reward = np.ones_like(base_pos)*np.sqrt(player_state[0])/reward_divisor
  
  out = np.concatenate(
    [reward, base_pos, ship_pos, ship_halite], -1)
  
  return out

def mirror_observation_edges(observation, num_mirror_dim):
  if num_mirror_dim > 0:
    # observation = np.arange(450).reshape((1,15,15,2)) # Debugging test
    assert len(observation.shape) == 4
    obs_shape = observation.shape
    grid_size = obs_shape[1]
    new_grid_size = grid_size + 2*num_mirror_dim
    mirrored_obs = np.full(
      (obs_shape[0], new_grid_size, new_grid_size, obs_shape[3]), np.nan)
    
    # Fill in the original data
    mirrored_obs[:, num_mirror_dim:(-num_mirror_dim),
                 num_mirror_dim:(-num_mirror_dim), :] = observation
    
    # Add top and bottom mirrored data
    mirrored_obs[:, :num_mirror_dim, num_mirror_dim:(-num_mirror_dim),
                 :] = observation[:, -num_mirror_dim:, :, :]
    mirrored_obs[:, -num_mirror_dim:, num_mirror_dim:(-num_mirror_dim),
                 :] = observation[:, :num_mirror_dim, :, :]
    
    # Add left and right mirrored data
    mirrored_obs[:, :, :num_mirror_dim, :] = mirrored_obs[
      :, :, -(2*num_mirror_dim):(-num_mirror_dim), :]
    mirrored_obs[:, :, -num_mirror_dim:, :] = mirrored_obs[
      :, :, num_mirror_dim:(2*num_mirror_dim), :]
    
    observation = mirrored_obs
  
  return observation

# Combine the state to a single numpy array so it can be fed in to the
# network
def state_to_input(observation, reward_halite_divisor=1e4,
                   cell_halite_divisor=1e4, ship_halite_divisor=1e4,
                   combine_opponent_data=True, mirror_at_edges=True,
                   num_mirror_dim=0):
  # Combine the Halite count, relative step fraction, my and opponent
  # ship and base information to a single numpy array
  halite_count = np.maximum(0, np.expand_dims(observation['halite'], -1))
  log_halite_count = np.log10(1+halite_count)
  halite_count = np.sqrt(halite_count)/cell_halite_divisor
  relative_step = np.ones_like(halite_count)*observation['relative_step']
  my_data = player_state_to_array(
    observation['rewards_bases_ships'][0], reward_halite_divisor,
    ship_halite_divisor)
  
  opponent_data = []
  for i in range(1, len(observation['rewards_bases_ships'])):
    opponent_data.append(player_state_to_array(
      observation['rewards_bases_ships'][i], reward_halite_divisor,
      ship_halite_divisor))
  
  # Optional: combine opponent data - sum bases and ships and drop opp rewards
  if combine_opponent_data:
    opponent_data = [np.stack(opponent_data).sum(0)[:, :, 1:]]
  
  state_inputs = np.concatenate(
    [halite_count, log_halite_count, relative_step, my_data] + opponent_data,
    -1).astype(np.float32)
  
  # Optional: mirror the observation at the boundaries (donut environment)
  if mirror_at_edges:
    state_inputs = mirror_observation_edges(
      np.expand_dims(state_inputs, 0), num_mirror_dim)[0]
  
  return state_inputs

# Custom class to reuse data of subsequent interations with the environment
# FIFO buffer. 
class ExperienceBuffer:
  def __init__(self, buffer_size):
    self.buffer_size = buffer_size
    self.episode_offset = 0
    self.data = []
    self.episode_ids = np.array([], dtype=np.int)
    
  def add(self, data):
    episode_ids = np.array([d.game_id for d in data])
    num_episodes = episode_ids[-1] + 1
    if num_episodes > self.buffer_size:
      # Keep most recent experience of the experience batch
      data = data[
        np.where(episode_ids == (num_episodes-self.buffer_size))[0][0]:]
      self.data = data
      self.episode_ids = episode_ids
      self.episode_offset = 0
      return
      
    episode_ids = episode_ids + self.episode_offset
    self.data += data
    self.episode_ids = np.concatenate([self.episode_ids, episode_ids])
    
    unique_episode_ids = pd.unique(self.episode_ids)
    if unique_episode_ids.size > self.buffer_size:
      cutoff_index = np.where(self.episode_ids == unique_episode_ids[
        unique_episode_ids.size-self.buffer_size])[0][0]
      self.data = self.data[cutoff_index:]
      self.episode_ids = self.episode_ids[cutoff_index:]
    self.episode_offset += num_episodes
    
  def get_data(self):
    return self.data
  
  def size(self):
    return len(self.data)
  

###############################################################################
# Vizualization utilities                                                     #
###############################################################################

def update_learning_progress(experiment_name, data_vals): 
  # Append a line to the learning progress line if the file exists. Otherwise:
  # create it
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Deep Learning Agents/' + experiment_name)
  progress_path = os.path.join(agents_folder, 'learning_progress.csv')
  
  if os.path.exists(progress_path):
    progress = pd.read_csv(progress_path)
    progress.loc[progress.shape[0]] = data_vals
  else:
    progress = pd.DataFrame(data_vals, index=[0])
  
  progress.to_csv(progress_path, index=False)
  
  
def record_videos(agent_path, num_agents_per_game, num_mirror_dim,
                  extension_override=None):
  print("Generating videos of iteration {}".format(agent_path))
  model = load_models([agent_path])[0]
  action_costs = get_action_costs()
  
  def my_agent(observation, env_configuration):
    active_id = observation.player
    current_observation = structured_env_obs(
      env_configuration, observation, active_id)
    player_obs = observation.players[active_id]
    
    # Preprocess the state so it can be fed in to the network
    obs_input = np.expand_dims(
      state_to_input(current_observation, num_mirror_dim=num_mirror_dim), 0)
    
    # Obtain the q values
    q_values = model(obs_input).numpy()[0]
    
    # Determine valid actions for each of the ships/shipyards
    all_key_q_valid = get_key_q_valid(
      q_values, player_obs, env_configuration,
      current_observation['rewards_bases_ships'])
    
    mapped_actions = {}
    action_budget = player_obs[0]
    
    for i, (k, q_sub_values, valid_sub_actions, r, c, _) in enumerate(
        all_key_q_valid):
      # Set actions we can't afford to invalid
      valid_sub_actions &= action_costs <= action_budget
      valid_sub_actions = np.where(valid_sub_actions)[0]
      best_q = q_sub_values[valid_sub_actions].max()
      best_a_id = np.where(q_sub_values[valid_sub_actions] == best_q)[0][0]
      action_id = valid_sub_actions[best_a_id]
      
      # Hard coded epsilon greedy exploration
      if np.random.uniform() < 0.05:
        action_id = np.random.choice(valid_sub_actions)
      
      action_budget -= action_costs[action_id]
      mapped_action = ACTION_MAPPING[action_id]
      if mapped_action == GO_NEAREST_BASE:
        mapped_action = get_direction_nearest_base(
          player_obs, r, c, env_configuration.size)
      if not mapped_action in [SHIP_NONE, BASE_NONE]:
        mapped_actions[k] = mapped_action
       
    return mapped_actions
  
  env = make_environment(
    "halite", configuration={"agentExec": "LOCAL"})#, configuration={"agentTimeout": 10000, "actTimeout": 10000})
  for video_type in ["random opponent", "self play"]:
    env.reset(num_agents=num_agents_per_game)
    agents = [my_agent]*num_agents_per_game if video_type == "self play" else [
      my_agent] + ["random"]*(num_agents_per_game-1)
    env.run(agents)
    
    # Save the HTML recording in the videos folder
    game_recording = env.render(mode="html", width=800, height=600)
    folder, extension = tuple(agent_path.rsplit('/', 1))
    videos_folder = os.path.join(folder, 'Videos')
    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    ext = extension[:-3] if extension_override is None else extension_override
    video_path = os.path.join(videos_folder, ext+' - '+video_type+'.html')
    with open(video_path,"w") as f:
      f.write(game_recording)
  
def store_config_on_first_run(config):
  pool_name = config['pool_name']
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Deep Learning Agents/' + pool_name)
  Path(agents_folder).mkdir(parents=True, exist_ok=True)
  f = os.path.join(agents_folder, 'initial_config.yml')
  if not os.path.exists(f):
    with open(f, 'w') as outfile:
      yaml.dump(config, outfile, default_flow_style=False)
      
# Allow dot access for dict
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError from e
            
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)

    
###############################################################################
# Submission utilities                                                        #
###############################################################################

# Bit 64 encoding of a pickled object
def serialize(value, verbose=True):
  serialized = pickle.dumps(value)
  if verbose:
    print('Length of serialized object:', len(serialized))
  return b64.b64encode(serialized)

# Bit 64 decoding followed by a pickle load
def deserialize(serialized):
  data_byte = b64.b64decode(serialized)
  value = pickle.loads(data_byte)
  return value
    
# make_keras_picklable()