import copy
import json
from kaggle_environments import make as make_environment
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy import stats
import seaborn as sns
import utils



CONVERT = "CONVERT"
SPAWN = "SPAWN"

MOVE_DIRECTIONS = [None, "NORTH", "SOUTH", "EAST", "WEST"]

HALITE_MULTIPLIER_CONFIG_ENTRIES = [
          "ship_halite_cargo_conversion_bonus_constant",
          "friendly_ship_halite_conversion_constant",
          "nearby_halite_conversion_constant",
          
          "halite_collect_constant",
          "nearby_halite_move_constant",
          "nearby_onto_halite_move_constant",
          "nearby_base_move_constant",
          "nearby_move_onto_base_constant",
          "halite_dropoff_constant",
          
          "nearby_ship_halite_spawn_constant",
          "nearby_halite_spawn_constant",
          ]

GAUSSIAN_2D_KERNELS = {}
for dim in range(3, 16, 2):
  # Modified from https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
  center_distance = np.floor(np.abs(np.arange(dim) - (dim-1)/2))
  horiz_distance = np.tile(center_distance, [dim, 1])
  vert_distance = np.tile(np.expand_dims(center_distance, 1), [1, dim])
  manh_distance = horiz_distance + vert_distance
  kernel = np.exp(-manh_distance/(dim/4))
  kernel[manh_distance > dim/2] = 0
  
  GAUSSIAN_2D_KERNELS[dim] = kernel

def get_action_costs():
  costs = utils.get_action_costs().tolist()
  actions = utils.INVERSE_ACTION_MAPPING.keys()
  
  return dict(zip(actions, costs))

def add_action_costs_to_config(config):
   config['action_costs'] = get_action_costs()

def store_config_on_first_run(config):
  pool_name = config['pool_name']
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(this_folder, '../Rule agents/' + pool_name)
  Path(agents_folder).mkdir(parents=True, exist_ok=True)
  f = os.path.join(agents_folder, 'initial_config.json')
  if not os.path.exists(f):
    with open(f, 'w') as outfile:
      outfile.write(json.dumps(config))
      
def update_learning_progress(experiment_name, data_vals): 
  # Append a line to the learning progress line if the file exists. Otherwise:
  # create it
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Rule agents/' + experiment_name)
  progress_path = os.path.join(agents_folder, 'learning_progress.csv')
  
  if os.path.exists(progress_path):
    progress = pd.read_csv(progress_path)
    progress.loc[progress.shape[0]] = data_vals
  else:
    progress = pd.DataFrame(data_vals, index=[0])
  
  progress.to_csv(progress_path, index=False)
  
def serialize_game_experience_for_learning(
    experience, only_store_first, config_keys):
  # Create a row for each config - result pair
  list_of_dicts_x = [c for d in experience for c in d.agent_configs]
  dict_of_lists_x = {
    k: [dic[k] for dic in list_of_dicts_x] for k in list_of_dicts_x[0]}
  
  list_of_dicts_sh = [{'terminal_num_ships': sh} for d in experience for sh in(
    d.terminal_num_ships)]
  dict_of_lists_sh = {
    k: [dic[k] for dic in list_of_dicts_sh] for k in list_of_dicts_sh[0]}
  
  list_of_dicts_b = [{'terminal_num_bases': b} for d in experience for b in (
    d.terminal_num_bases)]
  dict_of_lists_b = {
    k: [dic[k] for dic in list_of_dicts_b] for k in list_of_dicts_b[0]}
  
  list_of_dicts_sc = [{'episode_reward': sc} for d in experience for sc in (
    d.episode_rewards)]
  dict_of_lists_sc = {
    k: [dic[k] for dic in list_of_dicts_sc] for k in list_of_dicts_sc[0]}
  
  list_of_dicts_h = [{'terminal_halite': h} for d in experience for h in (
    d.terminal_halite)]
  dict_of_lists_h = {
    k: [dic[k] for dic in list_of_dicts_h] for k in list_of_dicts_h[0]}
  
  combined = {}
  for d in [dict_of_lists_x, dict_of_lists_sh, dict_of_lists_b,
            dict_of_lists_sc, dict_of_lists_h]:
    combined.update(d)
    
  combined_df = pd.DataFrame.from_dict(combined)
  if only_store_first:
    num_agents = len(experience[0].terminal_halite)
    combined_df = combined_df.iloc[::num_agents]
    
  for k in config_keys:
    if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
      combined_df[k] *= combined_df['halite_config_setting_divisor']
    
  return combined_df

def clip_ranges(values, ranges):
  for i in range(len(values)):
    values[i] = np.maximum(np.minimum(np.array(values[i]),
                                      ranges[i][1]), ranges[i][0]).tolist()
    
  return values

def get_self_play_experience_path(experiment_name):
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Rule agents/' + experiment_name)
  data_path = os.path.join(agents_folder, 'self_play_experience.csv')
  
  return data_path
  
def write_experience_data(experiment_name, data):
  # Append to the learning progress line if the file exists.
  # Otherwise: create it
  data_path = get_self_play_experience_path(experiment_name)
  
  if os.path.exists(data_path):
    old_data = pd.read_csv(data_path)
    save_data = pd.concat([old_data, data], axis=0)
  else:
    save_data = data
  
  save_data.to_csv(data_path, index=False)
  
  return data_path

def plot_reward_versus_features(
    features_rewards_path, data, plot_name_suffix,
    target_col="episode_reward"):
  # data = pd.read_csv(features_rewards_path)
  folder, _ = tuple(features_rewards_path.rsplit('/', 1))
  plots_folder = os.path.join(folder, 'Plots')
  Path(plots_folder).mkdir(parents=True, exist_ok=True)
  
  other_columns = data.columns.tolist()
  other_columns.remove(target_col)
  targets = data[target_col].values
  targets_rounded = np.round(targets, 3)
  
  num_plots = len(other_columns)
  grid_size = int(np.ceil(np.sqrt(num_plots)))
  fig = plt.figure(figsize=[3*6.4, 3*4.8])
  plt.subplots_adjust(hspace=0.8, wspace=0.4)
  
  for i, c in enumerate(other_columns):
    ax = fig.add_subplot(grid_size, grid_size, i+1)
    x_vals = data[c].values
    if np.abs(x_vals - x_vals.astype(np.int)).mean() < 1e-8:
      # Drop data points corresponding with tied results to make the
      # categorical distinction more apparent
      non_tied_ids = np.where(np.abs(np.mod(targets*3, 1)) < 1e-8)[0]
      sns.violinplot(x=targets_rounded[non_tied_ids], y=x_vals[non_tied_ids],
                     ax=ax).set(title=c)
    else:
      sns.regplot(x=x_vals, y=targets, ax=ax).set(title=c)
      
  fig = ax.get_figure()
  fig.savefig(os.path.join(
    plots_folder, 'combined ' + plot_name_suffix + '.png'))
  
def save_config(config, path):
  with open(path, 'w') as outfile:
    outfile.write(json.dumps(config))
    
# Load all json configs for the given paths
def load_configs(paths):
  configs = []
  for p in paths:
    with open(p) as f:
      configs.append(json.load(f))
    
  return configs

def evolve_config(config_path, data, initial_config_ranges,
                  target_col="episode_reward", min_slope_cutoff=0.2,
                  significant_range_reduction=0.95, relative_shift=0.2):
  config = load_configs([config_path])[0]
  y_vals = data[target_col].values
  for k in config:
    x_vals = data[k].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(
      x_vals, y_vals)
    
    if np.abs(r_value) < min_slope_cutoff:
      # Don't adjust the range if the slope is not significant
      pass
    else:
      current_range = config[k][0][1]-config[k][0][0]
      next_range = current_range*significant_range_reduction
      if r_value > 0:
        # Shift the config up and slightly shrink the search range
        min_val = config[k][0][0]+relative_shift*current_range
      else:
        min_val = max(config[k][2],
                      config[k][0][0]-relative_shift*current_range)
      config[k][0][0] = min_val
      config[k][0][1] = config[k][0][0]+next_range

  save_config(config, config_path)

def mirror_edges(observation, num_mirror_dim):
  if num_mirror_dim > 0:
    # observation = np.arange(225).reshape((15,15)) # Debugging test
    assert len(observation.shape) == 2
    grid_size = observation.shape[0]
    new_grid_size = grid_size + 2*num_mirror_dim
    mirrored_obs = np.full((new_grid_size, new_grid_size), np.nan)
    
    # Fill in the original data
    mirrored_obs[num_mirror_dim:(-num_mirror_dim),
                 num_mirror_dim:(-num_mirror_dim)] = observation
    
    # Add top and bottom mirrored data
    mirrored_obs[:num_mirror_dim, num_mirror_dim:(
      -num_mirror_dim)] = observation[-num_mirror_dim:, :]
    mirrored_obs[-num_mirror_dim:, num_mirror_dim:(
      -num_mirror_dim)] = observation[:num_mirror_dim, :]
    
    # Add left and right mirrored data
    mirrored_obs[:, :num_mirror_dim] = mirrored_obs[
      :, -(2*num_mirror_dim):(-num_mirror_dim)]
    mirrored_obs[:, -num_mirror_dim:] = mirrored_obs[
      :, num_mirror_dim:(2*num_mirror_dim)]
    
    observation = mirrored_obs
  
  return observation

def smooth2d(grid, smooth_kernel_dim=7, return_kernel=False):
  edge_augmented = mirror_edges(grid, smooth_kernel_dim-1)
  kernel = GAUSSIAN_2D_KERNELS[int(2*smooth_kernel_dim-1)]
  convolved = signal.convolve2d(edge_augmented, kernel, mode="valid")
  
  if return_kernel:
    return convolved, kernel
  else:
    return convolved

def decide_ship_convert_actions(
    config, observation, player_obs, env_config, verbose):
  spawn_cost = env_config.spawnCost
  convert_cost =  env_config.convertCost
  remaining_budget = player_obs[0]
  max_conversions = int((
     remaining_budget - config['min_spawns_after_conversions']*spawn_cost)/(
      convert_cost))
  max_conversions = min(max_conversions, config['max_conversions_per_step'])
  my_bases = observation['rewards_bases_ships'][0][1]
  my_next_ships = observation['rewards_bases_ships'][0][2]
  obs_halite = np.maximum(0, observation['halite'])
  max_conversions = min(max_conversions, int(obs_halite.sum()/4/convert_cost))

  if max_conversions == 0:
    return ({}, np.array(list(player_obs[2].keys())), my_bases, my_next_ships,
            obs_halite, remaining_budget)

  num_ships = len(player_obs[2])
  conversion_scores = np.zeros(num_ships)
  grid_size = obs_halite.shape[0]
  smoothed_friendly_ship_halite = smooth2d(
    observation['rewards_bases_ships'][0][3])
  smoothed_friendly_bases = smooth2d(my_bases)
  smoothed_halite = smooth2d(obs_halite)
  
  can_deposit_halite = my_bases.sum() > 0
  
  for i, ship_k in enumerate(player_obs[2]):
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    
    # Don't convert at an existing base
    conversion_scores[i] -= 1e9*my_bases[row, col]
    
    # Add the ship halite to the conversion scores (it is added to the score)
    conversion_scores[i] += player_obs[2][ship_k][1]*config[
      'ship_halite_cargo_conversion_bonus_constant']
    
    # Add distance adjusted friendly nearby ship halite to the conversion
    # scores
    conversion_scores[i] += (smoothed_friendly_ship_halite[row, col]-(
      observation['rewards_bases_ships'][0][3][row, col]))*config[
        'friendly_ship_halite_conversion_constant']
    
    # Subtract distance adjusted nearby friendly bases from the conversion
    # scores
    conversion_scores[i] -= (smoothed_friendly_bases[row, col])*(
      config['friendly_bases_conversion_constant'])
    
    # Add distance adjusted nearby halite to the conversion scores
    conversion_scores[i] += (smoothed_halite[row, col]-obs_halite[row, col])*(
      config['nearby_halite_conversion_constant'])
    
    if verbose:
      print(player_obs[2][ship_k][1]/config[
        'halite_config_setting_divisor']*can_deposit_halite,
        (smoothed_friendly_ship_halite[row, col]-(
        observation['rewards_bases_ships'][0][3][row, col]))*config[
          'friendly_ship_halite_conversion_constant'],
        (smoothed_friendly_bases[row, col])*(
        config['friendly_bases_conversion_constant']),
        (smoothed_halite[row, col]-obs_halite[row, col])*(
        config['nearby_halite_conversion_constant'])
        )
    
  if verbose:
    print("Conversion scores and threshold: {}; {}".format(
      conversion_scores, config['conversion_score_threshold']))
    
  # Convert the ships with the top conversion scores that stay within the
  # max conversion limits
  convert_ids = np.where(conversion_scores > config[
    'conversion_score_threshold'])[0][:max_conversions]
  
  mapped_actions = {}
  not_converted_ship_keys = []
  for i, ship_k in enumerate(player_obs[2]):
    if np.isin(i, convert_ids):
      mapped_actions[ship_k] = CONVERT
      row, col = utils.row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      obs_halite[row, col] = 0
      my_bases[row, col] = 1
      my_next_ships[row, col] = 0
      remaining_budget -= convert_cost
    else:
      not_converted_ship_keys.append(ship_k)
      
  return (mapped_actions, np.array(not_converted_ship_keys), my_bases,
          my_next_ships, obs_halite, remaining_budget)

def move_ship_row_col(row, col, direction, size):
  if direction == "NORTH":
    return (size-1 if row == 0 else row-1, col)
  elif direction == "SOUTH":
    return (row+1 if row < (size-1) else 0, col)
  elif direction == "EAST":
    return (row, col+1 if col < (size-1) else 0)
  elif direction == "WEST":
    return (row, size-1 if col == 0 else col-1)
  elif direction is None:
    return (row, col)
  
def add_warped_kernel(grid, kernel, row, col):
  kernel_dim = kernel.shape[0]
  assert kernel_dim % 2 == 1
  half_kernel_dim = int((kernel_dim-1)/2)
  
  addition = np.zeros_like(grid)
  addition[:kernel_dim, :kernel_dim] = kernel
  row_roll = row-half_kernel_dim
  col_roll = col-half_kernel_dim
  addition = np.roll(addition, row_roll, axis=0)
  addition = np.roll(addition, col_roll, axis=1)
  
  return grid + addition

def decide_not_converted_ship_actions(
    config, observation, player_obs, env_config, not_converted_ship_keys,
    my_next_bases, my_next_ships, obs_halite, verbose,
    convert_first_ship_on_None_action=True):
  ship_actions = {}
  num_ships = len(not_converted_ship_keys)
  halite_deposited = 0
  
  if not not_converted_ship_keys.size:
    return ship_actions, my_next_ships, halite_deposited
  
  # Select ship actions sequentially in the order of the most halite on board
  ship_halite = np.array(
    [player_obs[2][k][1] for k in not_converted_ship_keys])
  ship_keys_ordered_ids = np.argsort(-ship_halite)
  ship_halite = ship_halite[ship_keys_ordered_ids]
  
  ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships']]).sum(0)
  
  # Compute values that will be useful in the calculation for all ships
  grid_size = obs_halite.shape[0]
  smoothed_friendly_bases = smooth2d(my_next_bases)
  smoothed_ships, ships_kernel = smooth2d(ships, return_kernel=True)
  smoothed_halite = smooth2d(obs_halite)
  
  # Select the best option: go to most salient halite, return to base or stay
  # at base.
  my_next_ships = np.zeros_like(my_next_ships)
  
  # List all positions you definitely don't want to move to. Initially this
  # only contains enemy bases and eventually also earlier ships.
  bad_positions = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  
  # At the start of the game, halite can not be deposited and should therefore
  # induce different behavior
  can_deposit_halite = my_next_bases.sum() > 0 or num_ships > 1
  
  for ship_i, ship_k in enumerate(
      not_converted_ship_keys[ship_keys_ordered_ids]):
    row, col = utils.row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
    
    # Subtract the own influence of the ship from smoothed_ships
    smoothed_ships = add_warped_kernel(
      smoothed_ships, -ships_kernel, row, col)
    
    move_scores = np.zeros(len(MOVE_DIRECTIONS))
    for i, move_dir in enumerate(MOVE_DIRECTIONS):
      new_row, new_col = move_ship_row_col(row, col, move_dir, grid_size)
      
      # Going to a position where I already occupy a ship or a position of an
      # enemy ship is very bad
      if bad_positions[new_row, new_col]:
        move_scores[i] -= 1e9
      
      # TODO: take current ship halite into account - lower value of collecting
      # When there already is a lot of halite on board
      if move_dir is None:
        # Collecting halite is good, unless if there is only one ship and no
        # base
        move_scores[i] += config['halite_collect_constant']*max(
          0, obs_halite[row, col])*can_deposit_halite
        
        # Staying at my base is better, the more halite I have on board
        move_scores[i] += config['halite_dropoff_constant']*(
          ship_halite[ship_i])*my_next_bases[new_row, new_col]
      
      # Going closer to halite is good
      move_scores[i] += config['nearby_halite_move_constant']*(
        smoothed_halite[new_row, new_col])
      
      # Moving on top of halite is good, unless if there is only one ship and
      # no base
      move_scores[i] += config['nearby_onto_halite_move_constant']*(
        obs_halite[new_row, new_col])*can_deposit_halite
      
      # Going closer to my other ships is bad
      move_scores[i] -= config['nearby_ships_move_constant']*(
        smoothed_ships[new_row, new_col])
      
      # Going closer to my bases is good, the more halite I have on the ship
      move_scores[i] += config['nearby_base_move_constant']*(
        smoothed_friendly_bases[new_row, new_col])*ship_halite[ship_i]
      
      # Going right on top of one of my bases is good, the more halite I have
      # on board of the ship
      move_scores[i] += config['nearby_move_onto_base_constant']*(
        my_next_bases[new_row, new_col])*ship_halite[ship_i]
      
      if verbose:
        print(
          ship_k,
          move_dir,
          config['nearby_halite_move_constant']*(
          smoothed_halite[new_row, new_col]),
          config['nearby_onto_halite_move_constant']*(
          obs_halite[new_row, new_col])*can_deposit_halite,
          config['nearby_ships_move_constant']*(
          smoothed_ships[new_row, new_col]),
          config['nearby_base_move_constant']*(
          smoothed_friendly_bases[new_row, new_col])*ship_halite[ship_i],
          config['nearby_move_onto_base_constant']*(
          my_next_bases[new_row, new_col])*ship_halite[ship_i]
          )
      
    if verbose:
      print("Ship {} move scores: {}".format(ship_k, move_scores))
      
    move_id = np.argmax(move_scores)
    move_dir = MOVE_DIRECTIONS[move_id]
    new_row, new_col = move_ship_row_col(row, col, move_dir, grid_size)
    my_next_ships[new_row, new_col] = 1
    bad_positions[new_row, new_col] = 1
    if move_dir is None:
      halite_deposited += ship_halite[ship_i]*my_next_bases[new_row, new_col]
    else:
      ship_actions[str(ship_k)] = str(move_dir)
    
    # Add the updated influence of the moved ship to smoothed_ships
    if ship_i < (num_ships - 1):
      smoothed_ships = add_warped_kernel(
        smoothed_ships, ships_kernel, new_row, new_col)
      
  if not can_deposit_halite and convert_first_ship_on_None_action:
    convert_cost =  env_config.convertCost
    remaining_budget = player_obs[0]
    if num_ships == 1 and remaining_budget >= convert_cost:
      ship_actions[str(ship_k)] = CONVERT
      
  return ship_actions, my_next_ships, halite_deposited

def decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships, obs_halite,
    env_config, remaining_budget, verbose):

  spawn_cost = env_config.spawnCost
  max_spawns = int(remaining_budget/spawn_cost)
  max_spawns = min(max_spawns, config['max_spawns_per_step'])
  num_next_bases = my_next_bases.sum()
  num_ships = my_next_ships.sum()
  max_allowed_ships = num_next_bases*config['max_ship_to_base_ratio']
  max_spawns = min(max_spawns, int(max_allowed_ships - num_ships))
  max_spawns = min(max_spawns, int(obs_halite.sum()/2/spawn_cost))

  if max_spawns <= 0 or not player_obs[1]:
    return {}, remaining_budget
  
  num_bases = len(player_obs[1])
  spawn_scores = np.zeros(num_bases)
  obs_halite = observation['halite']
  grid_size = obs_halite.shape[0]
  smoothed_friendly_ship_halite = smooth2d(
    observation['rewards_bases_ships'][0][3])
  smoothed_halite = smooth2d(obs_halite)
  
  for i, base_k in enumerate(player_obs[1]):
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[1][base_k], grid_size)
    
    # Don't spawn when there will be a ship at the base
    spawn_scores[i] -= 1e9*my_next_ships[row, col]
    
    # Spawn less when the base is crowded with ships with a lot of halite
    # TODO: use the updated ship halite
    spawn_scores[i] -= smoothed_friendly_ship_halite[row, col]*(
      config['nearby_ship_halite_spawn_constant'])
        
    # Spawn more when there is a lot of nearby halite
    spawn_scores[i] += smoothed_halite[row, col]*(
      config['nearby_halite_spawn_constant'])
    
    # Spawn more when there is a lot of remaining budget available
    spawn_scores[i] += remaining_budget*(
      config['remaining_budget_spawn_constant'])
    
    if verbose:
        print(smoothed_friendly_ship_halite[row, col]*(
          config['nearby_ship_halite_spawn_constant']),
          smoothed_halite[row, col]*(
            config['nearby_halite_spawn_constant']),
          remaining_budget*(config['remaining_budget_spawn_constant']),
          )
    
  if verbose:
    print("Spawn scores and threshold: {}; {}".format(
      spawn_scores, config['spawn_score_threshold']))
    
  # Convert the ships with the top conversion scores that stay within the
  # max conversion limits
  spawn_ids = np.where(spawn_scores > config[
    'spawn_score_threshold'])[0][:max_spawns]
  
  mapped_actions = {}
  for i, base_k in enumerate(player_obs[1]):
    if np.isin(i, spawn_ids):
      mapped_actions[base_k] = SPAWN
      remaining_budget -= spawn_cost
      
  return mapped_actions, remaining_budget

def get_config_actions(config, observation, player_obs, env_config,
                       verbose=False):
  # Compute the added value of converting a ship to a base and record all
  # convert actions that are above the threshold and within the convert budget
  (mapped_actions, not_converted_ship_keys, my_next_bases, my_next_ships,
   my_next_halite, remaining_budget) = decide_ship_convert_actions(
    config, observation, player_obs, env_config, verbose)
  
  # Decide on the move actions for all ships that are not converted to bases
  # Options: go to most salient halite, return to base or stay at base.
  ship_actions, my_next_ships, halite_deposited = (
    decide_not_converted_ship_actions(
      config, observation, player_obs, env_config, not_converted_ship_keys,
      my_next_bases, my_next_ships, my_next_halite, verbose))
  mapped_actions.update(ship_actions)
  
  # Decide for all bases whether to spawn or keep the base available for 
  # returning ships
  base_actions, remaining_budget = decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships,
    my_next_halite, env_config, remaining_budget, verbose)
  mapped_actions.update(base_actions)
  halite_spent = player_obs[0]-remaining_budget
  
  return mapped_actions, halite_deposited, halite_spent

def get_next_config_settings(
    opt, config_keys, num_games, num_repeat_first_configs):
  num_suggested = int(np.ceil(num_games/num_repeat_first_configs))
  suggested = opt.ask(num_suggested)
  
  suggested_configs = [suggested_to_config(s, config_keys) for s in suggested]
  
  return suggested, suggested_configs

def suggested_to_config(suggested, config_keys):
  config = {}
  for i, k in enumerate(config_keys):
    config[k] = suggested[i]
  for k in config_keys:
    if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
      config[k] /= config['halite_config_setting_divisor']
      
  return config

def sample_from_config(config):
  sampled_config = {}
  for k in config:
    if config[k][1] == "float":
      sampled_config[k] = np.random.uniform(config[k][0][0], config[k][0][1])
    elif config[k][1] == "int":
      sampled_config[k] = np.random.randint(
        np.floor(config[k][0][0]), np.ceil(config[k][0][1]))
    else:
      raise ValueError("Unsupported sampling type '{}'".format(config[k][1]))
  
  for k in config:
    if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
      sampled_config[k] /= sampled_config['halite_config_setting_divisor']
  
  return sampled_config

def record_videos(agent_path, num_agents_per_game, extension_override=None,
                  config_override_agents=None):
  print("Generating videos of iteration {}".format(agent_path))
  env = make_environment(
    "halite", configuration={"agentExec": "LOCAL"})#, configuration={"agentTimeout": 10000, "actTimeout": 10000})
  config = load_configs([agent_path])[0]
  env_configuration = env.configuration
  
  def my_agent(observation, config_id):
    config = AGENT_CONFIGS[config_id]
    active_id = observation.player
    current_observation = utils.structured_env_obs(
      env_configuration, observation, active_id)
    player_obs = observation.players[active_id]
    
    mapped_actions, _, _ = get_config_actions(
      config, current_observation, player_obs, env_configuration)
       
    return mapped_actions
  
  if config_override_agents is None:
    AGENT_CONFIGS = []
    for i in range(num_agents_per_game):
      AGENT_CONFIGS.append(sample_from_config(config))
  else:
    AGENT_CONFIGS = config_override_agents
  
  # For some reason this needs to be verbose - list comprehension breaks the
  # stochasticity of the agents.
  config_id_agents = [
    lambda observation: my_agent(observation, 0),
    lambda observation: my_agent(observation, 1),
    lambda observation: my_agent(observation, 2),
    lambda observation: my_agent(observation, 3),
    ][:num_agents_per_game]
  
  for video_type in ["self play", "random opponent"]:
    env.reset(num_agents=num_agents_per_game)
    agents = config_id_agents if video_type == "self play" else [
      config_id_agents[0]] + ["random"]*(num_agents_per_game-1)
    
    env.run(agents)
    
    # Save the HTML recording in the videos folder
    game_recording = env.render(mode="html", width=800, height=600)
    folder, extension = tuple(agent_path.rsplit('/', 1))
    videos_folder = os.path.join(folder, 'Videos')
    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    ext = extension[:-5] if extension_override is None else extension_override
    video_path = os.path.join(videos_folder, ext+' - '+video_type+'.html')
    with open(video_path,"w") as f:
      f.write(game_recording)