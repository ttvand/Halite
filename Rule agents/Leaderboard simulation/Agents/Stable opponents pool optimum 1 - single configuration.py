from kaggle_environments import make as make_environment
import numpy as np
from scipy import signal

CONFIG = {
  'halite_config_setting_divisor': 4000.0,
  'max_ship_to_base_ratio': 5.8359129876498725,
  'min_spawns_after_conversions': 0,
  'max_conversions_per_step': 2,
  'ship_halite_cargo_conversion_bonus_constant': 4.437607013738509,
  'friendly_ship_halite_conversion_constant': 0.3,
  'friendly_bases_conversion_constant': 20.0,
  'nearby_halite_conversion_constant': 0.0,
  'conversion_score_threshold': 9.408671551583732,
  'halite_collect_constant': 7.623369052805619,
  'nearby_halite_move_constant': 2.0,
  'nearby_onto_halite_move_constant': 4.0,
  'nearby_ships_move_constant': 0.05,
  'nearby_base_move_constant': 12.253133554058168,
  'nearby_move_onto_base_constant': 10.0,
  'halite_dropoff_constant': 0.0,
  'max_spawns_per_step': 3,
  'nearby_ship_halite_spawn_constant': 0.0,
  'nearby_halite_spawn_constant': 2.0,
  'remaining_budget_spawn_constant': 0.016479878250335398,
  'spawn_score_threshold': 12.273082965711891
  }

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

##########################
# Observation generation #
##########################

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
  relative_step = step/num_episode_steps
  
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
    }


####################
# Action selection #
####################
  
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
  max_conversions = min(max_conversions,
                        int(config['max_conversions_per_step']))
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
    row, col = row_col_from_square_grid_pos(
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
      row, col = row_col_from_square_grid_pos(
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
    row, col = row_col_from_square_grid_pos(
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
  max_spawns = min(max_spawns, int(config['max_spawns_per_step']))
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
    row, col = row_col_from_square_grid_pos(
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


###############################################################################
for k in CONFIG:
  if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
    CONFIG[k] /= CONFIG['halite_config_setting_divisor']

def my_agent(observation, env_config):
  active_id = observation.player
  current_observation = structured_env_obs(env_config, observation, active_id)
  player_obs = observation.players[active_id]
  
  mapped_actions, _, _ = get_config_actions(
    CONFIG, current_observation, player_obs, env_config)
     
  return mapped_actions
