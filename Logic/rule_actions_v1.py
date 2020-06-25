# import copy
import numpy as np
import utils
import rule_utils


def decide_ship_convert_actions(
    config, observation, player_obs, env_config, verbose):
  spawn_cost = env_config.spawnCost
  convert_cost = env_config.convertCost
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
  num_ships = len(player_obs[2])

  # Override the maximum number of conversions on the last episode turn
  last_episode_turn = observation['relative_step'] == 1
  if last_episode_turn:
    max_conversions = num_ships

  if max_conversions == 0:
    return ({}, np.array(list(player_obs[2].keys())), my_bases, my_next_ships,
            obs_halite, remaining_budget)

  conversion_scores = np.zeros(num_ships)
  grid_size = obs_halite.shape[0]
  smoothed_friendly_ship_halite = rule_utils.smooth2d(
    observation['rewards_bases_ships'][0][3])
  smoothed_friendly_bases = rule_utils.smooth2d(my_bases)
  smoothed_halite = rule_utils.smooth2d(obs_halite)
  
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
    
    # Convert on the last turn if there is more halite on board than the
    # conversion cost.
    if last_episode_turn:
      conversion_scores[i] += 1e6*np.sign(
        player_obs[2][ship_k][1]-convert_cost)
    
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
      mapped_actions[ship_k] = rule_utils.CONVERT
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

def decide_not_converted_ship_actions(
    config, observation, player_obs, env_config, not_converted_ship_keys,
    my_next_bases, my_next_ships, obs_halite, verbose,
    convert_first_ship_on_None_action=True, nearby_ship_grid=3):
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
  
  my_ships = observation['rewards_bases_ships'][0][2].astype(np.int)
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  
  # Compute values that will be useful in the calculation for all ships
  grid_size = obs_halite.shape[0]
  smoothed_friendly_bases = rule_utils.smooth2d(my_next_bases)
  my_smoothed_ships, ships_kernel = rule_utils.smooth2d(
    my_ships, return_kernel=True)
  smoothed_halite = rule_utils.smooth2d(obs_halite)
  
  # Select the best option: go to most salient halite, return to base or stay
  # at base.
  my_next_ships = np.zeros_like(my_next_ships)
  
  # List all positions you definitely don't want to move to. Initially this
  # only contains enemy bases and eventually also earlier ships.
  bad_positions = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  
  # Fixed rule to decide when there should be a focus on collecting halite
  should_collect_halite = my_next_bases.sum() > 0 or num_ships > 1
  
  for ship_i, ship_k in enumerate(
      not_converted_ship_keys[ship_keys_ordered_ids]):
    row, col = utils.row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
    
    # Subtract the own influence of the ship from my_smoothed_ships
    my_smoothed_ships = rule_utils.add_warped_kernel(
      my_smoothed_ships, -ships_kernel, row, col)
    
    move_scores = np.zeros(len(rule_utils.MOVE_DIRECTIONS))
    for i, move_dir in enumerate(rule_utils.MOVE_DIRECTIONS):
      new_row, new_col = rule_utils.move_ship_row_col(
        row, col, move_dir, grid_size)
      
      # Going to a position where I already occupy a ship or a position of an
      # enemy ship is very bad
      if bad_positions[new_row, new_col]:
        move_scores[i] -= 1e9
      
      if move_dir is None:
        # Collecting halite is good, if we have a backup base in case the ship
        # gets destroyed
        move_scores[i] += config['halite_collect_constant']*max(
          0, obs_halite[row, col])*should_collect_halite
        
      # Going closer to halite is good
      move_scores[i] += config['nearby_halite_move_constant']*(
        smoothed_halite[row, col] - smoothed_halite[new_row, new_col])
      
      # Moving on top of halite is good, when the collect halite mode is active
      move_scores[i] += config['nearby_onto_halite_move_constant']*(
        obs_halite[new_row, new_col])*should_collect_halite
      
      # Going closer to my other ships is bad
      move_scores[i] -= config['nearby_ships_move_constant']*(
        my_smoothed_ships[new_row, new_col])
      
      # Going closer to my bases is good, the more halite I have on the ship
      move_scores[i] += config['nearby_base_move_constant']*(
        smoothed_friendly_bases[new_row, new_col])*ship_halite[ship_i]
      
      # Going right on top of one of my bases is good, the more halite I have
      # on board of the ship
      move_scores[i] += config['nearby_move_onto_base_constant']*(
        my_next_bases[new_row, new_col])*ship_halite[ship_i]
      
      # Consider nearby enemy ships in a nearby_ship_grid*nearby_ship_grid
      # from the new position and assign penalty/gain points for possible
      # collisions / approachments as a function of the halite on board of
      # other ships
      for row_increment in range(-nearby_ship_grid, nearby_ship_grid+1):
        for col_increment in range(-nearby_ship_grid, nearby_ship_grid+1):
          distance_to_new = np.abs(row_increment) + np.abs(col_increment)
          if distance_to_new <= nearby_ship_grid:
            other_row = (new_row + row_increment) % grid_size
            other_col = (new_col + col_increment) % grid_size
            if opponent_ships[other_row, other_col]:
              halite_diff = halite_ships[other_row, other_col] - halite_ships[
                row, col]
              if halite_diff == 0:
                # Equal halite - impose a penalty of half a spawn cost for
                # moving closer due to the risk of collision
                dist_kernel = 1/((distance_to_new+1)**2)
                move_scores[i] -= config[
                  'adjacent_opponent_ships_move_constant']*(
                    env_config.spawnCost/2*dist_kernel)
              elif halite_diff > 0:
                # I can steal the opponent's halite - moving closer is good,
                # proportional to the difference in cargo on board.
                dist_kernel = 1/((distance_to_new+1)**2)
                move_scores[i] += config[
                  'adjacent_opponent_ships_move_constant']*(
                    halite_diff*dist_kernel)
              else:
                # I risk losing my halite - moving closer is bad, proportional
                # to the difference in cargo on board.
                dist_kernel = 1/((max(0, distance_to_new-1)+1)**2)
                move_scores[i] += config[
                  'adjacent_opponent_ships_move_constant']*(
                    halite_diff*dist_kernel)
            
      
      if verbose:
        print(
          ship_k,
          move_dir,
          config['nearby_halite_move_constant']*(
          smoothed_halite[new_row, new_col]),
          config['nearby_onto_halite_move_constant']*(
          obs_halite[new_row, new_col])*should_collect_halite,
          config['nearby_ships_move_constant']*(
          my_smoothed_ships[new_row, new_col]),
          config['nearby_base_move_constant']*(
          smoothed_friendly_bases[new_row, new_col])*ship_halite[ship_i],
          config['nearby_move_onto_base_constant']*(
          my_next_bases[new_row, new_col])*ship_halite[ship_i]
          )
    
    if verbose:
      print("Ship {} move scores: {}".format(ship_k, move_scores))
      
    move_id = np.argmax(move_scores)
    move_dir = rule_utils.MOVE_DIRECTIONS[move_id]
    new_row, new_col = rule_utils.move_ship_row_col(
      row, col, move_dir, grid_size)
    my_next_ships[new_row, new_col] = 1
    bad_positions[new_row, new_col] = 1
    if move_dir is None:
      halite_deposited += ship_halite[ship_i]*my_next_bases[new_row, new_col]
    else:
      ship_actions[str(ship_k)] = str(move_dir)
    
    # Restore the updated influence of the moved ship to my_smoothed_ships
    if ship_i < (num_ships - 1):
      my_smoothed_ships = rule_utils.add_warped_kernel(
        my_smoothed_ships, ships_kernel, new_row, new_col)
      
  if not should_collect_halite and convert_first_ship_on_None_action:
    convert_cost =  env_config.convertCost
    remaining_budget = player_obs[0]
    if num_ships == 1 and remaining_budget >= convert_cost:
      ship_actions[str(ship_k)] = rule_utils.CONVERT
      
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
  smoothed_friendly_ship_halite = rule_utils.smooth2d(
    observation['rewards_bases_ships'][0][3])
  smoothed_halite = rule_utils.smooth2d(obs_halite)
  
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
      mapped_actions[base_k] = rule_utils.SPAWN
      remaining_budget -= spawn_cost
      
  return mapped_actions, remaining_budget

def get_config_actions(config, observation, player_obs, env_config, verbose):
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
  
  return mapped_actions, halite_spent