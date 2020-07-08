from collections import OrderedDict
import copy
import numpy as np
import utils
import rule_utils


NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
CONVERT = "CONVERT"
SPAWN = "SPAWN"
MOVE_DIRECTIONS = [None, NORTH, SOUTH, EAST, WEST]
RELATIVE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
MOVE_GATHER_OPTIONS = [(-1, 0, False), (1, 0, False), (0, -1, False),
                       (0, 1, False), (0, 0, True)]
TWO_STEP_THREAT_DIRECTIONS = {
  (-2, 0): [(-1, 0)],
  (-1, -1): [(-1, 0), (0, -1)],
  (-1, 0): [(-1, 0), (0, 0)],
  (-1, 1): [(-1, 0), (0, 1)],
  (0, -2): [(0, -1)],
  (0, -1): [(0, -1), (0, 0)],
  (0, 1): [(0, 1), (0, 0)],
  (0, 2): [(0, 1)],
  (1, -1): [(1, 0), (0, -1)],
  (1, 0): [(1, 0), (0, 0)],
  (1, 1): [(1, 0),(0, 1)],
  (2, 0): [(1, 0)],
  }

DISTANCE_MASKS = {}
HALF_PLANES_CATCH = {}
HALF_PLANES_RUN = {}
ROW_COL_DISTANCE_MASKS = {}
ROW_COL_MAX_DISTANCE_MASKS = {}
DISTANCE_MASK_DIM = 21
half_distance_mask_dim = int(DISTANCE_MASK_DIM/2)
for row in range(DISTANCE_MASK_DIM):
  for col in range(DISTANCE_MASK_DIM):
    # Modified from https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
    horiz_distance = np.minimum(
      np.abs(np.arange(DISTANCE_MASK_DIM) - col),
      np.abs(np.arange(DISTANCE_MASK_DIM) - col - DISTANCE_MASK_DIM))
    horiz_distance = np.minimum(
      horiz_distance,
      np.abs(np.arange(DISTANCE_MASK_DIM) - col + DISTANCE_MASK_DIM))
    
    vert_distance = np.minimum(
      np.abs(np.arange(DISTANCE_MASK_DIM) - row),
      np.abs(np.arange(DISTANCE_MASK_DIM) - row - DISTANCE_MASK_DIM))
    vert_distance = np.minimum(
      vert_distance,
      np.abs(np.arange(DISTANCE_MASK_DIM) - row + DISTANCE_MASK_DIM))
    
    horiz_distance = np.tile(horiz_distance, [DISTANCE_MASK_DIM, 1])
    vert_distance = np.tile(np.expand_dims(vert_distance, 1),
                            [1, DISTANCE_MASK_DIM])
    manh_distance = horiz_distance + vert_distance
    kernel = np.exp(-manh_distance/(DISTANCE_MASK_DIM/4))
    
    DISTANCE_MASKS[(row, col)] = kernel
    
    catch_distance_masks = {}
    run_distance_masks = {}
    
    for d in MOVE_DIRECTIONS:
      if d is None:
        catch_rows = np.array([]).astype(np.int)
        catch_cols = np.array([]).astype(np.int)
      if d == NORTH:
        catch_rows = np.mod(row - np.arange(half_distance_mask_dim) - 1,
                            DISTANCE_MASK_DIM)
        catch_cols = np.arange(DISTANCE_MASK_DIM)
      if d == SOUTH:
        catch_rows = np.mod(row + np.arange(half_distance_mask_dim) + 1,
                            DISTANCE_MASK_DIM)
        catch_cols = np.arange(DISTANCE_MASK_DIM)
      if d == WEST:
        catch_cols = np.mod(col - np.arange(half_distance_mask_dim) - 1,
                            DISTANCE_MASK_DIM)
        catch_rows = np.arange(DISTANCE_MASK_DIM)
      if d == EAST:
        catch_cols = np.mod(col + np.arange(half_distance_mask_dim) + 1,
                            DISTANCE_MASK_DIM)
        catch_rows = np.arange(DISTANCE_MASK_DIM)
        
      catch_mask = np.zeros((DISTANCE_MASK_DIM, DISTANCE_MASK_DIM),
                            dtype=np.bool)
      
      catch_mask[catch_rows[:, None], catch_cols] = 1
      run_mask = np.copy(catch_mask)
      run_mask[row, col] = 1
      
      catch_distance_masks[d] = catch_mask
      run_distance_masks[d] = run_mask
    
    HALF_PLANES_CATCH[(row, col)] = catch_distance_masks
    HALF_PLANES_RUN[(row, col)] = run_distance_masks

    for d in range(1, DISTANCE_MASK_DIM):
      ROW_COL_DISTANCE_MASKS[(row, col, d)] = manh_distance == d

    for d in range(half_distance_mask_dim):
      ROW_COL_MAX_DISTANCE_MASKS[(row, col, d)] = manh_distance <= d
      
def get_relative_position(row, col, other_row, other_col, size):
  if row >= other_row:
    if (other_row + size - row) < (row - other_row):
      row_diff = (other_row + size - row)
    else:
      row_diff = -(row - other_row)
  else:
    if (row + size - other_row) < (other_row - row):
      row_diff = -(row + size - other_row)
    else:
      row_diff = other_row - row
      
  if col >= other_col:
    if (other_col + size - col) < (col - other_col):
      col_diff = (other_col + size - col)
    else:
      col_diff = -(col - other_col)
  else:
    if (col + size - other_col) < (other_col - col):
      col_diff = -(col + size - other_col)
    else:
      col_diff = other_col - col
  
  return (row_diff, col_diff)

def update_scores_enemy_ships(
    config, collect_grid_scores, return_to_base_scores, establish_base_scores,
    attack_base_scores, opponent_ships, halite_ships, row, col, grid_size,
    spawn_cost, drop_None_valid, obs_halite, collect_rate, np_rng,
    opponent_ships_sensible_actions, min_dist=2):
  direction_halite_diff_distance={
    NORTH: None,
    SOUTH: None,
    EAST: None,
    WEST: None,
    }
  for row_shift in range(-min_dist, min_dist+1):
    considered_row = (row + row_shift) % grid_size
    for col_shift in range(-min_dist, min_dist+1):
      considered_col = (col + col_shift) % grid_size
      distance = np.abs(row_shift) + np.abs(col_shift)
      if distance <= min_dist:
        if opponent_ships[considered_row, considered_col]:
          relevant_dirs = []
          relevant_dirs += [] if row_shift >= 0 else [NORTH]
          relevant_dirs += [] if row_shift <= 0 else [SOUTH]
          relevant_dirs += [] if col_shift <= 0 else [EAST]
          relevant_dirs += [] if col_shift >= 0 else [WEST]
          
          halite_diff = halite_ships[row, col] - halite_ships[
            considered_row, considered_col]
          for d in relevant_dirs:
            halite_diff_dist = direction_halite_diff_distance[d]
            if halite_diff_dist is None:
              direction_halite_diff_distance[d] = (halite_diff, distance)
            else:
              max_halite_diff = max(halite_diff_dist[0], halite_diff)
              min_ship_dist = min(halite_diff_dist[1], distance)
              direction_halite_diff_distance[d] = (
                max_halite_diff, min_ship_dist)
                
  ship_halite = halite_ships[row, col]
  preferred_directions = []
  valid_directions = copy.copy(MOVE_DIRECTIONS)
  bad_directions = []
  ignore_catch = np_rng.uniform() < config['ignore_catch_prob']
  for direction, halite_diff_dist in direction_halite_diff_distance.items():
    if halite_diff_dist is not None:
      halite_diff = halite_diff_dist[0]
      if halite_diff >= 0:
        # I should avoid a collision
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = np.copy(HALF_PLANES_RUN[(row, col)][direction])
        valid_directions.remove(direction)
        bad_directions.append(direction)
        if halite_diff_dist[1] == 1:
          if halite_diff != 0:
            # Only suppress the stay still action if the opponent has something
            # to gain. TODO: consider making this a function of the game state
            if None in valid_directions:
              valid_directions.remove(None)
              bad_directions.append(None)
          else:
            mask_collect_return[row, col] = False

        # I can safely mine halite at the current square if the opponent ship
        # is >1 move away
        if halite_diff_dist[1] > 1:
          mask_collect_return[row, col] = False
          
        collect_grid_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
          config['collect_run_enemy_multiplier'])*distance_multiplier
        return_to_base_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
          config['return_base_run_enemy_multiplier'])
        attack_base_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
          config['attack_base_run_enemy_multiplier'])
        mask_establish = np.copy(mask_collect_return)
        mask_establish[row, col] = False
        establish_base_scores -= mask_establish*(ship_halite+spawn_cost)*(
          config['establish_base_run_enemy_multiplier'])
        
      elif halite_diff < 0 and not ignore_catch:
        # I would like a collision unless if there is another opponent ship
        # chasing me - risk avoiding policy for now: if there is at least
        # one ship in a direction that has less halite, I should avoid it
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = np.copy(HALF_PLANES_CATCH[(row, col)][direction])
        collect_grid_scores -= mask_collect_return*halite_diff*(
          config['collect_catch_enemy_multiplier'])*distance_multiplier
        return_to_base_scores -= mask_collect_return*halite_diff*(
          config['return_base_catch_enemy_multiplier'])*distance_multiplier
        attack_base_scores -= mask_collect_return*halite_diff*(
          config['attack_base_catch_enemy_multiplier'])*distance_multiplier
        mask_establish = np.copy(mask_collect_return)
        mask_establish[row, col] = False
        establish_base_scores -= mask_establish*halite_diff*(
          config['establish_base_catch_enemy_multiplier'])*distance_multiplier
        
        preferred_directions.append(direction)

  if drop_None_valid and None in valid_directions:
    valid_directions.remove(None)
    
  # For the remaining valid directions: compute a score that resembles the 
  # probability of being boxed in during the next step
  two_step_bad_directions = []
  for d in valid_directions:
    my_next_halite = halite_ships[row, col] if d != None else (
      halite_ships[row, col] + int(collect_rate*obs_halite[row, col]))
    move_row, move_col = rule_utils.move_ship_row_col(row, col, d, grid_size)
    opponent_mask = ROW_COL_MAX_DISTANCE_MASKS[(move_row, move_col, 3)]
    less_halite_threat_opponents = np.where(np.logical_and(
      opponent_mask, np.logical_and(
        opponent_ships, my_next_halite > halite_ships)))
    num_threat_ships = less_halite_threat_opponents[0].size
    if num_threat_ships > 1:
      all_dir_threat_counter = {
        (-1, 0): 0, (1, 0): 0, (0, -1): 0, (0, 1): 0, (0, 0): 0}
      for i in range(num_threat_ships):
        other_row = less_halite_threat_opponents[0][i]
        other_col = less_halite_threat_opponents[1][i]
        relative_other_pos = get_relative_position(
          move_row, move_col, other_row, other_col, grid_size)
        for diff_rel_row, diff_rel_col, other_gather in MOVE_GATHER_OPTIONS:
          # Only consider sensible opponent actions
          if (diff_rel_row, diff_rel_col) in opponent_ships_sensible_actions[
              other_row, other_col]:
            is_threat = (not other_gather) or (my_next_halite > (
              halite_ships[other_row, other_col] + int(collect_rate*obs_halite[
                other_row, other_col])))
            if is_threat:
              other_rel_row = relative_other_pos[0] + diff_rel_row
              other_rel_col = relative_other_pos[1] + diff_rel_col
              move_diff = np.abs(other_rel_row) + np.abs(other_rel_col)
              if move_diff < 3 and move_diff > 0:
                threat_dirs = TWO_STEP_THREAT_DIRECTIONS[
                  (other_rel_row, other_rel_col)]
                for threat_row_diff, threat_col_diff in threat_dirs:
                  all_dir_threat_counter[
                    (threat_row_diff, threat_col_diff)] += 1
      
      # Aggregate the threat count in all_dir_threat_counter
      threat_counts = np.array(list(all_dir_threat_counter.values()))
      threat_score = np.sqrt(threat_counts.prod())
      if threat_score > 0:
        # Disincentivize an action that can get me boxed in on the next step
        mask_avoid_two_steps = np.copy(HALF_PLANES_RUN[(row, col)][d])
        if d is not None:
          mask_avoid_two_steps[row, col] = False
        collect_grid_scores -= mask_avoid_two_steps*threat_score*(
          config['two_step_avoid_boxed_enemy_multiplier'])
        return_to_base_scores -= mask_avoid_two_steps*threat_score*(
          config['two_step_avoid_boxed_enemy_multiplier'])
        establish_base_scores -= mask_avoid_two_steps*threat_score*(
          config['two_step_avoid_boxed_enemy_multiplier'])
        
        two_step_bad_directions.append(d)
        
  return (collect_grid_scores, return_to_base_scores, establish_base_scores,
          attack_base_scores, preferred_directions, valid_directions,
          len(bad_directions) == 5, two_step_bad_directions)

# Update the scores as a function of blocking enemy bases
def update_scores_blockers(
    collect_grid_scores, return_to_base_scores, establish_base_scores,
    attack_base_scores, row, col, grid_size, blockers, valid_directions,
    early_base_direct_dir=None, blocker_max_distance=half_distance_mask_dim,
    update_attack_base=True):
  for d in MOVE_DIRECTIONS[1:]:
    if d == NORTH:
      rows = np.mod(row - (1 + np.arange(blocker_max_distance)), grid_size)
      cols = np.repeat(col, blocker_max_distance)
      considered_vals = blockers[rows, col]
    elif d == SOUTH:
      rows = np.mod(row + (1 + np.arange(blocker_max_distance)), grid_size)
      cols = np.repeat(col, blocker_max_distance)
      considered_vals = blockers[rows, col]
    elif d == WEST:
      rows = np.repeat(row, blocker_max_distance)
      cols = np.mod(col - (1 + np.arange(blocker_max_distance)), grid_size)
      considered_vals = blockers[row, cols]
    elif d == EAST:
      rows = np.repeat(row, blocker_max_distance)
      cols = np.mod(col + (1 + np.arange(blocker_max_distance)), grid_size)
      considered_vals = blockers[row, cols]

    if d == early_base_direct_dir:
      considered_vals[0] = 1
    
    if np.any(considered_vals):
      first_blocking_base_id = np.where(considered_vals)[0][0]
      mask_rows = rows[first_blocking_base_id:]
      mask_cols = cols[first_blocking_base_id:]
      
      collect_grid_scores[mask_rows, mask_cols] = -1e9
      return_to_base_scores[mask_rows, mask_cols] = -1e9
      establish_base_scores[mask_rows, mask_cols] = -1e9
      if update_attack_base:
        attack_base_scores[mask_rows, mask_cols] = -1e9
      
      if first_blocking_base_id == 0 and d in valid_directions:
        valid_directions.remove(d)
        
  # Additional logic for the use of avoiding collisions when there is only a
  # single escape direction
  if blockers[row, col]:
    collect_grid_scores[row, col] = -1e9
    return_to_base_scores[row, col] = -1e9
    establish_base_scores[row, col] = -1e9
    attack_base_scores[row, col] = -1e9
    if None in valid_directions:
      valid_directions.remove(None)
      
  return (collect_grid_scores, return_to_base_scores, establish_base_scores,
          attack_base_scores, valid_directions)

def set_scores_single_nearby_zero(scores, nearby, size, ship_row, ship_col,
                                  nearby_distance=1):
  nearby_pos = np.where(nearby)
  row = nearby_pos[0][0]
  col = nearby_pos[1][0]
  next_nearby_pos = None
  drop_None_valid = False
  
  for i in range(-nearby_distance, nearby_distance+1):
    near_row = (row + i) % size
    for j in range(-nearby_distance, nearby_distance+1):
      near_col = (col + j) % size
      if i != 0 or j != 0:
        # Don't gather near the base and don't move on top of it
        scores[near_row, near_col] = 0
        if near_row == ship_row and near_col == ship_col:
          next_nearby_pos = get_dir_from_target(
            ship_row, ship_col, row, col, size)[0]
      else:
        if near_row == ship_row and near_col == ship_col:
          # Don't stay on top of the base
          drop_None_valid = True
  
  return scores, next_nearby_pos, drop_None_valid

def grid_distance(r1, c1, r2, c2, size):
  horiz_diff = c2-c1
  horiz_distance = min(np.abs(horiz_diff),
    min(np.abs(horiz_diff-size), np.abs(horiz_diff+size)))
  vert_diff = r2-r1
  vert_distance = min(np.abs(vert_diff),
    min(np.abs(vert_diff-size), np.abs(vert_diff+size)))
  
  return horiz_distance+vert_distance

def override_early_return_base_scores(
    return_to_base_scores, my_bases, ship_row, ship_col, size, num_ships):
  base_pos = np.where(my_bases)
  base_row = base_pos[0][0]
  base_col = base_pos[1][0]
  
  dist_to_base = grid_distance(base_row, base_col, ship_row, ship_col, size)
  # Remember the rule that blocks spawning when a ship is about to return
  if dist_to_base <= 10-num_ships:
    return_to_base_scores[base_row, base_col] = 0
    
  return return_to_base_scores

def get_nearest_base_distance(player_obs, grid_size):
  base_dms = []
  for b in player_obs[1]:
    row, col = utils.row_col_from_square_grid_pos(player_obs[1][b], grid_size)
    base_dms.append(DISTANCE_MASKS[(row, col)])
  
  if base_dms:
    base_nearest_distance = np.stack(base_dms).max(0)
  else:
    base_nearest_distance = np.ones((grid_size, grid_size))
    
  return base_nearest_distance

def get_valid_opponent_ship_actions(rewards_bases_ships, halite_ships, size):
  valid_opponent_actions = {}
  num_agents = len(rewards_bases_ships)
  stacked_ships = np.stack([rbs[2] for rbs in rewards_bases_ships])
  for i in range(1, num_agents):
    opponent_ships = stacked_ships[i]
    enemy_ships = np.delete(stacked_ships, (i), axis=0).sum(0)
    ship_pos = np.where(opponent_ships)
    num_ships = ship_pos[0].size
    for j in range(num_ships):
      valid_rel_directions = copy.copy(RELATIVE_DIRECTIONS)
      row = ship_pos[0][j]
      col = ship_pos[1][j]
      ship_halite = halite_ships[row, col]
      for row_diff in range(-2, 3):
        for col_diff in range(-2, 3):
          distance = (np.abs(row_diff) + np.abs(col_diff))
          if distance == 1 or distance == 2:
            other_row = (row + row_diff) % size
            other_col = (col + col_diff) % size
            if enemy_ships[other_row, other_col]:
              hal_diff = halite_ships[other_row, other_col] - ship_halite
              rem_dirs = []
              rem_dirs += [(0, 0)] if distance == 1 and hal_diff < 0 else []
              rem_dirs += [(-1, 0)] if row_diff < 0 and hal_diff <= 0 else []
              rem_dirs += [(1, 0)] if row_diff > 0 and hal_diff <= 0 else []
              rem_dirs += [(0, -1)] if col_diff < 0 and hal_diff <= 0 else []
              rem_dirs += [(0, 1)] if col_diff > 0 and hal_diff <= 0 else []
              
              for d in rem_dirs:
                if d in valid_rel_directions:
                  valid_rel_directions.remove(d)
                  
      valid_opponent_actions[(row, col)] = valid_rel_directions
      
  return valid_opponent_actions


def get_ship_scores(config, observation, player_obs, env_config, np_rng,
                    verbose):
  convert_cost = env_config.convertCost
  spawn_cost = env_config.spawnCost
  my_bases = observation['rewards_bases_ships'][0][1]
  obs_halite = np.maximum(0, observation['halite'])
  # Clip obs_halite to zero when gathering it doesn't add to the score
  # code: delta_halite = int(cell.halite * configuration.collect_rate)
  collect_rate = env_config.collectRate
  obs_halite[obs_halite < 1/collect_rate] = 0
  num_ships = len(player_obs[2])
  first_base = num_ships == 1 and my_bases.sum() == 0
  max_ships = config['max_ships']
  early_game_return_boost_step = config['early_game_return_boost_step']
  step = observation['step']
  early_game_not_max_ships = (num_ships < max_ships) and (
    step < early_game_return_boost_step)
  early_game_return_boost = (early_game_return_boost_step-step)/(
    early_game_return_boost_step)*config[
      'early_game_return_base_additional_multiplier']*early_game_not_max_ships
  end_game_return_boost = (step > env_config['episodeSteps']*0.97)*config[
    'end_game_return_base_additional_multiplier']
      
  # Override the maximum number of conversions on the last episode turn
  last_episode_turn = observation['relative_step'] == 1

  grid_size = obs_halite.shape[0]
  # smoothed_friendly_ship_halite = rule_utils.smooth2d(
  #   observation['rewards_bases_ships'][0][3])
  smoothed_halite = rule_utils.smooth2d(obs_halite)
  can_deposit_halite = my_bases.sum() > 0
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  my_ship_count = len(player_obs[2])
  all_ship_count = opponent_ships.sum() + my_ship_count
  my_ship_fraction = my_ship_count/(1e-9+all_ship_count)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  enemy_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  
  # Flag to indicate I should not occupy/flood my base with early ships
  my_halite = observation['rewards_bases_ships'][0][0]
  avoid_base_early_game = my_halite >= spawn_cost and (
    observation['step'] < 20) and my_bases.sum() == 1 and (
      my_halite % 500) == 0 and num_ships < 9

  # Distance to nearest base mask - gathering closer to my base is better
  base_nearest_distance = get_nearest_base_distance(player_obs, grid_size)

  # Get opponent ship actions that avoid collisions with less halite ships
  opponent_ships_sensible_actions = get_valid_opponent_ship_actions(
    observation['rewards_bases_ships'], halite_ships, grid_size)

  ship_scores = {}
  for i, ship_k in enumerate(player_obs[2]):
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    dm = DISTANCE_MASKS[(row, col)]
    ship_halite = player_obs[2][ship_k][1]
    
    opponent_smoother_less_halite_ships = rule_utils.smooth2d(np.logical_and(
      opponent_ships, halite_ships <= ship_halite), smooth_kernel_dim=5)
    
    # Scores 1: collecting halite at row, col
    # Multiply the smoothed halite, added with the obs_halite with a distance
    # mask, specific for the current row and column
    collect_grid_scores = dm*(
      smoothed_halite * config['collect_smoothed_multiplier'] + 
      obs_halite * config['collect_actual_multiplier']) * (
        config['collect_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) * (
            base_nearest_distance ** config[
              'collect_base_nearest_distance_exponent'])
    
    # Override the collect score to 0 to avoid blocking the base early on in
    # the game: All squares right next to the initial base are set to 0
    if avoid_base_early_game:
      collect_grid_scores, early_next_base_dir, drop_None_valid = (
        set_scores_single_nearby_zero(
          collect_grid_scores, my_bases, grid_size, row, col))
    else:
      early_next_base_dir = None
      drop_None_valid = False
      
    # TODO: send the first N ships to the center by default in the first K
    # episode steps (?)
    
    # Scores 2: returning to any of my bases
    base_return_grid_multiplier = dm*ship_halite*(
      config['return_base_multiplier'] * (
        config['return_base_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) + early_game_return_boost + (
            end_game_return_boost))
    return_to_base_scores = my_bases*base_return_grid_multiplier
    
    # Override the return base score to 0 to avoid blocking the base early on
    # in the game.
    if avoid_base_early_game:
      return_to_base_scores = override_early_return_base_scores(
        return_to_base_scores, my_bases, row, col, grid_size, num_ships)
    
    # Scores 3: establish a new base
    establish_base_scores = dm*(
      smoothed_halite-obs_halite) * (config[
        'establish_base_smoothed_multiplier'] + first_base*config[
          'establish_first_base_smoothed_multiplier_correction'])*(
            1-((my_bases*dm).max()))*(1-my_bases) * (
        config['establish_base_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) - (
              convert_cost*can_deposit_halite) + min(
                ship_halite, convert_cost)*(
                  config['establish_base_deposit_multiplier'])
                  
    # Scores 4: attack an opponent base at row, col
    attack_base_scores = config['attack_base_multiplier']*dm*enemy_bases*(
      config['attack_base_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) - (config[
            'attack_base_halite_sum_multiplier'] * obs_halite.sum() / (
              all_ship_count))*int(my_ship_fraction < 0.5) - 1e9*(
                ship_halite > 0)
                
    # Update the scores as a function of nearby enemy ships to avoid collisions
    # with opposing ships that carry less halite and promote collisions with
    # enemy ships that carry less halite
    (collect_grid_scores, return_to_base_scores, establish_base_scores,
     attack_base_scores, preferred_directions, valid_directions,
     agent_surrounded, two_step_bad_directions) = update_scores_enemy_ships(
       config, collect_grid_scores, return_to_base_scores,
       establish_base_scores, attack_base_scores, opponent_ships, halite_ships,
       row, col, grid_size, spawn_cost, drop_None_valid, obs_halite,
       collect_rate, np_rng, opponent_ships_sensible_actions)
       
    # Update the scores as a function of blocking enemy bases and my early
    # game initial base
    (collect_grid_scores, return_to_base_scores, establish_base_scores,
     attack_base_scores, valid_directions) = update_scores_blockers(
       collect_grid_scores, return_to_base_scores, establish_base_scores,
       attack_base_scores, row, col, grid_size, enemy_bases, valid_directions,
       early_next_base_dir, update_attack_base=False)
            
    if last_episode_turn:
      # Convert all ships with more halite than the convert cost on the last
      # episode step
      establish_base_scores[row, col] = 1e9*(ship_halite > convert_cost)
        
    ship_scores[ship_k] = (
      collect_grid_scores, return_to_base_scores, establish_base_scores,
      attack_base_scores, preferred_directions, agent_surrounded,
      valid_directions, two_step_bad_directions)
    
  return ship_scores

def get_ship_plans(config, observation, player_obs, env_config, verbose,
                   all_ship_scores, convert_first_ship_on_None_action=True):
  my_bases = observation['rewards_bases_ships'][0][1]
  can_deposit_halite = my_bases.sum() > 0
  grid_size = observation['halite'].shape[0]
  ship_ids = list(player_obs[2])
  num_ships = len(player_obs[2])
  convert_cost = env_config.convertCost
  num_bases = my_bases.sum()
  new_bases = []
  
  # First, process the convert actions
  # TODO: make it a function of the game state - e.g. don't convert low halite
  # ships towards the end of a game.
  ship_plans = OrderedDict()
  for i, ship_k in enumerate(player_obs[2]):
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    
    ship_scores = all_ship_scores[ship_k]
    ship_halite = player_obs[2][ship_k][1]
    has_budget_to_convert = (ship_halite + player_obs[0]) >= convert_cost
    convert_surrounded_ship = ship_scores[5] and (
      ship_halite >= (convert_cost/2)) and has_budget_to_convert
    valid_directions = ship_scores[6]
    almost_boxed_in = not None in valid_directions and (len(
      valid_directions) == 1 or set(valid_directions) in [
        set([NORTH, SOUTH]), set([EAST, WEST])])
    if (has_budget_to_convert and ship_scores[2].max() >= max([
          ship_scores[0].max()*can_deposit_halite,
          ship_scores[1].max(),
          ship_scores[3].max(),
          ]) and (not almost_boxed_in)) or (convert_surrounded_ship):
            
      # Obtain the row and column of the new target base
      target_base = np.where(ship_scores[2] == ship_scores[2].max())
      target_row = target_base[0][0]
      target_col = target_base[1][0]
      
      if (target_row == row and target_col == col) or convert_surrounded_ship:
        ship_plans[ship_k] = CONVERT
        new_bases.append((row, col))
        my_bases[row, col] = True
        can_deposit_halite = True
      else:
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False)
        
  # Next, do another pass to coordinate the target squares. This is done in a
  # single pass for now where the selection order is determined based on the 
  # availability of > 1 direction in combination with the initial best score
  ship_priority_scores = np.zeros(num_ships)
  for i, ship_k in enumerate(player_obs[2]):
    ship_scores = all_ship_scores[ship_k]
    for (r, c) in new_bases:
      # TODO: Update the return to base score for new bases
      ship_scores[0][r, c] = 0
      ship_scores[2][r, c] = 0
    all_ship_scores[ship_k] = ship_scores
    
    ship_priority_scores[i] = np.stack([
      ship_scores[0], ship_scores[1], ship_scores[2], ship_scores[3]]).max()+(
        1e9*(len(ship_scores[6]) == 1))
    
  ship_order = np.argsort(-ship_priority_scores)
  occupied_target_squares = []
  single_escape_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  single_path_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  return_base_distances = []
  for i in range(num_ships):
    ship_k = ship_ids[ship_order[i]]
    if not ship_k in ship_plans:
      row, col = utils.row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      ship_scores = all_ship_scores[ship_k]
      valid_directions = ship_scores[6]
      
      for sq, d in [(single_escape_squares, 4), (single_path_squares, 1)]:
        if sq.sum():
          (s0, s1, s2, s3, s6) = update_scores_blockers(
            ship_scores[0], ship_scores[1], ship_scores[2], ship_scores[3],
            row, col, grid_size, sq, valid_directions, blocker_max_distance=d,
            update_attack_base=True)
          ship_scores = (s0, s1, s2, s3, ship_scores[4], ship_scores[5], s6,
                         ship_scores[7])
      
      for (r, c) in occupied_target_squares:
        ship_scores[0][r, c] = 0
        ship_scores[2][r, c] = 0
        ship_scores[3][r, c] = 0

      for (r, c, d) in return_base_distances:
        # This coordinates return to base actions and avoids base blocking
        if grid_distance(r, c, row, col, grid_size) == d:
          ship_scores[1][r, c] = 0
      
      best_collect_score = ship_scores[0].max()
      best_return_score = ship_scores[1].max()
      best_establish_score = ship_scores[2].max()
      best_attack_base_score = ship_scores[3].max()
      
      if best_collect_score >= max([
          best_return_score, best_establish_score, best_attack_base_score]):
        # Gather mode
        target_gather = np.where(ship_scores[0] == ship_scores[0].max())
        target_row = target_gather[0][0]
        target_col = target_gather[1][0]
        
        if target_row == row and target_col == col and num_ships == 1 and (
            num_bases == 0) and convert_first_ship_on_None_action:
          ship_plans[ship_k] = CONVERT
          my_bases[row, col] = True
          occupied_target_squares.append((row, col))
        else:
          ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False)
          occupied_target_squares.append((target_row, target_col))
      elif best_return_score >= max(
          best_establish_score, best_attack_base_score):
        # Return base mode
        target_return = np.where(ship_scores[1] == ship_scores[1].max())
        target_row = target_return[0][0]
        target_col = target_return[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False)
        base_distance = grid_distance(target_row, target_col, row, col,
                                      grid_size)
        return_base_distances.append((target_row, target_col, base_distance))
      elif best_establish_score >= best_attack_base_score:
        # Establish base mode
        target_base = np.where(ship_scores[2] == ship_scores[2].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False)
        occupied_target_squares.append((target_row, target_col))
      else:
        # Attack base mode
        target_base = np.where(ship_scores[3] == ship_scores[3].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], True)
        occupied_target_squares.append((target_row, target_col))
        
      if len(valid_directions) == 1 and not None in valid_directions and (
          target_row != row or target_col != col):
        # I have only one escape direction and must therefore take that path
        escape_square = rule_utils.move_ship_row_col(
          row, col, valid_directions[0], grid_size)
        single_escape_squares[escape_square[0], escape_square[1]] = 1
        
      elif (target_row == row) != (target_col == col):
        # I only have a single path to my target, so my next position, which is
        # deterministic, should be treated as an opponent base (don't set
        # plan targets behind the ship to avoid collisions with myself)
        move_dir = get_dir_from_target(
          row, col, target_row, target_col, grid_size)[0]
        next_square = rule_utils.move_ship_row_col(
          row, col, move_dir, grid_size)
        single_path_squares[next_square[0], next_square[1]] = 1
        
    all_ship_scores[ship_k] = ship_scores
      
  return ship_plans, my_bases, all_ship_scores

def get_dir_from_target(row, col, target_row, target_col, grid_size):
  if row == target_row and col == target_col:
    return [None]
  horiz_diff = target_col-col
  horiz_distance = min(np.abs(horiz_diff),
    min(np.abs(horiz_diff-grid_size), np.abs(horiz_diff+grid_size)))
  vert_diff = target_row-row
  vert_distance = min(np.abs(vert_diff),
    min(np.abs(vert_diff-grid_size), np.abs(vert_diff+grid_size)))
  
  half_grid = grid_size / 2
  shortest_directions = []
  if horiz_distance > 0:
    if target_col > col:
      shortest_dirs = [EAST if (target_col - col) <= half_grid else WEST]
    else:
      shortest_dirs = [WEST if (col - target_col) <= half_grid else EAST]
    if horiz_distance == grid_size/2:
      shortest_dirs = [EAST, WEST]
    shortest_directions.extend(shortest_dirs)
  if vert_distance > 0:
    if target_row > row:
      shortest_dirs = [SOUTH if (target_row - row) <= half_grid else NORTH]
    else:
      shortest_dirs = [NORTH if (row - target_row) <= half_grid else SOUTH]
    if vert_distance == grid_size/2:
      shortest_dirs = [NORTH, SOUTH]
    shortest_directions.extend(shortest_dirs)
    
  return shortest_directions

def map_ship_plans_to_actions(config, observation, player_obs, env_config,
                              verbose, ship_scores, ship_plans, np_rng):
  ship_actions = {}
  remaining_budget = player_obs[0]
  convert_cost = env_config.convertCost
  obs_halite = np.maximum(0, observation['halite'])
  # Clip obs_halite to zero when gathering it doesn't add to the score
  # code: delta_halite = int(cell.halite * configuration.collect_rate)
  obs_halite[obs_halite < 1/env_config.collectRate] = 0
  grid_size = obs_halite.shape[0]
  num_ships = len(player_obs[2])
  my_next_ships = np.zeros((grid_size, grid_size), dtype=np.bool)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  updated_ship_pos = {}
  
  # List all positions you definitely don't want to move to. Initially this
  # only contains enemy bases and eventually also earlier ships.
  bad_positions = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  
  # Order the ship plans based on the available valid direction count. Break
  # ties using the original order
  move_valid_actions = OrderedDict()
  ship_priority_scores = np.zeros(num_ships)
  ship_key_plans = list(ship_plans)
  for i, ship_k in enumerate(ship_key_plans):
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    valid_actions = []
    
    if not isinstance(ship_plans[ship_k], str):
      target_row, target_col, preferred_directions, ignore_base_collision = (
        ship_plans[ship_k])
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                             grid_size)
      
      # Filter out bad positions from the shortest actions
      for a in shortest_actions:
        move_row, move_col = rule_utils.move_ship_row_col(
          row, col, a, grid_size)
        if not bad_positions[move_row, move_col] or (
            ignore_base_collision and (
              move_row == target_row and move_col == target_col)):
          valid_actions.append(a)
      move_valid_actions[ship_k] = valid_actions
  
    ship_priority_scores[i] = -1e6*len(ship_scores[ship_k][6]) -1e3*len(
      valid_actions) - i
  
  ship_order = np.argsort(-ship_priority_scores)
  ordered_ship_plans = [ship_key_plans[o] for o in ship_order]
  
  for ship_k in ordered_ship_plans:
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    has_selected_action = False
    
    if isinstance(ship_plans[ship_k], str):
      # TODO: verify this is not buggy when the opponent has destroyed one of
      # my bases and there are no bases left
      if ship_plans[ship_k] == "CONVERT" and (
          halite_ships[row, col] < convert_cost/4) and (
            halite_ships[row, col] > 0) and num_ships > 1:
        # Override the convert logic - it's better to lose some ships than to
        # convert too often (good candidate for stateful logic)
        if not ship_scores[ship_k][6]:
          # It's better not to lose halite to some opponents
          # TODO: make it a function of the current score, ships and base count
          # Great candidate for stateful logic that learns a model of the
          # opponent behavior.
          target_row = np.mod(row + np_rng.choice([-1, 1]), grid_size)
          target_col = np.mod(col + np_rng.choice([-1, 1]), grid_size)
        else:
          target_row = np.mod(row + np_rng.choice([-1, 1]), grid_size)
          target_col = np.mod(col + np_rng.choice([-1, 1]), grid_size)
        ship_plans[ship_k] = (target_row, target_col, [], False)
      else:
        ship_actions[ship_k] = ship_plans[ship_k]
        obs_halite[row, col] = 0
        remaining_budget -= convert_cost
        has_selected_action = True
    if not has_selected_action:
      target_row, target_col, preferred_directions, ignore_base_collision = (
        ship_plans[ship_k])
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                             grid_size)
      
      # Filter out bad positions from the shortest actions
      valid_actions = []
      for a in shortest_actions:
        move_row, move_col = rule_utils.move_ship_row_col(
          row, col, a, grid_size)
        if (not bad_positions[move_row, move_col] and (
                a in ship_scores[ship_k][6])) or (
                  ignore_base_collision and (
                    move_row == target_row and move_col == target_col)):
          valid_actions.append(a)
      if valid_actions:
        if preferred_directions:
          # TODO: figure out if this is actually helpful (it makes the agent
          # quite predictable)
          intersect_directions = list(set(valid_actions) & set(
            preferred_directions))
          if intersect_directions:
            valid_actions = intersect_directions
        # TODO: if two actions are optimal, pick the one that moves into a
        # region with less other ships and prefer actions that lead to more
        # future options (e.g. prefer to move diagonally towards the base (?))
        # This is probably bad near the base since the current behavior makes
        # it hard to attack.
        action = str(np_rng.choice(valid_actions))
        action = None if action == 'None' else action
      else:
        # I can not move to my target using a shortest path action
        # Alternative: consider all valid, not bad actions
        valid_not_bad_actions = []
        for a in MOVE_DIRECTIONS:
          if a in ship_scores[ship_k][6]:
            move_row, move_col = rule_utils.move_ship_row_col(
                row, col, a, grid_size)
            if not bad_positions[move_row, move_col]:
              valid_not_bad_actions.append(a)
              
        # When attacking a base it is better to keep moving or stay still on
        # a square that has no halite - otherwise it becomes a target for
        # stealing halite.
        if valid_not_bad_actions and ignore_base_collision and obs_halite[
            row, col] > 0:
          if len(valid_not_bad_actions) > 1 and None in valid_not_bad_actions:
            valid_not_bad_actions.remove(None)
              
        if valid_not_bad_actions:
          action = np_rng.choice(valid_not_bad_actions)
        else:
          # Pick a random, not bad moving action
          shuffled_actions = np_rng.permutation(MOVE_DIRECTIONS[1:])
          found_non_bad = False
          for a in shuffled_actions:
            move_row, move_col = rule_utils.move_ship_row_col(
              row, col, a, grid_size)
            if not bad_positions[move_row, move_col]:
              action = str(a)
              found_non_bad = True
              break
          
          # If all actions are bad: do nothing - this is very rare since it
          # would mean being surrounded by my other ships and opponent bases
          if not found_non_bad:
            action = None
      
      # Update my_next_ships
      new_row, new_col = rule_utils.move_ship_row_col(
        row, col, action, grid_size)
      my_next_ships[new_row, new_col] = 1
      bad_positions[new_row, new_col] = 1
      updated_ship_pos[ship_k] = (new_row, new_col)
      if action is not None:
        ship_actions[ship_k] = action
  
  return (ship_actions, remaining_budget, my_next_ships, obs_halite,
          updated_ship_pos)

def decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships, obs_halite,
    env_config, remaining_budget, verbose, ship_plans, updated_ship_pos):

  spawn_cost = env_config.spawnCost
  num_ships = my_next_ships.sum()
  max_spawns = int(remaining_budget/spawn_cost)
  max_spawns = min(max_spawns, int(config['max_ships']-num_ships))
  max_allowed_ships = config['max_ships']
  total_ship_count = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships']]).sum()
  max_spawns = min(max_spawns, int(max_allowed_ships - num_ships))
  max_spawns = min(max_spawns, int(obs_halite.sum()/2/spawn_cost))
  relative_step = observation['relative_step']
  max_spawns = min(max_spawns, int(
    obs_halite.sum()/min(
      total_ship_count+1e-9, (num_ships+1e-9)*2)/spawn_cost*(
      1-relative_step)*398/config['max_spawn_relative_step_divisor']))
  last_episode_turn = observation['relative_step'] == 1

  if max_spawns <= 0 or not player_obs[1] or last_episode_turn:
    return {}, remaining_budget
  
  num_bases = len(player_obs[1])
  spawn_scores = np.zeros(num_bases)
  grid_size = obs_halite.shape[0]
  smoothed_friendly_ship_halite = rule_utils.smooth2d(
    observation['rewards_bases_ships'][0][3])
  smoothed_halite = rule_utils.smooth2d(obs_halite)
  
  for i, base_k in enumerate(player_obs[1]):
    row, col = utils.row_col_from_square_grid_pos(
      player_obs[1][base_k], grid_size)
    
    # Don't spawn when there will be a ship at the base
    spawn_scores[i] -= 1e9*my_next_ships[row, col]
    
    # Don't spawn when there is a returning ship that wants to enter the base
    # in two steps
    # Exception: if the base is not too crowded, it is ok to spawn in this
    # scenario.
    near_base_ship_count = np.logical_and(
      my_next_ships, ROW_COL_MAX_DISTANCE_MASKS[(row, col, 3)]).sum()
    if near_base_ship_count >= 3:
      for k in ship_plans:
        if ship_plans[k][0] == row and ship_plans[k][1] == col:
          updated_distance = grid_distance(row, col, updated_ship_pos[k][0],
                                           updated_ship_pos[k][1], grid_size)
          if updated_distance == 1:
            spawn_scores[i] -= 1e6
            break
    
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

def get_numpy_random_generator(
    config, observation, rng_action_seed, print_seed=False):
  if rng_action_seed is None:
    rng_action_seed = 0
  
  if observation['step'] == 0 and print_seed:
    print("Random acting seed: {}".format(rng_action_seed))
    
  return np.random.RandomState(rng_action_seed)

def get_config_actions(config, observation, player_obs, env_config,
                       rng_action_seed, verbose=False):
  # Set the random seed
  np_rng = get_numpy_random_generator(
    config, observation, rng_action_seed, print_seed=True)
  
  # Compute the ship scores for all high level actions
  ship_scores = get_ship_scores(config, observation, player_obs, env_config,
                                np_rng, verbose)
  
  # Compute the coordinated high level ship plan
  ship_plans, my_next_bases, plan_ship_scores = get_ship_plans(
    config, observation, player_obs, env_config, verbose,
    copy.deepcopy(ship_scores))
  
  # Translate the ship high level plans to basic move/convert actions
  (mapped_actions, remaining_budget, my_next_ships, my_next_halite,
   updated_ship_pos) = map_ship_plans_to_actions(
     config, observation, player_obs, env_config, verbose, ship_scores,
     ship_plans, np_rng)
  
  # Decide for all bases whether to spawn or keep the base available
  base_actions, remaining_budget = decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships,
    my_next_halite, env_config, remaining_budget, verbose, ship_plans,
    updated_ship_pos)
  
  mapped_actions.update(base_actions)
  halite_spent = player_obs[0]-remaining_budget
  
  step_details = {
    'ship_scores': ship_scores,
    'plan_ship_scores': plan_ship_scores,
    'ship_plans': ship_plans,
    'mapped_actions': mapped_actions,
    'observation': observation,
    'player_obs': player_obs,
    }
  
  return mapped_actions, halite_spent, step_details