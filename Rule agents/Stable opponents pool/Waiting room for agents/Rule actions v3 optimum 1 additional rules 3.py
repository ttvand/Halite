from collections import OrderedDict
import copy
import getpass
import itertools
import numpy as np
from scipy import signal
import time


LOCAL_MODE = getpass.getuser() == 'tom'

CONFIG = {
    'halite_config_setting_divisor': 1.0,
    'collect_smoothed_multiplier': 0.02,
    'collect_actual_multiplier': 5.0,
    'collect_less_halite_ships_multiplier_base': 0.55,
    'collect_base_nearest_distance_exponent': 0.1,
  
    'return_base_multiplier': 8.0,
    'return_base_less_halite_ships_multiplier_base': 0.85,
    'early_game_return_base_additional_multiplier': 0.1,
    'early_game_return_boost_step': 50,
    'establish_base_smoothed_multiplier': 0.0,
    
    'establish_first_base_smoothed_multiplier_correction': 2.5,
    'first_base_no_4_way_camping_spot_bonus': 500,
    'establish_base_deposit_multiplier': 1.0,
    'establish_base_less_halite_ships_multiplier_base': 1.0,
    'max_attackers_per_base': 3*1,
    
    'attack_base_multiplier': 200.0,
    'attack_base_less_halite_ships_multiplier_base': 0.9,
    'attack_base_halite_sum_multiplier': 2.0, #*0,
    'attack_base_run_enemy_multiplier': 1.0,
    'attack_base_catch_enemy_multiplier': 1.0,
    
    'collect_run_enemy_multiplier': 10.0,
    'return_base_run_enemy_multiplier': 2.0,
    'establish_base_run_enemy_multiplier': 2.5,
    'collect_catch_enemy_multiplier': 1.0,
    'return_base_catch_enemy_multiplier': 1.0,
    
    'establish_base_catch_enemy_multiplier': 0.5,
    'two_step_avoid_boxed_enemy_multiplier_base': 0.8,
    'n_step_avoid_boxed_enemy_multiplier_base': 0.45,
    'min_consecutive_chase_extrapolate': 5,
    'chase_return_base_exponential_bonus': 2.0,
    
    'ignore_catch_prob': 0.5,
    'max_ships': 20,
    'max_spawns_per_step': 1,
    'nearby_ship_halite_spawn_constant': 3.0,
    'nearby_halite_spawn_constant': 5.0,
    
    'remaining_budget_spawn_constant': 0.2,
    'spawn_score_threshold': 75.0,
    'boxed_in_halite_convert_divisor': 1.0,
    'n_step_avoid_min_die_prob_cutoff': 0.1,
    'n_step_avoid_window_size': 7,
    
    'influence_map_base_weight': 1.5,
    'influence_map_min_ship_weight': 0.0,
    'influence_weights_additional_multiplier': 4.0,
    'influence_weights_exponent': 8.0,
    'escape_influence_prob_divisor': 3.0,
    
    'rescue_ships_in_trouble': 1,
    'max_spawn_relative_step_divisor': 100.0,
    'no_spawn_near_base_ship_limit': 100,
    }


NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
CONVERT = "CONVERT"
SPAWN = "SPAWN"
NOT_NONE_DIRECTIONS = [NORTH, SOUTH, EAST, WEST]
MOVE_DIRECTIONS = [None, NORTH, SOUTH, EAST, WEST]
RELATIVE_DIR_MAPPING = {None: (0, 0), NORTH: (-1, 0), SOUTH: (1, 0),
                        EAST: (0, 1), WEST: (0, -1)}
RELATIVE_DIR_TO_DIRECTION_MAPPING = {
  v: k for k, v in RELATIVE_DIR_MAPPING.items()}
OPPOSITE_MAPPING = {None: None, NORTH: SOUTH, SOUTH: NORTH, EAST: WEST,
                    WEST: EAST}
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

GAUSSIAN_2D_KERNELS = {}
for dim in range(3, 20, 2):
  # Modified from https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
  center_distance = np.floor(np.abs(np.arange(dim) - (dim-1)/2))
  horiz_distance = np.tile(center_distance, [dim, 1])
  vert_distance = np.tile(np.expand_dims(center_distance, 1), [1, dim])
  manh_distance = horiz_distance + vert_distance
  kernel = np.exp(-manh_distance/(dim/4))
  kernel[manh_distance > dim/2] = 0
  
  GAUSSIAN_2D_KERNELS[dim] = kernel

DISTANCES = {}
DISTANCE_MASKS = {}
HALF_PLANES_CATCH = {}
HALF_PLANES_RUN = {}
ROW_COL_DISTANCE_MASKS = {}
ROW_COL_MAX_DISTANCE_MASKS = {}
ROW_COL_BOX_MAX_DISTANCE_MASKS = {}
BOX_DIRECTION_MASKS = {}
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
    DISTANCES[(row, col)] = manh_distance
    
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
      ROW_COL_BOX_MAX_DISTANCE_MASKS[(row, col, d)] = np.logical_and(
        horiz_distance <= d, vert_distance <= d)
      
    for dist in range(2, 10):
      dist_mask_dim = dist*2+1
      row_pos = np.tile(np.expand_dims(np.arange(dist_mask_dim), 1),
                        [1, dist_mask_dim])
      col_pos = np.tile(np.arange(dist_mask_dim), [dist_mask_dim, 1])
      for direction in NOT_NONE_DIRECTIONS:
        if direction == NORTH:
          box_mask = (row_pos < dist) & (
            np.abs(col_pos-dist) <= (dist-row_pos))
        if direction == SOUTH:
          box_mask = (row_pos > dist) & (
            np.abs(col_pos-dist) <= (row_pos-dist))
        if direction == WEST:
          box_mask = (col_pos < dist) & (
            np.abs(row_pos-dist) <= (dist-col_pos))
        if direction == EAST:
          box_mask = (col_pos > dist) & (
            np.abs(row_pos-dist) <= (col_pos-dist))
        BOX_DIRECTION_MASKS[(dist, direction)] = box_mask
      
def row_col_from_square_grid_pos(pos, size):
  col = pos % size
  row = pos // size
  
  return row, col
      
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
    attack_base_scores, opponent_ships, opponent_bases, halite_ships, row, col,
    grid_size, spawn_cost, drop_None_valid, obs_halite, collect_rate, np_rng,
    opponent_ships_sensible_actions, ignore_bad_attack_directions,
    observation, ship_k, my_bases, my_ships, steps_remaining, history,
    escape_influence_probs, player_ids, env_obs_ids, env_observation,
    main_base_distances, nearest_base_distances):
  direction_halite_diff_distance_raw = {
    NORTH: [], SOUTH: [], EAST: [], WEST: []}
  my_bases_or_ships = np.logical_or(my_bases, my_ships)
  
  chase_details = history['chase_counter'][0].get(ship_k, None)
  take_my_square_next_halite_diff = None
  take_my_next_square_dir = None
  for row_shift in range(-2, 3):
    considered_row = (row + row_shift) % grid_size
    for col_shift in range(-2, 3):
      considered_col = (col + col_shift) % grid_size
      distance = np.abs(row_shift) + np.abs(col_shift)
      if distance <= 2:
        if opponent_ships[considered_row, considered_col]:
          relevant_dirs = []
          halite_diff = halite_ships[row, col] - halite_ships[
            considered_row, considered_col]
          assume_take_my_square_next = False
          
          # Extrapolate the opponent behavior if we have been chased for a 
          # while and chasing is likely to continue
          if distance == 1 and chase_details is not None and (
              chase_details[1] >= config[
                'min_consecutive_chase_extrapolate']) and (
                  considered_row, considered_col) == (
                    chase_details[4], chase_details[5]):
            chaser_row = chase_details[4]
            chaser_col = chase_details[5]
            to_opponent_dir = get_dir_from_target(
              row, col, chaser_row, chaser_col, grid_size)[0]
            opp_to_me_dir = OPPOSITE_MAPPING[to_opponent_dir]
            rel_opp_to_me_dir = RELATIVE_DIR_MAPPING[opp_to_me_dir]
            opp_can_move_to_me = rel_opp_to_me_dir in (
              opponent_ships_sensible_actions[chaser_row, chaser_col])
            
            # There is a unique opponent id with the least amount of halite
            # on the chaser square or the chaser has at least one friendly
            # ship that can replace it
            chaser_can_replace = None
            chaser_is_chased_by_not_me = None
            if opp_can_move_to_me:
              chaser_id = player_ids[chaser_row, chaser_col]
              near_chaser = ROW_COL_MAX_DISTANCE_MASKS[
                chaser_row, chaser_col, 1]
              near_halite = halite_ships[near_chaser]
              near_chaser_friendly_halite = near_halite[
                (near_halite >= 0) & (player_ids[near_chaser] == chaser_id)]
              min_non_chaser_halite = near_halite[
                (near_halite >= 0) & (
                  player_ids[near_chaser] != chaser_id)].min()
              min_near_chaser_halite = near_halite[near_halite >= 0].min()
              opponent_min_hal_ids = player_ids[np.logical_and(
                near_chaser, halite_ships == min_near_chaser_halite)]
              
              near_me = ROW_COL_MAX_DISTANCE_MASKS[row, col, 1]
              near_me_threat_players = player_ids[np.logical_and(
                near_me, (halite_ships >= 0) & (
                  halite_ships < halite_ships[row, col]))]
              
              double_opp_chase = (near_me_threat_players.size > 1) and (
                np.all(near_me_threat_players == chaser_id))
              
              chaser_can_replace = ((opponent_min_hal_ids.size > 1) and (
                np.all(opponent_min_hal_ids == chaser_id) or (
                (opponent_min_hal_ids == chaser_id).sum() > 1)) or (
                  (near_chaser_friendly_halite <= (
                    min_non_chaser_halite)).sum() > 1)) or double_opp_chase
                
              if opp_can_move_to_me and not chaser_can_replace:
                chaser_players_index = env_obs_ids[chaser_id]
                chaser_k = [k for k, v in env_observation.players[
                  chaser_players_index][2].items() if v[0] == (
                    chaser_row*grid_size + chaser_col)][0]
                chaser_is_chased = chaser_k in history[
                  'chase_counter'][chaser_id]
                
                chaser_is_chased_by_not_me = chaser_is_chased
                if chaser_is_chased:
                  chaser_chaser = history['chase_counter'][chaser_id][chaser_k]
                  chaser_is_chased_by_not_me = (chaser_chaser[4] is None) or (
                    player_ids[chaser_chaser[4], chaser_chaser[5]] != 0)
            
            if opp_can_move_to_me and not chaser_can_replace and not (
                chaser_is_chased_by_not_me):
              assume_take_my_square_next = True
              take_my_square_next_halite_diff = halite_diff
              take_my_next_square_dir = to_opponent_dir
          
          opponent_id = player_ids[considered_row, considered_col]
          is_near_base = (
            nearest_base_distances[considered_row, considered_col] <= 2)
          careful_lookup_k = str(is_near_base) + '_' + str(distance) + (
            '_careful')
          can_ignore_ship = history['zero_halite_move_behavior'][opponent_id][
            careful_lookup_k] and (halite_ships[row, col] == 0) and (
              halite_ships[considered_row, considered_col] == 0)
          
          if not assume_take_my_square_next and not can_ignore_ship:
            relevant_dirs += [] if row_shift >= 0 else [NORTH]
            relevant_dirs += [] if row_shift <= 0 else [SOUTH]
            relevant_dirs += [] if col_shift <= 0 else [EAST]
            relevant_dirs += [] if col_shift >= 0 else [WEST]
          
          for d in relevant_dirs:
            direction_halite_diff_distance_raw[d].append(
              (halite_diff, distance))
            
  direction_halite_diff_distance = {}
  for k in direction_halite_diff_distance_raw:
    vals = np.array(direction_halite_diff_distance_raw[k])
    if vals.size:
      diffs = vals[:, 0]
      distances = vals[:, 1]
      max_diff = diffs.max()
      if max_diff > 0:
        greater_min_distance = distances[diffs > 0].min()
        direction_halite_diff_distance[k] = (max_diff, greater_min_distance)
      elif max_diff == 0:
        equal_min_distance = distances[diffs == 0].min()
        direction_halite_diff_distance[k] = (max_diff, equal_min_distance)
      else:
        min_diff = diffs.min()
        min_diff_min_distance = distances[diffs == min_diff].min()
        direction_halite_diff_distance[k] = (min_diff, min_diff_min_distance)
    else:
      direction_halite_diff_distance[k] = None
                
  ship_halite = halite_ships[row, col]
  preferred_directions = []
  valid_directions = copy.copy(MOVE_DIRECTIONS)
  one_step_valid_directions = copy.copy(MOVE_DIRECTIONS)
  bad_directions = []
  ignore_catch = np_rng.uniform() < config['ignore_catch_prob']
  
  # if observation['step'] == 13 and ship_k == '8-1':
  #   import pdb; pdb.set_trace()
  #   x=1
  
  for direction, halite_diff_dist in direction_halite_diff_distance.items():
    if halite_diff_dist is not None:
      halite_diff = halite_diff_dist[0]
      if halite_diff >= 0:
        # I should avoid a collision
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = np.copy(HALF_PLANES_RUN[(row, col)][direction])
        valid_directions.remove(direction)
        one_step_valid_directions.remove(direction)
        bad_directions.append(direction)
        if halite_diff_dist[1] == 1:
          if halite_diff != 0:
            # Only suppress the stay still action if the opponent has something
            # to gain. TODO: consider making this a function of the game state
            if None in valid_directions:
              valid_directions.remove(None)
              one_step_valid_directions.remove(None)
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
        base_nearby_in_direction_mask = np.logical_and(
          ROW_COL_MAX_DISTANCE_MASKS[(row, col, 2)], mask_collect_return)
        base_nearby_in_direction = np.logical_and(
          base_nearby_in_direction_mask, opponent_bases).sum() > 0
        if not ignore_bad_attack_directions and not base_nearby_in_direction:
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
        halite_diff = max(-spawn_cost/2, halite_diff)
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

  if take_my_square_next_halite_diff is not None and None in valid_directions:
    valid_directions.remove(None)
    one_step_valid_directions.remove(None)
    bad_directions.append(None)

  if drop_None_valid and None in valid_directions:
    valid_directions.remove(None)
    one_step_valid_directions.remove(None)
    
  valid_non_base_directions = []
  base_directions = []
  for d in valid_directions:
    move_row, move_col = move_ship_row_col(row, col, d, grid_size)
    if not opponent_bases[move_row, move_col] :
      valid_non_base_directions.append(d)
    else:
      base_directions.append(d)
    
  # For the remaining valid non base directions: compute a score that resembles
  # the probability of being boxed in during the next step
  two_step_bad_directions = []
  n_step_step_bad_directions = []
  n_step_bad_directions_die_probs = {}
  if steps_remaining > 1:
    for d in valid_non_base_directions:
      my_next_halite = halite_ships[row, col] if d != None else (
        halite_ships[row, col] + int(collect_rate*obs_halite[row, col]))
      move_row, move_col = move_ship_row_col(row, col, d, grid_size)
      my_next_halite = 0 if my_bases[move_row, move_col] else my_next_halite
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
                halite_ships[other_row, other_col] + int(
                  collect_rate*obs_halite[other_row, other_col])))
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
          collect_grid_scores[mask_avoid_two_steps] *= ((
            config['two_step_avoid_boxed_enemy_multiplier_base']) ** (
              threat_score))
          return_to_base_scores[mask_avoid_two_steps] *= ((
            config['two_step_avoid_boxed_enemy_multiplier_base']) ** (
              threat_score))
          establish_base_scores[mask_avoid_two_steps] *= ((
            config['two_step_avoid_boxed_enemy_multiplier_base']) ** (
              threat_score))
          
          two_step_bad_directions.append(d)
          
      if d not in two_step_bad_directions:
        # For the remaining valid directions: compute a score that resembles 
        # the probability of being boxed in sometime in the future
        opponent_mask_lt = ROW_COL_MAX_DISTANCE_MASKS[
          (move_row, move_col, min(
            steps_remaining, config['n_step_avoid_window_size']))]
        less_halite_threat_opponents_lt = np.where(np.logical_and(
          opponent_mask_lt, np.logical_and(
            opponent_ships, my_next_halite > halite_ships)))
        num_threat_ships_lt = less_halite_threat_opponents_lt[0].size
        
        # Ignore the box in threat if I have a base and at least one zero
        # halite ship one step from the move square
        ignore_threat = my_bases[
          ROW_COL_DISTANCE_MASKS[(move_row, move_col, 1)]].sum() > 0 and ((
            halite_ships[np.logical_and(
              my_ships,
              ROW_COL_DISTANCE_MASKS[move_row, move_col, 1])] == 0).sum() > 0)
        
        # if observation['step'] == 359 and ship_k == '67-1':
        #   import pdb; pdb.set_trace()
            
        if not ignore_threat:
          lt_catch_prob = {k: [] for k in RELATIVE_DIRECTIONS[:-1]}
          for i in range(num_threat_ships_lt):
            other_row = less_halite_threat_opponents_lt[0][i]
            other_col = less_halite_threat_opponents_lt[1][i]
            other_sensible_actions = opponent_ships_sensible_actions[
              other_row, other_col]
            relative_other_pos = get_relative_position(
              move_row, move_col, other_row, other_col, grid_size)
            
            # Give less weight to the other ship if there is a base of mine or
            # a/multiple less halite ships in between
            distance_move_other = np.abs(relative_other_pos).sum()
            mask_between_move_and_threat = np.logical_and(
                DISTANCES[(move_row, move_col)] < distance_move_other,
                DISTANCES[(other_row, other_col)] < distance_move_other)
            less_halite_ship_base_count = np.logical_and(
              np.logical_and(my_bases_or_ships, mask_between_move_and_threat),
              halite_ships <= halite_ships[other_row, other_col]).sum()
            my_material_defense_multiplier = 2**less_halite_ship_base_count
            
            for threat_dir in RELATIVE_DIRECTIONS[:-1]:
              nz_dim = int(threat_dir[0] == 0)
              dir_offset = relative_other_pos[nz_dim]*threat_dir[nz_dim]
              other_dir_abs_offset = np.abs(relative_other_pos[1-nz_dim])
              
              if dir_offset > 0 and (other_dir_abs_offset-1) <= dir_offset:
                # Ignore the threat if the ship is on the diagonal and can not
                # move in the direction of the threat dir
                if (other_dir_abs_offset-1) == dir_offset and len(
                    other_sensible_actions) < 5:
                  if nz_dim == 0:
                    threat_other_dir = (
                      0, 1 if relative_other_pos[1-nz_dim] < 0 else -1)
                  else:
                    threat_other_dir = (
                      1 if relative_other_pos[1-nz_dim] < 0 else -1, 0)
                  # import pdb; pdb.set_trace()
                  consider_this_threat = threat_other_dir in (
                    other_sensible_actions)
                else:
                  consider_this_threat = True
                
                if consider_this_threat:
                  lt_catch_prob[threat_dir].append((
                    other_dir_abs_offset+dir_offset)*(
                      my_material_defense_multiplier))
                    
          # Add a "bootstrapped" catch probability using the density of the
          # players towards the edge of the threat direction
          # Only add it if the next halite is > 0 (otherwise assume I can
          # always escape)
          if my_next_halite > 0:
            for threat_dir in RELATIVE_DIRECTIONS[:-1]:
              dens_threat_rows = np.mod(move_row + threat_dir[0]*(
                np.arange(config['n_step_avoid_window_size']//2,
                          config['n_step_avoid_window_size'])), grid_size)
              dens_threat_cols = np.mod(move_col + threat_dir[1]*(
                1+np.arange(config['n_step_avoid_window_size']//2,
                            config['n_step_avoid_window_size'])), grid_size)
              mean_escape_prob = escape_influence_probs[
                dens_threat_rows, dens_threat_cols].mean()
              lt_catch_prob[threat_dir].append(1/(1-mean_escape_prob+1e-9))
            
          # if observation['step'] == 359 and ship_k == '67-1':
          #   import pdb; pdb.set_trace()
          if np.all([len(v) > 0 for v in lt_catch_prob.values()]):
            survive_probs = np.array([
              (np.maximum(1, np.array(lt_catch_prob[k])-1)/np.array(
                lt_catch_prob[k])).prod() for k in lt_catch_prob])
            min_die_prob = 1-survive_probs.max()
            if main_base_distances.max() > 0:
              if main_base_distances[move_row, move_col] <= 2:
                min_die_prob = 0
              else:
                min_die_prob = max(
                  0, min_die_prob-0.33**main_base_distances[move_row, move_col])
            
            # if observation['step'] > 20:
            #   import pdb; pdb.set_trace()
            
            # Disincentivize an action that can get me boxed in during the next
            # N steps
            mask_avoid_n_steps = np.copy(HALF_PLANES_RUN[(row, col)][d])
            if d is not None:
              mask_avoid_n_steps[row, col] = False
            collect_grid_scores[mask_avoid_n_steps] *= ((
              config['n_step_avoid_boxed_enemy_multiplier_base']) ** (
                min_die_prob))
            return_to_base_scores[mask_avoid_n_steps] *= (
              config['n_step_avoid_boxed_enemy_multiplier_base']) ** (
                min_die_prob)
            establish_base_scores[mask_avoid_n_steps] *= (
              config['n_step_avoid_boxed_enemy_multiplier_base']) ** (
                min_die_prob)
                
            n_step_bad_directions_die_probs[d] = min_die_prob
            
            if min_die_prob > config['n_step_avoid_min_die_prob_cutoff']:
              n_step_step_bad_directions.append(d)
           
  # if observation['step'] == 136 and ship_k == '51-1':
  #   import pdb; pdb.set_trace()
    
  # Treat the chasing - replace chaser position as an n-step bad action.
  # Otherwise, we can get trapped in a loop of dumb behavior.
  if take_my_next_square_dir is not None and not take_my_next_square_dir in (
      two_step_bad_directions) and not take_my_next_square_dir in (
        n_step_step_bad_directions):
    n_step_step_bad_directions.append(take_my_next_square_dir)
    n_step_bad_directions_die_probs[take_my_next_square_dir] = 0.25
          
  if valid_non_base_directions:
    valid_not_preferred_dirs = list(set(
      two_step_bad_directions + n_step_step_bad_directions))
    if valid_not_preferred_dirs and (
      len(valid_non_base_directions) - len(valid_not_preferred_dirs)) > 0:
      # Drop 2 and n step bad directions if that leaves us with valid options
      bad_directions.extend(valid_not_preferred_dirs)
      valid_directions = list(
        set(valid_directions) - set(valid_not_preferred_dirs))
    else:
      # Drop 2 step bad directions if that leaves us with valid options
      valid_not_preferred_dirs = set(two_step_bad_directions)
      if valid_not_preferred_dirs and (
          len(valid_non_base_directions) - len(valid_not_preferred_dirs)) > 0:
        bad_directions.extend(valid_not_preferred_dirs)
        valid_directions = list(
          set(valid_directions) - set(valid_not_preferred_dirs))
  
  # if observation['step'] == 114 and ship_k == '64-1':
  #   import pdb; pdb.set_trace()
    
  return (collect_grid_scores, return_to_base_scores, establish_base_scores,
          attack_base_scores, preferred_directions, valid_directions,
          len(bad_directions) == len(MOVE_DIRECTIONS), two_step_bad_directions,
          n_step_step_bad_directions, one_step_valid_directions,
          n_step_bad_directions_die_probs)

# Update the scores as a function of blocking enemy bases
def update_scores_blockers(
    collect_grid_scores, return_to_base_scores, establish_base_scores,
    attack_base_scores, row, col, grid_size, blockers,
    blocker_max_distances_to_consider, valid_directions,
    one_step_valid_directions, early_base_direct_dir=None,
    blocker_max_distance=half_distance_mask_dim, update_attack_base=True):
  one_step_bad_directions = []
  for d in NOT_NONE_DIRECTIONS:
    if d == NORTH:
      rows = np.mod(row - (1 + np.arange(blocker_max_distance)), grid_size)
      cols = np.repeat(col, blocker_max_distance)
      considered_vals = blockers[rows, col]
      considered_max_distances = blocker_max_distances_to_consider[rows, col]
    elif d == SOUTH:
      rows = np.mod(row + (1 + np.arange(blocker_max_distance)), grid_size)
      cols = np.repeat(col, blocker_max_distance)
      considered_vals = blockers[rows, col]
      considered_max_distances = blocker_max_distances_to_consider[rows, col]
    elif d == WEST:
      rows = np.repeat(row, blocker_max_distance)
      cols = np.mod(col - (1 + np.arange(blocker_max_distance)), grid_size)
      considered_vals = blockers[row, cols]
      considered_max_distances = blocker_max_distances_to_consider[row, cols]
    elif d == EAST:
      rows = np.repeat(row, blocker_max_distance)
      cols = np.mod(col + (1 + np.arange(blocker_max_distance)), grid_size)
      considered_vals = blockers[row, cols]
      considered_max_distances = blocker_max_distances_to_consider[row, cols]

    if d == early_base_direct_dir:
      considered_vals[0] = 1
    
    is_blocking = np.logical_and(considered_vals, np.arange(
      blocker_max_distance) < considered_max_distances)
    
    if np.any(is_blocking):
      first_blocking_id = np.where(is_blocking)[0][0]
      mask_rows = rows[first_blocking_id:]
      mask_cols = cols[first_blocking_id:]
      
      collect_grid_scores[mask_rows, mask_cols] = -1e12
      return_to_base_scores[mask_rows, mask_cols] = -1e12
      establish_base_scores[mask_rows, mask_cols] = -1e12
      if update_attack_base:
        attack_base_scores[mask_rows, mask_cols] = -1e12
      
      if first_blocking_id == 0:
        one_step_bad_directions.append(d)
        if d in valid_directions:
          valid_directions.remove(d)
        if d in one_step_valid_directions:
          one_step_valid_directions.remove(d)
          
  # Lower the score for entire quadrants when the two quadrant directions are
  # blocking the movement
  num_bad_one_directions = len(one_step_bad_directions)
  if num_bad_one_directions > 1:
    for i in range(num_bad_one_directions-1):
      bad_direction_1 = one_step_bad_directions[i]
      for j in range(i+1, num_bad_one_directions):
        bad_direction_2 = one_step_bad_directions[j]
        if (bad_direction_1 in [NORTH, SOUTH]) != (
            bad_direction_2 in [NORTH, SOUTH]):
          bad_quadrant_mask = np.logical_and(
            HALF_PLANES_CATCH[row, col][bad_direction_1],
            HALF_PLANES_CATCH[row, col][bad_direction_2])
          collect_grid_scores[bad_quadrant_mask] = -1e12
          return_to_base_scores[bad_quadrant_mask] = -1e12
          establish_base_scores[bad_quadrant_mask] = -1e12
          if update_attack_base:
            attack_base_scores[bad_quadrant_mask] = -1e12 
        
  # Additional logic for the use of avoiding collisions when there is only a
  # single escape direction
  if blockers[row, col]:
    collect_grid_scores[row, col] = -1e12
    return_to_base_scores[row, col] = -1e12
    establish_base_scores[row, col] = -1e12
    attack_base_scores[row, col] = -1e12
    if None in valid_directions:
      valid_directions.remove(None)
    if None in one_step_valid_directions:
      one_step_valid_directions.remove(None)
      
  return (collect_grid_scores, return_to_base_scores, establish_base_scores,
          attack_base_scores, valid_directions, one_step_valid_directions,
          one_step_bad_directions)

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
        scores[near_row, near_col] = -1e7
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
    base_return_grid_multiplier, my_bases, ship_row, ship_col, size,
    my_ship_count):
  base_pos = np.where(my_bases)
  base_row = base_pos[0][0]
  base_col = base_pos[1][0]
  
  dist_to_base = grid_distance(base_row, base_col, ship_row, ship_col, size)
  # Remember the rule that blocks spawning when a ship is about to return
  if dist_to_base <= 10-my_ship_count:
    base_return_grid_multiplier[base_row, base_col] = 0
    
  return base_return_grid_multiplier

def get_nearest_base_distances(player_obs, grid_size):
  base_dms = []
  for b in player_obs[1]:
    row, col = row_col_from_square_grid_pos(player_obs[1][b], grid_size)
    base_dms.append(DISTANCE_MASKS[(row, col)])
  
  if base_dms:
    base_nearest_distance_scores = np.stack(base_dms).max(0)
  else:
    base_nearest_distance_scores = np.ones((grid_size, grid_size))
    
  return base_nearest_distance_scores

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

def scale_attack_scores_bases_ships(
    config, observation, player_obs, spawn_cost, main_base_distances,
    weighted_base_mask, steps_remaining, obs_halite, halite_ships,
    laplace_smoother_rel_ship_count=4, initial_normalize_ship_diff=10,
    final_normalize_ship_diff=2):
  stacked_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])
  my_bases = stacked_bases[0]
  stacked_opponent_bases = stacked_bases[1:]
  stacked_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])
  stacked_opponent_ships = stacked_ships[1:]
  ship_halite_per_player = np.stack([rbs[3] for rbs in observation[
    'rewards_bases_ships']]).sum((1, 2))
  scores = np.array([rbs[0] for rbs in observation['rewards_bases_ships']])
  base_counts = stacked_opponent_bases.sum((1, 2))
  my_ship_count = len(player_obs[2])
  ship_counts = stacked_opponent_ships.sum((1, 2))
  all_ship_count = my_ship_count+ship_counts.sum()
  grid_size = stacked_opponent_bases.shape[1]
  
  # Factor 1: an opponent with less bases is more attractive to attack
  base_count_multiplier = np.where(base_counts == 0, 0, 1/(base_counts+1e-9))
  
  # Factor 2: an opponent that is closer in score is more attractive to attack
  abs_ship_diffs = np.abs((scores[0] + ship_halite_per_player[0]) - (
    scores[1:] + ship_halite_per_player[1:]))/spawn_cost
  normalize_diff = initial_normalize_ship_diff - observation['relative_step']*(
    initial_normalize_ship_diff-final_normalize_ship_diff)
  rel_normalized_diff = np.maximum(
    0, (normalize_diff-abs_ship_diffs)/normalize_diff)
  rel_score_max_y = initial_normalize_ship_diff/normalize_diff
  rel_score_multiplier = rel_normalized_diff*rel_score_max_y
  
  # Factor 3: an opponent with less ships is more attractive to attack
  rel_ship_count_multiplier = (my_ship_count+laplace_smoother_rel_ship_count)/(
    ship_counts+laplace_smoother_rel_ship_count)
  
  # Additional term: attack bases nearby my main base
  opponent_bases = stacked_opponent_bases.sum(0).astype(np.bool)
  if opponent_bases.sum() > 0 and main_base_distances.max() > 0:
    # TODO: tune these weights - maybe add them to the config
    additive_nearby_main_base = min(
      10, 3/(observation['relative_step']+1e-9))/(1.5**main_base_distances)/(
      weighted_base_mask[my_bases].sum())
    additive_nearby_main_base[~opponent_bases] = 0
  else:
    additive_nearby_main_base = 0
  
  attack_multipliers = base_count_multiplier*rel_score_multiplier*(
    rel_ship_count_multiplier)
  tiled_multipliers = np.tile(attack_multipliers.reshape((-1, 1, 1)),
                              [1, grid_size, grid_size])
  
  opponent_bases_scaled = (stacked_opponent_bases*tiled_multipliers).sum(0) + (
    additive_nearby_main_base)
  
  # Compute the priority of attacking the ships of opponents
  approximate_scores = np.array([my_ship_count] + (ship_counts.tolist()))*(
    np.minimum(500, steps_remaining*obs_halite.sum()**0.6/(
      all_ship_count+1e-9))) + scores + (
        halite_ships[None]*stacked_ships).sum((1, 2))
  opponent_ships_scaled = np.maximum(0, 1 - np.abs(
    approximate_scores[0]-approximate_scores[1:])/steps_remaining/10)
  # print(observation['step'], opponent_ships_scaled, approximate_scores)
  
  return opponent_bases_scaled, opponent_ships_scaled

def get_influence_map(config, stacked_bases, stacked_ships, halite_ships,
                      observation, player_obs, smooth_kernel_dim=7):
  
  all_ships = stacked_ships.sum(0).astype(np.bool)
  my_ships = stacked_ships[0].astype(np.bool)
  
  if my_ships.sum() == 0:
    return None, None, None, None
  
  num_players = stacked_ships.shape[0]
  grid_size = my_ships.shape[0]
  ship_range = 1-config['influence_map_min_ship_weight']
  all_ships_halite = halite_ships[all_ships]
  unique_vals, unique_counts = np.unique(
    all_ships_halite, return_counts=True)
  assert np.all(np.diff(unique_vals) > 0)
  unique_halite_vals = np.sort(unique_vals).astype(np.int).tolist()
  num_ships = all_ships_halite.size
  
  # TODO: double check!
  halite_ranks = [np.array(
    [unique_halite_vals.index(hs) for hs in halite_ships[
      stacked_ships[i]]]) for i in range(num_players)]
  less_rank_cum_counts = np.cumsum(unique_counts)
  num_unique = unique_counts.size
  halite_rank_counts = [np.array(
    [less_rank_cum_counts[r-1] if r > 0 else 0 for r in (
      halite_ranks[i])]) for i in range(num_players)]
  ship_weights = [1 - r/(num_ships-1+1e-9)*ship_range for r in halite_rank_counts]
  
  
  
  raw_influence_maps = np.zeros((num_players, grid_size, grid_size))
  influence_maps = np.zeros((num_players, grid_size, grid_size))
  for i in range(num_players):
    raw_influence_maps[i][stacked_ships[i]] += ship_weights[i]
    raw_influence_maps[i][stacked_bases[i]] += config[
      'influence_map_base_weight']
    
    influence_maps[i] = smooth2d(raw_influence_maps[i],
                                 smooth_kernel_dim=smooth_kernel_dim)
  
  my_influence = influence_maps[0]
  max_other_influence = influence_maps[1:].max(0)
  influence_map = my_influence - max_other_influence
  
  # Define the escape influence map
  rem_other_influence = influence_maps[1:].sum(0) - max_other_influence
  escape_influence_map = 3*my_influence-(
    2*max_other_influence+rem_other_influence)
  escape_influence_probs = np.exp(np.minimum(0, escape_influence_map)/config[
    'escape_influence_prob_divisor'])
  
  # Derive the priority scores based on the influence map
  priority_scores = 1/(1+np.abs(influence_map))
  
  # Extract a dict of my ship weights
  ship_priority_weights = {}
  for ship_k in player_obs[2]:
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    ship_halite = halite_ships[row, col]
    halite_rank = unique_halite_vals.index(ship_halite)
    ship_priority_weights[ship_k] = 1 - halite_rank/(
      num_unique-1+1e-9)*ship_range
  
  return (influence_map, priority_scores, ship_priority_weights,
          escape_influence_probs)

# Compute the weighted base mask - the base with value one represents the
# main base and the values are used as a multiplier in the return to base
# scores. Only the base with weight one is defended and is used as a basis for
# deciding what nearby opponent bases to attack.
def get_weighted_base_mask(stacked_bases, stacked_ships, observation):
  my_bases = stacked_bases[0]
  num_bases = my_bases.sum()
  grid_size = stacked_bases.shape[1]
  if num_bases == 0:
    base_mask = np.ones((grid_size, grid_size))
    main_base_distances = -1*np.ones((grid_size, grid_size))
  elif num_bases >= 1:
    ship_diff_smoothed = smooth2d(stacked_ships[0] - stacked_ships[1:].sum(0))
    base_densities = ship_diff_smoothed[my_bases]
    highest_base_density = base_densities.max()
    best_ids = np.where(base_densities == highest_base_density)[0]
    
    # Subtract some small value of the non max densities to break rare ties
    main_base_row = np.where(my_bases)[0][best_ids[0]]
    main_base_col = np.where(my_bases)[1][best_ids[0]]
    main_base_distances = DISTANCES[main_base_row, main_base_col]
    all_densities = np.minimum(ship_diff_smoothed, highest_base_density-1e-5)
    all_densities[main_base_row, main_base_col] += 1e-5
    
    # Linearly compute the weighted base mask: 1 is my best base and 0 is the
    # lowest ship_diff_smoothed value
    all_densities -= all_densities.min()
    base_mask = all_densities/all_densities.max()
    
  return base_mask, main_base_distances

# Force returning to a base when the episode is almost over
def force_return_base_end_episode(
    my_bases, base_return_grid_multiplier, main_base_distances, row, col,
    steps_remaining, opponent_less_halite_ships, weighted_base_mask):
  num_bases = my_bases.sum()
  base_positions = np.where(my_bases)
  
  # List the bases I *can* return to
  can_return_scores = np.zeros(num_bases)
  for i in range(num_bases):
    base_row = base_positions[0][i]
    base_col = base_positions[1][i]
    base_distance = DISTANCES[row, col][base_row, base_col]
    threat_mask = np.logical_and(
      DISTANCES[(row, col)] <= base_distance,
      DISTANCES[(base_row, base_col)] <= base_distance)
    if base_distance > 1:
      threat_mask[row, col] = 0
      threat_mask[base_row, base_col] = 0
    threat_ships_mask = opponent_less_halite_ships[threat_mask]
    can_return_scores[i] = (base_distance <= steps_remaining)*(10+
      weighted_base_mask[base_row, base_col] - 5*threat_ships_mask.mean())
    
  # Force an emergency return if the best return scores demand an urgent
  # return in order to bring the halite home before the episode is over
  end_game_base_return = False
  if num_bases > 0:
    best_return_id = np.argmax(can_return_scores)
    best_base_row = base_positions[0][best_return_id]
    best_base_col = base_positions[1][best_return_id]
    best_base_distance = DISTANCES[row, col][best_base_row, best_base_col]
    
    end_game_base_return = best_base_distance in [
      steps_remaining-1, steps_remaining]
    if end_game_base_return:
      base_return_grid_multiplier[best_base_row, best_base_col] += 1e15
    
  return base_return_grid_multiplier, end_game_base_return

def edge_aware_square_subset_mask(data, row, col, window, box, grid_size):
  # Figure out how many rows to roll the data and box to end up with a
  # contiguous subset
  min_row = row - window
  max_row = row + window
  if min_row < 0:
    data = np.roll(data, -min_row, axis=0)
    box = np.roll(box, -min_row, axis=0)
  elif max_row >= grid_size:
    data = np.roll(data, grid_size-max_row-1, axis=0)
    box = np.roll(box, grid_size-max_row-1, axis=0)
    
  # Figure out how many columns to roll the data and box to end up with a
  # contiguous subset
  min_col = col - window
  max_col = col + window
  if min_col < 0:
    data = np.roll(data, -min_col, axis=1)
    box = np.roll(box, -min_col, axis=1)
  elif max_col >= grid_size:
    data = np.roll(data, grid_size-max_col-1, axis=1)
    box = np.roll(box, grid_size-max_col-1, axis=1)
    
  return data[box]

def update_scores_opponent_boxing_in(
    ship_scores, stacked_ships, observation, opponent_ships_sensible_actions,
    halite_ships, steps_remaining, player_obs, np_rng, opponent_ships_scaled,
    collect_rate, obs_halite, main_base_distances, history, on_rescue_mission,
    box_in_window=3, min_attackers_to_box=4):
  # Loop over the opponent ships and derive if I can box them in
  # For now this is just greedy. We should probably consider decoupling finding
  # targets from actually boxing in.
  opponent_positions = np.where(stacked_ships[1:].sum(0) > 0)
  opponent_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  num_opponent_ships = opponent_positions[0].size
  double_window = box_in_window*2
  dist_mask_dim = 2*double_window+1
  nearby_rows = np.tile(np.expand_dims(np.arange(dist_mask_dim), 1),
                [1, dist_mask_dim])
  nearby_cols = np.tile(np.arange(dist_mask_dim), [dist_mask_dim, 1])
  ships_available = np.copy(stacked_ships[0] & (~on_rescue_mission))
  grid_size = stacked_ships.shape[1]
  ship_pos_to_key = {v[0]: k for k, v in player_obs[2].items()}
  my_ship_density = smooth2d(ships_available, smooth_kernel_dim=2)
  
  # Compute the priorities of attacking each ship
  # Compute the minimum opponent halite in the neighborhood of each square
  # by looping over all opponent ships
  attack_ship_priorities = np.zeros(num_opponent_ships)
  near_opponent_min_halite = np.ones((grid_size, grid_size))*1e6
  near_opponent_2_min_halite = np.ones((grid_size, grid_size))*1e6
  should_attack = np.zeros(num_opponent_ships, dtype=np.bool)
  for i in range(num_opponent_ships):
    row = opponent_positions[0][i]
    col = opponent_positions[1][i]
    opponent_halite = halite_ships[row, col]
    clipped_opponent_halite = min(500, opponent_halite)
    opponent_id = np.where(stacked_ships[:, row, col])[0][0]
    attack_ship_priorities[i] = clipped_opponent_halite + 1000*(
      opponent_ships_scaled[opponent_id-1]) + 1000*my_ship_density[row, col]
    near_opp_mask = ROW_COL_MAX_DISTANCE_MASKS[(row, col, box_in_window)]
    near_opp_2_mask = ROW_COL_MAX_DISTANCE_MASKS[(row, col, 2)]
    near_opponent_min_halite[near_opp_mask] = np.minimum(
        opponent_halite, near_opponent_min_halite[near_opp_mask])
    near_opponent_2_min_halite[near_opp_2_mask] = np.minimum(
        opponent_halite, near_opponent_2_min_halite[near_opp_2_mask])
    should_attack[i] = main_base_distances[row, col] >= 9-(
      observation['relative_step']*6) or (opponent_halite < history[
        'inferred_boxed_in_conv_threshold'][opponent_id][0])
  
  opponent_ship_order = np.argsort(-attack_ship_priorities)
  for i in range(num_opponent_ships):
    opponent_ship_id = opponent_ship_order[i]
    row = opponent_positions[0][opponent_ship_id]
    col = opponent_positions[1][opponent_ship_id]
    target_halite = halite_ships[row, col]
    my_less_halite_mask = np.logical_and(
      halite_ships < target_halite, ships_available)
    
    # if observation['step'] == 93 and row == 6 and col == 2:
    #   import pdb; pdb.set_trace()
    
    # Drop non zero halite ships towards the end of a game (they should return)
    my_less_halite_mask = np.logical_and(
      my_less_halite_mask, np.logical_or(
        halite_ships == 0, steps_remaining > 20))
    max_dist_mask = ROW_COL_MAX_DISTANCE_MASKS[(row, col, double_window)]
    my_less_halite_mask &= max_dist_mask
    box_pos = ROW_COL_BOX_MAX_DISTANCE_MASKS[row, col, double_window]
    
    # if observation['step'] == 183 and row == 3 and col == 6:
    #   import pdb; pdb.set_trace()
    
    if my_less_halite_mask.sum() >= min_attackers_to_box and should_attack[
        opponent_ship_id]:
      # Look up the near opponent min halite in the square which is in the middle
      # between my attackers and the target - don't attack when there is a less
      # halite ship near that ship or if there is an equal halite ship near that
      # square and close to the opponent
      my_considered_pos = np.where(my_less_halite_mask)
      if my_considered_pos[0].size:
        considered_rows = my_considered_pos[0]
        considered_cols = my_considered_pos[1]
        mid_rows = np.where(
          np.abs(considered_rows-row) <= (grid_size // 2),
          np.round((considered_rows*(1-1e-9)+row*(1+1e-9))/2),
          np.where(considered_rows*(1-1e-9)+row*(1+1e-9) >= grid_size,
                   np.round(
                     (considered_rows*(1-1e-9)+row*(1+1e-9)-grid_size)/2),
                   np.mod(np.round(
                     (considered_rows*(1-1e-9)+row*(1+1e-9)+grid_size)/2),
                     grid_size))
          ).astype(np.int)
        mid_cols = np.where(
          np.abs(considered_cols-col) <= (grid_size // 2),
          np.round((considered_cols*(1-1e-9)+col*(1+1e-9))/2),
          np.where(considered_cols*(1-1e-9)+col*(1+1e-9) >= grid_size,
                   np.round(
                     (considered_cols*(1-1e-9)+col*(1+1e-9)-grid_size)/2),
                   np.mod(np.round(
                     (considered_cols*(1-1e-9)+col*(1+1e-9)+grid_size)/2),
                     grid_size))
          ).astype(np.int)
        
        drop_ids = (near_opponent_min_halite[(mid_rows, mid_cols)] < (
          halite_ships[(considered_rows, considered_cols)])) | (
            (near_opponent_min_halite[(mid_rows, mid_cols)] == (
              halite_ships[(considered_rows, considered_cols)])) & (
                near_opponent_2_min_halite[(row, col)] <= (
                  halite_ships[(considered_rows, considered_cols)])))
                
        if np.any(drop_ids):
          drop_row_ids = considered_rows[drop_ids]
          drop_col_ids = considered_cols[drop_ids]
          my_less_halite_mask[(drop_row_ids, drop_col_ids)] = 0
      
      my_less_halite_mask_box = edge_aware_square_subset_mask(
        my_less_halite_mask, row, col, double_window, box_pos,
        grid_size)
      nearby_less_halite_mask = my_less_halite_mask_box.reshape(
          (dist_mask_dim, dist_mask_dim))
      my_num_nearby = nearby_less_halite_mask.sum()
    else:
      my_num_nearby = 0
    if my_num_nearby >= min_attackers_to_box:
      # Check all directions to make sure I can box the opponent in
      can_box_in = True
      box_in_mask_dirs = np.zeros(
        (4, dist_mask_dim, dist_mask_dim), dtype=np.bool)
      for dim_id, d in enumerate(NOT_NONE_DIRECTIONS):
        dir_and_ships = BOX_DIRECTION_MASKS[(double_window, d)] & (
          nearby_less_halite_mask)
        if not np.any(dir_and_ships):
          can_box_in = False
          break
        else:
          box_in_mask_dirs[dim_id] = dir_and_ships
          
      # if observation['step'] >= 117 and row == 3:
      #   import pdb; pdb.set_trace()
        
      if can_box_in:
        # Sketch out the escape squares for the target ship
        opponent_distances = np.abs(nearby_rows-double_window) + np.abs(
          nearby_cols-double_window)
        opponent_euclid_distances = np.sqrt(
          (nearby_rows-double_window)**2 + (
          nearby_cols-double_window)**2)
        nearby_mask_pos = np.where(nearby_less_halite_mask)
        my_nearest_distances = np.stack([np.abs(
          nearby_rows-nearby_mask_pos[0][j]) + np.abs(
          nearby_cols-nearby_mask_pos[1][j]) for j in range(
            my_num_nearby)])
        my_nearest_euclid_distances = np.stack([np.sqrt((
          nearby_rows-nearby_mask_pos[0][j])**2 + (
          nearby_cols-nearby_mask_pos[1][j])**2) for j in range(
            my_num_nearby)])
        
        # No boxing in if the opponent has a base in one of the escape squares
        escape_squares = opponent_distances <= my_nearest_distances.min(0)
        opponent_id = np.where(stacked_ships[:, row, col])[0][0]
        if not np.any(observation['rewards_bases_ships'][opponent_id][1][
            box_pos][escape_squares.flatten()]):
          # Let's box the opponent in!
          # We should move towards the opponent if we can do so without opening
          # up an escape direction
          # import pdb; pdb.set_trace()
          # print(observation['step'], row, col, my_num_nearby)
          
          # Order the planning by priority of direction and distance to the
          # opponent
          # Reasoning: mid-distance ships plan first since that allows fast
          # boxing in - the nearby ships then just have to cover the remaining
          # directions.
          # Ships which cover hard to cover directions plan later.
          box_in_mask_dirs_sum = box_in_mask_dirs.sum((1, 2))
          ship_priorities = np.zeros(my_num_nearby)
          threatened_one_step = set()
          for j in range(my_num_nearby):
            my_row = nearby_mask_pos[0][j]
            my_col = nearby_mask_pos[1][j]
            box_directions = box_in_mask_dirs[:, my_row, my_col]
            opponent_distance = np.abs(my_row-double_window) + np.abs(
              my_col-double_window)
            ship_priorities[j] = 20/(
              box_in_mask_dirs_sum[box_directions].prod())+np.abs(
                opponent_distance**0.9-box_in_window**0.9)
                
            if opponent_distance == 2 and box_directions.sum() == 2 and np.all(
                box_in_mask_dirs_sum[box_directions] == 1):
              two_step_dirs = [MOVE_DIRECTIONS[move_id+1] for move_id in (
                np.where(box_directions)[0])]
              threatened_one_step.update(two_step_dirs)
              
          # DISCERN if we are just chasing or actually attacking the ship in
          # the next move - dummy rule to have at least K neighboring ships
          # for us to attack the position of the targeted ship - this makes it
          # hard to guess the escape direction
          ship_target_1_distances = my_nearest_distances[
            :, double_window, double_window] == 1
          next_step_attack = len(
            opponent_ships_sensible_actions[row, col]) == 0 and (
              ship_target_1_distances.sum() > 2)
              
          # if observation['step'] == 204:
          #   import pdb; pdb.set_trace()
              
          opponent_boxed_bases = edge_aware_square_subset_mask(
            opponent_bases, row, col, double_window, box_pos,
            grid_size).reshape((dist_mask_dim, dist_mask_dim))
          pos_taken = np.copy(opponent_boxed_bases)
          box_override_assignment_not_next_attack = {}
          if next_step_attack:
            # If there is a ship that can take the position of my attacker:
            # attack with that ship and replace its position.
            # Otherwise pick a random attacker and keep the others in place.
            # Initial approach: don't move with ships at distance 1.
            ship_target_2_distance_ids = np.where(my_nearest_distances[
              :, double_window, double_window] == 2)[0].tolist()
            move_ids_directions_next_attack = {}
            
            # Add the positions of the one step attackers
            for one_step_diff_id in np.where(ship_target_1_distances)[0]:
              my_row = nearby_mask_pos[0][one_step_diff_id]
              my_col = nearby_mask_pos[1][one_step_diff_id]
              pos_taken[my_row, my_col] = 1
              
            for two_step_diff_id in ship_target_2_distance_ids:
              my_row = nearby_mask_pos[0][two_step_diff_id]
              my_col = nearby_mask_pos[1][two_step_diff_id]
              
              # Consider the shortest directions towards the target
              shortest_directions = get_dir_from_target(
                my_row, my_col, double_window, double_window, grid_size=1000)
              
              has_selected_action = False
              for d in shortest_directions:
                # Prefer empty 1-step to target spaces over replacing a one
                # step threat
                move_row, move_col = move_ship_row_col(
                  my_row, my_col, d, size=1000)
                if not pos_taken[move_row, move_col]:
                  move_ids_directions_next_attack[two_step_diff_id] = d
                  has_selected_action = True
                  break
              if not has_selected_action:
                # Replace a 1-step threatening ship
                for d in shortest_directions:
                  move_row, move_col = move_ship_row_col(
                    my_row, my_col, d, size=1000)
                  if pos_taken[move_row, move_col] and not pos_taken[
                    double_window, double_window] and not opponent_boxed_bases[
                      move_row, move_col]:
                    move_ids_directions_next_attack[two_step_diff_id] = d
                    # Find the ids of the 1-step ship and make sure that ship
                    # attacks
                    replaced_id = np.where(my_nearest_distances[
                      :, move_row, move_col] == 0)[0][0]
                    one_step_attack_dir = get_dir_from_target(
                      move_row, move_col, double_window, double_window,
                      grid_size=1000)[0]
                    move_ids_directions_next_attack[replaced_id] = (
                      one_step_attack_dir)
                    pos_taken[double_window, double_window] = True
              
              
            one_step_diff_ids = np.where(ship_target_1_distances)[0]
            if pos_taken[double_window, double_window]:
              # Add the remaining one step attackers with stay in place actions
              # TODO: (maybe) add some randomness so that it is harder to escape.
              for one_step_diff_id in one_step_diff_ids:
                if not one_step_diff_id in move_ids_directions_next_attack:
                  move_ids_directions_next_attack[one_step_diff_id] = None
            else:
              one_step_attacker_id = np_rng.choice(one_step_diff_ids)
              # Pick a random one step attacker to attack the target and make
              # sure the remaining 1-step ships stay in place
              for one_step_diff_id in one_step_diff_ids:
                if one_step_diff_id == one_step_attacker_id:
                  my_row = nearby_mask_pos[0][one_step_diff_id]
                  my_col = nearby_mask_pos[1][one_step_diff_id]
                  attack_dir = get_dir_from_target(
                      my_row, my_col, double_window, double_window,
                      grid_size=1000)[0]
                else:
                  attack_dir = None
                move_ids_directions_next_attack[one_step_diff_id] = attack_dir
          elif len(opponent_ships_sensible_actions[row, col]) == 0:
            # Inspect what directions I can move right next to when the
            # opponent has no valid escape actions. Use a greedy search to
            # determine the action selection order
            can_box_immediately = []
            can_box_immediately_counts = np.zeros(4)
            for j in range(my_num_nearby):
              my_row = nearby_mask_pos[0][j]
              my_col = nearby_mask_pos[1][j]
              box_directions = box_in_mask_dirs[:, my_row, my_col]
              opponent_distance = np.abs(my_row-double_window) + np.abs(
                my_col-double_window)
              if opponent_distance <= 2:
                immediate_box_dirs = np.where(box_directions)[0]
                can_box_immediately.append((
                  j, immediate_box_dirs, box_directions, my_row, my_col))
                can_box_immediately_counts[box_directions] += 1
                
            can_box_progress = [list(cb) for cb in can_box_immediately]
            can_box_immediately_counts_progress = np.copy(
              can_box_immediately_counts)
            not_boxed_dirs = np.ones(4, dtype=np.bool)
            
            # Iteratively look for directions where I can box in in one step
            # when I have others that can box in the remaining directions
            # and nobody else can box in that direction
            # direction in
            box_in_mask_rem_dirs_sum = np.copy(box_in_mask_dirs_sum)
            while len(can_box_progress) > 0 and np.any(not_boxed_dirs) and (
                can_box_immediately_counts_progress.sum() > 0):
              considered_dir = np.argmin(
                can_box_immediately_counts_progress + 100*(
                  can_box_immediately_counts_progress == 0))
              considered_dir_ids = [(
                j, cb[0], box_in_mask_rem_dirs_sum[cb[1]], cb[1], cb[3],
                cb[4]) for j, cb in enumerate(can_box_progress) if (
                    considered_dir in cb[1] and np.all(
                  box_in_mask_rem_dirs_sum[cb[1]] >= 1))]
              num_considered_dir_ids = len(considered_dir_ids)
              if num_considered_dir_ids > 0:
                # Tie breaker: the one with the most ships in the other dir
                # support
                if num_considered_dir_ids > 1:
                  scores = np.zeros(num_considered_dir_ids)
                  for k in range(num_considered_dir_ids):
                    scores[k] = 100*len(considered_dir_ids[k][2]) + (
                      considered_dir_ids[k][2].sum())
                  picked_dir_id = np.argmin(scores)
                else:
                  picked_dir_id = 0
                picked = considered_dir_ids[picked_dir_id]
                box_override_assignment_not_next_attack[picked[1]] = (
                  considered_dir, picked[4], picked[5])
                can_box_immediately_counts_progress[picked[3]] = 0
                not_boxed_dirs[considered_dir] = 0
                box_in_mask_rem_dirs_sum[picked[3]] -= 1
                ship_priorities[picked[1]] -= 1e6
                del can_box_progress[picked[0]]
              else:
                break
              
          num_covered_directions = np.zeros(4, dtype=np.int)
          num_one_step_from_covered = np.zeros(4, dtype=np.bool)
          ship_order = np.argsort(ship_priorities)
          box_in_mask_rem_dirs_sum = np.copy(box_in_mask_dirs_sum)
          update_ship_scores = []
          one_square_threats = []
          almost_covered_dirs = []
          for j in range(my_num_nearby):
            attack_id = ship_order[j]
            my_row = nearby_mask_pos[0][attack_id]
            my_col = nearby_mask_pos[1][attack_id]
            my_abs_row = (row+my_row-double_window) % grid_size
            my_abs_col = (col+my_col-double_window) % grid_size
            ship_pos = my_abs_row*grid_size+my_abs_col
            ship_k = ship_pos_to_key[ship_pos]
            box_directions = box_in_mask_dirs[:, my_row, my_col]
            opponent_distance = np.abs(my_row-double_window) + np.abs(
              my_col-double_window)
            box_in_mask_rem_dirs_sum[box_directions] -= 1
            
            # if observation['step'] == 341:
            #   import pdb; pdb.set_trace()
            
            if next_step_attack:
              # Increase the ship scores for the planned actions
              if attack_id in move_ids_directions_next_attack:
                move_dir = move_ids_directions_next_attack[attack_id]
                move_row, move_col = move_ship_row_col(
                  my_abs_row, my_abs_col, move_dir, grid_size)
                
                # if observation['step'] == 204:
                #   import pdb; pdb.set_trace()
                update_ship_scores.append(
                  (ship_k, move_row, move_col, 1e5, None, None))
            else:
              # Figure out if we should use this ship to attack the target -
              # there is no point in using too many ships!
              
              # if observation['step'] == 201 and my_row == 6 and my_col == 7:
              #   import pdb; pdb.set_trace()
              
              if (opponent_distance > 2) and (
                  (num_covered_directions[box_directions] + 0.5*(
                   box_in_mask_rem_dirs_sum[box_directions])).min() >= 2 and (
                    np.all(num_covered_directions[box_directions] > 0)) or (
                      box_in_mask_rem_dirs_sum[box_directions].min() > 2) and (
                        opponent_distance > box_in_window)):
                # print("Dropping ship", my_abs_row, my_abs_col, "from attack")
                continue
              
              rel_pos_diff = (my_row-double_window, my_col-double_window)
              num_covered_attacker = num_covered_directions[box_directions]
              # Logic to cover a direction that is almost covered
              almost_covered_override = False
              if np.all((num_covered_attacker > 0) | (
                  box_in_mask_rem_dirs_sum[box_directions] >= 1)) & np.any(
                  num_one_step_from_covered) and (
                    box_directions.sum() == 1) and len(
                      threatened_one_step) > 0 and ((
                        np.abs(my_row - my_col) == 1) or (my_row + my_col in [
                          double_window-1, double_window+1])):
                            
                move_dir = None
                if my_row-my_col == -1:
                  if WEST in threatened_one_step and my_row < double_window:
                    almost_covered_dir = WEST
                    move_dir = SOUTH
                  elif SOUTH in threatened_one_step and my_row > double_window:
                    almost_covered_dir = SOUTH
                    move_dir = WEST
                elif my_row-my_col == 1:
                  if NORTH in threatened_one_step and my_row < double_window:
                    almost_covered_dir = NORTH
                    move_dir = EAST
                  elif EAST in threatened_one_step and my_row > double_window:
                    almost_covered_dir = EAST
                    move_dir = NORTH
                elif my_row+my_col == double_window-1:
                  if EAST in threatened_one_step and my_row < double_window:
                    almost_covered_dir = EAST
                    move_dir = SOUTH
                  elif SOUTH in threatened_one_step and my_row > double_window:
                    almost_covered_dir = SOUTH
                    move_dir = EAST
                elif my_row+my_col == double_window+1:
                  if NORTH in threatened_one_step and my_row < double_window:
                    almost_covered_dir = NORTH
                    move_dir = WEST
                  elif WEST in threatened_one_step and my_row > double_window:
                    almost_covered_dir = WEST
                    move_dir = NORTH
                  
                if move_dir is not None:
                  move_row, move_col = move_ship_row_col(
                    my_row, my_col, move_dir, grid_size)
                  
                  if not pos_taken[move_row, move_col]:
                    # Override: when we are next to the target: expect opponent
                    # to move
                    almost_covered_override = True
                    if opponent_distance == 1:
                      threat_dir = OPPOSITE_MAPPING[get_dir_from_target(
                        my_row, my_col, double_window, double_window, 1000)[0]]
                      one_square_threats.append(threat_dir)
                      move_dir = None
                    else:
                      # Make sure that the square we want to move to is
                      # available
                      almost_covered_dirs.append(almost_covered_dir)
                    
              if not almost_covered_override:
                if attack_id in box_override_assignment_not_next_attack:
                  attack_move_id = box_override_assignment_not_next_attack[
                    attack_id][0]
                  assert box_directions[attack_move_id]
                else:
                  attack_dir_scores = num_covered_attacker + 0.1*(
                    box_in_mask_rem_dirs_sum[box_directions])
                  attack_dir_id = np.argmin(attack_dir_scores)
                  attack_move_id = np.where(box_directions)[0][attack_dir_id]
              
                rel_pos_diff = (my_row-double_window, my_col-double_window)
                attack_cover_dir = np.array(NOT_NONE_DIRECTIONS)[
                  attack_move_id]
                one_hot_cover_dirs = np.zeros(4, dtype=bool)
                one_hot_cover_dirs[attack_move_id] = 1
                other_dirs_covered = one_hot_cover_dirs | (
                  num_covered_directions > 0) | (box_in_mask_rem_dirs_sum >= 1)
                wait_reinforcements = not np.all(other_dirs_covered) or (
                  opponent_distance == 1)
                
                # if observation['step'] == 249:
                #   print(my_row, my_col, threatened_one_step,
                #         num_covered_directions, num_one_step_from_covered)
                #   import pdb; pdb.set_trace()
                
                if wait_reinforcements:
                  # Move away from the target if staying would mean having more
                  # halite than the target
                  my_next_halite = halite_ships[my_abs_row, my_abs_col] + int(
                      collect_rate*obs_halite[my_abs_row, my_abs_col])
                  if my_next_halite > target_halite:
                    move_away_dirs = get_dir_from_target(
                      double_window, double_window, my_row, my_col,
                      grid_size=1000)
                    # import pdb; pdb.set_trace()
                    move_dir = np_rng.choice(move_away_dirs)
                  else:
                    move_dir = None
                else:
                  if num_covered_directions[attack_move_id] > 0:
                    # Move towards the target on the diagonal (empowerment)
                    move_penalties = 0.001*opponent_euclid_distances**4 + (
                      my_nearest_euclid_distances[attack_id]**4) + 1e3*(
                        pos_taken)
                    move_penalties[my_row, my_col] += 1e3
                    best_penalty_pos = np.where(
                      move_penalties == move_penalties.min())
                    target_move_row = best_penalty_pos[0][0]
                    target_move_col = best_penalty_pos[1][0]
                    move_dir = get_dir_from_target(
                      my_row, my_col, target_move_row, target_move_col,
                      grid_size=1000)[0]
                  if attack_cover_dir == NORTH:
                    if np.abs(rel_pos_diff[1]) < (np.abs(rel_pos_diff[0])-1):
                      move_dir = SOUTH
                    elif rel_pos_diff[1] < 0:
                      move_dir = EAST
                    else:
                      move_dir = WEST
                  elif attack_cover_dir == SOUTH:
                    if np.abs(rel_pos_diff[1]) < (np.abs(rel_pos_diff[0])-1):
                      move_dir = NORTH
                    elif rel_pos_diff[1] < 0:
                      move_dir = EAST
                    else:
                      move_dir = WEST
                  elif attack_cover_dir == EAST:
                    if np.abs(rel_pos_diff[0]) < (np.abs(rel_pos_diff[1])-1):
                      move_dir = WEST
                    elif rel_pos_diff[0] < 0:
                      move_dir = SOUTH
                    else:
                      move_dir = NORTH
                  elif attack_cover_dir == WEST:
                    if np.abs(rel_pos_diff[0]) < (np.abs(rel_pos_diff[1])-1):
                      move_dir = EAST
                    elif rel_pos_diff[0] < 0:
                      move_dir = SOUTH
                    else:
                      move_dir = NORTH
                    
              # Increase the ship scores for the planned actions
              moved_rel_dir = RELATIVE_DIR_MAPPING[move_dir]
              new_rel_pos = (rel_pos_diff[0] + moved_rel_dir[0],
                             rel_pos_diff[1] + moved_rel_dir[1])
              new_grid_pos = (double_window + new_rel_pos[0],
                              double_window + new_rel_pos[1])
              if new_grid_pos[0] < 0 or new_grid_pos[1] < 0 or new_grid_pos[
                  0] > 2*double_window or new_grid_pos[1] > 2*double_window:
                new_rel_pos = (rel_pos_diff[0], rel_pos_diff[1])
                new_grid_pos = (double_window + new_rel_pos[0],
                                double_window + new_rel_pos[1])
              if pos_taken[new_grid_pos] and opponent_distance == 2:
                # Override - if I can move right next to the target: do it.
                shortest_directions = get_dir_from_target(
                  my_row, my_col, double_window, double_window, grid_size=1000)
                
                for move_dir in shortest_directions:
                  moved_rel_dir = RELATIVE_DIR_MAPPING[move_dir]
                  new_rel_pos = (rel_pos_diff[0] + moved_rel_dir[0],
                                 rel_pos_diff[1] + moved_rel_dir[1])
                  new_grid_pos = (double_window + new_rel_pos[0],
                                  double_window + new_rel_pos[1])
                  
                  if not pos_taken[new_grid_pos]:
                    break
                
              move_row, move_col = move_ship_row_col(
                my_abs_row, my_abs_col, move_dir, grid_size)
              
              if not pos_taken[new_grid_pos] and not new_rel_pos == (0, 0):
                # Update the covered attack directions
                ship_covered_directions = np.zeros(4, dtype=np.bool)
                ship_one_step_from_covered_directions = np.zeros(
                  4, dtype=np.bool)
                for threat_dir in RELATIVE_DIRECTIONS[:-1]:
                  nz_dim = int(threat_dir[0] == 0)
                  dir_offset = new_rel_pos[nz_dim]*threat_dir[nz_dim]
                  other_dir_abs_offset = np.abs(new_rel_pos[1-nz_dim])
                  
                  if dir_offset > 0 and other_dir_abs_offset <= dir_offset:
                    covered_id = np.where(
                      RELATIVE_DIR_TO_DIRECTION_MAPPING[threat_dir] == (
                        np.array(NOT_NONE_DIRECTIONS)))[0][0]
                    ship_one_step_from_covered_directions[covered_id] = 1
                    
                    if other_dir_abs_offset < dir_offset:
                      ship_covered_directions[covered_id] = 1
                      
                # Join the attack - add actions to the list
                num_covered_directions[ship_covered_directions] += 1
                num_one_step_from_covered[
                  ship_one_step_from_covered_directions] = 1
                update_ship_scores.append(
                  (ship_k, move_row, move_col, 1e5, opponent_distance,
                   np.where(ship_covered_directions)[0]))
                pos_taken[new_grid_pos] = 1
          
          # We can almost box the opponent in and rely on the opponent not
          # taking risky actions to escape
          almost_attack_nearby_blockers = False
          if len(threatened_one_step) > 0 and (
              len(one_square_threats+almost_covered_dirs) > 0) and not np.all(
                num_covered_directions > 0) and not next_step_attack:
            not_covered_dirs = [MOVE_DIRECTIONS[i+1] for i in np.where(
              num_covered_directions == 0)[0]]
            if len(one_square_threats) > 0 and np.all(
                [d in threatened_one_step for d in not_covered_dirs]):
              almost_attack_nearby_blockers = True
            else:
              almost_attack_nearby_blockers = len(
                threatened_one_step.intersection(almost_covered_dirs)) > 0
              
          # if observation['step'] == 293:
          #   import pdb; pdb.set_trace()
          if next_step_attack or np.all(num_covered_directions > 0) or (
              almost_attack_nearby_blockers):
            # Prune the attackers: only keep the closest two in each direction
            if not next_step_attack:
              drop_rows = []
              distance_dir = np.array([[u[4], u[5][0]] for u in (
                update_ship_scores) if u[5].size > 0])
              for d_id in np.arange(4):
                if (distance_dir[:, 1] == d_id).sum() > 2:
                  dir_rows = np.where(distance_dir[:, 1] == d_id)[0]
                  drop_ids = np.argsort(distance_dir[dir_rows, 0])[2:]
                  drop_rows.extend(dir_rows[drop_ids].tolist())
                  
              for dr in np.sort(drop_rows)[::-1]:
                del update_ship_scores[dr]
            
            for ship_k, move_row, move_col, new_collect_score, _, _ in (
                update_ship_scores):
              ship_scores[ship_k][0][move_row, move_col] = new_collect_score
            
            # Flag the boxing in ships as unavailable for other hunts
            ships_available[box_pos & my_less_halite_mask] = 0
  
  return ship_scores

def get_no_zero_halite_neighbors(halite):
  no_zero_halite_neighbors = np.ones_like(halite, dtype=np.bool)
  
  for d in NOT_NONE_DIRECTIONS:
    if d == NORTH:
      shifted = np.concatenate([halite[None, -1], halite[:-1]])
    elif d == SOUTH:
      shifted = np.concatenate([halite[1:], halite[None, 0]])
    elif d == EAST:
      shifted = np.concatenate([halite[:, 1:], halite[:, 0, None]], 1)
    elif d == WEST:
      shifted = np.concatenate([halite[:, -1, None], halite[:, :-1]], 1)
    no_zero_halite_neighbors &= (shifted > 0)
  
  return no_zero_halite_neighbors


def get_ship_scores(config, observation, player_obs, env_config, np_rng,
                    ignore_bad_attack_directions, history,
                    env_obs_ids, env_observation, verbose):
  ship_scores_start_time = time.time()
  convert_cost = env_config.convertCost
  spawn_cost = env_config.spawnCost
  stacked_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships']])
  my_bases = stacked_bases[0]
  obs_halite = np.maximum(0, observation['halite'])
  # Clip obs_halite to zero when gathering it doesn't add to the score
  # code: delta_halite = int(cell.halite * configuration.collect_rate)
  collect_rate = env_config.collectRate
  obs_halite[obs_halite < 1/collect_rate] = 0
  my_ship_count = len(player_obs[2])
  first_base = my_ship_count == 1 and my_bases.sum() == 0
  max_ships = config['max_ships']
  early_game_return_boost_step = config['early_game_return_boost_step']
  step = observation['step']
  early_game_not_max_ships = (my_ship_count < max_ships) and (
    step < early_game_return_boost_step)
  early_game_return_boost = (early_game_return_boost_step-step)/(
    early_game_return_boost_step)*config[
      'early_game_return_base_additional_multiplier']*early_game_not_max_ships
  steps_remaining = env_config.episodeSteps-1-observation['step']
      
  # Override the maximum number of conversions on the last episode turn
  last_episode_turn = observation['relative_step'] == 1

  grid_size = obs_halite.shape[0]
  half_dim_grid_mask = np.ones((grid_size, grid_size))*half_distance_mask_dim
  # smoothed_friendly_ship_halite = smooth2d(
  #   observation['rewards_bases_ships'][0][3])
  smoothed_halite = smooth2d(obs_halite)
  can_deposit_halite = my_bases.sum() > 0
  stacked_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']])
  my_ships = stacked_ships[0]
  opponent_ships = stacked_ships[1:].sum(0) > 0
  all_ship_count = opponent_ships.sum() + my_ship_count
  my_ship_fraction = my_ship_count/(1e-9+all_ship_count)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  opponent_bases = stacked_bases[1:].sum(0)
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
    
  # Get the distance to the nearest base for all squares
  all_bases = stacked_bases.sum(0) > 0
  base_locations = np.where(all_bases)
  num_bases = all_bases.sum()
  all_base_distances = [DISTANCES[
    base_locations[0][i], base_locations[1][i]] for i in range(num_bases)] + [
        99*np.ones((grid_size, grid_size))]
  nearest_base_distances = np.stack(all_base_distances).min(0)
  
  # Flag to indicate I should not occupy/flood my base with early ships
  my_halite = observation['rewards_bases_ships'][0][0]
  avoid_base_early_game = my_halite >= spawn_cost and (
    observation['step'] < 20) and my_bases.sum() == 1 and (
      my_halite % 500) == 0 and my_ship_count < 9

  # Distance to nearest base mask - gathering closer to my base is better
  base_nearest_distance_scores = get_nearest_base_distances(
    player_obs, grid_size)

  # Get opponent ship actions that avoid collisions with less halite ships
  opponent_ships_sensible_actions = get_valid_opponent_ship_actions(
    observation['rewards_bases_ships'], halite_ships, grid_size)
  
  # Get the weighted base mask
  weighted_base_mask, main_base_distances = get_weighted_base_mask(
    stacked_bases, stacked_ships, observation)
  
  # Scale the opponent bases as a function of attack desirability
  opponent_bases_scaled, opponent_ships_scaled = (
    scale_attack_scores_bases_ships(
      config, observation, player_obs, spawn_cost, main_base_distances,
      weighted_base_mask, steps_remaining, obs_halite, halite_ships))

  # Get the influence map
  (influence_map, priority_scores, ship_priority_weights,
   escape_influence_probs) = get_influence_map(
    config, stacked_bases, stacked_ships, halite_ships, observation,
    player_obs)
     
  # Get the squares that have no zero halite neighbors - this makes it hard
  # to successfully camp out next to the base
  no_zero_halite_neighbors = get_no_zero_halite_neighbors(
    observation['halite'])
  
  ship_scores = {}
  for i, ship_k in enumerate(player_obs[2]):
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    dm = DISTANCE_MASKS[(row, col)]
    ship_halite = player_obs[2][ship_k][1]
    
    opponent_less_halite_ships = np.logical_and(
      opponent_ships, halite_ships <= ship_halite)
    opponent_smoother_less_halite_ships = smooth2d(
      opponent_less_halite_ships, smooth_kernel_dim=5)
    
    # Scores 1: collecting halite at row, col
    # Multiply the smoothed halite, added with the obs_halite with a distance
    # mask, specific for the current row and column
    collect_grid_scores = dm*(
      smoothed_halite * config['collect_smoothed_multiplier'] + 
      obs_halite * config['collect_actual_multiplier']) * (
        config['collect_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) * (
            base_nearest_distance_scores ** config[
              'collect_base_nearest_distance_exponent'])*((
                1+config['influence_weights_additional_multiplier']*(
                  ship_priority_weights[ship_k])**config[
                    'influence_weights_exponent']) ** priority_scores)
                    
    # if observation['step'] >= 14 and row == 2 and col in [9]:
    #   import pdb; pdb.set_trace()
    
    # Override the collect score to 0 to avoid blocking the base early on in
    # the game: All squares right next to the initial base are set to 0
    if avoid_base_early_game:
      collect_grid_scores, early_next_base_dir, drop_None_valid = (
        set_scores_single_nearby_zero(
          collect_grid_scores, my_bases, grid_size, row, col))
    else:
      early_next_base_dir = None
      drop_None_valid = False
      
    # Scores 2: returning to any of my bases
    base_return_grid_multiplier = dm*ship_halite*(
      config['return_base_multiplier'] * (
        config['return_base_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) + early_game_return_boost)*(
            weighted_base_mask)
    chase_details = history['chase_counter'][0].get(ship_k, None)
    if chase_details is not None:
      # Keep the relative order using the minimum in case the return to base
      # pull is big
      base_return_grid_multiplier = np.minimum(
        base_return_grid_multiplier+5e4, base_return_grid_multiplier*(config[
        'chase_return_base_exponential_bonus']**chase_details[1]))
            
    # Force returning to a base when the episode is almost over and I
    # have halite on board
    if ship_halite > 0 and steps_remaining <= grid_size:
      base_return_grid_multiplier, end_game_base_return = (
        force_return_base_end_episode(
          my_bases, base_return_grid_multiplier, main_base_distances, row, col,
          steps_remaining, opponent_less_halite_ships, weighted_base_mask))
    else:
      end_game_base_return = False
    
    # Override the return base score to 0 to avoid blocking the base early on
    # in the game.
    if avoid_base_early_game:
      base_return_grid_multiplier = override_early_return_base_scores(
        base_return_grid_multiplier, my_bases, row, col, grid_size,
        my_ship_count)
    
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
                  config['establish_base_deposit_multiplier']) + first_base*(
                    config['first_base_no_4_way_camping_spot_bonus']*(
                      no_zero_halite_neighbors))
                  
    # Scores 4: attack an opponent base at row, col
    attack_base_scores = config['attack_base_multiplier']*dm*(
      opponent_bases_scaled)*(
        config['attack_base_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) - (config[
            'attack_base_halite_sum_multiplier'] * obs_halite.sum()**0.8 / (
              all_ship_count))*int(my_ship_fraction < 0.5) - 1e12*(
                ship_halite > 0)
                
    # Update the scores as a function of nearby enemy ships to avoid collisions
    # with opposing ships that carry less halite and promote collisions with
    # enemy ships that carry less halite
    (collect_grid_scores, base_return_grid_multiplier, establish_base_scores,
     attack_base_scores, preferred_directions, valid_directions,
     agent_surrounded, two_step_bad_directions, n_step_step_bad_directions,
     one_step_valid_directions,
     n_step_bad_directions_die_probs) = update_scores_enemy_ships(
       config, collect_grid_scores, base_return_grid_multiplier,
       establish_base_scores, attack_base_scores, opponent_ships,
       opponent_bases, halite_ships, row, col, grid_size, spawn_cost,
       drop_None_valid, obs_halite, collect_rate, np_rng,
       opponent_ships_sensible_actions, ignore_bad_attack_directions,
       observation, ship_k, my_bases, my_ships, steps_remaining, history,
       escape_influence_probs, player_ids, env_obs_ids, env_observation,
       main_base_distances, nearest_base_distances)
       
    # Update the scores as a function of blocking enemy bases and my early
    # game initial base
    (collect_grid_scores, base_return_grid_multiplier, establish_base_scores,
     attack_base_scores, valid_directions, one_step_valid_directions,
     opponent_base_directions) = update_scores_blockers(
       collect_grid_scores, base_return_grid_multiplier, establish_base_scores,
       attack_base_scores, row, col, grid_size, opponent_bases,
       half_dim_grid_mask, valid_directions, one_step_valid_directions,
       early_next_base_dir, update_attack_base=False)
       
    if last_episode_turn:
      # Convert all ships with more halite than the convert cost on the last
      # episode step
      # TODO: don't do this if I can safely move to a best next to my square.
      last_episode_step_convert = ship_halite >= convert_cost
      establish_base_scores[row, col] = 1e12*int(last_episode_step_convert)
    else:
      last_episode_step_convert = False
      
    ship_scores[ship_k] = (
      collect_grid_scores, base_return_grid_multiplier, establish_base_scores,
      attack_base_scores, preferred_directions, agent_surrounded,
      valid_directions, two_step_bad_directions, n_step_step_bad_directions,
      one_step_valid_directions, opponent_base_directions, 0,
      end_game_base_return, last_episode_step_convert,
      n_step_bad_directions_die_probs, opponent_smoother_less_halite_ships)
    
  ship_scores_duration = time.time() - ship_scores_start_time
  return (ship_scores, opponent_ships_sensible_actions, weighted_base_mask,
          opponent_ships_scaled, main_base_distances, ship_scores_duration)

def get_mask_between_exclude_ends(r1, c1, r2, c2, grid_size):
  rel_pos = get_relative_position(r1, c1, r2, c2, grid_size)
  start_row = r2 if rel_pos[0] < 0 else r1
  rows = np.mod(
    np.arange(start_row, start_row+np.abs(rel_pos[0])+1), grid_size)
  start_col = c2 if rel_pos[1] < 0 else c1
  cols = np.mod(
    np.arange(start_col, start_col+np.abs(rel_pos[1])+1), grid_size)
  mask = np.zeros((grid_size, grid_size), dtype=np.bool)
  mask[rows[:, None], cols] = 1
  mask[r1, c1] = 0
  mask[r2, c2] = 0
  
  return mask

def consider_restoring_base(
    observation, env_config, all_ship_scores, player_obs, convert_cost, np_rng,
    max_considered_attackers=3, halite_on_board_mult=1e-6):
  obs_halite = np.maximum(0, observation['halite'])
  grid_size = obs_halite.shape[0]
  collect_rate = env_config.collectRate
  obs_halite[obs_halite < 1/collect_rate] = 0
  my_ships = observation['rewards_bases_ships'][0][2]
  my_ship_count = my_ships.sum()
  stacked_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']])
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  opponent_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  opponent_ship_count = opponent_ships.sum()
  all_ship_count = opponent_ship_count + my_ship_count
  my_ship_fraction = my_ship_count/(1e-9+all_ship_count)
  remaining_halite = obs_halite.sum()
  steps_remaining = env_config.episodeSteps-1-observation['step']
  ship_cargo = (np.minimum(convert_cost, halite_ships)*my_ships).sum()
  expected_payoff_conversion = ship_cargo*0.5 + (max(
    0, steps_remaining-20)**0.6)*(remaining_halite**0.9)*my_ship_fraction
  
  can_deposit_halite = expected_payoff_conversion > convert_cost
  restored_base_pos = None
  can_defend_converted = False
  if can_deposit_halite:
    # Decide what ship to convert - it should be relatively central, have high
    # ship halite on board and be far away from opponent ships and bases
    my_ship_density = smooth2d(my_ships, smooth_kernel_dim=10)
    opponent_base_density = smooth2d(opponent_bases, smooth_kernel_dim=5)
    opponent_ship_density = smooth2d(opponent_ships, smooth_kernel_dim=5)
    
    convert_priority_scores = np.zeros(my_ship_count)
    for i, ship_k in enumerate(player_obs[2]):
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      ship_halite = min(convert_cost, player_obs[2][ship_k][1])
      
      # Compute if the base can be defended after conversion
      my_ship_distances = np.sort(DISTANCES[row, col][my_ships]+(
        halite_on_board_mult*halite_ships[my_ships]))[1:]
      opponent_ship_distances = np.sort(DISTANCES[row, col][opponent_ships])+(
        halite_on_board_mult*halite_ships[opponent_ships])
      num_considered_distances = min([
        max_considered_attackers, my_ship_count-1, opponent_ship_count])
      can_defend = np.all(my_ship_distances[:num_considered_distances] <= (
        opponent_ship_distances[:num_considered_distances]))
      can_afford = (halite_ships[row, col] + player_obs[0]) >= convert_cost
      
      convert_priority_scores[i] = ship_halite + 100*(
        my_ship_density-opponent_ship_density)[row, col] - 200*(
          opponent_base_density[row, col]) - 1e12*int(
            not can_defend or not can_afford)
          
    can_defend_converted = convert_priority_scores.max() > -1e11
    
  if can_defend_converted:
    convert_k = list(player_obs[2].keys())[np.argmax(convert_priority_scores)]
    convert_row, convert_col = row_col_from_square_grid_pos(
        player_obs[2][convert_k][0], grid_size)
    restored_base_pos = (convert_row, convert_col)
    all_ship_scores[convert_k][2][convert_row, convert_col] = 1e12
    
  else:
    # Don't gather
    # Add some small positive noise to the establish base score, away from
    # the current square - this ensures ships keep moving around when I don't
    # plan on restoring my last destroyed base
    # Move ships closer to each other if we want to convert a base but they are
    # not able to defend it - send all to the least dense opponent point
    opponent_density = smooth2d(opponent_ships+opponent_bases,
                                smooth_kernel_dim=5)
    lowest_densities = np.where(opponent_density == opponent_density.min())
    halite_density = smooth2d(obs_halite)
    target_id = np.argmax(halite_density[lowest_densities])
    gather_row = lowest_densities[0][target_id]
    gather_col = lowest_densities[1][target_id]
    for ship_k in player_obs[2]:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      all_ship_scores[ship_k][0][:] *= 0
      if can_deposit_halite:
        # Gather with some low probability since we may not have enough halite
        # to convert a ship
        if obs_halite[row, col] > 0 and np_rng.uniform() < 0.2:
          all_ship_scores[ship_k][0][row, col] = 2000
        ensure_move_mask = 1000*DISTANCE_MASKS[(gather_row, gather_col)]
      else:
        ensure_move_mask = np_rng.uniform(0, 1e-9, (grid_size, grid_size))
        ensure_move_mask[row, col] = 0
      all_ship_scores[ship_k][2][:] += ensure_move_mask
      
    can_deposit_halite = False
    
  return all_ship_scores, can_deposit_halite, restored_base_pos

def protect_main_base(observation, env_config, all_ship_scores, player_obs,
                      defend_override_base_pos, max_considered_attackers=3,
                      halite_on_board_mult=1e-6):
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  defend_base_ignore_collision_key = None
  main_base_protected = True
  ignore_base_collision_ship_keys = []
  if opponent_ships.sum():
    opponent_ship_count = opponent_ships.sum()
    grid_size = opponent_ships.shape[0]
    obs_halite = np.maximum(0, observation['halite'])
    collect_rate = env_config.collectRate
    obs_halite[obs_halite < 1/collect_rate] = 0
    my_ship_count = len(player_obs[2])
    if defend_override_base_pos is None:
      base_row, base_col = row_col_from_square_grid_pos(
        list(player_obs[1].values())[0], grid_size)
    else:
      base_row, base_col = defend_override_base_pos
    stacked_ships = np.stack(
      [rbs[2] for rbs in observation['rewards_bases_ships']])
    halite_ships = np.stack([
      rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
    halite_ships[stacked_ships.sum(0) == 0] = -1e-9
    opponent_ship_distances = DISTANCES[(base_row, base_col)][opponent_ships]+(
      halite_on_board_mult*halite_ships[opponent_ships])
    sorted_opp_distance = np.sort(opponent_ship_distances)
    ship_keys = list(player_obs[2].keys())
    
    ship_base_distances = np.zeros((my_ship_count, 6))
    # Go over all my ships and approximately compute how far they are expected
    # to be from the base !with no halite on board! by the end of the next turn
    # Approximate since returning ships are expected to always move towards the
    # base and other ships are assumed to be moving away.
    for i, ship_k in enumerate(player_obs[2]):
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      ship_scores = all_ship_scores[ship_k]
      ship_halite = player_obs[2][ship_k][1]
      current_distance = DISTANCES[(base_row, base_col)][row, col]
      is_returning = ship_scores[1][base_row, base_col] > max([
        ship_scores[0].max(), ship_scores[2].max(), ship_scores[3].max()])
      to_base_directions = get_dir_from_target(
        row, col, base_row, base_col, grid_size)
      can_not_move_to_base = int((ship_halite != 0) and len(
        set(to_base_directions) & set(ship_scores[9])) == 0)
      ship_base_distances[i, 0] = current_distance
      ship_base_distances[i, 1] = current_distance + 1 - int(2*is_returning)
      ship_base_distances[i, 2] = ship_halite
      ship_base_distances[i, 3] = row
      ship_base_distances[i, 4] = col
      ship_base_distances[i, 5] = can_not_move_to_base
    
    weighted_distances = ship_base_distances[:, 1] + halite_on_board_mult*(
      ship_base_distances[:, 2])
    defend_distances = ship_base_distances[:, 0] + halite_on_board_mult*(
      ship_base_distances[:, 2]) + 100*ship_base_distances[:, 5]
    
    # Update the defend distances so we allow a ship with halite to move onto
    # the base when it is one step away, and the closest opponent is at least
    # two steps away, or is one step away with strictly more halite on board.
    if sorted_opp_distance[0] > defend_distances.min():
      # I have at least one ship that can be used to defend the base
      good_defense_ids = np.where(np.logical_and(
        np.floor(defend_distances) <= max(1, np.floor(defend_distances.min())),
        defend_distances < sorted_opp_distance[0]))[0]
      
      # Pick the maximum distance (max halite) of the min step ids that can
      # defend the base
      best_good_defense_id = np.argmax(defend_distances[good_defense_ids])
      if defend_distances[good_defense_ids[best_good_defense_id]] > 0:
        defend_distances[good_defense_ids[best_good_defense_id]] = -0.1
    
    next_ship_distances_ids = np.argsort(weighted_distances)
    next_ship_distances_sorted = weighted_distances[next_ship_distances_ids]
    worst_case_opponent_distances = sorted_opp_distance-1
    num_considered_distances = min([
      max_considered_attackers, my_ship_count, opponent_ship_count])
    opponent_can_attack_sorted = next_ship_distances_sorted[
      :num_considered_distances] > worst_case_opponent_distances[
        :num_considered_distances]
        
    main_base_protected = worst_case_opponent_distances[0] > 0
    
    # if observation['step'] == 114:
    #   import pdb; pdb.set_trace()
    
    if np.any(opponent_can_attack_sorted):
      # Update the defend distances to make sure that two zero halite ships
      # switch position when one is at the base and the other is at a distance
      # of one - that way the second ship has no halite and can defend the base
      # on the next step
      if num_considered_distances > 1 and opponent_can_attack_sorted[1]:
        num_defenders = defend_distances.size
        argsort_defend = np.argsort(defend_distances)
        sorted_manh_distances = ship_base_distances[argsort_defend, 0]
        if (sorted_manh_distances[0] == 0 and sorted_manh_distances[1] == 1):
          # Both should not have halite on board
          if defend_distances[argsort_defend[1]] == 1 or int(
              next_ship_distances_sorted[1] == 2):
            defend_distances[argsort_defend[1]] -= 1.2
        elif worst_case_opponent_distances[0] == 0 and (
            worst_case_opponent_distances[1] in [0, 1]) and (
              defend_distances.min() <= 0) and 2 in defend_distances and (
                np.logical_and(defend_distances > 1,
                               defend_distances < 2).sum() > 0):
          # TODO: remove, this looks like a special case of the next elif
          print("PROBABLY REDUNDANT - LOOK INTO ME")
          defend_distances[np.where(defend_distances==2)[0][0]] = 1
        elif num_defenders > 2 and (defend_distances[argsort_defend[0]]-1) <= (
            worst_case_opponent_distances[0]) and (
              defend_distances[argsort_defend[2]]-1) <= (
                worst_case_opponent_distances[1]) and ship_base_distances[
                  argsort_defend[1], 2] > 0:    
          # Switch the second and third defenders when the second defender has
          # halite on board and the third doesn't but can still defend the base
          defend_score_diff = defend_distances[argsort_defend[2]] - (
            defend_distances[argsort_defend[1]]) + halite_on_board_mult
          defend_distances[argsort_defend[1]] += defend_score_diff
      
      # Summon the closest K agents towards or onto the base to protect it.
      # When the ship halite is zero, we should aggressively attack base
      # raiders
      num_attackers = 1+np.where(opponent_can_attack_sorted)[0][-1]
      defend_distances_ids = np.argsort(defend_distances)
      for i in range(num_attackers):
        defend_id = defend_distances_ids[i]
        if opponent_can_attack_sorted[i] or defend_distances[defend_id] < 0:
          # Very simple defense strategy for now: prefer returning to the
          # base by increasing the gather score for all squares beween the
          # current position and the only base. If my ship is currently on the
          # base: keep it there
          ship_id = defend_distances_ids[i]
          distance_to_base = ship_base_distances[ship_id, 0]
          ship_k = ship_keys[ship_id]
          ship_scores = list(all_ship_scores[ship_k])
          ship_halite = int(ship_base_distances[ship_id, 2])
          row = int(ship_base_distances[ship_id, 3])
          col = int(ship_base_distances[ship_id, 4])
          if distance_to_base <= 1:
            # Stay or move to the base; or stay 1 step away
            # TODO: attack the attackers when the base is safe
            if i == 0:
              ship_scores[1][base_row, base_col] += 1e6*(
                1+max_considered_attackers-i)
            elif obs_halite[row, col] == 0 or (
                worst_case_opponent_distances[i] > distance_to_base):
              ship_scores[0][row, col] += 2e6
              if None in ship_scores[9]:
                # Stay close to the base when defending
                ship_scores[6].append(None)
            if halite_ships[row, col] == 0 or (i == 0 and (
                worst_case_opponent_distances[0] > halite_on_board_mult*(
                  ship_base_distances[defend_id, 2]))):
              ship_scores[11] = max_considered_attackers-i
              if defend_base_ignore_collision_key is None:
                defend_base_ignore_collision_key = ship_k
          else:
            # Set the base as the target and override the base synchronization
            ship_scores[1][base_row, base_col] += 1e6*(
                1+max_considered_attackers-i)
            ignore_base_collision_ship_keys.append(ship_k)
            
            # Defend the base without fear if I have no halite on board
            # Only consider staying at the current position or moving towards
            # the base in order to increase the action execution priority
            if ship_halite == 0:
              ship_scores[6] = copy.copy(MOVE_DIRECTIONS)
              ship_scores[7] = []
              ship_scores[8] = []
              ship_scores[9] = copy.copy(MOVE_DIRECTIONS)
              ship_scores[11] = max_considered_attackers-i
            else:
              # Still move towards the base to defend it when there is halite
              # on board as long as it does not mean selecting a 1-step bad
              # action
              base_defend_dirs = get_dir_from_target(
                row, col, base_row, base_col, grid_size)
              not_bad_defend_dirs = list(set(ship_scores[9]) & set(
                base_defend_dirs))
              ship_scores[6] = list(set(ship_scores[6] + not_bad_defend_dirs))
            
            # synchronization logic to avoid arriving at the same time
            # mask_between_current_and_base = get_mask_between_exclude_ends(
            #   row, col, base_row, base_col, grid_size)
            # mask_between_current_and_base = np.logical_and(
            #   DISTANCES[(row, col)] < distance_to_base,
            #   DISTANCES[(base_row, base_col)] < distance_to_base)
            # ship_scores[0][mask_between_current_and_base] += 1e6
            
          all_ship_scores[ship_k] = tuple(ship_scores)
  
  return (all_ship_scores, defend_base_ignore_collision_key,
          main_base_protected, ignore_base_collision_ship_keys)

def update_occupied_count(row, col, occupied_target_squares,
                          occupied_squares_count):
  k = (row, col)
  occupied_target_squares.append(k)
  if k in occupied_squares_count:
    occupied_squares_count[k] += 1
  else:
    occupied_squares_count[k] = 1
    
def update_scores_rescue_missions(
    config, all_ship_scores, stacked_ships, observation, halite_ships,
    steps_remaining, player_obs, obs_halite, history,
    opponent_ships_sensible_actions, weighted_base_mask, my_bases, np_rng,
    max_box_distance=5):
  grid_size = stacked_ships.shape[1]
  opponent_ships = stacked_ships[1:].sum(0) > 0
  my_zero_halite_ships = stacked_ships[0] & (halite_ships == 0)
  opponent_zero_halite_ships = opponent_ships & (halite_ships == 0)
  opponent_zero_halite_ship_density = smooth2d(
    opponent_zero_halite_ships, smooth_kernel_dim=4)
  zero_halite_pos = np.where(my_zero_halite_ships)
  on_rescue_mission = np.zeros((grid_size, grid_size), dtype=np.bool)
  pos_to_k = {v[0]: k for k, v in player_obs[2].items()}
  
  # TODO: consider rescuing before being chased - this is only sensible if
  # rescue missions remain rare
  chased_ships = list(history['chase_counter'][0].keys())
  for ship_k in chased_ships:
    # if observation['step'] == 38 and ship_k == '22-1':
    #   import pdb; pdb.set_trace()
    ship_scores = all_ship_scores[ship_k]
    valid_directions = ship_scores[6]
    if len(set(valid_directions) - set(ship_scores[7]+ship_scores[8])) == 0 and (
        history['chase_counter'][0][ship_k][1] > 3):
      # Only call for help when the considered ship is nearly boxed in in all
      # directions and has been chased for a while
      valid_directions = valid_directions if len(ship_scores[8]) == 0 else (
          ship_scores[8])
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      threat_opponents = opponent_ships & (halite_ships < halite_ships[
        row, col])
      nearly_boxed_in = True
      for d in NOT_NONE_DIRECTIONS:
        opposite_d = OPPOSITE_MAPPING[d]
        rel_move = RELATIVE_DIR_MAPPING[d]
        ref_square = ((row + max_box_distance*rel_move[0]) % grid_size,
                      (col + max_box_distance*rel_move[1]) % grid_size)
        dir_mask = HALF_PLANES_CATCH[ref_square][opposite_d] & (
          ROW_COL_MAX_DISTANCE_MASKS[
            ref_square[0], ref_square[1], max_box_distance])
        num_threats = (dir_mask & threat_opponents).sum()
        if num_threats == 0:
          nearly_boxed_in = False
          break
      
      if nearly_boxed_in and my_zero_halite_ships.sum():
        friendly_zero_halite_distances = DISTANCES[row, col][zero_halite_pos]
        min_halite_distance = friendly_zero_halite_distances.min()
        
        # Follow the nearest zero halite ship home if it is at a distance one
        # of the ship
        is_protected = False
        if min_halite_distance == 1:
          # Jointly move to the nearest weighted base and assume that the
          # former zero halite position won't be attacked
          dm = DISTANCE_MASKS[(row, col)]
          base_scores = dm*weighted_base_mask*my_bases
          target_base = np.where(base_scores == base_scores.max())
          
          nearest_halite_id = np.argmin(friendly_zero_halite_distances)
          rescuer_row = zero_halite_pos[0][nearest_halite_id]
          rescuer_col = zero_halite_pos[1][nearest_halite_id]
          all_ship_scores[ship_k][0][rescuer_row, rescuer_col] = 1e8
          to_rescuer_dir = get_dir_from_target(
            row, col, rescuer_row, rescuer_col, grid_size)[0]
          if to_rescuer_dir not in all_ship_scores[ship_k][6]:
            all_ship_scores[ship_k][6].append(to_rescuer_dir)
          rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
          increase_mask = get_mask_between_exclude_ends(
            target_base[0], target_base[1], rescuer_row, rescuer_col,
            grid_size)
          for score_id in range(3):
            all_ship_scores[rescuer_k][score_id][increase_mask] += 1e4
          
          history['escort_to_base_list'].append(
            (ship_k, rescuer_k, True, 3, 15))
          
          # Add the ship pair to the escort-list for a fixed number of steps
          is_protected = True
          
        elif min_halite_distance == 2:
          considered_zero_2_ship_ids = np.argsort(
            friendly_zero_halite_distances)[:3]
          for zero_halite_ship_id in considered_zero_2_ship_ids:
            rescuer_distance = friendly_zero_halite_distances[
              zero_halite_ship_id]
            if rescuer_distance == 2:
              rescuer_row = zero_halite_pos[0][zero_halite_ship_id]
              rescuer_col = zero_halite_pos[1][zero_halite_ship_id]
              rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
              
              # Figure out if the chased ship can move to the rescuer
              valid_move_dirs = []
              for d in valid_directions:
                if d is not None:
                  if HALF_PLANES_RUN[row, col][d][rescuer_row, rescuer_col]:
                    valid_move_dirs.append(d)
              
              # Plan A: Let the non zero halite ship wait (stupid, this never
              # happens)
              if None in valid_directions:
                rescue_dirs = get_dir_from_target(
                  rescuer_row, rescuer_col, row, col, grid_size)
                safe_rescue_dirs = set(rescue_dirs) & set(
                  all_ship_scores[rescuer_k][6])
                if len(safe_rescue_dirs) > 0:
                  rescue_dirs = list(safe_rescue_dirs)
                rescue_dir = np_rng.choice(rescue_dirs)
                rescuer_move_row, rescuer_move_col = move_ship_row_col(
                  rescuer_row, rescuer_col, rescue_dir, grid_size)
                move_row = row
                move_col = col
                is_protected = True
              
              # Plan B: Let the zero halite ship wait if there is no halite at
              # the considered square, and the chased ship can move to the
              # rescuer.
              if not is_protected and obs_halite[
                  rescuer_row, rescuer_col] == 0:
                to_rescuer_dirs = get_dir_from_target(
                  row, col, rescuer_row, rescuer_col, grid_size)
                valid_to_rescuer_dirs = list(set(to_rescuer_dirs) & set(
                  valid_directions))
                if len(valid_to_rescuer_dirs) > 0:
                  to_rescuer_dir = np_rng.choice(valid_to_rescuer_dirs)
                  move_row, move_col = move_ship_row_col(
                    row, col, to_rescuer_dir, grid_size)
                  rescuer_move_row = rescuer_row
                  rescuer_move_col = rescuer_col
                  is_protected = True
                
              # Plan C: move towards the rescuer move square and have the
              # rescuer move to another neighboring zero halite square
              if not is_protected:
                safe_zero_halite_squares_dirs = []
                for d in NOT_NONE_DIRECTIONS:
                  if d in all_ship_scores[rescuer_k][6]:
                    rescue_r, rescue_c = move_ship_row_col(
                      rescuer_row, rescuer_col, d, grid_size)
                    if obs_halite[rescue_r, rescue_c] == 0:
                      safe_zero_halite_squares_dirs.append((
                        d, rescue_r, rescue_c))
                
                for rescue_d, rescue_r, rescue_c in (
                    safe_zero_halite_squares_dirs):
                  # Figure out if the chased ship can move to the new rescuer
                  # square
                  valid_to_rescuer_moved_dirs = []
                  for d in valid_directions:
                    if d is not None:
                      if HALF_PLANES_RUN[row, col][d][rescue_r, rescue_c]:
                        valid_to_rescuer_moved_dirs.append(d)
                  
                  if valid_to_rescuer_moved_dirs:
                    to_rescuer_dir = np_rng.choice(valid_to_rescuer_moved_dirs)
                    move_row, move_col = move_ship_row_col(
                      row, col, to_rescuer_dir, grid_size)
                    rescuer_move_row = rescue_r
                    rescuer_move_col = rescue_c
                    is_protected = True
                    break
                  
              # # Plan D: Consider other rescuers up to distance 3 if I
              # # can't wait at the current/nearby square
              # import pdb; pdb.set_trace()
              # x=1
              
            if is_protected:
              all_ship_scores[ship_k][0][move_row, move_col] = 1e8
              all_ship_scores[rescuer_k][0][
                rescuer_move_row, rescuer_move_col] = 1e8
              break
        
        if not is_protected and len(valid_directions) > 0:
          # Only consider zero halite ships in the directions I can move to
          valid_rescue_mask = np.zeros_like(my_zero_halite_ships)
          for d in valid_directions:
            valid_rescue_mask[HALF_PLANES_RUN[row, col][d]] = 1
            
          valid_zero_halite_ships = np.copy(my_zero_halite_ships)*(
            valid_rescue_mask)
          
          valid_zero_halite_pos = np.where(valid_zero_halite_ships)
          valid_friendly_zero_halite_distances = DISTANCES[row, col][
            valid_zero_halite_pos]
          if valid_zero_halite_pos[0].size:
            min_valid_distance = valid_friendly_zero_halite_distances.min()
          else:
            min_valid_distance = grid_size
          
          if min_valid_distance <= 6:
            # Consider rescuing the ship if there is a nearby zero halite ship
            # that can move to me and is in a valid direction of the move pos
            considered_zero_ship_ids = np.argsort(
              valid_friendly_zero_halite_distances)[:5]
            for zero_halite_ship_id in considered_zero_ship_ids:
              rescuer_distance = valid_friendly_zero_halite_distances[
                zero_halite_ship_id]
              if rescuer_distance > 2:
                rescuer_row = valid_zero_halite_pos[0][zero_halite_ship_id]
                rescuer_col = valid_zero_halite_pos[1][zero_halite_ship_id]
                rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
                valid_move_dirs = []
                for d in valid_directions:
                  if d is not None:
                    if HALF_PLANES_RUN[row, col][d][rescuer_row, rescuer_col]:
                      valid_move_dirs.append(d)
                      
                # Break valid move ties using moving towards my weighted bases
                # and a region where I have more 0 halite ships and preferably
                # a lower die probability while moving along the diagonal
                # (empowerment)
                if len(valid_move_dirs) > 1:
                  my_valid_zero_halite_ship_density = smooth2d(
                    valid_zero_halite_ships, smooth_kernel_dim=8)
                  move_scores = np.zeros(len(valid_move_dirs))
                  for i, d in enumerate(valid_move_dirs):
                    move_row, move_col = move_ship_row_col(
                      row, col, d, grid_size)
                    dm = DISTANCE_MASKS[(row, col)]
                    dir_penalty = ship_scores[14][d] if (
                      d in ship_scores[14]) else 1
                    horiz_diff = move_col-rescuer_col
                    horiz_distance = min(np.abs(horiz_diff),
                      min(np.abs(horiz_diff-grid_size),
                          np.abs(horiz_diff+grid_size)))
                    vert_diff = move_row-rescuer_row
                    vert_distance = min(np.abs(vert_diff),
                      min(np.abs(vert_diff-grid_size),
                          np.abs(vert_diff+grid_size)))
                    empowerment_bonus = min(
                      vert_distance, horiz_distance)/2
                    # import pdb; pdb.set_trace()
                    move_scores[i] = 0.5*my_valid_zero_halite_ship_density[
                      move_row, move_col] + empowerment_bonus + (
                        dm*weighted_base_mask*my_bases).sum() - dir_penalty
                  
                  move_dir = valid_move_dirs[np.argmax(move_scores)]
                  move_row, move_col = move_ship_row_col(
                    row, col, move_dir, grid_size)
                  all_ship_scores[ship_k][0][move_row, move_col] = 1e4
                else:
                  move_dir = valid_move_dirs[0]
                  
                if valid_move_dirs:
                  move_row, move_col = move_ship_row_col(
                    row, col, move_dir, grid_size)
                  
                  # Check if the zero halite ship can move to the move position
                  rescue_dirs = get_dir_from_target(
                    rescuer_row, rescuer_col, move_row, move_col, grid_size)
                  
                  valid_rescue_dirs = [d for d in rescue_dirs if (
                    d in all_ship_scores[rescuer_k][6])]
                   
                  rescuer_should_wait = rescuer_distance in [4, 6] and (
                    obs_halite[rescuer_row, rescuer_col] == 0) and not (
                      my_bases[rescuer_row, rescuer_col])
                  if valid_rescue_dirs or rescuer_should_wait or (
                      rescuer_distance == 3):
                    if rescuer_should_wait:
                      # The rescuer should wait on a zero halite square when
                      # the distance is 4 or 6 and there is no halite at the
                      # waiting square
                      rescuer_move_row = rescuer_row
                      rescuer_move_col = rescuer_col
                    else:
                      if rescuer_distance == 3 and len(valid_rescue_dirs) == 0:
                        valid_rescue_dirs = rescue_dirs
                      # Both ships should move to each other
                      # Break rescuer ties using the lower 0 halite opponent
                      # density and moving along the diagonal (empowerment)
                      # Strongly prefer zero halite squares when the current
                      # distance is 4
                      # TODO: maybe reconsider potentially reckless rescuing
                      # behavior?
                      rescuer_move_scores = np.zeros(len(valid_rescue_dirs))
                      for i, d in enumerate(valid_rescue_dirs):
                        rescuer_move_row, rescuer_move_col = move_ship_row_col(
                          rescuer_row, rescuer_col, d, grid_size)
                        move_zero_halite_bonus = int((rescuer_distance == 4)*(
                          obs_halite[rescuer_move_row, rescuer_move_col] == 0))
                        horiz_diff = rescuer_move_col-move_col
                        horiz_distance = min(np.abs(horiz_diff),
                          min(np.abs(horiz_diff-grid_size),
                              np.abs(horiz_diff+grid_size)))
                        vert_diff = rescuer_move_row-move_row
                        vert_distance = min(np.abs(vert_diff),
                          min(np.abs(vert_diff-grid_size),
                              np.abs(vert_diff+grid_size)))
                        empowerment_bonus = min(
                          vert_distance, horiz_distance)/2
                        # import pdb; pdb.set_trace()
                        rescuer_move_scores[i] = (
                          -opponent_zero_halite_ship_density[
                            rescuer_move_row, rescuer_move_col]) + (
                              move_zero_halite_bonus) + empowerment_bonus
                      rescuer_dir = valid_rescue_dirs[np.argmax(
                        rescuer_move_scores)]
                      rescuer_move_row, rescuer_move_col = move_ship_row_col(
                          rescuer_row, rescuer_col, rescuer_dir, grid_size)
                      
                    all_ship_scores[ship_k][0][move_row, move_col] = 1e8
                    all_ship_scores[rescuer_k][0][
                      rescuer_move_row, rescuer_move_col] = 1e8
                    break
          else:
            # If I have no valid zero halite ships nearby - prefer moving
            # towards my weighted bases and a region where I have more 0
            # halite ships and preferably a lower die probability
            if len(valid_directions) > 1:
              my_valid_zero_halite_ship_density = smooth2d(
                valid_zero_halite_ships, smooth_kernel_dim=8)
              move_scores = np.zeros(len(valid_directions))
              for i, d in enumerate(valid_directions):
                move_row, move_col = move_ship_row_col(
                  row, col, d, grid_size)
                dm = DISTANCE_MASKS[(row, col)]
                dir_penalty = ship_scores[14][d] if d in ship_scores[14] else 1
                move_scores[i] = 0.5*my_valid_zero_halite_ship_density[
                  move_row, move_col] + (dm*weighted_base_mask*my_bases).sum(
                    ) - dir_penalty
              
              move_dir = valid_directions[np.argmax(move_scores)]
              move_row, move_col = move_ship_row_col(
                row, col, move_dir, grid_size)
              all_ship_scores[ship_k][0][move_row, move_col] = 1e4
            else:
              move_dir = valid_directions[0]
            
            # Slightly incentivize the nearest zero halite ship to move
            # towards my move square
            if valid_zero_halite_pos[0].size:
              move_row, move_col = move_ship_row_col(
                row, col, move_dir, grid_size)
              nearest_halite_id = np.argmin(
                valid_friendly_zero_halite_distances)
              rescuer_row = valid_zero_halite_pos[0][nearest_halite_id]
              rescuer_col = valid_zero_halite_pos[1][nearest_halite_id]
              rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
              increase_mask = get_mask_between_exclude_ends(
                move_row, move_col, rescuer_row, rescuer_col, grid_size)
              for score_id in range(3):
                all_ship_scores[rescuer_k][score_id][increase_mask] += 1e2
        
  # Escort previously chased ships to the base
  new_escort_list = []
  for (ship_k, rescuer_k, rescue_executed, min_escort_steps_remaining,
       max_escort_steps_remaining) in history[
      'escort_to_base_list']:
    if ship_k in player_obs[2] and (rescuer_k in player_obs[2]):
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      
      dm = DISTANCE_MASKS[(row, col)]
      base_scores = dm*weighted_base_mask*my_bases
      target_base = np.where(base_scores == base_scores.max())
      abort_rescue = False
      if not rescue_executed and base_scores.max() > 0:
        # Jointly move to the nearest weighted base and assume that the
        # former zero halite position won't be attacked
        rescuer_row, rescuer_col = row_col_from_square_grid_pos(
          player_obs[2][rescuer_k][0], grid_size)
        
        # Abort the rescue operation when the rescuing ship has gobbled up
        # another non-zero halite ships or if both are no longer next to
        # each other
        abort_rescue = (DISTANCES[row, col][rescuer_row, rescuer_col] > 1) or (
          halite_ships[rescuer_row, rescuer_col] > 0)
        if not abort_rescue:
          all_ship_scores[ship_k][0][rescuer_row, rescuer_col] = 1e8
          to_rescuer_dir = get_dir_from_target(
            row, col, rescuer_row, rescuer_col, grid_size)[0]
          if to_rescuer_dir not in all_ship_scores[ship_k][6]:
            all_ship_scores[ship_k][6].append(to_rescuer_dir)
          increase_mask = get_mask_between_exclude_ends(
            target_base[0], target_base[1], rescuer_row, rescuer_col,
            grid_size)
          for score_id in range(3):
            all_ship_scores[rescuer_k][score_id][increase_mask] += 1e4
          
      if not abort_rescue:
        if min_escort_steps_remaining > 1:
          new_escort_list.append((ship_k, rescuer_k, False,
                                  min_escort_steps_remaining-1,
                                  max_escort_steps_remaining-1))
        elif max_escort_steps_remaining > 1:
          # Consider if I should keep escorting the ship (there is no safe action)
          ship_scores = all_ship_scores[ship_k]
          valid_directions = ship_scores[6]
          if len(set(valid_directions) - set(ship_scores[7]+ship_scores[8])) == 0:
            new_escort_list.append(
              (ship_k, rescuer_k, False, 1, max_escort_steps_remaining-1))
  history['escort_to_base_list'] = new_escort_list
  
  return all_ship_scores, on_rescue_mission, history

def get_ship_plans(config, observation, player_obs, env_config, verbose,
                   all_ship_scores, np_rng, weighted_base_mask,
                   steps_remaining, opponent_ships_sensible_actions,
                   opponent_ships_scaled, main_base_distances, history,
                   convert_first_ship_on_None_action=True):
  ship_plans_start_time = time.time()
  my_bases = observation['rewards_bases_ships'][0][1]
  opponent_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  can_deposit_halite = my_bases.sum() > 0
  stacked_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']])
  all_ships = stacked_ships.sum(0) > 0
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[~all_ships] = -1e-9
  grid_size = observation['halite'].shape[0]
  ship_ids = list(player_obs[2])
  my_ship_count = len(player_obs[2])
  my_non_converted_ship_count = my_ship_count
  convert_cost = env_config.convertCost
  obs_halite = np.maximum(0, observation['halite'])
  collect_rate = env_config.collectRate
  num_bases = my_bases.sum()
  new_bases = []
  base_attackers = {}
  max_attackers_per_base = config['max_attackers_per_base']
  
  # Update ship scores to make sure that the plan does not contradict with
  # invalid actions when the plan is executed (map_ship_plans_to_actions)
  for ship_k in all_ship_scores:
    # if observation['step'] == 398 and ship_k == '74-1':
    #   import pdb; pdb.set_trace()
    
    bad_directions = list(set(MOVE_DIRECTIONS) - set(
      all_ship_scores[ship_k][6]))
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    if max_attackers_per_base <= 0:
      all_ship_scores[ship_k][3][:] = -1e12
    if len(bad_directions) < len(MOVE_DIRECTIONS) and not all_ship_scores[
        ship_k][13] and not (
          all_ship_scores[ship_k][12] and steps_remaining == 1):
      for d in bad_directions:
        mask_avoid = np.copy(HALF_PLANES_RUN[(row, col)][d])
        if d is not None:
          mask_avoid[row, col] = False
        for i in range(3):
          all_ship_scores[ship_k][i][mask_avoid] -= 1e5
        if d not in all_ship_scores[ship_k][10]:
          move_row, move_col = move_ship_row_col(row, col, d, grid_size)
          if opponent_bases.sum() > 0 and DISTANCES[(move_row, move_col)][
              opponent_bases].min() > 1:
            # TODO: make sure this makes our base attack strategy not too
            # dominant
            if not d in all_ship_scores[ship_k][9]:
              all_ship_scores[ship_k][3][mask_avoid] -= 1e5
            
  # Decide whether to build a new base after my last base has been destroyed.
  if num_bases == 0 and my_ship_count > 1:
    all_ship_scores, can_deposit_halite, defend_override_base_pos = (
      consider_restoring_base(
        observation, env_config, all_ship_scores, player_obs, convert_cost,
        np_rng))
    num_restored_bases = int(can_deposit_halite)
    num_bases = num_restored_bases
  elif num_bases > 1:
    can_deposit_halite = True
    num_restored_bases = 0
    max_base_pos = np.where(weighted_base_mask == weighted_base_mask.max())
    defend_override_base_pos = (max_base_pos[0][0], max_base_pos[1][0])
  else:
    can_deposit_halite = num_bases > 0
    num_restored_bases = 0
    defend_override_base_pos = None
    
  # Decide to redirect ships to the base to avoid the last base being destroyed
  # by opposing ships
  defend_base_ignore_collision_key = None
  ignore_base_collision_ship_keys = []
  should_defend = (my_ship_count-num_restored_bases) > min(
    4, 2 + steps_remaining/5)
  if num_bases >= 1 and should_defend:
    (all_ship_scores, defend_base_ignore_collision_key,
     main_base_protected, ignore_base_collision_ship_keys) = protect_main_base(
       observation, env_config, all_ship_scores, player_obs,
       defend_override_base_pos)
  else:
    main_base_protected = True
    
  # Decide on redirecting ships to friendly ships that are boxed in/chased and
  # can not return to any of my bases
  if main_base_distances.max() > 0:
    (all_ship_scores, on_rescue_mission,
     history) = update_scores_rescue_missions(
        config, all_ship_scores, stacked_ships, observation, halite_ships,
        steps_remaining, player_obs, obs_halite, history,
        opponent_ships_sensible_actions, weighted_base_mask, my_bases, np_rng)
  else:
    on_rescue_mission = np.zeros((grid_size, grid_size), dtype=np.bool)
    
  # Coordinate box in actions of opponent more halite ships
  box_start_time = time.time()
  if main_base_distances.max() > 0:
    all_ship_scores = update_scores_opponent_boxing_in(
      all_ship_scores, stacked_ships, observation,
      opponent_ships_sensible_actions, halite_ships, steps_remaining,
      player_obs, np_rng, opponent_ships_scaled, collect_rate, obs_halite,
      main_base_distances, history, on_rescue_mission)
  box_in_duration = time.time() - box_start_time
  
  # TODO: coordinate pack hunting (hoard in fixed directions)
    
  # First, process the convert actions
  # TODO: Make it a function of the attacker strength - e.g. don't convert when
  # I can probably escape.
  ship_plans = OrderedDict()
  for i, ship_k in enumerate(player_obs[2]):
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    
    ship_scores = all_ship_scores[ship_k]
    ship_halite = player_obs[2][ship_k][1]
    has_budget_to_convert = (ship_halite + player_obs[0]) >= convert_cost
    convert_surrounded_ship = ship_scores[5] and (
      ship_halite >= (convert_cost/config[
            'boxed_in_halite_convert_divisor'])) and has_budget_to_convert and(
              my_non_converted_ship_count > 1)
    valid_directions = ship_scores[6]
    almost_boxed_in = not None in valid_directions and (len(
      valid_directions) == 1 or set(valid_directions) in [
        set([NORTH, SOUTH]), set([EAST, WEST])])
    if (has_budget_to_convert and (
        my_ship_count > 1 or observation['step'] < 20 or (
          steps_remaining == 1 and ship_halite >= convert_cost and (
            ship_halite + player_obs[0]) >= 2*convert_cost)) and (
              ship_scores[2].max()) >= max([
          ship_scores[0].max()*can_deposit_halite,
          (ship_scores[1]*my_bases).max(),
          ship_scores[3].max(),
          ]) and (not almost_boxed_in)) or convert_surrounded_ship or (
            ship_scores[13]):
      # Obtain the row and column of the new target base
      target_base = np.where(ship_scores[2] == ship_scores[2].max())
      target_row = target_base[0][0]
      target_col = target_base[1][0]
      my_non_converted_ship_count -= 1
      
      if (target_row == row and target_col == col) or convert_surrounded_ship:
        ship_plans[ship_k] = CONVERT
        new_bases.append((row, col))
        my_bases[row, col] = True
        can_deposit_halite = True
      else:
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False,
                              row, col)
        
  # Next, do another pass to coordinate the target squares. This is done in a
  # double pass for now where the selection order is determined based on the 
  # availability of > 1 direction in combination with the initial best score.
  # The priorities are recomputed as target squares are taken by higher
  # priority ships.
  best_ship_scores = {}
  ship_priority_scores = np.zeros(my_ship_count)
  for i, ship_k in enumerate(player_obs[2]):
    if ship_k in ship_plans:
      # Make sure that already planned ships stay on top of the priority Q
      ship_priority_scores[i] = 1e20
    else:
      ship_scores = all_ship_scores[ship_k]
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      
      # if observation['step'] == 136 and ship_k == '51-1':
      #   import pdb; pdb.set_trace()
      
      # Incorporate new bases in the return to base scores
      ship_scores = list(ship_scores)
      ship_scores[1][np.logical_not(my_bases)] = -1e7
      ship_scores = tuple(ship_scores)
      for (r, c) in new_bases:
        if r != row or c != col:
          ship_scores[0][r, c] = -1e7
          ship_scores[2][r, c] = -1e7
      all_ship_scores[ship_k] = ship_scores
      
      num_non_immediate_bad_directions = len(set(
        ship_scores[6] + ship_scores[8]))
      
      num_two_step_neighbors = all_ships[
        ROW_COL_MAX_DISTANCE_MASKS[(row, col, 2)]].sum() - 1
      
      best_score = np.stack([
        ship_scores[0], ship_scores[1], ship_scores[2], ship_scores[3]]).max()
      best_ship_scores[ship_k] = best_score
      
      ship_priority_scores[i] = best_score + 1e12*(
          (len(ship_scores[6]) == 1)) - 1e6*(
            num_non_immediate_bad_directions) + 1e4*(
              len(ship_scores[8])) + 1e2*(
              num_two_step_neighbors) - 1e5*(
                len(ship_scores[9])) + 1e7*(
                ship_scores[11])
         
      # Old low level action planning:
      # ship_priority_scores[i] = -1e6*num_non_immediate_bad_directions -1e3*len(
    #   valid_actions) + 1e4*len(ship_scores[ship_k][8]) - 1e5*len(
    #     before_plan_ship_scores[ship_k][9]) - i + 1e7*ship_scores[ship_k][11]
            
  ship_order = np.argsort(-ship_priority_scores)
  occupied_target_squares = []
  occupied_squares_count = {}
  single_path_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  single_path_max_block_distances = np.ones(
    (grid_size, grid_size), dtype=np.int)
  return_base_distances = []
  chain_conflict_resolution = []
  ship_conflict_resolution = []
  
  # if observation['step'] == 235:
  #   print([ship_ids[o] for o in ship_order])
  #   import pdb; pdb.set_trace()
  
  for i in range(my_ship_count):
    ship_k = ship_ids[ship_order[i]]
    ship_scores = all_ship_scores[ship_k]
    ship_halite = player_obs[2][ship_k][1]
    if not ship_k in ship_plans:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      valid_directions = ship_scores[6]
      
      # if observation['step'] == 136 and ship_k in ['51-1']:
      #   import pdb; pdb.set_trace()
      
      for sq, sq_max_d in [
          (single_path_squares, single_path_max_block_distances)]:
        if sq.sum() and not ship_scores[12]:
          (s0, s1, s2, s3, _, _, _) = update_scores_blockers(
            ship_scores[0], ship_scores[1], ship_scores[2], ship_scores[3],
            row, col, grid_size, sq, sq_max_d, valid_directions,
            ship_scores[9], update_attack_base=True)
          ship_scores = (s0, s1, s2, s3, ship_scores[4], ship_scores[5],
                         ship_scores[6], ship_scores[7], ship_scores[8],
                         ship_scores[9], ship_scores[10], ship_scores[11],
                         ship_scores[12], ship_scores[13], ship_scores[14],
                         ship_scores[15])
      
      best_collect_score = ship_scores[0].max()
      best_return_score = ship_scores[1].max()
      best_establish_score = ship_scores[2].max()
      best_attack_base_score = ship_scores[3].max()
      
      # if observation['step'] == 167 and ship_k == '4-1':
      #   import pdb; pdb.set_trace()
      
      if best_collect_score >= max([
          best_return_score, best_establish_score, best_attack_base_score]):
        # 1) Gather mode
        target_gather = np.where(ship_scores[0] == ship_scores[0].max())
        target_row = target_gather[0][0]
        target_col = target_gather[1][0]
        
        if target_row == row and target_col == col and my_ship_count == 1 and (
            num_bases == 0) and convert_first_ship_on_None_action:
          ship_plans[ship_k] = CONVERT
          my_bases[row, col] = True
          update_occupied_count(
            row, col, occupied_target_squares, occupied_squares_count)
        else:
          ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False,
                                row, col)
          
          if best_collect_score > 1e5:
            # If there is only one path to defend the base: treat it as if
            # there is only one valid action:
            defend_dirs = get_dir_from_target(
              row, col, target_row, target_col, grid_size)
            if len(defend_dirs) == 1:
              move_row, move_col = move_ship_row_col(
                row, col, defend_dirs[0], grid_size)
              single_path_squares[move_row, move_col] = 1
          
          update_occupied_count(
            target_row, target_col, occupied_target_squares,
            occupied_squares_count)
      elif best_return_score >= max(
          best_establish_score, best_attack_base_score):
        # 2) Return base mode
        target_return = np.where(ship_scores[1] == ship_scores[1].max())
        target_row = target_return[0][0]
        target_col = target_return[1][0]
        
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4],
                              defend_base_ignore_collision_key == ship_k and (
                                not main_base_protected),
                              row, col)
        base_distance = grid_distance(target_row, target_col, row, col,
                                      grid_size)
        
        if not main_base_protected:
          main_base_protected = base_distance==0
        if not ship_k in ignore_base_collision_ship_keys:
          return_base_distances.append((target_row, target_col, base_distance))
      elif best_establish_score >= best_attack_base_score:
        # 3) Establish base mode
        target_base = np.where(ship_scores[2] == ship_scores[2].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False,
                              row, col)
        update_occupied_count(
            target_row, target_col, occupied_target_squares,
            occupied_squares_count)
      else:
        # 4) Attack base mode
        # import pdb; pdb.set_trace()
        target_base = np.where(ship_scores[3] == ship_scores[3].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        base_distance = DISTANCES[(row, col)][target_row, target_col]
        attack_tuple = (base_distance, ship_halite, ship_k, row, col)
        if (target_row, target_col) in base_attackers:
          base_attackers[(target_row, target_col)].append(attack_tuple)
        else:
          base_attackers[(target_row, target_col)] = [attack_tuple]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], True,
                              row, col)
        update_occupied_count(
            target_row, target_col, occupied_target_squares,
            occupied_squares_count)
        
      deterministic_next_pos = None
      if target_row == row and target_col == col:
        deterministic_next_pos = (row, col)
        single_path_squares[row, col] = 1
      elif len(valid_directions) == 1 and not None in valid_directions and (
          target_row != row or target_col != col):
        # I have only one escape direction and must therefore take that path
        escape_square = move_ship_row_col(
          row, col, valid_directions[0], grid_size)
        deterministic_next_pos = escape_square
        single_path_squares[escape_square[0], escape_square[1]] = 1
      elif (target_row == row) != (target_col == col):
        # I only have a single path to my target, so my next position, which is
        # deterministic, should be treated as an opponent base (don't set
        # plan targets behind the ship to avoid collisions with myself)
        move_dir = get_dir_from_target(
          row, col, target_row, target_col, grid_size)[0]
        next_square = move_ship_row_col(row, col, move_dir, grid_size)
        deterministic_next_pos = next_square
        single_path_squares[next_square[0], next_square[1]] = 1
      else:
        # Check if higher priority ships have already selected one of my two
        # possible paths to the target and keep track of the domino effect of
        # selected optimal squares for two-path targets
        move_dirs = get_dir_from_target(
          row, col, target_row, target_col, grid_size)
        if len(move_dirs) == 2:
          square_taken = []
          considered_squares = []
          for square_id, d in enumerate(move_dirs):
            move_square = move_ship_row_col(row, col, d, grid_size)
            taken = single_path_squares[move_square]
            square_taken.append(taken)
            considered_squares.append(move_square)
            if not taken:
              not_taken_square = move_square
          
          if square_taken[0] != square_taken[1]:
            single_path_squares[not_taken_square] = 1
          else:
            if not square_taken[0]:
              first_pair = (considered_squares[0], considered_squares[1])
              second_pair = (considered_squares[1], considered_squares[0])
              chain_conflict_resolution.append(first_pair)
              chain_conflict_resolution.append(second_pair)
              
              # If two ships move in opposite diagonal directions, both squares
              # are definitely occupied
              if ((first_pair, second_pair) in ship_conflict_resolution) and (
                    first_pair in chain_conflict_resolution) and (
                      second_pair in chain_conflict_resolution):
                deterministic_next_pos = considered_squares[0]
                ship_conflict_resolution.remove((first_pair, second_pair))
                ship_conflict_resolution.remove((second_pair, first_pair))
              else:
                ship_conflict_resolution.append((first_pair, second_pair))
                ship_conflict_resolution.append((second_pair, first_pair))
          
      # if observation['step'] == 55 and (row, col) in [
      #     (3, 4), (4, 5), (5, 4)]:
      #   import pdb; pdb.set_trace()
                  
      if deterministic_next_pos is not None:
        det_stack = [deterministic_next_pos]
        while det_stack:
          det_pos = det_stack.pop()
          single_path_squares[det_pos] = 1
          chained_pos = []
          del_pairs = []
          for sq1, sq2 in chain_conflict_resolution:
            if det_pos == sq1:
              if not sq2 in chained_pos:
                chained_pos.append(sq2)
              del_pairs.append((sq1, sq2))
              del_pairs.append((sq2, sq1))
          if chained_pos:
            det_stack += chained_pos
            chain_conflict_resolution = list(
              set(chain_conflict_resolution)-set(del_pairs))
        
    all_ship_scores[ship_k] = ship_scores
    
    # Update the ship scores for future ships - largely redundant but likely
    # not to be a performance bottleneck
    for j in range(i+1, my_ship_count):
      order_id = ship_order[j]
      ship_k_future = ship_ids[order_id]
      if not ship_k_future in ship_plans and not all_ship_scores[
          ship_k_future][12]:
        future_ship_scores = all_ship_scores[ship_k_future]
        future_row, future_col = row_col_from_square_grid_pos(
          player_obs[2][ship_k_future][0], grid_size)
        for (r, c) in occupied_target_squares:
          future_ship_scores[0][r, c] = -1e7
          future_ship_scores[2][r, c] = -1e7
          if occupied_squares_count[(r, c)] >= max_attackers_per_base:
            future_ship_scores[3][r, c] = -1e7
  
        for (r, c, d) in return_base_distances:
          # This coordinates return to base actions and avoids base blocking
          if grid_distance(r, c, future_row, future_col, grid_size) == d:
            future_ship_scores[1][r, c] = -1e7
        
        updated_best_score = np.stack([
          future_ship_scores[0], future_ship_scores[1], future_ship_scores[2],
          future_ship_scores[3]]).max()
        
        # Lower the priority for future ships using the updated ship scores
        priority_change = updated_best_score - best_ship_scores[ship_k_future]
        assert priority_change <= 0 
        ship_priority_scores[order_id] += priority_change
        
        all_ship_scores[ship_k_future] = future_ship_scores
    
    # Update the ship order - this works since priorities can only be lowered
    # and we only consider future ships when downgrading priorities
    # Make sure no ships get skipped by the +1 hack
    ship_priority_scores[ship_order[:(i+1)]] += 1
    ship_order = np.argsort(-ship_priority_scores)
    
  # if observation['step'] == 55:
  #   import pdb; pdb.set_trace()
    
  ship_plans_duration = time.time() - ship_plans_start_time
  return (ship_plans, my_bases, all_ship_scores, base_attackers,
          box_in_duration, history, ship_plans_duration)

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

def base_can_be_defended(base_attackers, target_row, target_col, stacked_bases,
                         stacked_ships, halite_ships):
  attackers = base_attackers[(target_row, target_col)]
  num_attackers = len(attackers)
  attack_distances = np.array([a[0] for a in attackers])
  distance_argsort = np.argsort(attack_distances)
  attack_distances_sorted = np.maximum(
    attack_distances[distance_argsort], 1+np.arange(num_attackers))
  attack_halite_sorted = np.array([
    attackers[distance_argsort[i]][1] for i in range(num_attackers)])
  
  attack_scores = np.sort(attack_distances_sorted+1e-6*attack_halite_sorted)
  opponent_ids = np.where(stacked_bases[:, target_row, target_col])[0]
  can_defend = False
  if opponent_ids:
    opponent_id = opponent_ids[0]
    opponent_ships = stacked_ships[opponent_id]
    defend_scores = 1e-6*halite_ships[opponent_ships] + np.maximum(
      1, DISTANCES[(target_row, target_col)][opponent_ships])
    sorted_defend_scores = np.sort(defend_scores)
    max_ships = min(attack_scores.size, defend_scores.size)
    can_defend = not np.any(
      attack_scores[:max_ships] < sorted_defend_scores[:max_ships])
    if can_defend:
      # Check that the opponent can defend against a sequence of sacrifices
      in_sequence_near_base = attack_distances_sorted[:max_ships] == (
        1+np.arange(max_ships))
      num_lined_up = (np.cumprod(in_sequence_near_base) == 1).sum()
      opponent_zero_halite_sorted_base_distances = np.sort(DISTANCES[
        (target_row, target_col)][np.logical_and(
          opponent_ships, halite_ships == 0)])
      can_defend = opponent_zero_halite_sorted_base_distances.size >= (
        num_lined_up) and not np.any(
          opponent_zero_halite_sorted_base_distances[:num_lined_up] > (
            1+np.arange(num_lined_up)))
    
  return can_defend

def get_opponent_blocked_escape_dir(
  bad_positions, opponent_ships_sensible_actions, row, col, np_rng, grid_size,
  observation, ship_k):
  escape_actions = []
  for a in MOVE_DIRECTIONS:
    move_row, move_col = move_ship_row_col(row, col, a, grid_size)
    
    if not bad_positions[move_row, move_col]:
      # Check all five ways the move row, move col can be reached. If none is
      # in the opponent sensible actions: take that action
      opponent_can_attack_square = False
      for b in MOVE_DIRECTIONS:
        neighbor_row, neighbor_col = move_ship_row_col(
          move_row, move_col, b, grid_size)
        neighbor_k = (neighbor_row, neighbor_col)
        if neighbor_k in opponent_ships_sensible_actions:
          threat_opponent_rel_dir = RELATIVE_DIR_MAPPING[OPPOSITE_MAPPING[b]]
          if threat_opponent_rel_dir in opponent_ships_sensible_actions[
              neighbor_k]:
            opponent_can_attack_square = True
            break
          
      if not opponent_can_attack_square:
        escape_actions.append(a)
        
  return escape_actions

def map_ship_plans_to_actions(
    config, observation, player_obs, env_observation, env_config, verbose,
    ship_scores, before_plan_ship_scores, ship_plans, np_rng,
    ignore_bad_attack_directions, base_attackers, steps_remaining,
    opponent_ships_sensible_actions, history, env_obs_ids,
    opponent_ships_scaled, main_base_distances):
  ship_map_start_time = time.time()
  ship_actions = {}
  remaining_budget = player_obs[0]
  convert_cost = env_config.convertCost
  obs_halite = np.maximum(0, observation['halite'])
  # Clip obs_halite to zero when gathering it doesn't add to the score
  # code: delta_halite = int(cell.halite * configuration.collect_rate)
  obs_halite[obs_halite < 1/env_config.collectRate] = 0
  grid_size = obs_halite.shape[0]
  my_ship_count = len(player_obs[2])
  my_next_ships = np.zeros((grid_size, grid_size), dtype=np.bool)
  stacked_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships']])
  stacked_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']])
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  updated_ship_pos = {}
  my_non_zero_halite_ship_density = smooth2d(np.logical_and(
    stacked_ships[0], halite_ships > 0), smooth_kernel_dim=3)
  my_zero_halite_ship_density = smooth2d(np.logical_and(
    stacked_ships[0], halite_ships == 0), smooth_kernel_dim=5)
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
  
  # For debugging - the order in which actions are planned
  ordered_debug_ship_plans = [[k]+list(v) for k, v in ship_plans.items()]
  ordered_debug_ship_plans = ordered_debug_ship_plans
  
  for target_base in base_attackers:
    attackers = base_attackers[target_base]
    num_attackers = len(attackers)
    if num_attackers > 1 and steps_remaining > 20:
      # If the base can not be defended: don't bother synchronizing the attack
      # if observation['step'] == 360:
      #   import pdb; pdb.set_trace()
      #   x=1
        
      can_defend = base_can_be_defended(
        base_attackers, target_base[0], target_base[1], stacked_bases,
        stacked_ships, halite_ships)
      
      # Synchronize the attackers
      if can_defend:
        attack_distances = np.array([a[0] for a in attackers])
        argsort_distances = np.argsort(attack_distances)
        sorted_distances = attack_distances[argsort_distances]
        
        if not np.all(np.diff(sorted_distances) == 1):
          # First try the simplest strategy: pause the attackers that are the
          # closest to the target on squares with no halite
          should_wait = np.zeros((num_attackers), dtype=np.bool)
          if sorted_distances[0] == sorted_distances[1]:
            should_wait[argsort_distances[1]] = 1
            next_distance = sorted_distances[0]
          elif sorted_distances[0] == sorted_distances[1]-1:
            should_wait[argsort_distances[0]] = 1
            should_wait[argsort_distances[1]] = 1
            next_distance = sorted_distances[1]
          else:
            should_wait[argsort_distances[0]] = 1
            next_distance = sorted_distances[1]-1
            
          for i in range(2, num_attackers):
            if next_distance == sorted_distances[i]-2:
              # I should close the ranks and move along
              next_distance = sorted_distances[i]-1
            else:
              # I should wait for other ships to catch up
              should_wait[argsort_distances[i]] = 1
              next_distance = sorted_distances[i]
              
          # Verify that all ships that should wait can wait
          can_all_wait = True
          should_wait_ids = np.where(should_wait)[0]
          for should_wait_id in should_wait_ids:
            row = attackers[should_wait_id][3]
            col = attackers[should_wait_id][4]
            if obs_halite[row, col] > 0:
              can_all_wait = False
              break
          if can_all_wait:
            for should_wait_id in should_wait_ids:
              ship_k = attackers[should_wait_id][2]
              row = attackers[should_wait_id][3]
              col = attackers[should_wait_id][4]
              ship_plans[ship_k] = (row, col, [], False, row, col)
  
  # List all positions you definitely don't want to move to. Initially this
  # only contains enemy bases and eventually also earlier ships.
  opponent_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  bad_positions = np.copy(opponent_bases)
  
  # Order the ship plans based on the available valid direction count. Break
  # ties using the original order.
  # move_valid_actions = OrderedDict()
  shortest_path_count = {}
  ship_priority_scores = np.zeros(my_ship_count)
  ship_key_plans = list(ship_plans)
  for i, ship_k in enumerate(ship_key_plans):
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    valid_actions = []
    if not isinstance(ship_plans[ship_k], str):
      (target_row, target_col, preferred_directions, ignore_base_collision,
       _, _) = ship_plans[ship_k]
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                             grid_size)
      
      # Filter out bad positions from the shortest actions
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        if not bad_positions[move_row, move_col] or (
            ignore_base_collision and (
              move_row == target_row and move_col == target_col)) or (
                not opponent_bases[move_row, move_col] and ship_scores[
                  ship_k][12]):
          valid_actions.append(a)
      for a in valid_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        path_lookup_k = (move_row, move_col)
        square_weight = 1 if len(valid_actions) > 1 else 1.1
        if path_lookup_k in shortest_path_count:
          shortest_path_count[path_lookup_k] += square_weight
        else:
          shortest_path_count[path_lookup_k] = square_weight
      # move_valid_actions[ship_k] = valid_actions
  
    # num_non_immediate_bad_directions = len(set(
    #   ship_scores[ship_k][6] + ship_scores[ship_k][8]))
    
    # Just keep the order from the planning - this is cleaner and works better!
    ship_priority_scores[i] = -i
    
    # ship_priority_scores[i] = -1e6*num_non_immediate_bad_directions -1e3*len(
    #   valid_actions) + 1e4*len(ship_scores[ship_k][8]) - 1e5*len(
    #     before_plan_ship_scores[ship_k][9]) - i + 1e7*ship_scores[ship_k][11]
  
  ship_order = np.argsort(-ship_priority_scores)
  ordered_ship_plans = [ship_key_plans[o] for o in ship_order]
  
  # Keep track of all my ship positions and rearrange the action planning when
  # one of my ships only has one remaining option that does not self destruct.
  ship_non_self_destructive_actions = {}
  for ship_k in ordered_ship_plans:
    ship_non_self_destructive_actions[ship_k] = copy.copy(MOVE_DIRECTIONS)
  
  num_ships = len(ordered_ship_plans)
  action_overrides = np.zeros((7))
  for i in range(num_ships):
    ship_k = ordered_ship_plans[i]
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    has_selected_action = False
    
    if isinstance(ship_plans[ship_k], str):
      if ship_plans[ship_k] == CONVERT and (
          halite_ships[row, col] < convert_cost/config[
            'boxed_in_halite_convert_divisor']) and (
            halite_ships[row, col] > 0) and my_ship_count > 1 and (
              ship_scores[ship_k][2][row, col] < 1e6):
        # Override the convert logic - it's better to lose some ships than to
        # convert too often (good candidate for stateful logic)
        # This can happen when the base is reconstructed
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
          
        shortest_actions = get_dir_from_target(
          row, col, target_row, target_col, grid_size)
        # Filter out bad positions from the shortest actions
        for a in shortest_actions:
          move_row, move_col = move_ship_row_col(row, col, a, grid_size)
          if not bad_positions[move_row, move_col] or (not opponent_bases[
              move_row, move_col] and ship_scores[ship_k][12]):
            path_lookup_k = (move_row, move_col)
            if not path_lookup_k in shortest_path_count:
              shortest_path_count[path_lookup_k] = 0
        ship_plans[ship_k] = (target_row, target_col, [], False, row, col)
      else:
        ship_actions[ship_k] = ship_plans[ship_k]
        obs_halite[row, col] = 0
        remaining_budget -= convert_cost
        has_selected_action = True
        del ship_non_self_destructive_actions[ship_k]
        
    if not has_selected_action:
      (target_row, target_col, preferred_directions, ignore_base_collision,
       _, _) = ship_plans[ship_k]
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                             grid_size)
      
      if ignore_base_collision and not ignore_bad_attack_directions and (
          target_row, target_col) in base_attackers:
        
        can_defend = base_can_be_defended(
          base_attackers, target_row, target_col, stacked_bases, stacked_ships,
          halite_ships)
          
        if can_defend:
          ignore_base_collision = False
        else:
          ignore_bad_attack_directions = True
        
      # Remove own ships from the shortest action bad positions when the ships
      # are returning to the base at the end of the game
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        if bad_positions[move_row, move_col] and not opponent_bases[
            move_row, move_col] and ship_scores[ship_k][12]:
          bad_positions[move_row, move_col] = False
        
      # Filter out bad positions from the shortest actions
      valid_actions = []
      valid_move_positions = []
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        if (not bad_positions[move_row, move_col] and (
            a in ship_scores[ship_k][6])) or (ignore_base_collision and ((
              move_row == target_row and move_col == target_col) or (
                ignore_bad_attack_directions and not bad_positions[
                  move_row, move_col])) and not my_next_ships[
                    move_row, move_col]) or (
                      ship_scores[ship_k][12] and steps_remaining == 1):
          valid_actions.append(a)
          valid_move_positions.append((move_row, move_col))
          path_lookup_k = (move_row, move_col)
          if not path_lookup_k in shortest_path_count:
            print("Path key lookup fail step", observation['step'], row, col,
                  ship_k, i)
            shortest_path_count[path_lookup_k] = 1

      if valid_actions:
        # Prefer actions that conflict with the lowest number of optimal
        # ship trajectories
        if len(valid_actions) > 1:
          shortest_path_counts = np.array([
            shortest_path_count[k] for k in valid_move_positions])
          shortest_path_ids = np.where(
            shortest_path_counts == shortest_path_counts.min())[0].tolist()
          valid_actions = [a for i, a in enumerate(valid_actions) if (
            i in shortest_path_ids)]
          
        # Take a preferred action when it is among the shortest path options
        if len(valid_actions) > 1 and preferred_directions:
          intersect_directions = list(set(valid_actions) & set(
            preferred_directions))
          if intersect_directions:
            valid_actions = intersect_directions
            
        # Prefer a non invalid one step action when defending the base
        if len(valid_actions) > 1:
          intersect_directions = list(set(valid_actions) & set(
            before_plan_ship_scores[ship_k][9]))
          if intersect_directions:
            valid_actions = intersect_directions
            
        # When all actions are equally valid, move to a square with lower non
        # zero halite ship density in order to avoid double box-ins.
        # Only do this for non zero halite ships
        if len(valid_actions) > 1 and halite_ships[row, col] > 0:
          considered_densities = np.zeros(len(valid_actions))
          for a_id, a in enumerate(valid_actions):
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            considered_densities[a_id] = my_non_zero_halite_ship_density[
              move_row, move_col]
          valid_actions = [valid_actions[np.argmin(considered_densities)]]
            
        if len(valid_actions) > 1:
          # Do this in order to obtain reproducible results - the set intersect
          # logic is flaky.
          valid_actions.sort()
        action = str(np_rng.choice(valid_actions))
        action = None if action == 'None' else action
      else:
        # I can not move to my target using a shortest path action
        # Alternative: consider all valid, not bad actions
        # import pdb; pdb.set_trace()
        action_overrides[0] += 1
        valid_not_bad_actions = []
        for a in np_rng.permutation(MOVE_DIRECTIONS):
          if a in ship_scores[ship_k][6]:
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            if not bad_positions[move_row, move_col]:
              valid_not_bad_actions.append(a)
              
        # if valid_not_bad_actions:
        #   import pdb; pdb.set_trace()
              
        # Pick a direction where my opponent should not go to since I can
        # attack that square with one of my less halite ships
        if not valid_not_bad_actions:
          action_overrides[1] += 1
          self_escape_actions = get_opponent_blocked_escape_dir(
            bad_positions, opponent_ships_sensible_actions, row, col, np_rng,
            grid_size, observation, ship_k)
         
          if self_escape_actions:
            if before_plan_ship_scores[ship_k][9]:
              # Filter out 1-step bad actions if that leaves us with options
              self_escape_actions_not_1_step_bad = list(
                set(self_escape_actions) & set(
                  before_plan_ship_scores[ship_k][9]))
              if self_escape_actions_not_1_step_bad:
                self_escape_actions = self_escape_actions_not_1_step_bad
            
            if ship_scores[ship_k][7]:
              # Filter out 2-step bad actions if that leaves us with options
              self_escape_actions_not_2_step_bad = list(
                set(self_escape_actions) - set(ship_scores[ship_k][7]))
              if self_escape_actions_not_2_step_bad:
                self_escape_actions = self_escape_actions_not_2_step_bad
                
            # Filter out n-step bad actions if that leaves us with options
            if ship_scores[ship_k][8]:
              self_escape_actions_not_n_step_bad = list(
                set(self_escape_actions) - set(ship_scores[ship_k][8]))
              if self_escape_actions_not_n_step_bad:
                self_escape_actions = self_escape_actions_not_n_step_bad
                
            # Select the shortest actions if that leaves us with options (that
            # way we stick to the ship plan)
            if bool(self_escape_actions) and bool(shortest_actions):
              intersect_directions = list(set(self_escape_actions) & set(
                shortest_actions))
              if intersect_directions:
                self_escape_actions = intersect_directions
                
            # Pick the least bad of the n-step bad directions if that is all we
            # are choosing from
            if len(ship_scores[ship_k][8]) > 1 and len(
                self_escape_actions) > 1:
              if np.all(
                  [a in ship_scores[ship_k][8] for a in self_escape_actions]):
                die_probs = np.array([ship_scores[ship_k][14][a] for a in (
                  self_escape_actions)])
                self_escape_actions = [
                  self_escape_actions[np.argmin(die_probs)]]
               
            valid_not_bad_actions = self_escape_actions
              
        # There is no valid direction available; consider n step bad actions
        # This check is most probably obsolete since it would appear in
        # self_escape_actions
        if not valid_not_bad_actions:
          action_overrides[2] += 1
          for a in np_rng.permutation(MOVE_DIRECTIONS):
            if a in ship_scores[ship_k][8]:
              move_row, move_col = move_ship_row_col(row, col, a, grid_size)
              if not bad_positions[move_row, move_col]:
                valid_not_bad_actions.append(a)
                
        # There is still no valid direction available; consider 2 step bad
        # actions
        # This check is most probably obsolete since it would appear in
        # self_escape_actions
        if not valid_not_bad_actions:
          action_overrides[3] += 1
          for a in np_rng.permutation(MOVE_DIRECTIONS):
            if a in ship_scores[ship_k][7]:
              move_row, move_col = move_ship_row_col(row, col, a, grid_size)
              if not bad_positions[move_row, move_col]:
                valid_not_bad_actions.append(a)
              
        # When attacking a base it is better to keep moving or stay still on
        # a square that has no halite - otherwise it becomes a target for
        # stealing halite.
        # if observation['step'] == 258:
        #   import pdb; pdb.set_trace()
        if valid_not_bad_actions and ignore_base_collision and obs_halite[
            row, col] > 0:
          if len(valid_not_bad_actions) > 1 and None in valid_not_bad_actions:
            valid_not_bad_actions.remove(None)
              
        if valid_not_bad_actions:
          if len(valid_not_bad_actions) > 1:
            # Do this in order to obtain reproducible results - the set
            # intersect logic is flaky.
            valid_not_bad_actions = [str(a) for a in valid_not_bad_actions]
            valid_not_bad_actions.sort()
            valid_not_bad_actions = [
              a if a != "None" else None for a in valid_not_bad_actions]
          action = np_rng.choice(valid_not_bad_actions)
        else:
          action_overrides[4] += 1
          
          # By default: pick a random, not bad moving action
          found_non_bad = False
          # When being chased: consider replacing the postion of the chaser
          if ship_k in history['chase_counter'][0]:
            chase_details = history['chase_counter'][0][ship_k]
            # If the opponent can move towards me and has no other ships that
            # can take the position of the chaser: take the place of the chaser
            chaser_row = chase_details[4]
            chaser_col = chase_details[5]
            num_opp_chase_step_counter = chase_details[1]
            if num_opp_chase_step_counter > 2:
              to_opponent_dir = get_dir_from_target(
                  row, col, chaser_row, chaser_col, grid_size)[0]
              opp_to_me_dir = OPPOSITE_MAPPING[to_opponent_dir]
              rel_opp_to_me_dir = RELATIVE_DIR_MAPPING[opp_to_me_dir]
              opp_can_move_to_me = rel_opp_to_me_dir in (
                opponent_ships_sensible_actions[chaser_row, chaser_col])
              
              # There is a unique opponent id with the least amount of halite
              # on the chaser square or the chaser has at least one friendly
              # ship that can replace it
              chaser_id = player_ids[chaser_row, chaser_col]
              near_chaser = ROW_COL_MAX_DISTANCE_MASKS[
                chaser_row, chaser_col, 1]
              near_halite = halite_ships[near_chaser]
              near_chaser_friendly_halite = near_halite[
                (near_halite >= 0) & (player_ids[near_chaser] == chaser_id)]
              min_non_chaser_halite = near_halite[
                (near_halite >= 0) & (
                  player_ids[near_chaser] != chaser_id)].min()
              min_near_chaser_halite = near_halite[near_halite >= 0].min()
              opponent_min_hal_ids = player_ids[np.logical_and(
                near_chaser, halite_ships == min_near_chaser_halite)]
              
              near_me = ROW_COL_MAX_DISTANCE_MASKS[row, col, 1]
              near_me_threat_players = player_ids[np.logical_and(
                near_me, (halite_ships >= 0) & (
                  halite_ships < halite_ships[row, col]))]
              
              double_opp_chase = (near_me_threat_players.size > 1) and (
                np.all(near_me_threat_players == chaser_id))
              
              chaser_can_replace = ((opponent_min_hal_ids.size > 1) and (
                np.all(opponent_min_hal_ids == chaser_id) or (
                (opponent_min_hal_ids == chaser_id).sum() > 1)) or (
                  (near_chaser_friendly_halite <= (
                    min_non_chaser_halite)).sum() > 1)) or double_opp_chase
              
              chaser_players_index = env_obs_ids[chaser_id]
              chaser_k = [k for k, v in env_observation.players[
                chaser_players_index][2].items() if v[0] == (
                  chaser_row*grid_size + chaser_col)][0]
              chaser_is_chased = chaser_k in history[
                'chase_counter'][chaser_id]
              chaser_is_chased_by_not_me = chaser_is_chased
              if chaser_is_chased:
                chaser_chaser = history['chase_counter'][chaser_id][chaser_k]
                chaser_is_chased_by_not_me = (chaser_chaser[4] is None) or (
                  player_ids[chaser_chaser[4], chaser_chaser[5]] != 0)
              
              # if observation['step'] == 179:
              #   import pdb; pdb.set_trace()
              
              if opp_can_move_to_me and not chaser_can_replace and not (
                  chaser_is_chased_by_not_me):
                # Move to the position of the chaser
                action = str(to_opponent_dir)
                found_non_bad = True
          
          # if observation['step'] == 161:
          #   import pdb; pdb.set_trace()
          
          if not found_non_bad:
            action_scores = np.zeros(4)
            for a_id, a in enumerate(NOT_NONE_DIRECTIONS):
              move_row, move_col = move_ship_row_col(row, col, a, grid_size)
              # There is always only a single opponent that can safely attack
              # my move_square. First determine the opponent and count the
              # number of potential attack ships
              potential_threat_ships = []
              for d_move in MOVE_DIRECTIONS:
                other_row, other_col = move_ship_row_col(
                  move_row, move_col, d_move, grid_size)
                other_player = player_ids[other_row, other_col]
                if other_player > 0:
                  potential_threat_ships.append(
                    (other_player, halite_ships[other_row, other_col]))
              other_ships = np.array(potential_threat_ships)
              if len(other_ships) == 0:
                opponent_threat_count = 0
                opponent_id_penalty = 0
                
                # I have already taken the square
                if my_next_ships[move_row, move_col]:
                  attack_base_bonus = 0
                else:
                  # This is an opponent base - add an attack base bonus if
                  # it can not be defended
                  # TODO: remove in submission
                  attack_base_bonus = 0
                  print("BOXED IN NEXT TO AN OPPONENT BASE!")
                  assert opponent_bases[move_row, move_col]
                  opponent_base_id = np.where(
                    stacked_bases[:, move_row, move_col])[0][0]
                  opponent_id_bonus = opponent_ships_scaled[
                    opponent_base_id-1]
                  base_distance_my_main_base = main_base_distances[
                    move_row, move_col]
                  
                  # import pdb; pdb.set_trace()
                  opponent_can_move_to_base = False
                  for base_dir in MOVE_DIRECTIONS:
                    base_defend_row, base_defend_col = move_ship_row_col(
                      move_row, move_col, base_dir, grid_size)
                    if stacked_ships[
                        opponent_base_id, base_defend_row, base_defend_col]:
                      defend_dir = RELATIVE_DIR_MAPPING[
                        OPPOSITE_MAPPING[base_dir]]
                      print("OPPONENT NEXT TO BASE!")
                      if defend_dir in opponent_ships_sensible_actions[
                          base_defend_row, base_defend_col]:
                        print("OPPONENT CAN MOVE TO BASE!")
                        opponent_can_move_to_base = True
                        break
                      
                  if not opponent_can_move_to_base:
                    attack_base_bonus = 1e6*int(
                      base_distance_my_main_base <= 5 or (
                        opponent_id_bonus > 0)) + (20*opponent_id_bonus-5) + (
                          max(0, 6-base_distance_my_main_base))
              else:
                attack_base_bonus = 0
                min_halite_player = int(
                  other_ships[np.argmin(other_ships[:, 1]), 0])
                if np.all(other_ships[:, 0] == min_halite_player):
                  min_other_halite = 1e10
                else:
                  min_other_halite = other_ships[other_ships[:, 0] != (
                    min_halite_player), 1].min()
                my_move_neighbor_halite_mask = stacked_ships[0] & (
                  ROW_COL_DISTANCE_MASKS[move_row, move_col, 1])
                min_other_halite = min(min_other_halite, halite_ships[
                  my_move_neighbor_halite_mask].min())
                opponent_threat_count = (other_ships[
                  other_ships[:, 0] == min_halite_player, 1] < (
                    min_other_halite)).sum()
                opponent_id_penalty = opponent_ships_scaled[
                  min_halite_player-1]
              
              # import pdb; pdb.set_trace()
              action_scores[a_id] = -1e6*bad_positions[
                move_row, move_col] - main_base_distances[
                  move_row, move_col] - 3*opponent_threat_count -2*(
                    opponent_id_penalty*halite_ships[row, col]/250) + (
                      attack_base_bonus) - 0.5*ship_scores[ship_k][15][
                        move_row, move_col] - 0.5*(
                          my_non_zero_halite_ship_density[
                            move_row, move_col]) + my_zero_halite_ship_density[
                              move_row, move_col]
                
            best_action_score = action_scores.max()
            if best_action_score > -1e5:
              best_ids = np.where(action_scores == best_action_score)[0]
              select_id = np_rng.choice(best_ids)
              action = str(NOT_NONE_DIRECTIONS[select_id])
              found_non_bad = True
          
          # If all actions are bad: do nothing - this is very rare since it
          # would mean being surrounded by my other ships and opponent bases
          if not found_non_bad:
            action_overrides[5] += 1
            action = None
      
      # Update my_next_ships
      new_row, new_col = move_ship_row_col(row, col, action, grid_size)
      my_next_ships[new_row, new_col] = 1
      bad_positions[new_row, new_col] = 1
      updated_ship_pos[ship_k] = (new_row, new_col)
      if action is not None:
        ship_actions[ship_k] = action
        
      # Update the shortest path counts for the remaining ships
      shortest_path_count = {}
      for future_ship_k in ordered_ship_plans[(i+1):]:
        row, col = row_col_from_square_grid_pos(
          player_obs[2][future_ship_k][0], grid_size)
        
        # if observation['step'] == 390 and row == 0 and col == 8 and i == 5:
        #   import pdb; pdb.set_trace()
        
        if not isinstance(ship_plans[future_ship_k], str):
          (target_row, target_col, _, ignore_base_collision,
           _, _) = ship_plans[future_ship_k]
          shortest_actions = get_dir_from_target(
            row, col, target_row, target_col, grid_size)
          
          # Filter out bad positions from the shortest actions
          valid_actions = []
          for a in shortest_actions:
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            if not bad_positions[move_row, move_col] or (
                ignore_base_collision and (
                  move_row == target_row and move_col == target_col)) or (
                not opponent_bases[move_row, move_col] and ship_scores[
                  future_ship_k][12]):
              valid_actions.append(a)
          for a in valid_actions:
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            path_lookup_k = (move_row, move_col)
            square_weight = 1 if len(valid_actions) > 1 else 1.1
            if path_lookup_k in shortest_path_count:
              shortest_path_count[path_lookup_k] += square_weight
            else:
              shortest_path_count[path_lookup_k] = square_weight
        
      # Update the non self destructive actions for ships where no action is
      # planned yet
      del ship_non_self_destructive_actions[ship_k]
      rearrange_self_destruct_ships = []
      if not ship_scores[ship_k][12]:
        for j in range(i+1, num_ships):
          other_ship_k = ordered_ship_plans[j]
          if ship_plans[other_ship_k] != CONVERT:
            other_row = ship_plans[other_ship_k][4]
            other_col = ship_plans[other_ship_k][5]
          
            # If I move to a distance of <= 1 of the other ship: update valid
            # non self destruct actions
            # Exception: end of episode base return
            distance = grid_distance(
              new_row, new_col, other_row, other_col, grid_size)
            if distance <= 1:
              remove_dir = get_dir_from_target(
                other_row, other_col, new_row, new_col, grid_size)[0]
              if remove_dir in ship_non_self_destructive_actions[other_ship_k]:
                ship_non_self_destructive_actions[other_ship_k].remove(
                  remove_dir)
              
                if len(ship_non_self_destructive_actions[other_ship_k]) == 1:
                  rearrange_self_destruct_ships.append(other_ship_k)
            
      # Place ships that only have a single non self destruct action to the
      # front of the queue.
      if rearrange_self_destruct_ships:
        remaining_ships = [s for s in ordered_ship_plans[(i+1):] if (
          s not in rearrange_self_destruct_ships)]
        
        ordered_ship_plans = ordered_ship_plans[:(i+1)] + (
          rearrange_self_destruct_ships) + remaining_ships
  
  map_duration = time.time() - ship_map_start_time
  return (ship_actions, remaining_budget, my_next_ships, obs_halite,
          updated_ship_pos, -np.diff(action_overrides), map_duration)

def decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships, obs_halite,
    env_config, remaining_budget, verbose, ship_plans, updated_ship_pos,
    weighted_base_mask):

  spawn_cost = env_config.spawnCost
  my_ship_count = my_next_ships.sum()
  max_spawns = int(remaining_budget/spawn_cost)
  max_spawns = min(max_spawns, int(config['max_ships']-my_ship_count))
  max_allowed_ships = config['max_ships']
  total_ship_count = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships']]).sum()
  max_spawns = min(max_spawns, int(max_allowed_ships - my_ship_count))
  max_spawns = min(max_spawns, int(obs_halite.sum()/2/spawn_cost))
  relative_step = observation['relative_step']
  max_spawns = min(max_spawns, int(
    obs_halite.sum()/min(
      total_ship_count+1e-9, (my_ship_count+1e-9)*2)/spawn_cost*(
      1-relative_step)*(env_config.episodeSteps-2)/config[
        'max_spawn_relative_step_divisor']))
  last_episode_turn = observation['relative_step'] == 1

  if max_spawns <= 0 or not player_obs[1] or last_episode_turn:
    return {}, remaining_budget
  
  num_bases = len(player_obs[1])
  spawn_scores = np.zeros(num_bases)
  grid_size = obs_halite.shape[0]
  smoothed_friendly_ship_halite = smooth2d(
    observation['rewards_bases_ships'][0][3])
  smoothed_halite = smooth2d(obs_halite)
  
  for i, base_k in enumerate(player_obs[1]):
    row, col = row_col_from_square_grid_pos(player_obs[1][base_k], grid_size)
    # Don't spawn if it is not the main base
    spawn_scores[i] -= 1e12*int(weighted_base_mask[row, col] < 1)
    
    # Don't spawn when there will be a ship at the base
    spawn_scores[i] -= 1e12*my_next_ships[row, col]
    
    # Don't spawn when there is a returning ship that wants to enter the base
    # in two steps
    # Exception: if the base is not too crowded, it is ok to spawn in this
    # scenario.
    near_base_ship_count = np.logical_and(
      my_next_ships, ROW_COL_MAX_DISTANCE_MASKS[(row, col, 3)]).sum()
    if near_base_ship_count >= config['no_spawn_near_base_ship_limit']:
      for k in ship_plans:
        if ship_plans[k][0] == row and ship_plans[k][1] == col:
          updated_distance = grid_distance(row, col, updated_ship_pos[k][0],
                                           updated_ship_pos[k][1], grid_size)
          if updated_distance == 1:
            spawn_scores[i] -= 1e6
            break
    
    # Spawn less when the base is crowded with ships with a lot of halite
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

def get_env_obs_ids(env_observation):
  num_players = len(env_observation.players)
  my_id = env_observation.player
  env_obs_ids = [i for i in range(num_players)]
  env_obs_ids.remove(my_id)
  env_obs_ids = [my_id] + env_obs_ids
  
  return env_obs_ids

def update_chase_counter(history, observation, env_observation, stacked_ships,
                         other_halite_ships, player_ids, env_obs_ids):
  grid_size = stacked_ships.shape[1]
  num_players = stacked_ships.shape[0]
  
  if observation['step'] == 0:
    history['chase_counter'] = [{} for _ in range(num_players)]
    history['escort_to_base_list'] = []
  else:
    for player_id in range(num_players):
      # Remove converted or destroyed ships from the chase counter
      player_obs = env_observation.players[env_obs_ids[player_id]]
      delete_keys = []
      for ship_k in history['chase_counter'][player_id]:
        if not ship_k in player_obs[2]:
          # if player_id > 0 or (not history['prev_step']['my_ship_actions'][
          #     ship_k] == CONVERT):
          #   # import pdb; pdb.set_trace()
          #   print("Destroyed ship player", player_id, "step",
          #         observation['step'],
          #         history['chase_counter'][player_id][ship_k])
          delete_keys.append(ship_k)
      for del_k in delete_keys:
        del history['chase_counter'][player_id][del_k]
          
      # Increment the chase counter of ships that are 1 step away from a less
      # halite ship, delete the counter for other ships
      for ship_k in player_obs[2]:
        row, col = row_col_from_square_grid_pos(
          player_obs[2][ship_k][0], grid_size)
        ship_halite = player_obs[2][ship_k][1]
        step_1_mask = ROW_COL_DISTANCE_MASKS[row, col, 1]
        threats = ship_halite > other_halite_ships[step_1_mask]
        direct_threat = np.any(threats)
        
        # if observation['step'] == 88 and row == 11 and col == 8:
        #   import pdb; pdb.set_trace()
        
        if direct_threat:
          chaser_ids = player_ids[step_1_mask][threats]
          if (player_id == 0 and (
              history['prev_step']['my_ship_actions'][ship_k] is None)) or (
              not ship_k in history['chase_counter'][player_id]):
            history['chase_counter'][player_id][ship_k] = (
                chaser_ids, 1, row, col, None, None)
          else:
            prev_chasers, prev_count, prev_row, prev_col, _, _ = history[
              'chase_counter'][player_id][ship_k]
            prev_pos_opponent = player_ids[prev_row, prev_col]
            if ship_halite > other_halite_ships[prev_row, prev_col] and (
                prev_pos_opponent > 0) and prev_pos_opponent in prev_chasers:
              # if player_id == 0 and prev_count > 20:
                # import pdb; pdb.set_trace()
                # print(prev_count, observation['step'], row, col)
              history['chase_counter'][player_id][ship_k] = (
                np.array([prev_pos_opponent]), prev_count+1, row, col, prev_row,
                prev_col)
            else:
              history['chase_counter'][player_id][ship_k] = (
                chaser_ids, 1, row, col, None, None)
        else:
          if ship_k in history['chase_counter'][player_id]:
            del history['chase_counter'][player_id][ship_k]
        
  return history

def list_of_combs(arr):
    """returns a list of all subsets of a list"""
    combs = []
    for i in range(len(arr)):
      listing = [list(x) for x in itertools.combinations(arr, i+1)]
      combs.extend(listing)
    return combs

def infer_player_conversions(player_obs, prev_player_obs, env_config,
                             observation, env_obs_id):
  # By considering the score change, the gather behavior and the number of
  # spawns, the number of conversions can be aproximately inferred (not exact
  # because of base attacks)
  convert_cost = env_config.convertCost
  score_change = prev_player_obs[0] - player_obs[0]
  
  # Consider new, disappeared and remaining ships
  current_ships = set(player_obs[2].keys())
  prev_ships = set(prev_player_obs[2].keys())
  disappeared_ships = list(prev_ships - current_ships)
  new_ships = list(current_ships-prev_ships)
  new_ship_count = len(new_ships)
  spent_spawns = new_ship_count*env_config.spawnCost
  not_destroyed_ships = set(current_ships & prev_ships)
  
  # Consider new bases
  current_bases = set(player_obs[1].keys())
  current_base_pos = [player_obs[1][b] for b in current_bases]
  prev_bases = set(prev_player_obs[1].keys())
  new_bases = set(current_bases-prev_bases)
  spent_new_bases = len(new_bases)*convert_cost
  
  deposited = 0
  for k in not_destroyed_ships:
    if player_obs[2][k][0] in current_base_pos:
      deposited += prev_player_obs[2][k][1]
    
  prev_pos_to_k = {v[0]: k for k, v in prev_player_obs[2].items()}
  converted_ships = [prev_pos_to_k[player_obs[1][b]] for b in new_bases]
  for k in converted_ships:
    deposited += prev_player_obs[2][k][1]
    
  unexplained_score_change = (
    score_change-spent_spawns-spent_new_bases+deposited)
  
  if unexplained_score_change != 0:
    unexplained_ship_scores = [convert_cost-prev_player_obs[2][k][1] for k in (
      disappeared_ships)]
    num_disappeared = len(disappeared_ships)
    if num_disappeared < 7:
      # This is to avoid spending too much time in a crazy conversion scenario
      combination_found = False
      for comb in list_of_combs(np.arange(num_disappeared)):
        unexplained_sum = np.array(
          [unexplained_ship_scores[c_id] for c_id in comb]).sum()
        if unexplained_sum == unexplained_score_change:
          converted_ships.extend([disappeared_ships[c_id] for c_id in comb])
          combination_found = True
          break
      
      # One scenario where no combination can be found is when one of the ships
      # self-collides when returning to a base
      # if not combination_found:
      #   print("No convert resolution combination found", observation['step'],
      #         env_obs_id)
      combination_found = combination_found
  
  return converted_ships
  
def update_box_in_counter(history, observation, env_observation, stacked_ships,
                          env_obs_ids, env_config):
  grid_size = stacked_ships.shape[1]
  num_players = stacked_ships.shape[0]
  
  if observation['step'] in [0, env_config.episodeSteps-2]:
    history['raw_box_data'] = [[] for _ in range(num_players)]
    history['inferred_boxed_in_conv_threshold'] = [[
      env_config.convertCost/2, env_config.convertCost] for _ in range(
        num_players)]
  else:
    prev_opponent_sensible_actions = history['prev_step'][
      'opponent_ships_sensible_actions']
    for player_id in range(1, num_players):
      # Consider all boxed in ships and infer the action that each player took
      env_obs_id = env_obs_ids[player_id]
      player_obs = env_observation.players[env_obs_id]
      prev_player_obs = history['prev_step']['env_observation'].players[
          env_obs_id]
      converted_ships = infer_player_conversions(
        player_obs, prev_player_obs, env_config, observation,
        env_obs_id)
      
      for k in prev_player_obs[2]:
        row, col = row_col_from_square_grid_pos(
          prev_player_obs[2][k][0], grid_size)
        if len(prev_opponent_sensible_actions[row, col]) == 0:
          prev_halite = prev_player_obs[2][k][1]
          did_convert = k in converted_ships
          history['raw_box_data'][player_id].append((prev_halite, did_convert))
      
      # Infer the halite convert threshold when being boxed in
      if history['raw_box_data'][player_id]:
        box_data = np.array(history['raw_box_data'][player_id])
        current_thresholds = history['inferred_boxed_in_conv_threshold'][
          player_id]
        
        if np.any(
            (box_data[:, 1] == 0) & (box_data[:, 0] > current_thresholds[0])):
          history['inferred_boxed_in_conv_threshold'][player_id][0] = (
            box_data[box_data[:, 1] == 0, 0].max())
        
        for j in range(2):
          if np.any(
              (box_data[:, 1] == 1) & (box_data[:, 0] < current_thresholds[j])):
            history['inferred_boxed_in_conv_threshold'][player_id][j] = (
              box_data[box_data[:, 1] == 1, 0].min())
        
        if np.any(
            (box_data[:, 1] == 1) & (box_data[:, 0] < current_thresholds[1])):
          history['inferred_boxed_in_conv_threshold'][player_id][1] = (
            box_data[box_data[:, 1] == 1, 0].min())
          
  return history

def update_zero_halite_ship_behavior(
    history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config, near_base_distance=2):
  grid_size = stacked_ships.shape[1]
  num_players = stacked_ships.shape[0]
  
  # TODO (maybe): incorporate aggressive opponent 0 halite behavior at distance
  # 1 (attacking my zero halite ships).
  
  # Minimum number of required examples to be able to estimate the opponent's
  # zero halite ship behavior. Format ('nearbase_shipdistance')
  min_considered_types = {
    'False_1': 8,
    'False_2': 15,
    'True_1': 8,
    'True_2': 15,
      }
  
  if observation['step'] == 0:
    history['raw_zero_halite_move_data'] = [[] for _ in range(num_players)]
    history['zero_halite_move_behavior'] = [{} for _ in range(num_players)]
    
    initial_aggressive_behavior = {}
    for near_base in [False, True]:
      for considered_distance in [1, 2]:
        dict_k = str(near_base) + '_' + str(considered_distance)
        dict_k_careful = dict_k + '_careful'
        dict_k_real_count = dict_k + '_real_count'
        initial_aggressive_behavior[dict_k] = 1.0
        initial_aggressive_behavior[dict_k_careful] = False
        initial_aggressive_behavior[dict_k_real_count] = 0
    for player_id in range(1, num_players):
      history['zero_halite_move_behavior'][player_id] = (
        copy.copy(initial_aggressive_behavior))
    
  else:
    prev_stacked_bases = history['prev_step']['stacked_bases']
    all_prev_bases = prev_stacked_bases.sum(0) > 0
    prev_stacked_ships = history['prev_step']['stacked_ships']
    all_prev_ships = np.sum(prev_stacked_ships, 0) > 0
    prev_base_locations = np.where(all_prev_bases)
    num_prev_bases = all_prev_bases.sum()
    if num_prev_bases > 0:
      all_prev_base_distances = [DISTANCES[
        prev_base_locations[0][i], prev_base_locations[1][i]] for i in range(
          num_prev_bases)] + [
          99*np.ones((grid_size, grid_size))]
      stacked_prev_base_distances = np.stack(all_prev_base_distances)
      nearest_prev_base_distances = stacked_prev_base_distances.min(0)
      prev_base_player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
      for i in range(prev_stacked_bases.shape[0]):
        prev_base_player_ids[prev_stacked_bases[i]] = i
      prev_ship_player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
      for i in range(prev_stacked_ships.shape[0]):
        prev_ship_player_ids[prev_stacked_ships[i]] = i
      prev_opponent_sensible_actions = history['prev_step'][
        'opponent_ships_sensible_actions']
      for player_id in range(1, num_players):
        # Consider all boxed in ships and infer the action that each player
        # took
        env_obs_id = env_obs_ids[player_id]
        player_obs = env_observation.players[env_obs_id]
        prev_player_obs = history['prev_step']['env_observation'].players[
            env_obs_id]
        
        for k in prev_player_obs[2]:
          if k in player_obs[2] and prev_player_obs[2][k][1] == 0:
            prev_row, prev_col = row_col_from_square_grid_pos(
              prev_player_obs[2][k][0], grid_size)
            row, col = row_col_from_square_grid_pos(
              player_obs[2][k][0], grid_size)
            nearest_prev_base_distance = nearest_prev_base_distances[
              prev_row, prev_col]
            nearest_prev_base_id = np.argmin(stacked_prev_base_distances[
              :, prev_row, prev_col])
            nearest_prev_base_row = prev_base_locations[0][
              nearest_prev_base_id]
            nearest_prev_base_col = prev_base_locations[1][
              nearest_prev_base_id]
            nearest_base_player = prev_base_player_ids[
              nearest_prev_base_row, nearest_prev_base_col]
            friendly_prev_nearest_base = (nearest_base_player == player_id)
            
            if len(prev_opponent_sensible_actions[prev_row, prev_col]) < 5:
              # Loop over all zero halite opponent ships at a distance of max 2
              # and log the distance, None action count, move towards count and
              # move away count as well as the distance to the nearest base.
              # Also record whether the nearest base is friendly or not.
              considered_threat_data = []
              for row_shift in range(-2, 3):
                considered_row = (prev_row + row_shift) % grid_size
                for col_shift in range(-2, 3):
                  considered_col = (prev_col + col_shift) % grid_size
                  distance = np.abs(row_shift) + np.abs(col_shift)
                  if distance <= 2:
                    if all_prev_ships[considered_row, considered_col] and (
                        prev_ship_player_ids[
                          considered_row, considered_col] != player_id) and (
                            history['prev_step']['halite_ships'][
                              considered_row, considered_col] == 0):
                      
                      # Compute the distance of the considered ship, relative
                      # to the threat
                      moved_distance = DISTANCES[row, col][
                        considered_row, considered_col]
                      considered_threat_data.append((
                        distance, moved_distance, nearest_prev_base_distance,
                        friendly_prev_nearest_base, observation['step']))
                      
              # Aggregate the per-ship behavior - only consider the nearest
              # opponent threats
              num_considered_threats = len(considered_threat_data)
              if num_considered_threats == 1:
                history['raw_zero_halite_move_data'][player_id].append(
                  considered_threat_data[0])
              else:
                threat_data = np.array(considered_threat_data)
                min_distance = threat_data[:, 0].min()
                for row_id in range(num_considered_threats):
                  if threat_data[row_id, 0] == min_distance:
                    history['raw_zero_halite_move_data'][player_id].append(
                      considered_threat_data[row_id])
        
        # Infer the zero halite behavior as a function of distance to opponent
        # base and distance to other zero halite ships
        if history['raw_zero_halite_move_data'][player_id]:
          zero_halite_data = np.array(history['raw_zero_halite_move_data'][
            player_id])
          aggregate_data = {}
          for near_base in [False, True]:
            for considered_distance in [1, 2]:
              relevant_rows = (zero_halite_data[:, 0] == considered_distance)
              if near_base:
                relevant_rows &= (zero_halite_data[:, 2] <= near_base_distance)
              else:
                relevant_rows &= (zero_halite_data[:, 2] > near_base_distance)
              num_relevant = relevant_rows.sum()
              aggressive_relevant_count = (
                relevant_rows & (zero_halite_data[:, 1] <= 1)).sum()
              
              dict_k = str(near_base) + '_' + str(considered_distance)
              dict_k_careful = dict_k + '_careful'
              dict_k_real_count = dict_k + '_real_count'
              min_considered = min_considered_types[dict_k]
              num_aggressive_added = min_considered-num_relevant
              if num_aggressive_added > 0:
                num_aggressive_added = min_considered-num_relevant
                num_relevant += num_aggressive_added
                aggressive_relevant_count += num_aggressive_added
                
              aggregate_data[dict_k] = aggressive_relevant_count/num_relevant
              aggregate_data[dict_k_careful] = (
                aggressive_relevant_count == 0)
              aggregate_data[dict_k_real_count] = (
                num_relevant - num_aggressive_added)
                
          history['zero_halite_move_behavior'][player_id] = aggregate_data
          
  return history

def update_history_start_step(
    history, observation, env_observation, env_obs_ids, env_config):
  history_start_time = time.time()
  stacked_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])
  opponent_ships = stacked_ships[1:].sum(0) > 0
  other_halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']])[1:].sum(0)
  other_halite_ships[~opponent_ships] = 1e9
  grid_size = opponent_ships.shape[0]
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
  
  # Update the counter that keeps track of how long ships are chased
  history = update_chase_counter(
    history, observation, env_observation, stacked_ships, other_halite_ships,
    player_ids, env_obs_ids)
  
  # Update the data that keeps track of opponent behavior when being boxed in
  history = update_box_in_counter(
    history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config)
  
  # Update the data that keeps track of zero halite ship opponent behavior as a
  # function of opponent zero halite ships
  history = update_zero_halite_ship_behavior(
    history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config)
    
  return history, (time.time()-history_start_time)

def update_history_end_step(
    history, observation, ship_actions, opponent_ships_sensible_actions,
    ship_plans, player_obs, env_observation):
  none_included_ship_actions = {k: (ship_actions[k] if (
    k in ship_actions) else None) for k in player_obs[2]}
  stacked_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])
  stacked_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  
  history['prev_step'] = {
    'my_ship_actions': none_included_ship_actions,
    'opponent_ships_sensible_actions': opponent_ships_sensible_actions,
    'ship_plans': ship_plans,
    'env_observation': env_observation,
    'stacked_bases': stacked_bases,
    'stacked_ships': stacked_ships,
    'halite_ships': halite_ships,
    'observation': observation,
    }
  return history

def get_numpy_random_generator(
    config, observation, rng_action_seed, print_seed=False):
  if rng_action_seed is None:
    rng_action_seed = 0
  
  if observation['step'] == 0 and print_seed:
    print("Random acting seed: {}".format(rng_action_seed))
    
  # Add the observation step to the seed so we are less predictable
  step_seed = int(rng_action_seed+observation['step'])
  return np.random.RandomState(step_seed)

def get_config_actions(config, observation, player_obs, env_observation,
                       env_config, history, rng_action_seed, verbose=False):
  # Set the random seed
  np_rng = get_numpy_random_generator(
    config, observation, rng_action_seed, print_seed=True)
  
  # Obtain the ordered player ids (myself in the first position)
  env_obs_ids = get_env_obs_ids(env_observation)
  
  # Decide how many ships I can have attack bases aggressively
  steps_remaining = env_config.episodeSteps-1-observation['step']
  max_aggressive_attackers = int(len(player_obs[2]) - (3+0.25*steps_remaining))
  ignore_bad_attack_directions = max_aggressive_attackers > 0
  
  # Update the history based on what happened during the past observation
  history, history_start_duration = update_history_start_step(
    history, observation, env_observation, env_obs_ids, env_config)
  
  # Compute the ship scores for all high level actions
  (ship_scores, opponent_ships_sensible_actions, weighted_base_mask,
   opponent_ships_scaled, main_base_distances,
   ship_scores_duration) = get_ship_scores(
    config, observation, player_obs, env_config, np_rng,
    ignore_bad_attack_directions, history, env_obs_ids, env_observation,
    verbose)
  
  # Compute the coordinated high level ship plan
  (ship_plans, my_next_bases, plan_ship_scores, base_attackers,
   box_in_duration, history, ship_plans_duration) = get_ship_plans(
    config, observation, player_obs, env_config, verbose,
    copy.deepcopy(ship_scores), np_rng, weighted_base_mask, steps_remaining,
    opponent_ships_sensible_actions, opponent_ships_scaled,
    main_base_distances, history)
  
  # Translate the ship high level plans to basic move/convert actions
  (mapped_actions, remaining_budget, my_next_ships, my_next_halite,
   updated_ship_pos, action_overrides,
   ship_map_duration) = map_ship_plans_to_actions(
     config, observation, player_obs, env_observation, env_config, verbose,
     plan_ship_scores, ship_scores, ship_plans, np_rng,
     ignore_bad_attack_directions, base_attackers, steps_remaining,
     opponent_ships_sensible_actions, history, env_obs_ids,
     opponent_ships_scaled, main_base_distances)
  ship_actions = copy.copy(mapped_actions)
  
  # Decide for all bases whether to spawn or keep the base available
  base_actions, remaining_budget = decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships,
    my_next_halite, env_config, remaining_budget, verbose, ship_plans,
    updated_ship_pos, weighted_base_mask)
  
  # Add data to my history so I can update it appropriately at the beginning of
  # the next step.
  history = update_history_end_step(
    history, observation, ship_actions, opponent_ships_sensible_actions,
    ship_plans, player_obs, env_observation)
  
  mapped_actions.update(base_actions)
  
  return mapped_actions, history

def get_base_pos(base_data, grid_size):
  base_pos = np.zeros((grid_size, grid_size), dtype=np.bool)
  for _, v in base_data.items():
    row, col = row_col_from_square_grid_pos(v, grid_size)
    base_pos[row, col] = 1
  
  return base_pos

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


###############################################################################

HISTORY = {}
def my_agent(observation, env_config, **kwargs):
  global HISTORY
  rng_action_seed = kwargs.get('rng_action_seed', 0)
  active_id = observation.player
  current_observation = structured_env_obs(env_config, observation, active_id)
  player_obs = observation.players[active_id]
  
  mapped_actions, HISTORY = get_config_actions(
    CONFIG, current_observation, player_obs, observation, env_config, HISTORY,
    rng_action_seed)
     
  if LOCAL_MODE:
    # This is to allow for debugging of the history outside of the agent
    return mapped_actions, copy.deepcopy(HISTORY)
  else:
    return mapped_actions
