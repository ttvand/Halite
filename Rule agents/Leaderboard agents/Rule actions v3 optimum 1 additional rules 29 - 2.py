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
    'collect_smoothed_multiplier': 0.0,
    'collect_actual_multiplier': 5.0,
    'collect_less_halite_ships_multiplier_base': 0.55,
    'collect_base_nearest_distance_exponent': 0.2,
  
    'return_base_multiplier': 8.0,
    'return_base_less_halite_ships_multiplier_base': 0.85,
    'early_game_return_base_additional_multiplier': 0.1,
    'early_game_return_boost_step': 50,
    'establish_base_smoothed_multiplier': 0.0,
    
    'establish_first_base_smoothed_multiplier_correction': 2.0,
    'establish_base_dm_exponent': 1.1,
    'first_base_no_4_way_camping_spot_bonus': 300*0,
    'start_camp_if_not_winning': 0,
    'max_camper_ship_budget': 2*1,
    
    'relative_step_start_camping': 0.15,
    'establish_base_deposit_multiplier': 1.0,
    'establish_base_less_halite_ships_multiplier_base': 1.0,
    'max_attackers_per_base': 3*1,
    'attack_base_multiplier': 300.0,
    
    'attack_base_less_halite_ships_multiplier_base': 0.9,
    'attack_base_halite_sum_multiplier': 2.0,
    'attack_base_run_opponent_multiplier': 1.0,
    'attack_base_catch_opponent_multiplier': 1.0,
    'collect_run_opponent_multiplier': 10.0,
    
    'return_base_run_opponent_multiplier': 2.5,
    'establish_base_run_opponent_multiplier': 2.5,
    'collect_catch_opponent_multiplier': 1.0,
    'return_base_catch_opponent_multiplier': 1.0,
    'establish_base_catch_opponent_multiplier': 0.5,
    
    'two_step_avoid_boxed_opponent_multiplier_base': 0.7,
    'n_step_avoid_boxed_opponent_multiplier_base': 0.45,
    'min_consecutive_chase_extrapolate': 6,
    'chase_return_base_exponential_bonus': 2.0,
    'ignore_catch_prob': 0.3,
    
    'max_initial_ships': 60,
    'max_final_ships': 60,
    'max_standard_ships_decided_end_pack_hunting': 2,
    'nearby_ship_halite_spawn_constant': 3.0,
    'nearby_halite_spawn_constant': 5.0,
    
    'remaining_budget_spawn_constant': 0.2,
    'spawn_score_threshold': 75.0,
    'boxed_in_halite_convert_divisor': 1.0,
    'n_step_avoid_min_die_prob_cutoff': 0.05,
    'n_step_avoid_window_size': 7,
    
    'influence_map_base_weight': 2.0,
    'influence_map_min_ship_weight': 0.0,
    'influence_weights_additional_multiplier': 2.0,
    'influence_weights_exponent': 8.0,
    'escape_influence_prob_divisor': 3.0,
    
    'rescue_ships_in_trouble': 1,
    'target_strategic_base_distance': 8.0,
    'target_strategic_num_bases_ship_divisor': 9,
    'target_strategic_triangle_weight': 20.0,  # initially: 20
    'target_strategic_independent_base_distance_multiplier': 8.0,  # initially 8.0
    
    'target_strategic_influence_desirability_multiplier': 1.0,  # initially: 1.0
    'target_strategic_potential_divisor': 15.0,  # initially: 15.0
    'max_spawn_relative_step_divisor': 12.0,
    'no_spawn_near_base_ship_limit': 100,
    'avoid_cycles': 1,
    
    'max_risk_n_step_risky': 0.5,
    'max_steps_n_step_risky': 70,
    'log_near_base_distance': 2,
    'max_recent_considered_relevant_zero_move_count': 120,
    'near_base_2_step_risky_min_count': 50,
    
    'relative_stand_still_collect_boost': 1.5,
    'initial_collect_boost_away_from_base': 2.0,
    'start_hunting_season_relative_step': 0.1875,
    'end_hunting_season_relative_step': 0.75,
    'early_hunting_season_less_collect_relative_step': 0.375,
    
    'max_standard_ships_early_hunting_season': 2,
    'late_hunting_season_more_collect_relative_step': 0.5,
    'late_hunting_season_collect_max_n_step_risk': 0.2,
    'after_hunting_season_collect_max_n_step_risk': 0.5,
    'late_hunting_season_standard_min_fraction': 0.7,
    
    'max_standard_ships_late_hunting_season': 15,
    'collect_on_safe_return_relative_step': 0.075,
    'min_halite_to_stop_early_hunt': 15000.0,
    'early_best_opponent_relative_step': 0.5,
    'surrounding_ships_cycle_extrapolate_step_count': 5,
    
    'surrounding_ships_extended_cycle_extrapolate_step_count': 7,
    }


NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
CONVERT = "CONVERT"
SPAWN = "SPAWN"
NOT_NONE_DIRECTIONS = [NORTH, SOUTH, EAST, WEST]
MOVE_DIRECTIONS = [None, NORTH, SOUTH, EAST, WEST]
MOVE_DIRECTIONS_TO_ID = {None: 0, NORTH: 1, SOUTH: 2, EAST: 3, WEST: 4}
RELATIVE_DIR_MAPPING = {None: (0, 0), NORTH: (-1, 0), SOUTH: (1, 0),
                        EAST: (0, 1), WEST: (0, -1)}
RELATIVE_DIR_TO_DIRECTION_MAPPING = {
  v: k for k, v in RELATIVE_DIR_MAPPING.items()}
OPPOSITE_MAPPING = {None: None, NORTH: SOUTH, SOUTH: NORTH, EAST: WEST,
                    WEST: EAST}
RELATIVE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
RELATIVE_NOT_NONE_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
ROW_COL_BOX_DIR_MAX_DISTANCE_MASKS = {}
BOX_DIR_MAX_DISTANCE = 4
BOX_DIRECTION_MASKS = {}
ROW_MASK = {}
COLUMN_MASK = {}
DISTANCE_MASK_DIM = 21
half_distance_mask_dim = int(DISTANCE_MASK_DIM/2)
for row in range(DISTANCE_MASK_DIM):
  row_mask = np.zeros((DISTANCE_MASK_DIM, DISTANCE_MASK_DIM), dtype=np.bool)
  row_mask[row] = 1
  col_mask = np.zeros((DISTANCE_MASK_DIM, DISTANCE_MASK_DIM), dtype=np.bool)
  col_mask[:, row] = 1
  ROW_MASK [row] = row_mask
  COLUMN_MASK[row] = col_mask
  for col in range(DISTANCE_MASK_DIM):
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
        box_dir_rows = np.mod(row + np.arange(BOX_DIR_MAX_DISTANCE) + 1,
                              DISTANCE_MASK_DIM)
        box_dir_cols = np.mod(col + np.arange(
          2*(BOX_DIR_MAX_DISTANCE+1)-1) - BOX_DIR_MAX_DISTANCE,
          DISTANCE_MASK_DIM)
      if d == SOUTH:
        catch_rows = np.mod(row + np.arange(half_distance_mask_dim) + 1,
                            DISTANCE_MASK_DIM)
        catch_cols = np.arange(DISTANCE_MASK_DIM)
        box_dir_rows = np.mod(row - np.arange(BOX_DIR_MAX_DISTANCE) - 1,
                              DISTANCE_MASK_DIM)
        box_dir_cols = np.mod(col + np.arange(
          2*(BOX_DIR_MAX_DISTANCE+1)-1) - BOX_DIR_MAX_DISTANCE,
          DISTANCE_MASK_DIM)
      if d == WEST:
        catch_cols = np.mod(col - np.arange(half_distance_mask_dim) - 1,
                            DISTANCE_MASK_DIM)
        catch_rows = np.arange(DISTANCE_MASK_DIM)
        box_dir_cols = np.mod(col + np.arange(BOX_DIR_MAX_DISTANCE) + 1,
                              DISTANCE_MASK_DIM)
        box_dir_rows = np.mod(row + np.arange(
          2*(BOX_DIR_MAX_DISTANCE+1)-1) - BOX_DIR_MAX_DISTANCE,
          DISTANCE_MASK_DIM)
      if d == EAST:
        catch_cols = np.mod(col + np.arange(half_distance_mask_dim) + 1,
                            DISTANCE_MASK_DIM)
        catch_rows = np.arange(DISTANCE_MASK_DIM)
        box_dir_cols = np.mod(col - np.arange(BOX_DIR_MAX_DISTANCE) - 1,
                              DISTANCE_MASK_DIM)
        box_dir_rows = np.mod(row + np.arange(
          2*(BOX_DIR_MAX_DISTANCE+1)-1) - BOX_DIR_MAX_DISTANCE,
          DISTANCE_MASK_DIM)
        
      catch_mask = np.zeros((DISTANCE_MASK_DIM, DISTANCE_MASK_DIM),
                            dtype=np.bool)
      
      catch_mask[catch_rows[:, None], catch_cols] = 1
      run_mask = np.copy(catch_mask)
      run_mask[row, col] = 1
      
      catch_distance_masks[d] = catch_mask
      run_distance_masks[d] = run_mask
      
      if d is not None:
        box_dir_mask = np.zeros((DISTANCE_MASK_DIM, DISTANCE_MASK_DIM),
                                dtype=np.bool)
        box_dir_mask[box_dir_rows[:, None], box_dir_cols] = 1
        if d in [NORTH, SOUTH]:
          box_dir_mask &= (horiz_distance <= vert_distance)
        else:
          box_dir_mask &= (horiz_distance >= vert_distance)
        ROW_COL_BOX_DIR_MAX_DISTANCE_MASKS[(row, col, d)] = box_dir_mask
    
    HALF_PLANES_CATCH[(row, col)] = catch_distance_masks
    HALF_PLANES_RUN[(row, col)] = run_distance_masks

    for d in range(1, DISTANCE_MASK_DIM):
      ROW_COL_DISTANCE_MASKS[(row, col, d)] = manh_distance == d

    for d in range(half_distance_mask_dim):
      ROW_COL_MAX_DISTANCE_MASKS[(row, col, d)] = manh_distance <= d
      ROW_COL_BOX_MAX_DISTANCE_MASKS[(row, col, d)] = np.logical_and(
        horiz_distance <= d, vert_distance <= d)
      
    for dist in range(2, half_distance_mask_dim+1):
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
        
CONSIDERED_OTHER_DISTANCES = [13]
OTHER_DISTANCES = {}
for other_distance in CONSIDERED_OTHER_DISTANCES:
  for row in range(other_distance):
    for col in range(other_distance):
      horiz_distance = np.minimum(
        np.abs(np.arange(other_distance) - col),
        np.abs(np.arange(other_distance) - col - other_distance))
      horiz_distance = np.minimum(
        horiz_distance,
        np.abs(np.arange(other_distance) - col + other_distance))
      
      vert_distance = np.minimum(
        np.abs(np.arange(other_distance) - row),
        np.abs(np.arange(other_distance) - row - other_distance))
      vert_distance = np.minimum(
        vert_distance,
        np.abs(np.arange(other_distance) - row + other_distance))
      
      horiz_distance = np.tile(horiz_distance, [other_distance, 1])
      vert_distance = np.tile(np.expand_dims(vert_distance, 1),
                              [1, other_distance])
      manh_distance = horiz_distance + vert_distance
      
      OTHER_DISTANCES[(row, col, other_distance)] = manh_distance
      
D2_ROW_COL_SHIFTS_DISTANCES = [
  (-2, 0, 2),
  (-1, -1, 2), (-1, 0, 1), (-1, 1, 2),
  (0, -2, 2), (0, -1, 1), (0, 1, 1), (0, 2, 2),
  (1, -1, 2), (1, 0, 1), (1, 1, 2),
  (2, 0, 2),
  ]
      
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
  
def get_directional_distance(r1, c1, r2, c2, size, d):
  relative_pos = get_relative_position(r1, c1, r2, c2, size)
  if d == NORTH:
    directional_distance = -relative_pos[0]
  elif d == SOUTH:
    directional_distance = relative_pos[0]
  elif d == EAST:
    directional_distance = relative_pos[1]
  elif d == WEST:
    directional_distance = -relative_pos[1]
    
  return directional_distance
  
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

def update_scores_opponent_ships(
    config, collect_grid_scores, return_to_base_scores, establish_base_scores,
    attack_base_scores, opponent_ships, opponent_bases, halite_ships, row, col,
    grid_size, spawn_cost, drop_None_valid, obs_halite, collect_rate, np_rng,
    opponent_ships_sensible_actions, opponent_ships_sensible_actions_no_risk,
    ignore_bad_attack_directions, observation, ship_k, my_bases, my_ships,
    steps_remaining, history, escape_influence_probs, player_ids, env_obs_ids,
    env_observation, main_base_distances, nearest_base_distances,
    end_game_base_return, camping_override_strategy,
    attack_campers_override_strategy, boxed_in_attack_squares,
    safe_to_collect, boxed_in_zero_halite_opponents, ignore_convert_positions,
    avoid_attack_squares_zero_halite, n_step_avoid_min_die_prob_cutoff,
    safe_to_return_halites, safe_to_return_base_halites,
    my_nearest_base_distances):
  direction_halite_diff_distance_raw = {
    NORTH: [], SOUTH: [], EAST: [], WEST: []}
  my_bases_or_ships = np.logical_or(my_bases, my_ships)
  chase_details = history['chase_counter'][0].get(ship_k, None)
  take_my_square_next_halite_diff = None
  take_my_next_square_dir = None
  wide_cycle_mask = ROW_COL_MAX_DISTANCE_MASKS[row, col, 3]
  tight_cycle_mask = ROW_COL_MAX_DISTANCE_MASKS[row, col, 2]
  opponents_in_cycle = np.any(opponent_ships[tight_cycle_mask]) and (
    np.all(history['empty_or_cycled_positions'][wide_cycle_mask]) or (
      np.all(history['empty_or_extended_cycled_positions'][tight_cycle_mask])))
  if opponents_in_cycle:
    print("EXTRAPOLATING OPPONENT CYCLIC BEHAVIOR", observation['step'], row,
          col)
  if len(camping_override_strategy) == 0:
    navigation_zero_halite_risk_threshold = 0
  else:
    navigation_zero_halite_risk_threshold = camping_override_strategy[0]
    if camping_override_strategy[1].max() >= 1e4:
      collect_grid_scores = 1e-4*collect_grid_scores + (
        camping_override_strategy[1])
    else:
      collect_grid_scores += camping_override_strategy[1]
    attack_base_scores += camping_override_strategy[2]
    
  if len(attack_campers_override_strategy) > 0:
    ignore_opponent_row = attack_campers_override_strategy[0]
    ignore_opponent_col = attack_campers_override_strategy[1]
    ignore_opponent_distance = attack_campers_override_strategy[5]
    collect_grid_scores[ignore_opponent_row, ignore_opponent_col] += (
      attack_campers_override_strategy[2])
    navigation_zero_halite_risk_threshold = max(
      navigation_zero_halite_risk_threshold,
      attack_campers_override_strategy[6])
  else:
    ignore_opponent_row = None
    ignore_opponent_col = None
    ignore_opponent_distance = None
    
  # Identify directions where I can certainly reach the base in time and always
  # mark them as valid
  ship_halite = halite_ships[row, col]
  safe_return_base_directions = []
  if ship_halite < safe_to_return_halites[row, col]:
    for base_safe_return_halite, base_location in safe_to_return_base_halites:
      if ship_halite < base_safe_return_halite[row, col]:
        for d in get_dir_from_target(
            row, col, base_location[0], base_location[1], grid_size):
          if not d is None and not d in safe_return_base_directions:
            safe_return_base_directions.append(d)
    
  # if observation['step'] == 131 and ship_k in ['63-1']:
  #   import pdb; pdb.set_trace()
    
  can_stay_still_zero_halite = True
  for row_shift, col_shift, distance in D2_ROW_COL_SHIFTS_DISTANCES:
    considered_row = (row + row_shift) % grid_size
    considered_col = (col + col_shift) % grid_size
    if opponent_ships[considered_row, considered_col] and (
        ignore_opponent_row is None or (((
          considered_row != ignore_opponent_row) or (
            considered_col != ignore_opponent_col)) and (
              ignore_opponent_distance > 2))):
                
      relevant_dirs = []
      halite_diff = halite_ships[row, col] - halite_ships[
        considered_row, considered_col]
      assume_take_my_square_next = False
      
      # if observation['step'] == 266 and row == 11 and col == 15:
      #   import pdb; pdb.set_trace()
      
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
          opponent_ships_sensible_actions_no_risk[chaser_row, chaser_col])
        
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
      
      # if observation['step'] == 96 and ship_k in ['80-1']:
      #   import pdb; pdb.set_trace()
      
      can_ignore_ship = False
      if (considered_row, considered_col) in boxed_in_zero_halite_opponents:
        can_stay_still_zero_halite = can_stay_still_zero_halite and (
          distance == 2)
      else:
        if halite_ships[row, col] == halite_ships[
            considered_row, considered_col]:
          opponent_id = player_ids[considered_row, considered_col]
          # Note: use the opponent distance because the opponent model is
          # learned using the opponent distance to the nearest base (with near
          # base distance cutoff typically at 2)
          is_near_base = nearest_base_distances[
            considered_row, considered_col] <= config['log_near_base_distance']
          risk_lookup_k = str(is_near_base) + '_' + str(distance)
          if distance == 2:
            can_ignore_ship = history['zero_halite_move_behavior'][
              opponent_id][risk_lookup_k] <= (
                navigation_zero_halite_risk_threshold)
          else:
            risk_lookup_k_dist_zero = str(is_near_base) + '_' + str(0)
            d1_threat = history['zero_halite_move_behavior'][
              opponent_id][risk_lookup_k] > (
                navigation_zero_halite_risk_threshold)
            d0_threat = history['zero_halite_move_behavior'][
              opponent_id][risk_lookup_k_dist_zero] > (
                navigation_zero_halite_risk_threshold)
            can_stay_still_zero_halite = can_stay_still_zero_halite and (
              not d0_threat)
            # if is_near_base and history['zero_halite_move_behavior'][
            #   opponent_id][str(is_near_base) + '_' + str(0) + '_ever_risky']:
            #   import pdb; pdb.set_trace()
            can_ignore_ship = not (d0_threat or d1_threat)
      
      if not assume_take_my_square_next and not can_ignore_ship:
        relevant_dirs += [] if row_shift >= 0 else [NORTH]
        relevant_dirs += [] if row_shift <= 0 else [SOUTH]
        relevant_dirs += [] if col_shift <= 0 else [EAST]
        relevant_dirs += [] if col_shift >= 0 else [WEST]
      
      # When the opponents are in a cycle: only consider the direction I
      # expect my opponent to be at in the next step (if any)
      if opponents_in_cycle:
        relevant_dirs = []
        opponent_ship_key = history['opponent_ship_pos_to_key'][(
          considered_row, considered_col)]
        opponent_id = player_ids[considered_row, considered_col]
        likely_opponent_action = history['opponent_cycle_counters'][
          opponent_id-1][opponent_ship_key][1][0]
        likely_opponent_next_pos = move_ship_row_col(
          considered_row, considered_col, likely_opponent_action, grid_size)
        relative_other_pos = get_relative_position(
          row, col, likely_opponent_next_pos[0], likely_opponent_next_pos[1],
          grid_size)
        current_opp_relative_dir = get_relative_position(
          row, col, considered_row, considered_col, grid_size)
        if np.abs(relative_other_pos[0]) + np.abs(
            relative_other_pos[1]) <= 1:
          # At distance 1 or 0
          # import pdb; pdb.set_trace()
          if relative_other_pos[0] == 0 and relative_other_pos[1] == 0:
            relevant_dirs = [RELATIVE_DIR_TO_DIRECTION_MAPPING[
              current_opp_relative_dir]]
          elif relative_other_pos == (0, 0):
            relevant_dirs = [RELATIVE_DIR_TO_DIRECTION_MAPPING[
              relative_other_pos]]
      
      # if observation['step'] == 215 and ship_k == '2-2':
      #   import pdb; pdb.set_trace()
      
      for d in relevant_dirs:
        direction_halite_diff_distance_raw[d].append(
          (halite_diff, distance))
            
  direction_halite_diff_distance = {}
  for d in direction_halite_diff_distance_raw:
    vals = np.array(direction_halite_diff_distance_raw[d])
    if vals.size:
      diffs = vals[:, 0]
      distances = vals[:, 1]
      max_diff = diffs.max()
      if max_diff > 0:
        if can_stay_still_zero_halite:
          greater_min_distance = distances[diffs > 0].min()
        else:
          # My halite is > 0 and I have a threat at D1 of an aggressive equal
          # halite ships and a threat of a less halite ship at D2
          greater_min_distance = distances[diffs >= 0].min()
        direction_halite_diff_distance[d] = (max_diff, greater_min_distance)
      elif max_diff == 0:
        equal_min_distance = distances[diffs == 0].min()
        direction_halite_diff_distance[d] = (max_diff, equal_min_distance)
      else:
        min_diff = diffs.min()
        min_diff_min_distance = distances[diffs == min_diff].min()
        direction_halite_diff_distance[d] = (min_diff, min_diff_min_distance)
    else:
      direction_halite_diff_distance[d] = None
                
  preferred_directions = []
  strongly_preferred_directions = []
  valid_directions = copy.copy(MOVE_DIRECTIONS)
  one_step_valid_directions = copy.copy(MOVE_DIRECTIONS)
  bad_directions = []
  ignore_catch = np_rng.uniform() < config['ignore_catch_prob']
  
  # if observation['step'] == 221 and ship_k == '54-1':
  #   import pdb; pdb.set_trace()
  #   x=1
  
  for direction, halite_diff_dist in direction_halite_diff_distance.items():
    if halite_diff_dist is not None:
      move_row, move_col = move_ship_row_col(row, col, direction, grid_size)
      no_escape_bonus = 0 if not (
        boxed_in_attack_squares[move_row, move_col]) else 5e3
      halite_diff = halite_diff_dist[0]
      if halite_diff >= 0:
        # I should avoid a collision
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = np.copy(HALF_PLANES_RUN[(row, col)][direction])
        valid_directions.remove(direction)
        one_step_valid_directions.remove(direction)
        bad_directions.append(direction)
        if halite_diff_dist[1] == 1:
          if halite_diff > 0 or not can_stay_still_zero_halite:
            # Only suppress the stay still action if the opponent has something
            # to gain.
            # Exception: the opponent may aggressively attack my zero halite
            # ships
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
          config['collect_run_opponent_multiplier'])*distance_multiplier
        return_to_base_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
          config['return_base_run_opponent_multiplier'])
        base_nearby_in_direction_mask = np.logical_and(
          ROW_COL_MAX_DISTANCE_MASKS[(row, col, 2)], mask_collect_return)
        base_nearby_in_direction = np.logical_and(
          base_nearby_in_direction_mask, opponent_bases).sum() > 0
        if not ignore_bad_attack_directions and not base_nearby_in_direction:
          attack_base_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
            config['attack_base_run_opponent_multiplier'])
        mask_establish = np.copy(mask_collect_return)
        mask_establish[row, col] = False
        establish_base_scores -= mask_establish*(ship_halite+spawn_cost)*(
          config['establish_base_run_opponent_multiplier'])
        
      elif halite_diff < 0 and (
          not ignore_catch or no_escape_bonus > 0) and (not (
            move_row, move_col) in ignore_convert_positions):
        # I would like a collision unless if there is another opponent ship
        # chasing me - risk avoiding policy for now: if there is at least
        # one ship in a direction that has less halite, I should avoid it
        if no_escape_bonus > 0:
          halite_diff = max(-spawn_cost/2, halite_diff) - no_escape_bonus
        else:
          halite_diff = 0  # Dubious choice, likely not very important
          # halite_diff = max(-spawn_cost/2, halite_diff) - no_escape_bonus
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = np.copy(HALF_PLANES_CATCH[(row, col)][direction])
        collect_grid_scores -= mask_collect_return*halite_diff*(
          config['collect_catch_opponent_multiplier'])*distance_multiplier
        return_to_base_scores -= mask_collect_return*halite_diff*(
          config['return_base_catch_opponent_multiplier'])*distance_multiplier
        attack_base_scores -= mask_collect_return*halite_diff*(
          config['attack_base_catch_opponent_multiplier'])*distance_multiplier
        mask_establish = np.copy(mask_collect_return)
        mask_establish[row, col] = False
        establish_base_scores -= mask_establish*halite_diff*(
          config['establish_base_catch_opponent_multiplier'])*(
            distance_multiplier)
        
        if no_escape_bonus > 0:
          strongly_preferred_directions.append(direction)
        
        if boxed_in_attack_squares[row, col] and no_escape_bonus > 0 and (
            ship_halite > 0 or obs_halite[row, col] == 0):
          # Also incentivize the None action when it is a possible escape
          # square of an opponent - divide by 2 to make the None action less
          # dominant (likely check in several directions)
          collect_grid_scores[row, col] += no_escape_bonus/2
          if not None in strongly_preferred_directions:
            strongly_preferred_directions.append(None)
        
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
  n_step_bad_directions = []
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
      if num_threat_ships > 1 and not d in safe_return_base_directions:
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
        
        # if observation['step'] == 112 and ship_k == '76-1':
        #   import pdb; pdb.set_trace()
        
        # Aggregate the threat count in all_dir_threat_counter
        threat_counts = np.array(list(all_dir_threat_counter.values()))
        threat_score = np.sqrt(threat_counts.prod())
        if threat_score > 0:
          # Disincentivize an action that can get me boxed in on the next step
          mask_avoid_two_steps = np.copy(HALF_PLANES_RUN[(row, col)][d])
          if d is not None:
            mask_avoid_two_steps[row, col] = False
          collect_grid_scores[mask_avoid_two_steps] *= ((
            config['two_step_avoid_boxed_opponent_multiplier_base']) ** (
              threat_score))
          return_to_base_scores[mask_avoid_two_steps] *= ((
            config['two_step_avoid_boxed_opponent_multiplier_base']) ** (
              threat_score))
          establish_base_scores[mask_avoid_two_steps] *= ((
            config['two_step_avoid_boxed_opponent_multiplier_base']) ** (
              threat_score))
          
          two_step_bad_directions.append(d)
          
      if d not in two_step_bad_directions and not end_game_base_return and (
          my_next_halite > 0) and (not d in safe_return_base_directions) and (
            d is not None or not safe_to_collect[row, col]):
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
          lt_catch_prob = {k: [] for k in RELATIVE_NOT_NONE_DIRECTIONS}
          for i in range(num_threat_ships_lt):
            other_row = less_halite_threat_opponents_lt[0][i]
            other_col = less_halite_threat_opponents_lt[1][i]
            other_sensible_actions = opponent_ships_sensible_actions[
              other_row, other_col]
            relative_other_pos = get_relative_position(
              move_row, move_col, other_row, other_col, grid_size)
            
            # Give less weight to the other ship if there is a base of mine or
            # a/multiple less halite ships in between
            # FUTURE WORK: Also give additional move leeway if I have nearby
            # bases? Especially relevant for None (collect) actions
            distance_move_other = np.abs(relative_other_pos).sum()
            mask_between_move_and_threat = np.logical_and(
                DISTANCES[(move_row, move_col)] < distance_move_other,
                DISTANCES[(other_row, other_col)] < distance_move_other)
            less_halite_ship_base_count = np.logical_and(
              np.logical_and(my_bases_or_ships, mask_between_move_and_threat),
              halite_ships <= halite_ships[other_row, other_col]).sum() + 0*(
                my_bases[ROW_COL_MAX_DISTANCE_MASKS[
                  move_row, move_col, 2]].sum())
            my_material_defense_multiplier = 2**less_halite_ship_base_count
            
            for threat_dir in RELATIVE_NOT_NONE_DIRECTIONS:
              nz_dim = int(threat_dir[0] == 0)
              dir_offset = relative_other_pos[nz_dim]*threat_dir[nz_dim]
              other_dir_abs_offset = np.abs(relative_other_pos[1-nz_dim])
              
              # if observation['step'] == 155 and ship_k == '63-2':
              #   import pdb; pdb.set_trace()
              
              if dir_offset >= 0 and (other_dir_abs_offset-1) <= dir_offset:
                # Ignore the threat if the ship is on the diagonal and can not
                # move in the direction of the threat dir
                if (other_dir_abs_offset-1) == dir_offset and len(
                    other_sensible_actions) < len(MOVE_DIRECTIONS):
                  if nz_dim == 0:
                    threat_other_dir = (
                      0, 1 if relative_other_pos[1-nz_dim] < 0 else -1)
                  else:
                    threat_other_dir = (
                      1 if relative_other_pos[1-nz_dim] < 0 else -1, 0)
                  threat_other_dirs = [threat_other_dir, threat_dir]
                  threats_actionable = np.array([
                    t in other_sensible_actions for t in threat_other_dirs])
                  consider_this_threat = np.any(threats_actionable)
                  if threats_actionable[1] and not threats_actionable[0]:
                    # Lower the threat weight - the opponent can not directly
                    # attack the considered threat direction and can only move
                    # along the threat direction
                    other_dir_abs_offset += 2
                else:
                  consider_this_threat = True
                
                if other_dir_abs_offset == 0 and dir_offset == 0:
                  # The scenario where a one step threat is ignored due to 
                  # being chased for a while and moving to the threat is
                  # currently considered.
                  # This avoids division by zero but is overridden later anyway
                  other_dir_abs_offset = 2
                
                if consider_this_threat:
                  lt_catch_prob[threat_dir].append(max(2,
                    other_dir_abs_offset+dir_offset)*(
                      my_material_defense_multiplier))
                    
          # Add a "bootstrapped" catch probability using the density of the
          # players towards the edge of the threat direction
          # Only add it if the next halite is > 0 (otherwise assume I can
          # always escape)
          # Also factor in the distance to my nearest non abandoned base
          if my_next_halite > 0:
            current_nearest_base_distance = my_nearest_base_distances[row, col]
            moved_nearest_base_distance = my_nearest_base_distances[
              move_row, move_col]
            move_distance_difference = current_nearest_base_distance - (
              moved_nearest_base_distance)
            for threat_dir in RELATIVE_NOT_NONE_DIRECTIONS:
              dens_threat_rows = np.mod(move_row + threat_dir[0]*(
                np.arange(config['n_step_avoid_window_size']//2,
                          config['n_step_avoid_window_size'])), grid_size)
              dens_threat_cols = np.mod(move_col + threat_dir[1]*(
                1+np.arange(config['n_step_avoid_window_size']//2,
                            config['n_step_avoid_window_size'])), grid_size)
              escape_probs = escape_influence_probs[
                dens_threat_rows, dens_threat_cols]
              
              
              mean_escape_prob = escape_probs.mean()
              if escape_probs[:2].min() < 1:
                if move_distance_difference > 0:
                  # When in trouble, it is typically better to move towards one
                  # of my bases. The move closer distance is of course 1.
                  mean_escape_prob *= 1.25
                if mean_escape_prob < 1:
                  lt_catch_prob[threat_dir].append(1/(1-mean_escape_prob+1e-9))
            
          # if observation['step'] == 75 and ship_k == '64-1' and d in [
          #     EAST, WEST]:
          #   import pdb; pdb.set_trace()
            
          # if observation['step'] == 112 and ship_k == '76-1':
          #   import pdb; pdb.set_trace()
          if np.all([len(v) > 0 for v in lt_catch_prob.values()]):
            # Interpretation: for a threat at distance d, I have a probability
            # of surviving it of (d-1)/d. The probability of surviving all
            # threat is the product of all individual threats
            survive_probs = np.array([
              (np.maximum(0.2, (np.array(lt_catch_prob[k])-1)/np.array(
                lt_catch_prob[k]))).prod() for k in lt_catch_prob])
            min_die_prob = 1-survive_probs.max()
            if main_base_distances.max() > 0:
              if main_base_distances[move_row, move_col] <= 2:
                min_die_prob = 0
              else:
                min_die_prob = max(
                  0, min_die_prob-0.33**main_base_distances[
                    move_row, move_col])
            
            # if observation['step'] == 155 and ship_k in ['63-2', '63-1']:
            #   import pdb; pdb.set_trace()
            
            # Disincentivize an action that can get me boxed in during the next
            # N steps
            mask_avoid_n_steps = np.copy(HALF_PLANES_RUN[(row, col)][d])
            if d is not None:
              mask_avoid_n_steps[row, col] = False
            collect_grid_scores[mask_avoid_n_steps] *= ((
              config['n_step_avoid_boxed_opponent_multiplier_base']) ** (
                min_die_prob))
            return_to_base_scores[mask_avoid_n_steps] *= (
              config['n_step_avoid_boxed_opponent_multiplier_base']) ** (
                min_die_prob)
            establish_base_scores[mask_avoid_n_steps] *= (
              config['n_step_avoid_boxed_opponent_multiplier_base']) ** (
                min_die_prob)
                
            n_step_bad_directions_die_probs[d] = min_die_prob
            
            # Correction to act with more risk towards the end of the game
            die_prob_cutoff = (n_step_avoid_min_die_prob_cutoff + 0.01*max(
                  0, 50-steps_remaining))
            if d is None:
              if observation['relative_step'] > config[
                'end_hunting_season_relative_step']:
                die_prob_cutoff = max(die_prob_cutoff, config[
                  'after_hunting_season_collect_max_n_step_risk'])
              elif observation['relative_step'] > config[
                'late_hunting_season_more_collect_relative_step']:
                die_prob_cutoff = max(die_prob_cutoff, config[
                  'late_hunting_season_collect_max_n_step_risk'])
            # print(observation['step'], die_prob_cutoff)
            if min_die_prob > die_prob_cutoff:
              n_step_bad_directions.append(d)
              
  # if observation['step'] == 215 and ship_k == '2-2':
  #   import pdb; pdb.set_trace()
              
  # Corner case: if I have a zero halite ship that is boxed in by other zero
  # halite ships on a zero halite square: compute the risk for all available
  # actions and only retain the actions with the lowest collision risks
  if halite_ships[row, col] == 0 and len(valid_directions) == 0 and (
      obs_halite[row, col] == 0):
    risk_scores = np.zeros(len(MOVE_DIRECTIONS))
    for risk_id, d in enumerate(MOVE_DIRECTIONS):
      move_row, move_col = move_ship_row_col(row, col, d, grid_size)
      for potential_threat_dir in MOVE_DIRECTIONS:
        threat_row, threat_col = move_ship_row_col(
          move_row, move_col, potential_threat_dir, grid_size)
        if opponent_ships[threat_row, threat_col] and halite_ships[
            threat_row, threat_col] == 0:
          opponent_id = player_ids[threat_row, threat_col]
          is_near_base = nearest_base_distances[
            threat_row, threat_col] <= config['log_near_base_distance']
          distance = int(d is not None) + int(potential_threat_dir is not None)
          risk_lookup_k = str(is_near_base) + '_' + str(distance)
          risk_scores[risk_id] = max(
            risk_scores[risk_id], history['zero_halite_move_behavior'][
              opponent_id][risk_lookup_k])
    
    best_risk_score = risk_scores.min()
    if best_risk_score < 0.05:
      valid_directions = [d for d_id, d in enumerate(
        MOVE_DIRECTIONS) if risk_scores[d_id] == best_risk_score]
    else:
      valid_directions = [None]
      
    one_step_valid_directions = copy.copy(valid_directions)
    bad_directions = list(set(MOVE_DIRECTIONS) - set(valid_directions))
         
  # if observation['step'] == 169 and ship_k == '65-2':
  #   import pdb; pdb.set_trace()
    
  # Corner case: if I have a zero halite ship that is boxed in by other zero
  # halite ships on a non-zero halite square: prefer moving in directions where
  # there is a lower risk of losing the ship as a function of opponent zero
  # halite behavior
  if halite_ships[row, col] == 0 and obs_halite[row, col] > 0 and (
      (len(valid_directions) == 1 and (valid_directions[0] is None)) or (
        len(valid_directions) == 0)):
    
    risk_scores = np.zeros(len(MOVE_DIRECTIONS))
    risk_scores[0] = 1 # Definitely don't stand still
    for risk_id, d in enumerate(MOVE_DIRECTIONS):
      move_row, move_col = move_ship_row_col(row, col, d, grid_size)
      for potential_threat_dir in MOVE_DIRECTIONS:
        threat_row, threat_col = move_ship_row_col(
          move_row, move_col, potential_threat_dir, grid_size)
        if opponent_ships[threat_row, threat_col] and halite_ships[
            threat_row, threat_col] == 0:
          opponent_id = player_ids[threat_row, threat_col]
          is_near_base = nearest_base_distances[
            threat_row, threat_col] <= config['log_near_base_distance']
          distance = int(d is not None) + int(potential_threat_dir is not None)
          risk_lookup_k = str(is_near_base) + '_' + str(distance)
          risk_scores[risk_id] = max(
            risk_scores[risk_id], history['zero_halite_move_behavior'][
              opponent_id][risk_lookup_k])
    
    best_risk_score = risk_scores.min()
    valid_directions = [d for d_id, d in enumerate(
        MOVE_DIRECTIONS) if risk_scores[d_id] == best_risk_score]
    
    one_step_valid_directions = copy.copy(valid_directions)
    bad_directions = list(set(MOVE_DIRECTIONS) - set(valid_directions))
      
  # Treat attack squares I should avoid with a zero halite ship as N-step bad
  # directions, if that leaves us with options
  if np.any(avoid_attack_squares_zero_halite) and halite_ships[
      row, col] == 0 and steps_remaining > 1:
    avoid_attack_directions = []
    for d in valid_non_base_directions:
      move_row, move_col = move_ship_row_col(row, col, d, grid_size)
      if avoid_attack_squares_zero_halite[move_row, move_col]:
        avoid_attack_directions.append(d)
      
    if len(avoid_attack_directions):
      all_bad_dirs = set(bad_directions + (
          two_step_bad_directions + n_step_bad_directions))
      updated_bad_dirs = all_bad_dirs.union(set(avoid_attack_directions))
      if len(updated_bad_dirs) > len(all_bad_dirs) and len(
          updated_bad_dirs) < len(MOVE_DIRECTIONS):
        new_bad_directions = list(updated_bad_dirs.difference(all_bad_dirs))
        # import pdb; pdb.set_trace()
        n_step_bad_directions.extend(new_bad_directions)
        for new_bad_dir in new_bad_directions:
          if not new_bad_dir in n_step_bad_directions_die_probs:
            n_step_bad_directions_die_probs[new_bad_dir] = 0
           
  # Corner case: if I can replace a chaser position and there are only very
  # bad two step escape directions left: replace the chaser
  if take_my_next_square_dir is not None and (
      take_my_next_square_dir in two_step_bad_directions):
    make_chase_replace_n_bad = True
    for d in NOT_NONE_DIRECTIONS:
      if not d == take_my_next_square_dir:
        if d in n_step_bad_directions:
          if n_step_bad_directions_die_probs[d] < 0.6:
            make_chase_replace_n_bad = False
            break
        elif d in valid_directions:
          make_chase_replace_n_bad = False
          break 
    
    if make_chase_replace_n_bad:
      print("CHASE: turning two step bad into n step bad", observation['step'],
            row, col)
      two_step_bad_directions.remove(take_my_next_square_dir)
    
  # Treat the chasing - replace chaser position as an n-step bad action.
  # Otherwise, we can get trapped in a loop of dumb behavior.
  if take_my_next_square_dir is not None and not take_my_next_square_dir in (
      two_step_bad_directions) and not take_my_next_square_dir in (
        n_step_bad_directions):
    n_step_bad_directions.append(take_my_next_square_dir)
    n_step_bad_directions_die_probs[take_my_next_square_dir] = 1/4
          
  # If all valid non base directions are n step bad actions: drop n step bad
  # actions (call them 2 step bad) that are significantly worse than other n
  # step bad actions
  all_original_n_step_bad_directions = copy.copy(n_step_bad_directions)
  all_n_step_bad_directions_die_probs = copy.copy(
    n_step_bad_directions_die_probs)
  if len(n_step_bad_directions) > 1 and len(
      n_step_bad_directions) == len(valid_non_base_directions) and np.all(
        np.array([d in n_step_bad_directions for d in (
          valid_non_base_directions)])):
    die_probs = np.array(list(n_step_bad_directions_die_probs.values()))
    max_die_prob = min(die_probs.min()*2, die_probs.min()+0.1)
    delete_from_n_step_bad = []
    for d in n_step_bad_directions:
      if n_step_bad_directions_die_probs[d] > max_die_prob and (
          not d in safe_return_base_directions):
        delete_from_n_step_bad.append(d)
    for d in delete_from_n_step_bad:
      two_step_bad_directions.append(d)
      n_step_bad_directions.remove(d)
      del n_step_bad_directions_die_probs[d]
    
  if valid_non_base_directions:
    valid_not_preferred_dirs = list(set(
      two_step_bad_directions + n_step_bad_directions))
    if valid_not_preferred_dirs and (
      len(valid_non_base_directions) - len(valid_not_preferred_dirs)) > 0:
      # Drop 2 and n step bad directions if that leaves us with valid options
      bad_directions.extend(valid_not_preferred_dirs)
      bad_directions = list(set(bad_directions))
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
        
  # Only keep the strongly preferred directions if there are any
  if len(strongly_preferred_directions) > 0:
    preferred_directions = strongly_preferred_directions
        
  # Drop repetitive actions if that leaves us with valid options
  if ship_k in history['avoid_cycle_actions']:
    repetitive_action = history['avoid_cycle_actions'][ship_k]
    
    if repetitive_action in valid_directions and len(valid_directions) > 1:
      valid_directions.remove(repetitive_action)
      if repetitive_action in preferred_directions:
        preferred_directions.remove(repetitive_action)
      if repetitive_action in one_step_valid_directions:
        one_step_valid_directions.remove(repetitive_action)
      bad_directions.append(repetitive_action)
  
  # if observation['step'] == 180 and ship_k == '10-2':
  #   import pdb; pdb.set_trace()
    
  return (collect_grid_scores, return_to_base_scores, establish_base_scores,
          attack_base_scores, preferred_directions, valid_directions,
          len(bad_directions) == len(MOVE_DIRECTIONS), two_step_bad_directions,
          n_step_bad_directions, one_step_valid_directions,
          n_step_bad_directions_die_probs, all_original_n_step_bad_directions,
          all_n_step_bad_directions_die_probs)

# Update the scores as a function of blocking opponent bases
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
    base_return_grid_multiplier, my_bases, ship_row, ship_col, my_ship_count):
  base_pos = np.where(my_bases)
  base_row = base_pos[0][0]
  base_col = base_pos[1][0]
  
  dist_to_base = DISTANCES[base_row, base_col][ship_row, ship_col]
  # Remember the rule that blocks spawning when a ship is about to return
  if dist_to_base <= 10-my_ship_count:
    base_return_grid_multiplier[base_row, base_col] = 0
    
  return base_return_grid_multiplier

def get_nearest_base_distances(grid_size, ignore_abandoned, observation):
  base_dms = []
  base_distances = []
  # for b in player_obs[1]:
  #   row, col = row_col_from_square_grid_pos(player_obs[1][b], grid_size)
  #   if not (row, col) in ignore_abandoned:
  #     base_dms.append(DISTANCE_MASKS[(row, col)])
  #     base_distances.append(DISTANCES[(row, col)])
  my_bases = np.copy(observation['rewards_bases_ships'][0][1])
  for r, c in ignore_abandoned:
    my_bases[r, c] = 0
  num_my_bases = my_bases.sum()
  if num_my_bases > 0:
    my_base_positions = np.where(my_bases)
    for base_id in range(num_my_bases):
      base_row = my_base_positions[0][base_id]
      base_col = my_base_positions[1][base_id]
      base_dms.append(DISTANCE_MASKS[(base_row, base_col)])
      base_distances.append(DISTANCES[(base_row, base_col)])
  
  if base_dms:
    base_nearest_distance_scores = np.stack(base_dms).max(0)
    all_base_distances = np.stack(base_distances)
  else:
    base_nearest_distance_scores = np.ones((grid_size, grid_size))
    all_base_distances = 99*np.ones((1, grid_size, grid_size))
  
  nearest_base_distances = np.min(all_base_distances, 0)
    
  return (base_nearest_distance_scores, all_base_distances,
          nearest_base_distances)

def get_valid_opponent_ship_actions(
    config, rewards_bases_ships, halite_ships, size, history,
    nearest_base_distances, observation, env_config):
  opponent_ships_sensible_actions = {}
  opponent_ships_sensible_actions_no_risk = {}
  boxed_in_zero_halite_opponents = []
  likely_convert_opponent_positions = []
  possible_convert_opponent_positions = []
  num_agents = len(rewards_bases_ships)
  convert_cost = env_config.convertCost
  stacked_bases = np.stack([rbs[1] for rbs in rewards_bases_ships])
  stacked_ships = np.stack([rbs[2] for rbs in rewards_bases_ships])
  num_players = stacked_ships.shape[0]
  grid_size = stacked_ships.shape[1]
  player_base_ids = -1*np.ones((grid_size, grid_size))
  boxed_in_attack_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  boxed_in_opponent_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  opponent_single_escape_pos = np.zeros(
    (grid_size, grid_size), dtype=np.bool)
  single_escape_mapping = {}
  for i in range(num_players):
    player_base_ids[stacked_bases[i]] = i
  for i in range(1, num_agents):
    opponent_ships = stacked_ships[i]
    enemy_ships = np.delete(stacked_ships, (i), axis=0).sum(0)
    ship_pos = np.where(opponent_ships)
    num_ships = ship_pos[0].size
    for j in range(num_ships):
      valid_rel_directions = copy.copy(RELATIVE_DIRECTIONS)
      valid_rel_directions_no_move_risk = copy.copy(RELATIVE_DIRECTIONS)
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
              
              # if observation['step'] == 189 and row == 14 and col == 2:
              #   import pdb; pdb.set_trace()
              
              ignores_move_collision = False
              risky_stay_still_collision = False
              if halite_ships[row, col] == halite_ships[
                  other_row, other_col]:
                # Note: use the opponent distance because the opponent model is
                # learned using the opponent distance to the nearest base (with 
                # near base distance cutoff typically at 2)
                is_near_base = nearest_base_distances[
                  other_row, other_col] <= config['log_near_base_distance']
                risk_lookup_k = str(is_near_base) + '_' + str(distance) + (
                  '_ever_risky')
                if distance == 2:
                  ignores_move_collision = history[
                    'zero_halite_move_behavior'][i][risk_lookup_k]
                else:
                  risk_lookup_k_dist_zero = str(is_near_base) + '_' + str(
                    0) + '_ever_risky'
                  risky_stay_still_collision = history[
                    'zero_halite_move_behavior'][i][risk_lookup_k]
                  ignores_move_collision = history[
                    'zero_halite_move_behavior'][i][risk_lookup_k_dist_zero]
              
              # if ignores_move_collision and distance == 1:
              #   import pdb; pdb.set_trace()
              #   x=1
              
              rem_dirs = []
              if risky_stay_still_collision:
                rem_dirs += [(0, 0)] if distance == 1 and hal_diff <= 0 else []
              else:
                rem_dirs += [(0, 0)] if distance == 1 and hal_diff < 0 else []
              if not ignores_move_collision:
                rem_dirs += [(-1, 0)] if row_diff < 0 and hal_diff <= 0 else []
                rem_dirs += [(1, 0)] if row_diff > 0 and hal_diff <= 0 else []
                rem_dirs += [(0, -1)] if col_diff < 0 and hal_diff <= 0 else []
                rem_dirs += [(0, 1)] if col_diff > 0 and hal_diff <= 0 else []
              
              for d in rem_dirs:
                if d in valid_rel_directions:
                  valid_rel_directions.remove(d)
                  
              # if observation['step'] == 146 and row == 13 and col == 13:
              #   import pdb; pdb.set_trace()
                  
              # Don't check for risky opponent zero halite behavior
              rem_dirs = []
              if risky_stay_still_collision:
                rem_dirs += [(0, 0)] if distance == 1 and hal_diff <= 0 else []
              else:
                rem_dirs += [(0, 0)] if distance == 1 and hal_diff < 0 else []
              rem_dirs += [(-1, 0)] if row_diff < 0 and hal_diff <= 0 else []
              rem_dirs += [(1, 0)] if row_diff > 0 and hal_diff <= 0 else []
              rem_dirs += [(0, -1)] if col_diff < 0 and hal_diff <= 0 else []
              rem_dirs += [(0, 1)] if col_diff > 0 and hal_diff <= 0 else []
              
              for d in rem_dirs:
                if d in valid_rel_directions_no_move_risk:
                  valid_rel_directions_no_move_risk.remove(d)
                  
                  
      # Prune for opponent base positions
      rem_dirs = []
      for rel_dir in valid_rel_directions:
        d = RELATIVE_DIR_TO_DIRECTION_MAPPING[rel_dir]
        move_row, move_col = move_ship_row_col(row, col, d, grid_size)
        move_base_id = player_base_ids[move_row, move_col]
        if move_base_id >= 0 and move_base_id != i:
          rem_dirs.append(rel_dir)
      for d in rem_dirs:
        valid_rel_directions.remove(d)
        
      # if observation['step'] == 146 and row == 14 and col == 13:
      #   import pdb; pdb.set_trace()
        
      rem_dirs = []
      for rel_dir in valid_rel_directions_no_move_risk:
        d = RELATIVE_DIR_TO_DIRECTION_MAPPING[rel_dir]
        move_row, move_col = move_ship_row_col(row, col, d, grid_size)
        move_base_id = player_base_ids[move_row, move_col]
        if move_base_id >= 0 and move_base_id != i:
          rem_dirs.append(rel_dir)
      for d in rem_dirs:
        valid_rel_directions_no_move_risk.remove(d)
        
      if len(valid_rel_directions) == 0:
        player_halite_budget = observation['rewards_bases_ships'][i][0]
        if ((ship_halite + player_halite_budget) >= convert_cost):
          if ship_halite >= history['inferred_boxed_in_conv_threshold'][i][1]:
            likely_convert_opponent_positions.append((row, col))
          if ship_halite >= history['inferred_boxed_in_conv_threshold'][i][0]:
            possible_convert_opponent_positions.append((row, col))
        
        if ship_halite > 0:
          for d in MOVE_DIRECTIONS:
            move_row, move_col = move_ship_row_col(row, col, d, grid_size)
            boxed_in_attack_squares[move_row, move_col] = True
            boxed_in_opponent_ids[move_row, move_col] = i
                  
      if ship_halite == 0 and len(valid_rel_directions_no_move_risk) == 1 and (
          valid_rel_directions_no_move_risk[0] == (0, 0)):
        boxed_in_zero_halite_opponents.append((row, col))
        
      if len(valid_rel_directions_no_move_risk) == 1:
        escape_dir = RELATIVE_DIR_TO_DIRECTION_MAPPING[
          valid_rel_directions_no_move_risk[0]]
        escape_square = move_ship_row_col(row, col, escape_dir, grid_size)
        opponent_single_escape_pos[escape_square] = 1
        single_escape_mapping[(row, col)] = escape_square
        
      opponent_ships_sensible_actions[(row, col)] = valid_rel_directions
      opponent_ships_sensible_actions_no_risk[(row, col)] = (
        valid_rel_directions_no_move_risk)
      
    # if observation['step'] == 146:
    #   import pdb; pdb.set_trace()
      
    # Do another pass over the zero halite ships to figure if they are boxed in
    # by the escape squares of their own non zero halite ships - these ships
    # will very likely take risky actions and should therefore be avoided
    if np.any(opponent_single_escape_pos):
      for j in range(num_ships):
        row = ship_pos[0][j]
        col = ship_pos[1][j]
        ship_halite = halite_ships[row, col]
        if ship_halite == 0:
          valid_rel_directions = opponent_ships_sensible_actions[(row, col)]
          valid_rel_directions_no_move_risk = (
            opponent_ships_sensible_actions_no_risk[row, col])
          # if observation['step'] == 146 and row == 15 and col == 12:
          #   import pdb; pdb.set_trace()
          # if observation['step'] == 146 and row == 14 and col == 13:
          #   import pdb; pdb.set_trace()
          for d in NOT_NONE_DIRECTIONS:
            move_row, move_col = move_ship_row_col(row, col, d, grid_size)
            if opponent_single_escape_pos[move_row, move_col]:
              my_escape_square = False
              if (row, col) in single_escape_mapping:
                my_escape_square = (move_row, move_col) == (
                  single_escape_mapping[row, col])
                if my_escape_square:
                  # Still treat it as a bad direction if there is another ship
                  # that has my escape square as it's only escape square
                  num_escape_count = np.array(
                    [v == (move_row, move_col) for v in (
                      single_escape_mapping.values())]).sum()
                  my_escape_square = num_escape_count == 1
              if not my_escape_square:
                avoid_rel_direction = RELATIVE_DIR_MAPPING[d]
                if avoid_rel_direction in valid_rel_directions:
                  valid_rel_directions.remove(avoid_rel_direction)
                if avoid_rel_direction in valid_rel_directions_no_move_risk:
                  valid_rel_directions_no_move_risk.remove(avoid_rel_direction)
          
          if (len(valid_rel_directions_no_move_risk) == 0 or (
              len(valid_rel_directions_no_move_risk) == 1 and (
                valid_rel_directions_no_move_risk[0] == (0, 0)))) and (
                  not (row, col) in boxed_in_zero_halite_opponents):
            # print("AVOIDING chained zero halite collision",
            #       observation['step'], row, col)
            boxed_in_zero_halite_opponents.append((row, col))
          
          opponent_ships_sensible_actions[(row, col)] = valid_rel_directions
          opponent_ships_sensible_actions_no_risk[(row, col)] = (
            valid_rel_directions_no_move_risk)
      
  return (opponent_ships_sensible_actions,
          opponent_ships_sensible_actions_no_risk, boxed_in_attack_squares,
          boxed_in_opponent_ids, boxed_in_zero_halite_opponents,
          likely_convert_opponent_positions,
          possible_convert_opponent_positions)

def scale_attack_scores_bases_ships(
    config, observation, player_obs, spawn_cost, non_abandoned_base_distances,
    weighted_base_mask, steps_remaining, obs_halite, halite_ships, history,
    smoothed_halite, player_influence_maps,
    nearest_base_distances_with_my_excluded, player_ids,
    laplace_smoother_rel_ship_count=4, initial_normalize_ship_diff=10,
    final_normalize_ship_diff=3):
  stacked_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])
  my_bases = stacked_bases[0]
  # Exclude bases that are persistently camped by opponents
  for base_pos in history['my_base_not_attacked_positions']:
    my_bases[base_pos] = 0
  stacked_opponent_bases = stacked_bases[1:]
  stacked_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])
  stacked_opponent_ships = stacked_ships[1:]
  base_counts = stacked_opponent_bases.sum((1, 2))
  my_ship_count = len(player_obs[2])
  ship_counts = stacked_opponent_ships.sum((1, 2))
  grid_size = stacked_opponent_bases.shape[1]
  approximate_scores = history['current_scores']
  num_players = stacked_bases.shape[0]
  player_ranks = np.zeros(num_players)
  for i in range(num_players):
    player_ranks[i] = (approximate_scores >= approximate_scores[i]).sum()
  # print(approximate_scores)
  
  # Factor 1: an opponent with less bases is more attractive to attack
  base_count_multiplier = np.where(base_counts == 0, 0, 1/(base_counts+1e-9))
  
  # Factor 2: an opponent that is closer in score is more attractive to attack
  spawn_diffs = (approximate_scores[0] - approximate_scores[1:])/spawn_cost
  abs_spawn_diffs = np.abs(spawn_diffs)
  currently_winning = approximate_scores[0] >= approximate_scores[1:]
  approximate_score_diff = approximate_scores[0] - approximate_scores[1:]
  normalize_diff = initial_normalize_ship_diff - observation['relative_step']*(
    initial_normalize_ship_diff-final_normalize_ship_diff)
  abs_rel_normalized_diff = np.maximum(
    0, (normalize_diff-abs_spawn_diffs)/normalize_diff)
  rel_score_max_y = initial_normalize_ship_diff/normalize_diff
  rel_score_multiplier = abs_rel_normalized_diff*rel_score_max_y
  
  # Factor 3: an opponent with less ships is more attractive to attack since it
  # is harder for them to defend the base
  rel_ship_count_multiplier = (my_ship_count+laplace_smoother_rel_ship_count)/(
    ship_counts+laplace_smoother_rel_ship_count)
  
  # Additional term: attack bases nearby my main base
  opponent_bases = stacked_opponent_bases.sum(0).astype(np.bool)
  if opponent_bases.sum() > 0 and non_abandoned_base_distances.max() > 0:
    additive_nearby_main_base = 3/max(0.15, observation['relative_step'])/(
        1.5**non_abandoned_base_distances)/(
      weighted_base_mask[my_bases].sum())
    additive_nearby_main_base[~opponent_bases] = 0
  else:
    additive_nearby_main_base = 0
  
  attack_multipliers = base_count_multiplier*rel_score_multiplier*(
    rel_ship_count_multiplier)
  tiled_multipliers = np.tile(attack_multipliers.reshape((-1, 1, 1)),
                              [1, grid_size, grid_size])
  
  # if observation['step'] == 391:
  #   import pdb; pdb.set_trace()
  
  opponent_bases_scaled = (stacked_opponent_bases*tiled_multipliers).sum(0) + (
    additive_nearby_main_base)
  
  # Compute the priority of attacking the ships of opponents
  opponent_ships_scaled = np.maximum(0, 1 - np.abs(
    approximate_scores[0]-approximate_scores[1:])/steps_remaining/10)
  # print(observation['step'], opponent_ships_scaled, approximate_scores)
  
  # if observation['step'] == 300:
  #   import pdb; pdb.set_trace()
  
  # If I am winning by a considerable margin before the game is over, and the
  # number three is far behind the number two: go ballistic on the number two
  # Prefer opponent bases that are close to my bases and halite, and where the
  # opponent has a relatively low density
  # Make sure to guarantee some continuity with a start and stop mode
  # Ballistic scenarios:
  #    - I am well ahead of all opponents: target the initial best agent
  #    - I am winning with a solid margin and the number three is far behind
  #      the number two: target the number two
  #    - I am in a close fight with the number two/one and the number three is
  #      very far behind: target the number two
  winning_massively = np.all(spawn_diffs >= (
    18-9*observation['relative_step']))
  if not winning_massively:
    history['ballistic_early_best_target_mode'] = False
  winning_very_clearly = np.all(spawn_diffs >= (
    14-7*observation['relative_step']))
  winning_clearly = np.all(spawn_diffs >= (8-4*observation['relative_step']))
  winning_considerably = np.all(spawn_diffs >= (
    6-4*observation['relative_step'] + int(history[
      'ballistic_early_best_target_mode'])))
  winning_massively_near_end_game = winning_massively and observation[
    'relative_step'] > 0.75
  winning_massively_before_end_game = winning_massively and not (
    winning_massively_near_end_game)
  first_opp_id = np.argsort(spawn_diffs)[0]
  second_opp_id = np.argsort(spawn_diffs)[1]
  second_third_spawn_diff = spawn_diffs[second_opp_id] - spawn_diffs[
    first_opp_id]
  very_tight_fight_for_first = np.abs(spawn_diffs[first_opp_id]) < 1 and (
    spawn_diffs[second_opp_id] >= (12-8*observation['relative_step']))
  tight_fight_for_first = np.abs(spawn_diffs[first_opp_id]) < 3 and (
    spawn_diffs[second_opp_id] >= (8-6*observation['relative_step']))
  
  prev_ballistic_mode = history['ballistic_mode']
  should_start_ballistic = (not winning_massively_before_end_game) and (
    winning_clearly and second_third_spawn_diff > (
      7-2*observation['relative_step']) or very_tight_fight_for_first or (
        winning_massively_near_end_game)) and (
      my_ship_count >= 15-max(0, 40*(observation['relative_step']-0.8)))
  should_continue_ballistic = not (
    winning_massively_before_end_game) and (winning_very_clearly or (
    winning_clearly and (second_third_spawn_diff > 1)) or (
      winning_considerably and (
        second_third_spawn_diff > (2-observation['relative_step']))) or (
        tight_fight_for_first)
    ) and (my_ship_count >= 10-max(0, 20*(observation['relative_step']-0.8)))
  ballistic_mode = should_start_ballistic or (
    prev_ballistic_mode and should_continue_ballistic)
  # Select the next target in line if the opponent has no bases and no ships
  if history['ballistic_early_best_targets_sorted'] is not None:
    for opponent_id in history['ballistic_early_best_targets_sorted']:
      ballistic_early_best_target_mode_target = opponent_id
      num_opponent_bases = stacked_bases[opponent_id+1].sum()
      num_opponent_ships = stacked_ships[opponent_id+1].sum()
      if num_opponent_bases > 0 or num_opponent_ships > 0:
        break
  else:
    ballistic_early_best_target_mode_target = first_opp_id
  # print(observation['step'], ballistic_early_best_target_mode_target)
  
  # if observation['step'] == 146:
  #   import pdb; pdb.set_trace()
      
  # Ballistic early best target mode override of the opponent id: prefer to
  # attack opponents that have a base which is close to one of my non
  # abandoned bases
  opponent_base_positions = np.where(stacked_opponent_bases.sum(0) > 0)
  opponent_near_my_base_distances = nearest_base_distances_with_my_excluded[
    opponent_base_positions]
  targeted_base_override = None
  if np.any(opponent_base_positions) and winning_very_clearly and (
      opponent_near_my_base_distances.min() < 6):
    prev_ballistic_target_override = history['prev_ballistic_target_override']
    if history['prev_ballistic_target_override'] is not None and (
        opponent_bases[prev_ballistic_target_override]):
      targeted_base_override = prev_ballistic_target_override
    else:
      # Sort annoying bases by score: prefer to attack opponent bases that
      # belong to the best opponent
      smoothed_halite = smooth2d(observation['halite'])
      opponent_near_my_base_scores = opponent_near_my_base_distances + 0.6*(
        player_ranks[player_ids[opponent_base_positions]-1]) - 1e-9*(
          smoothed_halite[opponent_base_positions])
      target_base_id = np.argmin(opponent_near_my_base_scores)
      targeted_base_override = (
        opponent_base_positions[0][target_base_id],
        opponent_base_positions[1][target_base_id])
  history['prev_ballistic_target_override'] = targeted_base_override
  
  if ballistic_mode and not prev_ballistic_mode and (
      winning_massively_near_end_game):
    # Switch to early best target mode - override of the target id
    print(observation['step'], "Start attack on early best target",
          ballistic_early_best_target_mode_target+1)
    ballistic_early_best_target_mode = True
    ballistic_target_id = ballistic_early_best_target_mode_target
  elif ballistic_mode:
    ballistic_early_best_target_mode = history[
      'ballistic_early_best_target_mode'] and winning_very_clearly
    if ballistic_early_best_target_mode or winning_massively_near_end_game:
      # Early best target mode
      ballistic_target_id = ballistic_early_best_target_mode_target
    else:
      # Standard ballistic mode
      ballistic_target_id = first_opp_id
    # print(observation['step'], "Winning massively near end?",
    #       winning_massively_near_end_game, ballistic_target_id)
  else:
    ballistic_early_best_target_mode = False
    
    # Consider going ballistic on the nearest contender for the second place
    # when the first place no longer seems possible
    first_out_of_reach = spawn_diffs.min() <= (
      -40+36*observation['relative_step']) # This should be conservative
    if first_out_of_reach and np.abs(spawn_diffs[first_opp_id]) > np.abs(
        spawn_diffs[second_opp_id]):
      ballistic_target_id = second_opp_id
      third_opp_id = np.argsort(spawn_diffs)[2]
      spawn_diffs_not_best = np.array([spawn_diffs[i] for i in range(3) if (
        not i == first_opp_id)])
      winning_clearly_second = np.all(
        spawn_diffs_not_best >= (8-4*observation['relative_step']))
      winning_considerably_second = np.all(spawn_diffs_not_best >= (
        6-4*observation['relative_step']))
      third_fourth_spawn_diff = spawn_diffs[third_opp_id] - (
        spawn_diffs[second_opp_id])
      very_tight_fight_for_second = (
        np.abs(spawn_diffs[second_opp_id]) < np.abs(
          spawn_diffs[third_opp_id])/2) and (
            spawn_diffs[third_opp_id] >= (12-8*observation['relative_step']))
      tight_fight_for_second = (
        np.abs(spawn_diffs[second_opp_id]) < np.abs(
          spawn_diffs[third_opp_id])) and (
            spawn_diffs[third_opp_id] >= (10-7*observation['relative_step']))
          
      should_start_ballistic = (
        winning_clearly_second and third_fourth_spawn_diff > (
          4-2*observation['relative_step']) or (
            very_tight_fight_for_second)) and (
          my_ship_count >= 15-max(0, 40*(observation['relative_step']-0.8)))
      should_continue_ballistic = ((
        winning_clearly_second and (third_fourth_spawn_diff > 1)) or (
          winning_considerably_second and (
            third_fourth_spawn_diff > (2-observation['relative_step']))) or (
            tight_fight_for_second)
        ) and (my_ship_count >= 10-max(
          0, 20*(observation['relative_step']-0.8)))
      ballistic_mode = should_start_ballistic or (
        prev_ballistic_mode and should_continue_ballistic)
      
      # if observation['step'] == 363:
      #   import pdb; pdb.set_trace()
      
      # if ballistic_mode:
      #   print("SECOND BALLISTIC MODE", observation['step'],
      #         ballistic_target_id)
    
    if not ballistic_mode:
      ballistic_target_id = 0 # This could be 1 or 2 as well
    
  history['ballistic_early_best_target_mode'] = (
    ballistic_early_best_target_mode)
  
  if not ballistic_mode and targeted_base_override is not None:
    print("Go ballistic on nearby base", observation['step'],
          targeted_base_override)
    ballistic_mode = True
    ballistic_target_id = np.argmax(base_counts)
  
  num_target_bases = base_counts[ballistic_target_id]
  ballistic_attack_base_targets = []
  if ballistic_mode and num_target_bases > 0:
    target_bases = stacked_opponent_bases[ballistic_target_id]
    target_base_locations = np.where(target_bases)
    attack_target_base_scores = np.zeros(num_target_bases)
    my_base_density = smooth2d(my_bases, 10)
    for base_id in range(num_target_bases):
      base_row = target_base_locations[0][base_id]
      base_col = target_base_locations[1][base_id]
      attack_target_base_scores[base_id] = 5e-4*smoothed_halite[
        base_row, base_col] + player_influence_maps[0, base_row, base_col] - (
          player_influence_maps[ballistic_target_id+1, base_row, base_col]) + (
            100*int((base_row, base_col) in history['prev_step'][
              'ballistic_attack_base_targets'])) + 10*my_base_density[
                base_row, base_col]
      
    # import pdb; pdb.set_trace()
    ordered_base_ids = np.argsort(-attack_target_base_scores)
    num_attacked_bases = 1 # 1 is plenty of aggression for the world
    for attack_id in range(num_attacked_bases):
      if attack_id == 0 and targeted_base_override is not None:
        # print("Targeted base override", observation['step'],
        #       targeted_base_override)
        base_row, base_col = targeted_base_override
      else:
        base_id = ordered_base_ids[attack_id]
        base_row = target_base_locations[0][base_id]
        base_col = target_base_locations[1][base_id]
      opponent_bases_scaled[base_row, base_col] = 1e4
      ballistic_attack_base_targets.append((base_row, base_col))
    
  del_keys = []
  for k in history['camping_ships_strategy']:
    if history['camping_ships_strategy'][k][5] in (
        ballistic_attack_base_targets):
      del_keys.append(k)
    
  for k in del_keys:
    del history['camping_ships_strategy'][k]
    
  # print(observation['step'], ballistic_mode, ballistic_attack_base_targets)
  history['ballistic_mode'] = ballistic_mode
    
  return (opponent_bases_scaled, opponent_ships_scaled,
          abs_rel_normalized_diff, currently_winning, approximate_score_diff,
          history, ballistic_attack_base_targets)

def get_influence_map(config, stacked_bases, stacked_ships, halite_ships,
                      observation, player_obs, smooth_kernel_dim=7):
  # FUTURE WORK: incorporate the number of ships in computing the base weights
  # Reasoning: a base without ships is not really a threat
  all_ships = stacked_ships.sum(0).astype(np.bool)
  my_ships = stacked_ships[0].astype(np.bool)
  
  if my_ships.sum() == 0:
    return None, None, None, None, None, None
  
  num_players = stacked_ships.shape[0]
  grid_size = my_ships.shape[0]
  ship_range = 1-config['influence_map_min_ship_weight']
  all_ships_halite = halite_ships[all_ships]
  unique_vals, unique_counts = np.unique(
    all_ships_halite, return_counts=True)
  assert np.all(np.diff(unique_vals) > 0)
  unique_halite_vals = np.sort(unique_vals).astype(np.int).tolist()
  num_ships = all_ships_halite.size
  
  halite_ranks = [np.array(
    [unique_halite_vals.index(hs) for hs in halite_ships[
      stacked_ships[i]]]) for i in range(num_players)]
  less_rank_cum_counts = np.cumsum(unique_counts)
  num_unique = unique_counts.size
  halite_rank_counts = [np.array(
    [less_rank_cum_counts[r-1] if r > 0 else 0 for r in (
      halite_ranks[i])]) for i in range(num_players)]
  ship_weights = [1 - r/(num_ships-1+1e-9)*ship_range for r in (
    halite_rank_counts)]
  
  raw_influence_maps = np.zeros((num_players, grid_size, grid_size))
  raw_influence_maps_unweighted = np.zeros((num_players, grid_size, grid_size))
  influence_maps = np.zeros((num_players, grid_size, grid_size))
  influence_maps_unweighted = np.zeros((num_players, grid_size, grid_size))
  for i in range(num_players):
    raw_influence_maps[i][stacked_ships[i]] += ship_weights[i]
    raw_influence_maps[i][stacked_bases[i]] += config[
      'influence_map_base_weight']
    raw_influence_maps_unweighted[i][stacked_ships[i]] += 1
    raw_influence_maps_unweighted[i][stacked_bases[i]] += 1
    
    influence_maps[i] = smooth2d(raw_influence_maps[i],
                                 smooth_kernel_dim=smooth_kernel_dim)
    influence_maps_unweighted[i] = smooth2d(
      raw_influence_maps_unweighted[i], smooth_kernel_dim=smooth_kernel_dim)
  
  my_influence = influence_maps[0]
  max_other_influence = influence_maps[1:].max(0)
  influence_map = my_influence - max_other_influence
  influence_map_unweighted = influence_maps_unweighted[0] - (
    influence_maps_unweighted[1:].sum(0))
  
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
  
  return (influence_map, influence_map_unweighted, influence_maps,
          priority_scores, ship_priority_weights, escape_influence_probs)

# Compute the weighted base mask - the base with value one represents the
# main base and the values are used as a multiplier in the return to base
# scores.
def get_weighted_base_mask(stacked_bases, stacked_ships, observation,
                           history, consistent_main_base_bonus=3):
  my_bases = stacked_bases[0]
  # Exclude bases that are persistently camped by opponents
  for base_pos in history['my_base_not_attacked_positions']:
    my_bases[base_pos] = 0
  num_bases = my_bases.sum()
  my_base_locations = np.where(my_bases)
  grid_size = stacked_bases.shape[1]
  ship_diff_smoothed = smooth2d(stacked_ships[0] - stacked_ships[1:].sum(0))
  if num_bases == 0:
    base_mask = np.ones((grid_size, grid_size))
    main_base_distances = -1*np.ones((grid_size, grid_size))
    non_abandoned_base_distances = -1*np.ones((grid_size, grid_size))
  elif num_bases >= 1:
    all_non_abandoned_base_distances = []
    for base_id in range(num_bases):
      base_row = my_base_locations[0][base_id]
      base_col = my_base_locations[1][base_id]
      all_non_abandoned_base_distances.append(DISTANCES[
        base_row, base_col])
    non_abandoned_base_distances = np.stack(
      all_non_abandoned_base_distances).min(0)
    
    # Add a bonus to identify the main base id, but don't include the bonus
    # in the base scaling
    ship_diff_smoothed_with_bonus = np.copy(ship_diff_smoothed)
    prev_main_base_location = history['prev_step']['my_main_base_location']
    # print(observation['step'], prev_main_base_location, num_bases)
    if prev_main_base_location[0] >= 0:
      ship_diff_smoothed_with_bonus[prev_main_base_location] += (
        consistent_main_base_bonus)
    base_densities = ship_diff_smoothed[my_base_locations]
    base_densities_with_bonus = ship_diff_smoothed_with_bonus[
      my_base_locations]
    highest_base_density_with_bonus = base_densities_with_bonus.max()
    best_ids = np.where(
      base_densities_with_bonus == highest_base_density_with_bonus)[0]
    highest_base_density = base_densities[best_ids[0]]
    
    # Subtract some small value of the non max densities to break rare ties
    main_base_row = my_base_locations[0][best_ids[0]]
    main_base_col = my_base_locations[1][best_ids[0]]
    main_base_distances = DISTANCES[main_base_row, main_base_col]
    all_densities = np.minimum(ship_diff_smoothed, highest_base_density-1e-5)
    all_densities[main_base_row, main_base_col] += 1e-5
    
    # Linearly compute the weighted base mask: 1 is my best base and 0 is the
    # lowest ship_diff_smoothed value
    all_densities -= all_densities.min()
    base_mask = all_densities/all_densities.max()
    
  return (base_mask, main_base_distances, non_abandoned_base_distances,
          ship_diff_smoothed)

# Force returning to a base when the episode is almost over
def force_return_base_end_episode(
    my_bases, base_return_grid_multiplier, main_base_distances, row, col,
    steps_remaining, opponent_less_halite_ships, weighted_base_mask,
    safe_to_collect):
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
      max(int(safe_to_collect[row, col]),
          weighted_base_mask[base_row, base_col]) - 5*(
            threat_ships_mask.mean()) - base_distance/30)
    
  # if observation['step'] == 384 and row == 8 and col == 11:
  #   import pdb; pdb.set_trace()
    
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
    all_ship_scores, stacked_ships, observation, env_config,
    opponent_ships_sensible_actions, halite_ships, steps_remaining, player_obs,
    np_rng, opponent_ships_scaled, collect_rate, obs_halite,
    main_base_distances, history, on_rescue_mission,
    my_defend_base_ship_positions, env_observation, player_influence_maps,
    override_move_squares_taken, ignore_convert_positions,
    convert_unavailable_positions, always_attack_opponent_id,
    num_non_abandoned_bases, likely_convert_opponent_positions,
    possible_convert_opponent_positions, my_current_base_distances,
    box_in_window=3, min_attackers_to_box=4):
  # Loop over the opponent ships and derive if I can box them in
  # For now this is just greedy. We should probably consider decoupling finding
  # targets from actually boxing in.
  # FUTURE WORK: proper handling of opponent bases
  opponent_positions = np.where(stacked_ships[1:].sum(0) > 0)
  opponent_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  num_opponent_ships = opponent_positions[0].size
  double_window = box_in_window*2
  dist_mask_dim = 2*double_window+1
  nearby_rows = np.tile(np.expand_dims(np.arange(dist_mask_dim), 1),
                [1, dist_mask_dim])
  nearby_cols = np.tile(np.arange(dist_mask_dim), [dist_mask_dim, 1])
  ships_available = np.copy(stacked_ships[0]) & (~on_rescue_mission) & (
    ~my_defend_base_ship_positions) & (~convert_unavailable_positions)
  boxing_in = np.zeros_like(on_rescue_mission)
  grid_size = stacked_ships.shape[1]
  # ship_pos_to_key = {v[0]: k for k, v in player_obs[2].items()}
  prev_step_boxing_in_ships = history['prev_step_boxing_in_ships']
  num_players = stacked_ships.shape[0]
  spawn_cost = env_config.spawnCost
  ship_pos_to_key = {}
  for i in range(num_players):
    ship_pos_to_key.update({
      v[0]: k for k, v in env_observation.players[i][2].items()})
  
  # Loop over the camping ships and exclude the ones from the available mask
  # that have flagged they are not available for boxing in
  camping_ships_strategy = history['camping_ships_strategy']
  for ship_k in camping_ships_strategy:
    if not camping_ships_strategy[ship_k][3]:
      camping_row, camping_col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      ships_available[camping_row, camping_col] = 0 
      
  # Loop over the ships that attack opponent camplers and exclude them from the
  # available mask
  attack_opponent_campers = history['attack_opponent_campers']
  for ship_k in attack_opponent_campers:
    attacking_camper_row, attacking_camper_col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    ships_available[attacking_camper_row, attacking_camper_col] = 0
    
  # Loop over the ships that are stuck in a loop and mark them as unavailable
  for ship_k in history['avoid_cycle_actions']:
    cycle_row, cycle_col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    ships_available[cycle_row, cycle_col] = 0
      
  original_ships_available = np.copy(ships_available)
  my_ship_density = smooth2d(ships_available, smooth_kernel_dim=2)
  
  # Compute the priorities of attacking each ship
  # Compute the minimum opponent halite in the neighborhood of each square
  # by looping over all opponent ships
  attack_ship_priorities = np.zeros(num_opponent_ships)
  near_opponent_min_halite = np.ones((grid_size, grid_size))*1e6
  near_opponent_2_min_halite = np.ones((grid_size, grid_size))*1e6
  near_opponent_specific_2_min_halite = [
    np.ones((grid_size, grid_size))*1e6 for _ in range(num_players)]
  should_attack = np.zeros(num_opponent_ships, dtype=np.bool)
  for i in range(num_opponent_ships):
    row = opponent_positions[0][i]
    col = opponent_positions[1][i]
    opponent_ship_k = ship_pos_to_key[row*grid_size+col]
    boxing_in_prev_step = opponent_ship_k in prev_step_boxing_in_ships
    opponent_halite = halite_ships[row, col]
    clipped_opponent_halite = min(spawn_cost, opponent_halite)
    opponent_id = np.where(stacked_ships[:, row, col])[0][0]
    attack_ship_priorities[i] = 1e5*boxing_in_prev_step + (
      clipped_opponent_halite) + 1000*(
        opponent_ships_scaled[opponent_id-1]) + 1000*my_ship_density[row, col]
    near_opp_mask = ROW_COL_MAX_DISTANCE_MASKS[(row, col, box_in_window)]
    near_opp_2_mask = ROW_COL_MAX_DISTANCE_MASKS[(row, col, 2)]
    near_opponent_min_halite[near_opp_mask] = np.minimum(
        opponent_halite, near_opponent_min_halite[near_opp_mask])
    near_opponent_2_min_halite[near_opp_2_mask] = np.minimum(
      opponent_halite, near_opponent_2_min_halite[near_opp_2_mask])
    near_opponent_specific_2_min_halite[opponent_id][near_opp_2_mask] = (
      np.minimum(opponent_halite,
                 near_opponent_specific_2_min_halite[opponent_id][
                   near_opp_2_mask]))
    
    # if observation['step'] == 163 and row == 2:
    #   import pdb; pdb.set_trace()
    
    should_attack[i] = (main_base_distances[row, col] >= (9-(
      observation['relative_step']*6 + 3*num_non_abandoned_bases)) or (
        opponent_halite < history[
          'inferred_boxed_in_conv_threshold'][opponent_id][0]) or (
            always_attack_opponent_id == opponent_id)) and not (
              (row, col) in ignore_convert_positions)
  
  box_opponent_positions = []
  boxing_in_ships = []
  ships_on_box_mission = {}
  opponent_ship_order = np.argsort(-attack_ship_priorities)
  for i in range(num_opponent_ships):
    opponent_ship_id = opponent_ship_order[i]
    row = opponent_positions[0][opponent_ship_id]
    col = opponent_positions[1][opponent_ship_id]
    opponent_id = np.where(stacked_ships[:, row, col])[0][0]
    opponent_ship_k = ship_pos_to_key[row*grid_size+col]
    sensible_target_actions = opponent_ships_sensible_actions[row, col]
    target_halite = halite_ships[row, col]
    my_less_halite_mask = np.logical_and(
      halite_ships < target_halite, ships_available)
    
    # if observation['step'] == 210 and row == 1 and col == 8:
    #   import pdb; pdb.set_trace()
    
    # Drop non zero halite ships towards the end of a game (they should return)
    my_less_halite_mask = np.logical_and(
      my_less_halite_mask, np.logical_or(
        halite_ships == 0, steps_remaining > 20))
    max_dist_mask = ROW_COL_MAX_DISTANCE_MASKS[(row, col, double_window)]
    my_less_halite_mask &= max_dist_mask
    box_pos = ROW_COL_BOX_MAX_DISTANCE_MASKS[row, col, double_window]
    
    # if observation['step'] == 157 and row == 13 and col == 1:
    #   import pdb; pdb.set_trace()
    
    if my_less_halite_mask.sum() >= min_attackers_to_box and should_attack[
        opponent_ship_id]:
      # Look up the near opponent min halite in the square which is in the
      # middle between my attackers and the target - don't attack when there is
      # a less halite ship near that ship or if there is an equal halite ship
      # near that square and close to the opponent
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
        
        # Only box in with ships that can safely do so without becoming a
        # target themselves. Take more risk when the halite on board is
        # equal to that of other target surrounding ships (typically 0 halite)
        considered_to_target_distances = DISTANCES[(row, col)][
          (considered_rows, considered_cols)]
        considered_min_halite_limits = np.where(
          considered_to_target_distances < 3, near_opponent_2_min_halite[
            (mid_rows, mid_cols)], near_opponent_min_halite[
              (mid_rows, mid_cols)])
        drop_ids = (considered_min_halite_limits < (
          halite_ships[(considered_rows, considered_cols)])) | (
            (considered_min_halite_limits == (
              halite_ships[(considered_rows, considered_cols)])) & (
                near_opponent_specific_2_min_halite[opponent_id][
                  (row, col)] <= (
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
      
      # if observation['step'] == 32:
      #   import pdb; pdb.set_trace()
      
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
        cropped_distances = OTHER_DISTANCES[
          (double_window, double_window, dist_mask_dim)]
        for dim_id, d in enumerate(NOT_NONE_DIRECTIONS):
          box_dir_mask = BOX_DIRECTION_MASKS[(double_window, d)]
          closest_dim_distance = cropped_distances[
            box_in_mask_dirs[dim_id]].min()
          escape_squares[box_dir_mask] &= (
            cropped_distances[box_dir_mask] <= closest_dim_distance)
        
        if not np.any(observation['rewards_bases_ships'][opponent_id][1][
            box_pos][escape_squares.flatten()]):
          # Let's box the opponent in!
          # We should move towards the opponent if we can do so without opening
          # up an escape direction
          
          # if observation['step'] == 32:
          #   import pdb; pdb.set_trace()
          
          # Order the planning by priority of direction and distance to the
          # opponent
          # Reasoning: mid-distance ships plan first since that allows fast
          # boxing in - the nearby ships then just have to cover the remaining
          # directions.
          # Ships which cover hard to cover directions plan later.
          box_in_mask_dirs_sum = box_in_mask_dirs.sum((1, 2))
          ship_priorities = np.zeros(my_num_nearby)
          must_attack_converting_square = ((row, col) in (
            likely_convert_opponent_positions)) and not (
              (row, col) in ignore_convert_positions) and ((
                always_attack_opponent_id == opponent_id) or (
                  my_current_base_distances[:, row, col].min() < 5))
     
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
              
          # I can always attack all escape squares if I have at least 5 ships
          # at a maximum distance of two with at least one attacker on each
          # half plane
          vert_diff = double_window-nearby_mask_pos[0]
          horiz_diff = double_window-nearby_mask_pos[1]
          distances = np.abs(vert_diff) + np.abs(horiz_diff)
          is_near = distances <= 2
          near_vert_diff = vert_diff[is_near]
          near_horiz_diff = horiz_diff[is_near]
          i_can_attack_all_escape_squares = distances.min() == 1 and (
            is_near.sum() >= 5) and np.sign(near_vert_diff).ptp() == 2 and (
              np.sign(near_horiz_diff).ptp() == 2)
          if i_can_attack_all_escape_squares and (distances == 1).sum() == 1:
            # I can only attack all escape squares if my attacker can be
            # replaced
            one_step_diff_id = np.argmin(distances)
            single_attack_row = nearby_mask_pos[0][one_step_diff_id]
            single_attack_col = nearby_mask_pos[1][one_step_diff_id]
            can_replace = False
            for row_offset in [-1, 1]:
              for col_offset in [-1, 1]:
                if nearby_less_halite_mask[single_attack_row + row_offset,
                                           single_attack_col + col_offset]:
                  can_replace = True
                  break
            i_can_attack_all_escape_squares = can_replace
            
          # DISCERN if we are just chasing or actually attacking the ship in
          # the next move - dummy rule to have at least K neighboring ships
          # for us to attack the position of the targeted ship - this makes it
          # hard to guess the escape direction
          ship_target_1_distances = my_nearest_distances[
            :, double_window, double_window] == 1
          next_step_attack = (len(sensible_target_actions) == 0 and (
              ship_target_1_distances.sum() > 2)) or (
                i_can_attack_all_escape_squares) or (
                  must_attack_converting_square and np.any(
                    ship_target_1_distances))
          # if next_step_attack and not (
          #     (len(sensible_target_actions) == 0 and (
          #       ship_target_1_distances.sum() > 2)) or (
          #         i_can_attack_all_escape_squares)):
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
            
            # Reorder ship_target_2_distance_ids so that the ones that can
            # replace a 1 step threat are considered last, except when there is
            # only a single 1 step threat (it would always move to the target).
            # Also prefer to consider ships that only have a single option
            # to move to the target first
            two_step_distance_scores = np.zeros(
              len(ship_target_2_distance_ids))
            for two_step_id, two_step_diff_id in enumerate(
                ship_target_2_distance_ids):
              my_row = nearby_mask_pos[0][two_step_diff_id]
              my_col = nearby_mask_pos[1][two_step_diff_id]
              mask_between = get_mask_between_exclude_ends(
                my_row, my_col, double_window, double_window, dist_mask_dim)
              two_step_distance_scores[two_step_id] = mask_between.sum() + 10*(
                nearby_less_halite_mask[mask_between].sum())*(
                  ship_target_1_distances.sum() > 1)
                  
            # if observation['step'] == 134:
            #   import pdb; pdb.set_trace()
                  
            ship_target_2_distance_ids = np.array(
              ship_target_2_distance_ids)[
                np.argsort(two_step_distance_scores)].tolist()
            
            # Add the positions of the one step attackers
            for one_step_diff_id in np.where(ship_target_1_distances)[0]:
              my_row = nearby_mask_pos[0][one_step_diff_id]
              my_col = nearby_mask_pos[1][one_step_diff_id]
              
              # If I only have one ship that can attack the target: attack with
              # that ship!
              if ship_target_1_distances.sum() == 1:
                attack_direction = get_dir_from_target(
                  my_row, my_col, double_window, double_window,
                  grid_size=1000)[0]
                pos_taken[double_window, double_window] = True
                move_ids_directions_next_attack[one_step_diff_id] = (
                  attack_direction)
              else:
                pos_taken[my_row, my_col] = 1
              
            # if observation['step'] == 176:
            #   import pdb; pdb.set_trace()
              
            two_step_pos_taken = []
            while ship_target_2_distance_ids:
              two_step_diff_id = ship_target_2_distance_ids.pop(0)
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
                if not pos_taken[move_row, move_col] and (not (
                    (move_row, move_col) in two_step_pos_taken)):
                  two_step_pos_taken.append((move_row, move_col))
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
                    
              # Recompute the priority of the remaining two step ships
              # Prefer ships with the lowest pos_taken shortest actions
              two_step_distance_scores = np.zeros(
                len(ship_target_2_distance_ids))
              for two_step_id, two_step_diff_id in enumerate(
                  ship_target_2_distance_ids):
                my_row = nearby_mask_pos[0][two_step_diff_id]
                my_col = nearby_mask_pos[1][two_step_diff_id]
                shortest_directions = get_dir_from_target(
                  my_row, my_col, double_window, double_window,
                  grid_size=1000)
                for d in shortest_directions:
                  move_row, move_col = move_ship_row_col(
                    my_row, my_col, d, size=1000)
                  two_step_distance_scores[two_step_id] += int(
                    not (pos_taken[move_row, move_col] or (
                      (move_row, move_col) in two_step_pos_taken)))
                      
              ship_target_2_distance_ids = np.array(
                ship_target_2_distance_ids)[
                  np.argsort(two_step_distance_scores)].tolist()
              
            one_step_diff_ids = np.where(ship_target_1_distances)[0]
            if pos_taken[double_window, double_window]:
              # Add the remaining one step attackers with stay in place actions
              for one_step_diff_id in one_step_diff_ids:
                if not one_step_diff_id in move_ids_directions_next_attack:
                  move_ids_directions_next_attack[one_step_diff_id] = None
            else:
              
              # Prefer to avoid stay in place actions with zero halite ships
              real_mask_pos = (
                np.mod(nearby_mask_pos[0]+row-double_window, grid_size),
                np.mod(nearby_mask_pos[1]+col-double_window, grid_size)
                )
              one_step_halite_on_board = halite_ships[real_mask_pos][
                one_step_diff_ids]
              one_step_halite_on_square = obs_halite[real_mask_pos][
                one_step_diff_ids]
              prefers_box_in = (one_step_halite_on_board == 0) & (
                one_step_halite_on_square > 0)
              if np.all(~prefers_box_in):
                one_step_diff_ids_attack = one_step_diff_ids
              else:
                one_step_diff_ids_attack = one_step_diff_ids[
                  prefers_box_in]
                
              # Of the remaining attack options: prefer an attacker from the
              # direction where we have the highest influence, relative to the
              # targeted opponent
              # one_step_attacker_id = np_rng.choice(one_step_diff_ids_attack)
              my_influences = player_influence_maps[0][real_mask_pos][
                one_step_diff_ids_attack]
              opponent_influences = player_influence_maps[opponent_id][
                real_mask_pos][one_step_diff_ids_attack]
              influence_differences = my_influences - opponent_influences
              one_step_attacker_id = one_step_diff_ids_attack[
                np.argmax(influence_differences)]
              
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
          elif len(sensible_target_actions) == 0 or (
              len(sensible_target_actions) == 1 and (
                sensible_target_actions[0] == (0, 0))):
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
            
            # if observation['step'] == 97:
            #   import pdb; pdb.set_trace()
            
            # Iteratively look for directions where I can box in in one step
            # when I have others that can box in the remaining directions
            # and nobody else can box that direction in
            box_in_mask_rem_dirs_sum = np.copy(box_in_mask_dirs_sum)
            while len(can_box_progress) > 0 and np.any(not_boxed_dirs) and (
                can_box_immediately_counts_progress.sum() > 0):
              considered_dir = np.argmin(
                can_box_immediately_counts_progress + 100*(
                  can_box_immediately_counts_progress <= 0) + 1e-2*(
                      box_in_mask_rem_dirs_sum))
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
                    scores[k] = 100*len(considered_dir_ids[k][2]) - (
                      considered_dir_ids[k][2].sum())
                  picked_dir_id = np.argmin(scores)
                else:
                  picked_dir_id = 0
                picked = considered_dir_ids[picked_dir_id]
                box_override_assignment_not_next_attack[picked[1]] = (
                  considered_dir, picked[4], picked[5])
                
                # If I move closer with a diagonal ship: subtract the
                # immediate box counter for the other direction
                picked_other_immediate_box_dirs = picked[3][
                  picked[3] != considered_dir]
                can_box_immediately_counts_progress[considered_dir] = 0
                can_box_immediately_counts_progress[
                  picked_other_immediate_box_dirs] -= 1
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
                  (ship_k, move_row, move_col, 2e6, opponent_distance, None,
                   my_abs_row, my_abs_col))
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
                
                # if observation['step'] == 357:
                #   import pdb; pdb.set_trace()
                #   print(my_row, my_col, threatened_one_step,
                #         num_covered_directions, num_one_step_from_covered)
                
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
                for threat_dir in RELATIVE_NOT_NONE_DIRECTIONS:
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
                      
                # if observation['step'] == 210 and row == 1 and col == 8:
                #   import pdb; pdb.set_trace()
                      
                # Join the attack - add actions to the list
                num_covered_directions[ship_covered_directions] += 1
                num_one_step_from_covered[
                  ship_one_step_from_covered_directions] = 1
                update_ship_scores.append(
                  (ship_k, move_row, move_col, 2e6, opponent_distance,
                   np.where(ship_covered_directions)[0], my_abs_row,
                   my_abs_col))
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
              
          # if observation['step'] == 87:
          #   import pdb; pdb.set_trace()
          if next_step_attack or np.all(num_covered_directions > 0) or (
              almost_attack_nearby_blockers and np.any(
                num_covered_directions > 0)):
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
            
            # if observation['step'] == 237:
            #   import pdb; pdb.set_trace()
            
            box_opponent_positions.append((row, col))
            boxing_in_ships.append(opponent_ship_k)
            for (ship_k, move_row, move_col, new_collect_score,
                 distance_to_target, _, my_abs_row, my_abs_col) in (
                   update_ship_scores):
              # Only update the ship scores if the box in action is in my one
              # step valid actions
              box_dir = get_dir_from_target(
                my_abs_row, my_abs_col, move_row, move_col, grid_size)[0]
              if box_dir in all_ship_scores[ship_k][9]:
                all_ship_scores[ship_k][0][move_row, move_col] = (
                  new_collect_score)
                # Flag the boxing in ships as unavailable for other hunts
                ships_available[my_abs_row, my_abs_col] = 0
                boxing_in[my_abs_row, my_abs_col] = 1
                ships_on_box_mission[ship_k] = distance_to_target
                override_move_squares_taken[move_row, move_col] = 1
            
  # Make sure that I attack all squares where an opponent is converting that
  # I can not allow to happen
  for (row, col) in possible_convert_opponent_positions:
    if not (row, col) in ignore_convert_positions:
      my_base_distances = my_current_base_distances[:, row, col]
      must_attack_converting_square = my_base_distances.min() < (3.5 - (
        observation['relative_step']))
      if must_attack_converting_square and not override_move_squares_taken[
          row, col]:
        # Look for nearby ships of mine that can attack the converting ship
        for d in NOT_NONE_DIRECTIONS:
          my_row, my_col = move_ship_row_col(row, col, d, grid_size)
          if original_ships_available[my_row, my_col] or (
              my_defend_base_ship_positions[my_row, my_col]):
            to_target_dir = OPPOSITE_MAPPING[d]
            ship_pos = my_row*grid_size+my_col
            ship_k = ship_pos_to_key[ship_pos]
            if to_target_dir in all_ship_scores[ship_k][6]:
              all_ship_scores[ship_k][0][row, col] = 1e9
              boxing_in[my_row, my_col] = 1
              print("ATTACKING POSSIBLY CONVERTING SHIP", observation['step'],
                    row, col, my_row, my_col)
              break
            
  history['prev_step_boxing_in_ships'] = boxing_in_ships
      
  return (all_ship_scores, boxing_in, box_opponent_positions,
          override_move_squares_taken, ships_on_box_mission)

def update_scores_pack_hunt(
    all_ship_scores, config, stacked_ships, observation,
    opponent_ships_sensible_actions, halite_ships, steps_remaining,
    player_obs, np_rng, opponent_ships_scaled, collect_rate, obs_halite,
    main_base_distances, history, on_rescue_mission, boxing_in_mission,
    my_defend_base_ship_positions, env_observation, box_opponent_positions,
    override_move_squares_taken, player_influence_maps,
    ignore_convert_positions, convert_unavailable_positions,
    early_hunting_season, late_hunting_season, safe_collect_margin, spawn_cost,
    change_standard_consecutive_steps=5):
  available_pack_hunt_ships = np.copy(stacked_ships[0])
  grid_size = available_pack_hunt_ships.shape[0]
  hunting_season_started = history['hunting_season_started']
  prev_standard_ships = history['hunting_season_standard_ships']
  # # FUTURE WORK: Make the number of standard ships a function of the hunt
  # # success?
  # # FUTURE WORK: Make the number of standard ships a function of ship losses?
  
  if early_hunting_season:
    max_standard_ships_hunting_season = config[
      'max_standard_ships_early_hunting_season']
  elif late_hunting_season:
    max_standard_ships_hunting_season = max(config[
      'max_standard_ships_late_hunting_season'], int(len(player_obs[2])*(
        config['late_hunting_season_standard_min_fraction'])))
  else:
    max_standard_ships_hunting_season = max(10, int(len(player_obs[2])/2.5))
  # print(observation['step'], len(player_obs[2]),
  #       max_standard_ships_hunting_season)
    
  # print(observation['step'], opponent_hunt_fraction, num_my_ships,
  #       my_target_standard_ships, max_standard_ships_hunting_season)
  
  # Determine if I should preferably target a specific opponent.
  # In games where there is a clear difference between the top two agents and
  # my agent, where I am one of those two: mostly harrass/hoard the other top
  # agent
  current_scores = history['current_scores']
  spawn_diffs = (current_scores[0] - current_scores[1:])/spawn_cost
  first_opponent_id = np.argsort(spawn_diffs)[0]
  second_opponent_id = np.argsort(spawn_diffs)[1]
  my_agent_in_top_two = (spawn_diffs < 0).sum() <= 1
  spawn_diff_first = np.abs(spawn_diffs[first_opponent_id])
  spawn_diff_second = np.abs(spawn_diffs[second_opponent_id])
  
  prev_targeted_hoard_mode = history['targeted_hoard_mode']
  should_start_targeted_hoard_mode = my_agent_in_top_two and (
    spawn_diff_second > 2*spawn_diff_first) and (spawn_diff_second > 6)
  should_continue_targeted_hoard_mode = my_agent_in_top_two and (
    spawn_diff_second > spawn_diff_first) and (spawn_diff_second > 4)
  targeted_hoard_mode = should_start_targeted_hoard_mode or (
    prev_targeted_hoard_mode and should_continue_targeted_hoard_mode)
  history['targeted_hoard_mode'] = targeted_hoard_mode
  preferred_victim = None
  if targeted_hoard_mode:
    preferred_victim = first_opponent_id+1
  if should_start_targeted_hoard_mode and not prev_targeted_hoard_mode:
    print(observation['step'], "Start selective hoarding of opponent",
          preferred_victim)
  
  prev_step_opponent_ship_moves = history['prev_step_opponent_ship_moves']
  num_players = stacked_ships.shape[0]
  ship_pos_to_key = {}
  for i in range(num_players):
    ship_pos_to_key.update({
      v[0]: k for k, v in env_observation.players[i][2].items()})
  ship_key_to_pos = {v: k for k, v in ship_pos_to_key.items()}
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
  
  not_available_due_to_camping = np.zeros_like(available_pack_hunt_ships)
  # Loop over the camping ships and exclude the ones from the available mask
  # that have flagged they are not available for boxing in
  camping_ships_strategy = history['camping_ships_strategy']
  for ship_k in camping_ships_strategy:
    if not camping_ships_strategy[ship_k][3]:
      camping_row, camping_col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      not_available_due_to_camping[camping_row, camping_col] = 1 
      
  # Loop over the ships that attack opponent camplers and exclude them from the
  # available mask
  attack_opponent_campers = history['attack_opponent_campers']
  for ship_k in attack_opponent_campers:
    attacking_camper_row, attacking_camper_col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    not_available_due_to_camping[
      attacking_camper_row, attacking_camper_col] = 1
    
  # Loop over the ships that are stuck in a loop and mark them as unavailable
  not_available_due_to_cycle = np.zeros_like(available_pack_hunt_ships)
  for ship_k in history['avoid_cycle_actions']:
    cycle_row, cycle_col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    not_available_due_to_cycle[cycle_row, cycle_col] = 1
    
  # Loop over the ships that are temporarily assigned a collect task
  not_available_due_to_temp_collect = np.zeros_like(available_pack_hunt_ships)
  delete_keys = []
  for ship_k in history['temporary_hoarding_collect_ships']:
    if ship_k in player_obs[2]:
      collect_row, collect_col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      not_available_due_to_temp_collect[collect_row, collect_col] = 1
    else:
      delete_keys.append(ship_k)
  for k in delete_keys:
    history['temporary_hoarding_collect_ships'].remove(k)
  
  # List the ships that are definitely not available for the pack hunt
  # In this group:
  #  - Opponent base camping
  #  - Attack opponent base campers
  #  - Ships that are are on a rescue mission (rescuer and rescued)
  #  - Base defense emergency ships
  #  - Boxing in other ships
  available_pack_hunt_ships &= (~not_available_due_to_camping)
  available_pack_hunt_ships &= (~on_rescue_mission)
  available_pack_hunt_ships &= (~my_defend_base_ship_positions)
  available_pack_hunt_ships &= (~boxing_in_mission)
  available_pack_hunt_ships &= (~convert_unavailable_positions)
  available_pack_hunt_ships &= (~not_available_due_to_cycle)
  available_pack_hunt_ships &= (~not_available_due_to_temp_collect)
  
  # Of the remaining list: identify 'max_standard_ships_hunting_season' ships
  # that are available to gather halite/attack bases.
  # Preferably select ships that were also selected for these modes in the
  # previous step and have halite on board.
  # Only change the gather/attack ships if one of my gatherers was destroyed
  # Assign a new gatherer if my gatherer is assigned to the base camping
  # attack or defense (These ships tend to be indefinitely unavailable), or if
  # the ship was destroyed.
  # Prefer non-zero halite ships for the initial gathering ships.
  my_ship_pos_to_k = {v[0]: k for k, v in player_obs[2].items()}
  available_positions = np.where(available_pack_hunt_ships)
  num_available_ships = available_pack_hunt_ships.sum()
  standard_ships = []
  if num_available_ships > 0:
    best_standard_scores = np.zeros(num_available_ships)
    pos_keys = []
    for i in range(num_available_ships):
      row = available_positions[0][i]
      col = available_positions[1][i]
      pos_key = my_ship_pos_to_k[row*grid_size+col]
      best_standard_scores[i] = all_ship_scores[pos_key][0].max() - 1e6*(
        halite_ships[row, col] == 0)
      pos_keys.append(pos_key)
    if hunting_season_started:
      already_included_ids = np.zeros(num_available_ships, dtype=np.bool)
      for ship_k in prev_standard_ships:
        if ship_k in player_obs[2]:
          # The ship still exists and was a standard ship in the previous step
          row, col = row_col_from_square_grid_pos(
            player_obs[2][ship_k][0], grid_size)
          if my_defend_base_ship_positions[row, col] or boxing_in_mission[
              row, col] or on_rescue_mission[row, col] or (
                not_available_due_to_cycle[row, col]):
            # We can use the ship for collecting soon (now it is rescuing or
            # boxing in or defending the base)
            standard_ships.append(ship_k)
          elif available_pack_hunt_ships[row, col]:
            # The ship is available now. Flag it for exclusion so it doesn't
            # get added twice
            standard_ships.append(ship_k)
            match_id = np.where((available_positions[0] == row) & (
              available_positions[1] == col))[0][0]
            already_included_ids[match_id] = True
          else:
            # The ship is now used for base camping or base conversion or
            # additional collection
            # Exclude it from the standard ships group
            assert not_available_due_to_camping[row, col] or (
              convert_unavailable_positions[row, col]) or (
                not_available_due_to_temp_collect[row, col])
            
      best_standard_scores = best_standard_scores[~already_included_ids]
      available_positions = (available_positions[0][~already_included_ids],
                             available_positions[1][~already_included_ids])
      pos_keys = np.array(pos_keys)[~already_included_ids].tolist()
      num_unassigned = max_standard_ships_hunting_season - len(standard_ships)
      # if num_unassigned > 0:
      #   import pdb; pdb.set_trace()
      num_to_assign = min(num_unassigned, num_available_ships)
      num_to_assign_phase_1 = max(0, num_to_assign - config[
        'max_standard_ships_decided_end_pack_hunting'])
      num_to_assign_phase_2 = num_to_assign-num_to_assign_phase_1
      num_available_ships = best_standard_scores.size
    else:
      num_to_assign = max_standard_ships_hunting_season
      num_to_assign_phase_1 = min(num_to_assign, num_available_ships)
      num_to_assign_phase_2 = 0
      
    # Assign the remaining standard ships
    # Assign the available ships with the highest collect score for collecting
    # (preferably non zero halite ships)
    best_standard_ids = np.argsort(-best_standard_scores)[
      :num_to_assign_phase_1]
    for standard_id in best_standard_ids:
      standard_row = available_positions[0][standard_id]
      standard_col = available_positions[1][standard_id]
      standard_key = pos_keys[standard_id]
      assert not standard_key in standard_ships
      standard_ships.append(standard_key)
      
    # Mark the standard ships as unavailable for pack hunting
    for standard_key in standard_ships:
      standard_row, standard_col = row_col_from_square_grid_pos(
        player_obs[2][standard_key][0], grid_size)
      available_pack_hunt_ships[standard_row, standard_col] = 0
    
  # The remaining ships are considered for pack hunting. Send the ones with
  # halite on board to a base.
  considered_hunting_ships_pos = np.where(available_pack_hunt_ships)
  num_available_ships = available_pack_hunt_ships.sum()
  for i in range(num_available_ships):
    row = considered_hunting_ships_pos[0][i]
    col = considered_hunting_ships_pos[1][i]
    if halite_ships[row, col] > 0:
      available_pack_hunt_ships[row, col] = 0
      ship_k = my_ship_pos_to_k[row*grid_size+col]
      # Let the ship collect at will (but prefer to go to a base sooner rather
      # than later) before joining the pack hunt (after touching base)
      for j in [2, 3]:
        all_ship_scores[ship_k][j][:] = -1e6
        
      # Let a hoarding ship gather when it is safe to do so
      if (obs_halite[row, col] < 100 or safe_collect_margin[
          row, col] <= 0) and (
            not ship_k in history['temporary_hoarding_collect_ships']):
        # FUTURE WORK: Make the multiplier a function of the opponent
        # aggression level?
        all_ship_scores[ship_k][1][:] *= 4
      elif not ship_k in history['temporary_hoarding_collect_ships'] and (
          safe_collect_margin[row, col] > 0):
        history['temporary_hoarding_collect_ships'].append(ship_k)
        # print(observation['step'], row, col, ship_k,
        #       "Temporarily collecting")
      
  # Ignore ships for hunting that are already being boxed in with my box-in
  # ships
  box_opponent_mask = np.zeros((grid_size, grid_size), dtype=np.bool)
  for boxed_target_row, boxed_target_col in box_opponent_positions:
    box_opponent_mask[boxed_target_row, boxed_target_col] = 1
      
  # Consider what to do with the zero halite ships that are available for
  # hunting.
  # First attempt: do something similar as mzotkiew
  # Main idea: move to the nearest non zero halite opponent if that direction
  # is valid.
  opponent_ships = stacked_ships[1:].sum(0) > 0
  potential_targets = opponent_ships & (halite_ships > 0) & (
    ~box_opponent_mask)
  hunting_ships_pos = np.where(available_pack_hunt_ships)
  num_hunting_ships = available_pack_hunt_ships.sum()
  
  # Exclude targets that I am willfully letting convert
  for (ignore_convert_row, ignore_convert_col) in ignore_convert_positions:
    potential_targets[ignore_convert_row, ignore_convert_col] = 0
  
  # Exclude targets that have a safe path to one of their bases
  if num_hunting_ships > 0:
    stacked_bases = np.stack(
      [rbs[1] for rbs in observation['rewards_bases_ships']])
    nearest_opponent_base_distances = [None]
    for opponent_id in range(1, num_players):
      num_bases = stacked_bases[opponent_id].sum()
      opponent_base_locations = np.where(stacked_bases[opponent_id])
      all_opponent_base_distances = [DISTANCES[
        opponent_base_locations[0][i], opponent_base_locations[1][i]] for i in (
          range(num_bases))] + [99*np.ones((grid_size, grid_size))]
      nearest_opponent_base_distances.append(np.stack(
        all_opponent_base_distances).min(0))
    considered_targets_pos = np.where(potential_targets)
    for j in range(potential_targets.sum()):
      target_row = considered_targets_pos[0][j]
      target_col = considered_targets_pos[1][j]
      opponent_id = np.where(stacked_ships[:, target_row, target_col])[0][0]
      opp_nearest_base_distances = nearest_opponent_base_distances[opponent_id]
      target_distance_to_nearest_base = opp_nearest_base_distances[
        target_row, target_col]
      my_min_distance_to_opp_nearest_base = opp_nearest_base_distances[
        hunting_ships_pos].min()
      
      # if target_row == 20 and target_col == 12 and observation['step'] == 160:
      #   import pdb; pdb.set_trace()
      
      if target_distance_to_nearest_base < my_min_distance_to_opp_nearest_base:
        potential_targets[target_row, target_col] = 0
  
  potential_targets_pos = np.where(potential_targets)
  num_potential_targets = potential_targets.sum()
  # print(observation['step'], num_hunting_ships, num_potential_targets)
  hoarded_one_step_opponent_keys = []
  if num_potential_targets > 0 and num_hunting_ships > 0:
    # print(observation['step'])
    ordered_ship_keys = []
    all_target_distances = np.zeros((num_hunting_ships, num_potential_targets))
    for i in range(num_hunting_ships):
      row = hunting_ships_pos[0][i]
      col = hunting_ships_pos[1][i]
      ship_k = my_ship_pos_to_k[row*grid_size+col]
      ordered_ship_keys.append(ship_k)
      potential_target_distances = DISTANCES[row, col][potential_targets]
      
      # Update the target distances to only include potential targets that
      # correspond with valid actions
      potential_targets_rows = potential_targets_pos[0]
      potential_targets_cols = potential_targets_pos[1]
      south_dist = np.where(
        potential_targets_rows >= row, potential_targets_rows-row,
        potential_targets_rows-row+grid_size)
      east_dist = np.where(
        potential_targets_cols >= col, potential_targets_cols-col,
        potential_targets_cols-col+grid_size)
      valid_directions = all_ship_scores[ship_k][6]
      valid_move_counts = 2*np.ones(num_potential_targets)
      for d in NOT_NONE_DIRECTIONS:
        if not d in valid_directions:
          if d == NORTH:
            decrement_ids = south_dist >= grid_size/2
          elif d == SOUTH:
            decrement_ids = (south_dist <= grid_size/2) & (south_dist > 0)
          elif d == EAST:
            decrement_ids = (east_dist <= grid_size/2) & (east_dist > 0)
          elif d == WEST:
            decrement_ids = east_dist >= grid_size/2
          valid_move_counts[decrement_ids] -= 1
        
      # Handle the case of being in the same row or column    
      valid_move_counts[south_dist == 0] -= 1
      valid_move_counts[east_dist == 0] -= 1
        
      # if observation['step'] == 91 and row == 6 and col == 3:
      #   import pdb; pdb.set_trace()
      
      assert np.all(valid_move_counts >= 0)
      potential_target_distances[valid_move_counts == 0] += 100
      all_target_distances[i] = potential_target_distances
      
    opponent_num_escape_directions = np.zeros(num_potential_targets)
    for j in range(num_potential_targets):
      target_row = potential_targets_pos[0][j]
      target_col = potential_targets_pos[1][j]
      opponent_num_escape_directions[j] = len(opponent_ships_sensible_actions[
        target_row, target_col])
      
    # First coordinate my hunters to ships that have no valid escape directions
    hunting_ships_available = np.ones(num_hunting_ships, dtype=np.bool)
    for j in range(num_potential_targets):
      num_escape_dirs = opponent_num_escape_directions[j]
      if num_escape_dirs == 0:
        target_row = potential_targets_pos[0][j]
        target_col = potential_targets_pos[1][j]
        
        # Loop over my hunting ships at a distance of max two and take as many
        # of the potential escape squares as possible
        my_near_ships = np.where((all_target_distances[:, j] <= 2) & (
          hunting_ships_available))[0]
        num_my_near_ships = my_near_ships.size
        if num_my_near_ships > 0:
          # You can always attack at most 2 escape squares. Keep track of what
          # escape squares each of my ships can attack without collecting
          # halite on the next turn (ignoring ship collision halite gain)
          my_target_relative_attack_dirs = np.zeros((num_my_near_ships, 5))
          for loop_id, my_ship_id in enumerate(my_near_ships):
            row = hunting_ships_pos[0][my_ship_id]
            col = hunting_ships_pos[1][my_ship_id]
            ship_k = my_ship_pos_to_k[row*grid_size+col]
            valid_attack_dirs = all_ship_scores[ship_k][6]
            considered_attack_dirs = get_dir_from_target(
              row, col, target_row, target_col, grid_size)
            if all_target_distances[my_ship_id, j] == 1 and (
                obs_halite[row, col] == 0):
              considered_attack_dirs.append(None)
            attack_dirs = list(set(considered_attack_dirs) & set(
              valid_attack_dirs))
              
            # Get the relative directions from the target that I can attack
            for d in attack_dirs:
              move_row, move_col = move_ship_row_col(
                row, col, d, grid_size)
              relative_covered_dir = MOVE_DIRECTIONS_TO_ID[get_dir_from_target(
                target_row, target_col, move_row, move_col, grid_size)[0]]
              my_target_relative_attack_dirs[loop_id, relative_covered_dir] = 1
            
          direction_covered = np.zeros(len(MOVE_DIRECTIONS), dtype=np.bool)
          for dir_id, d in enumerate(MOVE_DIRECTIONS):
            rel_target_row, rel_target_col = move_ship_row_col(
              target_row, target_col, d, grid_size)
            if override_move_squares_taken[rel_target_row, rel_target_col]:
              direction_covered[dir_id] = 1
            
          # First, handle the ships that can only attack a single square that
          # is not covered yet
          my_target_relative_attack_dirs[:, direction_covered] = 0
          
          # if observation['step'] == 149:
          #   import pdb; pdb.set_trace()
          
          # Greedily loop over directions by ordering the count of the number
          # of ships that cover. Prefer low but strictly positive directions.
          while my_target_relative_attack_dirs.sum() > 0:
            ship_num_possible_attacks = my_target_relative_attack_dirs.sum(1)
            dir_num_possible_attacks = my_target_relative_attack_dirs.sum(0)
            # The None action is slightly preferred since that guarantees a max
            # distance of 1 on the next turn
            dir_num_possible_attacks[0] -= 0.1
            
            non_zero_min_count = dir_num_possible_attacks[
              dir_num_possible_attacks > 0].min()
            best_dir_ids = np.where(dir_num_possible_attacks == (
              non_zero_min_count))[0]
            dir_id = np_rng.choice(best_dir_ids)
            considered_ships = np.where(
              my_target_relative_attack_dirs[:, dir_id])[0]
            
            # Break ties with the number of directions each ship covers
            cover_ship_scores = ship_num_possible_attacks[considered_ships]
            considered_ships_attacker_id = considered_ships[
              np.argmin(cover_ship_scores)]
            attacker_id = my_near_ships[considered_ships_attacker_id]
            
            # Move my ship to the relative position of the target
            rel_target_row, rel_target_col = move_ship_row_col(
              target_row, target_col, MOVE_DIRECTIONS[dir_id], grid_size)
            attacker_row = hunting_ships_pos[0][attacker_id]
            attacker_col = hunting_ships_pos[1][attacker_id]
            ship_k = my_ship_pos_to_k[attacker_row*grid_size+attacker_col]
            all_ship_scores[ship_k][0][rel_target_row, rel_target_col] = 2e5
            override_move_squares_taken[rel_target_row, rel_target_col] = 1
            hunting_ships_available[attacker_id] = 0
            
            # Update the attack dir counts
            my_target_relative_attack_dirs[considered_ships_attacker_id] = 0
            my_target_relative_attack_dirs[:, dir_id] = 0
      
    # if observation['step'] == 188:
    #   import pdb; pdb.set_trace()
      
    # Next, coordinate my hunters to ships that have a single moving escape
    # direction.
    # These ship are preferred targets since it is likely that I can soon box
    # them in, especially if it is me who cuts off three of the move directions
    # Order the ships so that the ships that had a single escape direction in
    # the previous step are handled first, so we can coordinate the
    # interception
    one_step_opponent_ids = np.arange(num_potential_targets).tolist()
    priority_ids = []
    for opponent_ship_k in history['prev_step_hoarded_one_step_opponent_keys']:
      if opponent_ship_k in ship_key_to_pos:
        opponent_pos = ship_key_to_pos[opponent_ship_k]
        target_row, target_col = row_col_from_square_grid_pos(
          opponent_pos, grid_size)
        if potential_targets[target_row, target_col]:
          # We need to check because the target may no longer be available for
          # pack hunting due to boxing in or getting close to a friendly base
          opponent_priority_id = np.where(
            (potential_targets_pos[0] == target_row) & (
              potential_targets_pos[1] == target_col))[0][0]
          priority_ids.append(opponent_priority_id)
    remaining_ids = list(set(one_step_opponent_ids) - set(priority_ids))
    remaining_ids.sort()  # Set intersect can be flaky
    one_step_opponent_ids = priority_ids + remaining_ids
      
    one_step_opponent_positions_directions = []
    for j in one_step_opponent_ids:
      target_row = potential_targets_pos[0][j]
      target_col = potential_targets_pos[1][j]
      target_escape_directions = opponent_ships_sensible_actions[
        target_row, target_col]
      move_escape_directions = copy.copy(target_escape_directions)
      if (0, 0) in move_escape_directions:
        move_escape_directions.remove((0, 0))
      num_move_escape_dirs = len(move_escape_directions)
      # nearest_target_distances = np.tile(
      #   all_target_distances.min(1)[:, None], [1, num_potential_targets])
      
      if num_move_escape_dirs == 1:
        # The <= ensures we consider piling up on inidividual ships
        potential_nearby_attackers = np.where(hunting_ships_available & (
          all_target_distances[:, j] <= 2))[0]
        if potential_nearby_attackers.size >= 2:
          
          # Figure out if I have at least one available ship at a max distance
          # of 2 that can push the opponent in one direction
          escape_dir = RELATIVE_DIR_TO_DIRECTION_MAPPING[
            move_escape_directions[0]]
          potential_nearby_distances = all_target_distances[
            potential_nearby_attackers, j]
          if potential_nearby_distances.min() == 1:
            # The None direction is covered - verify the other directions
            uncovered_dirs = copy.copy(NOT_NONE_DIRECTIONS)
            uncovered_dirs.remove(escape_dir)
            ignore_attacker_ids = []
            d1_hunters = []
            for attacker_id in potential_nearby_attackers:
              attacker_row = hunting_ships_pos[0][attacker_id]
              attacker_col = hunting_ships_pos[1][attacker_id]
              ship_k = my_ship_pos_to_k[attacker_row*grid_size+attacker_col]
              valid_directions = all_ship_scores[ship_k][6]
              if escape_dir in valid_directions:
                threat_dirs = get_dir_from_target(
                  target_row, target_col, attacker_row, attacker_col,
                  grid_size)
                uncovered_dirs = list(set(uncovered_dirs) - set(threat_dirs))
                if DISTANCES[target_row, target_col][
                    attacker_row, attacker_col] == 1:
                  d1_hunters.append(attacker_id)
              else:
                ignore_attacker_ids.append(attacker_id)
            
            if len(uncovered_dirs) == 0 or (
                len(uncovered_dirs) == 1 and len(d1_hunters) > 1):
              one_step_opponent_positions_directions.append((
                target_row, target_col, escape_dir))
              opponent_ship_k = ship_pos_to_key[
                target_row*grid_size+target_col]
              hoarded_one_step_opponent_keys.append(opponent_ship_k)
              
              if len(uncovered_dirs) > 0:
                potential_nearby_attackers = d1_hunters
              
              # Move the attackers in the single escape direction
              # import pdb; pdb.set_trace()
              for attacker_id in potential_nearby_attackers:
                if not attacker_id in ignore_attacker_ids:
                  attacker_row = hunting_ships_pos[0][attacker_id]
                  attacker_col = hunting_ships_pos[1][attacker_id]
                  move_row, move_col = move_ship_row_col(
                    attacker_row, attacker_col, escape_dir, grid_size)
                  ship_k = my_ship_pos_to_k[
                    attacker_row*grid_size+attacker_col]
                  all_ship_scores[ship_k][0][move_row, move_col] = 2e5
                  override_move_squares_taken[move_row, move_col] = 1
                  hunting_ships_available[attacker_id] = 0
          
    # Try to get into a position where the opponent can only move in one
    # direction (from two to one escape direction)
    for j in range(num_potential_targets):
      num_escape_dirs = opponent_num_escape_directions[j]
      if num_escape_dirs == 2:
        target_row = potential_targets_pos[0][j]
        target_col = potential_targets_pos[1][j]
        potential_nearby_attackers = np.where(hunting_ships_available & (
          all_target_distances[:, j] == 1))[0]
        attack_selected = False
        if potential_nearby_attackers.size == 2:
          escape_directions = opponent_ships_sensible_actions[
            target_row, target_col]
          if (escape_directions[0][0] == 0 and (
              escape_directions[1][0] == 0)) or (
                escape_directions[0][1] == 0 and (
                  escape_directions[1][1] == 0)):
            # Scenario: ME | OPPONENT | ME - guess the action and then chase
                  
            # Guess the opponent's next action
            opponent_id = np.where(
              stacked_ships[:, target_row, target_col])[0][0]
            escape_dir_scores = np.zeros(2)
            for escape_id, escape_dir in enumerate(escape_directions):
              move_row, move_col = move_ship_row_col(
                target_row, target_col, RELATIVE_DIR_TO_DIRECTION_MAPPING[
                  escape_dir], grid_size)
              opponent_influence = player_influence_maps[opponent_id][
                move_row, move_col]
              my_influence = player_influence_maps[0][move_row, move_col]
              escape_dir_scores[escape_id] = opponent_influence-my_influence
              
            likely_opponent_move = RELATIVE_DIR_TO_DIRECTION_MAPPING[
              escape_directions[np.argmax(escape_dir_scores)]]
            
            # Only continue if both my ships can move in the selected
            # directions 
            both_can_move = True
            can_stay = np.zeros(2, dtype=np.bool)
            for attacker_0_or_1, attacker_id in enumerate(
                potential_nearby_attackers):
              attacker_row = hunting_ships_pos[0][attacker_id]
              attacker_col = hunting_ships_pos[1][attacker_id]
              ship_k = my_ship_pos_to_k[attacker_row*grid_size+attacker_col]
              both_can_move = both_can_move and likely_opponent_move in (
                all_ship_scores[ship_k][6])
              can_stay[attacker_0_or_1] = obs_halite[
                attacker_row, attacker_col] == 0
              
            if both_can_move:
              # If both are on non zero halite squares: move both in the likely
              # escape direction. Otherwise, select a random ship to move in
              # the escape direction where the ship that remains in place has
              # no halite at the considered square
              if not np.any(can_stay):
                stay_in_place_ids = []
              else:
                stay_in_place_ids = [np_rng.choice(potential_nearby_attackers[
                  can_stay])]
                
              for attacker_id in potential_nearby_attackers:
                # import pdb; pdb.set_trace()
                attacker_row = hunting_ships_pos[0][attacker_id]
                attacker_col = hunting_ships_pos[1][attacker_id]
                move_dir = None if attacker_id in stay_in_place_ids else (
                  likely_opponent_move)
                move_row, move_col = move_ship_row_col(
                  attacker_row, attacker_col, move_dir, grid_size)
                ship_k = my_ship_pos_to_k[
                  attacker_row*grid_size+attacker_col]
                all_ship_scores[ship_k][0][move_row, move_col] = 2e5
                override_move_squares_taken[move_row, move_col] = 1
                hunting_ships_available[attacker_id] = 0
              attack_selected = True
                
        escape_directions = opponent_ships_sensible_actions[
          target_row, target_col]
        if not attack_selected and not (0, 0) in escape_directions and len(
            escape_directions) == 2:
          # Scenario: ME | OPPONENT | |  ME - guess the action and then chase
          available_nearby = np.where(hunting_ships_available & (
            all_target_distances[:, j] <= 2))[0]
          
          if available_nearby.size >= 2:
            attacker_rows = hunting_ships_pos[0][available_nearby]
            attacker_cols = hunting_ships_pos[1][available_nearby]
            north_dist = np.where(
              target_row >= attacker_rows, target_row-attacker_rows,
              target_row-attacker_rows+grid_size)
            vert_rel_pos = np.where(
              north_dist < 3, north_dist, north_dist-grid_size)
            west_dist = np.where(
              target_col >= attacker_cols, target_col-attacker_cols,
              target_col-attacker_cols+grid_size)
            horiz_rel_pos = np.where(
              west_dist < 3, west_dist, west_dist-grid_size)
            same_row_ids = (vert_rel_pos == 0)
            same_col_ids = (horiz_rel_pos == 0)
            
            consider_attack = False
            if np.any(horiz_rel_pos[same_row_ids] < 0) and np.any(
                horiz_rel_pos[same_row_ids] > 0):
              if np.any(horiz_rel_pos[same_row_ids] == 1) and np.any(
                  horiz_rel_pos[same_row_ids] == -2):
                move_to_target_id = available_nearby[same_row_ids][np.where(
                  horiz_rel_pos[same_row_ids] == -2)[0][0]]
                move_escape_id = available_nearby[same_row_ids][np.where(
                  horiz_rel_pos[same_row_ids] == 1)[0][0]]
                consider_attack = True
              elif np.any(horiz_rel_pos[same_row_ids] == -1) and np.any(
                  horiz_rel_pos[same_row_ids] == 2):
                move_to_target_id = available_nearby[same_row_ids][np.where(
                  horiz_rel_pos[same_row_ids] == 2)[0][0]]
                move_escape_id = available_nearby[same_row_ids][np.where(
                  horiz_rel_pos[same_row_ids] == -1)[0][0]]
                consider_attack = True
            elif np.any(vert_rel_pos[same_col_ids] < 0) and np.any(
                vert_rel_pos[same_col_ids] > 0):
              if np.any(vert_rel_pos[same_col_ids] == 1) and np.any(
                  vert_rel_pos[same_col_ids] == -2):
                move_to_target_id = available_nearby[same_col_ids][np.where(
                  vert_rel_pos[same_col_ids] == -2)[0][0]]
                move_escape_id = available_nearby[same_col_ids][np.where(
                  vert_rel_pos[same_col_ids] == 1)[0][0]]
                consider_attack = True
              elif np.any(vert_rel_pos[same_col_ids] == -1) and np.any(
                  vert_rel_pos[same_col_ids] == 2):
                move_to_target_id = available_nearby[same_col_ids][np.where(
                  vert_rel_pos[same_col_ids] == 2)[0][0]]
                move_escape_id = available_nearby[same_col_ids][np.where(
                  vert_rel_pos[same_col_ids] == -1)[0][0]]
                consider_attack = True
              
            if consider_attack:
              opponent_id = np.where(
                stacked_ships[:, target_row, target_col])[0][0]
              escape_dir_scores = np.zeros(2)
              for escape_id, escape_dir in enumerate(escape_directions):
                move_row, move_col = move_ship_row_col(
                  target_row, target_col, RELATIVE_DIR_TO_DIRECTION_MAPPING[
                    escape_dir], grid_size)
                opponent_influence = player_influence_maps[opponent_id][
                  move_row, move_col]
                my_influence = player_influence_maps[0][move_row, move_col]
                escape_dir_scores[escape_id] = opponent_influence-my_influence
                
              likely_opponent_move = RELATIVE_DIR_TO_DIRECTION_MAPPING[
                escape_directions[np.argmax(escape_dir_scores)]]
              
              # print(observation['step'], target_row, target_col)
              attacker_escape_row = hunting_ships_pos[0][move_escape_id]
              attacker_escape_col = hunting_ships_pos[1][move_escape_id]
              attacker_to_target_row = hunting_ships_pos[0][move_to_target_id]
              attacker_to_target_col = hunting_ships_pos[1][move_to_target_id]
              move_escape_row, move_escape_col = move_ship_row_col(
                attacker_escape_row, attacker_escape_col, likely_opponent_move,
                grid_size)
              to_target_dir = get_dir_from_target(
                  attacker_to_target_row, attacker_to_target_col,
                  target_row, target_col, grid_size)[0]
              move_to_target_row, move_to_target_col = move_ship_row_col(
                attacker_to_target_row, attacker_to_target_col,
                to_target_dir, grid_size)
              ship_escape_k = my_ship_pos_to_k[
                attacker_escape_row*grid_size+attacker_escape_col]
              ship_to_target_k = my_ship_pos_to_k[
                attacker_to_target_row*grid_size+attacker_to_target_col]
              
              if likely_opponent_move in all_ship_scores[ship_escape_k][6] and(
                  to_target_dir in all_ship_scores[ship_to_target_k][6]) and(
                    not override_move_squares_taken[
                      move_escape_row, move_escape_col]) and not (
                        override_move_squares_taken[
                          move_to_target_row, move_to_target_col]):
                all_ship_scores[ship_escape_k][0][
                  move_escape_row, move_escape_col] = 2e5
                all_ship_scores[ship_to_target_k][0][
                  move_to_target_row, move_to_target_col] = 2e5
                override_move_squares_taken[
                  move_escape_row, move_escape_col] = 1
                override_move_squares_taken[
                  move_to_target_row, move_to_target_col] = 1
                hunting_ships_available[move_escape_id] = 0
                hunting_ships_available[move_to_target_id] = 0
            
        
    # Intercept ships that are pushed in one direction to avoid chasing forever
    for target_row, target_col, escape_dir in (
        one_step_opponent_positions_directions):
      # Try to move perpendicular to the escaping ship if I can catch it in
      # time
      attacker_rows = hunting_ships_pos[0]
      attacker_cols = hunting_ships_pos[1]
      north_dist = np.where(
        target_row >= attacker_rows, target_row-attacker_rows,
        target_row-attacker_rows+grid_size)
      west_dist = np.where(
        target_col >= attacker_cols, target_col-attacker_cols,
        target_col-attacker_cols+grid_size)
      if escape_dir in [NORTH, SOUTH]:
        perpendicular_distances = np.minimum(west_dist, grid_size-west_dist)
        if escape_dir == SOUTH:
          direction_distances = grid_size-north_dist
        else:
          direction_distances = north_dist
      else:
        perpendicular_distances = np.minimum(north_dist, grid_size-north_dist)
        if escape_dir == EAST:
          direction_distances = grid_size-west_dist
        else:
          direction_distances = west_dist
        
      potential_nearby_attackers = np.where(hunting_ships_available & (
        direction_distances >= perpendicular_distances))[0]
      if potential_nearby_attackers.size > 0:
        potential_crossing_min_steps = np.ceil((
          direction_distances[potential_nearby_attackers]+(
            perpendicular_distances[potential_nearby_attackers]))/2)
        min_crossing_distance = potential_crossing_min_steps.min().astype(np.int)
        # FUTURE WORK: discard if there is a base on the escape track
        if min_crossing_distance <= 6:
          attacker_id = potential_nearby_attackers[
            np.argmin(potential_crossing_min_steps)]
          attacker_row = hunting_ships_pos[0][attacker_id]
          attacker_col = hunting_ships_pos[1][attacker_id]
          if escape_dir == NORTH:
            intersect_row = (target_row-min_crossing_distance) % grid_size
            intersect_col = target_col
          elif escape_dir == SOUTH:
            intersect_row = (target_row+min_crossing_distance) % grid_size
            intersect_col = target_col
          elif escape_dir == EAST:
            intersect_row = target_row
            intersect_col = (target_col+min_crossing_distance) % grid_size
          elif escape_dir == WEST:
            intersect_row = target_row
            intersect_col = (target_col-min_crossing_distance) % grid_size
          ship_k = my_ship_pos_to_k[
            attacker_row*grid_size+attacker_col]
          intersect_bonus_mask = get_mask_between_exclude_ends(
            attacker_row, attacker_col, intersect_row, intersect_col,
            grid_size)
          intersect_bonus = 1e5*intersect_bonus_mask
          if intersect_bonus_mask.sum() > 1:
            # Prefer to move to low halite squares in order to avoid conflicts
            # with collect ships
            intersect_bonus[intersect_bonus_mask] -= 10*(np.minimum(
              1000, 10*obs_halite[intersect_bonus_mask])+obs_halite[
                intersect_bonus_mask]/10)
            # Give a small penalty to same rows or columns in order to allow more
            # move options downstream
            intersect_bonus[row] -= 1
            intersect_bonus[:, col] -= 1
          
          all_ship_scores[ship_k][0][:] += intersect_bonus
          # override_move_squares_taken[move_row, move_col] = 1
          # import pdb; pdb.set_trace()
          hunting_ships_available[attacker_id] = 0
          
    # Try to cut off the preferred escape direction for opponent ships that
    # only have a stand still valid action
    for j in range(num_potential_targets):
      target_row = potential_targets_pos[0][j]
      target_col = potential_targets_pos[1][j]
      target_escape_directions = opponent_ships_sensible_actions[
        target_row, target_col]
      if (0, 0) in target_escape_directions and len(
          target_escape_directions) == 1:
        potential_nearby_attackers = np.where(hunting_ships_available & (
          all_target_distances[:, j] <= 2))[0]
        num_potential_attackers = potential_nearby_attackers.size
        if num_potential_attackers > 0:
          can_cover_dirs = {d: [] for d in NOT_NONE_DIRECTIONS}
          for attacker_id in potential_nearby_attackers:
            attacker_row = hunting_ships_pos[0][attacker_id]
            attacker_col = hunting_ships_pos[1][attacker_id]
            ship_k = my_ship_pos_to_k[attacker_row*grid_size+attacker_col]
            for d in NOT_NONE_DIRECTIONS:
              if d in all_ship_scores[ship_k][6]:
                move_attacker_row, move_attacker_col = move_ship_row_col(
                  attacker_row, attacker_col, d, grid_size)
                if DISTANCES[target_row, target_col][
                    move_attacker_row, move_attacker_col] == 1:
                  covered_dir = get_dir_from_target(
                      target_row, target_col, move_attacker_row,
                      move_attacker_col, grid_size)[0]
                  can_cover_dirs[covered_dir].append(attacker_id)
                
          cover_dir_scores = np.zeros(len(NOT_NONE_DIRECTIONS))
          opponent_id = np.where(
              stacked_ships[:, target_row, target_col])[0][0]
          for dir_id, d in enumerate(NOT_NONE_DIRECTIONS):
            move_row, move_col = move_ship_row_col(
                target_row, target_col, d, grid_size)
            opponent_influence = player_influence_maps[opponent_id][
                move_row, move_col]
            not_opponent_influence = player_influence_maps[
              :, move_row, move_col].sum() - opponent_influence
            cover_dir_scores[dir_id] = (
              opponent_influence-not_opponent_influence)
            
          # Greedily cover escape directions by the order of preference for the
          # target ship
          cover_dir_argsort = np.argsort(-cover_dir_scores)
          for dir_id in cover_dir_argsort:
            d = NOT_NONE_DIRECTIONS[dir_id]
            covered_row, covered_col = move_ship_row_col(
                target_row, target_col, d, grid_size)
            considered_attacker_ids = can_cover_dirs[d]
            # if considered_attacker_ids and override_move_squares_taken[
            #     covered_row, covered_col]:
            #   import pdb; pdb.set_trace()
            
            if considered_attacker_ids and not override_move_squares_taken[
                covered_row, covered_col]:
              # Prefer my attackers that can not attack other escape squares
              # (on the same row or col at a distance of 2)
              num_considered_attackers = len(considered_attacker_ids)
              considered_attack_scores = np.zeros(num_considered_attackers)
              for cons_attack_id in range(num_considered_attackers):
                attacker_id = considered_attacker_ids[cons_attack_id]
                attacker_row = hunting_ships_pos[0][attacker_id]
                attacker_col = hunting_ships_pos[1][attacker_id]
                considered_attack_scores[cons_attack_id] = int(
                  (attacker_row != target_row) and (
                    attacker_col != target_col))
              selected_attacker_id = considered_attacker_ids[
                np.argmin(considered_attack_scores)]
              hunting_ships_available[selected_attacker_id] = 0
              attacker_row = hunting_ships_pos[0][selected_attacker_id]
              attacker_col = hunting_ships_pos[1][selected_attacker_id]
              ship_k = my_ship_pos_to_k[attacker_row*grid_size+attacker_col]
              # import pdb; pdb.set_trace()
              all_ship_scores[ship_k][0][covered_row, covered_col] = 2e5
              override_move_squares_taken[covered_row, covered_col] = 1
              for d_ in NOT_NONE_DIRECTIONS:
                if selected_attacker_id in can_cover_dirs[d_]:
                  can_cover_dirs[d_].remove(selected_attacker_id)
              
      # FUTURE WORK: Try to cut off the preferred escape direction for opponent
      # ships that only have a single non stand still valid action
      
    # Assign the remaining standard ships
    if num_to_assign_phase_2 > 0 and hunting_ships_available.sum() > 0:
      available_hunting_ids = np.where(hunting_ships_available)[0]
      num_remaining_available_ships = available_hunting_ids.size
      best_standard_scores = np.zeros(num_remaining_available_ships)
      pos_keys = []
      for i in range(num_remaining_available_ships):
        row = hunting_ships_pos[0][available_hunting_ids[i]]
        col = hunting_ships_pos[1][available_hunting_ids[i]]
        pos_key = my_ship_pos_to_k[row*grid_size+col]
        best_standard_scores[i] = all_ship_scores[pos_key][0].max() - 1e6*(
          halite_ships[row, col] == 0)
        pos_keys.append(pos_key)
        
      # Assign the remaining collect ships
      # Assign the available ships with the highest collect score for collecting
      # (preferably non zero halite ships)
      best_standard_ids = np.argsort(
        -best_standard_scores)[:num_to_assign_phase_2]
      for standard_id in best_standard_ids:
        standard_row = hunting_ships_pos[0][available_hunting_ids[standard_id]]
        standard_col = hunting_ships_pos[1][available_hunting_ids[standard_id]]
        # print("Newly assigned standard ship", standard_row, standard_col)
        standard_key = pos_keys[standard_id]
        assert not standard_key in standard_ships
        standard_ships.append(standard_key)
        hunting_ships_available[available_hunting_ids[standard_id]] = False
      
    # Coordinate the remaining hunting actions based on the potential target
    # distances. Balance with staying closer to my center of mass of ships
    # with halite and my bases
    # FUTURE WORK: make this work
    # my_vulnerable_score = (
    #   observation['rewards_bases_ships'][0][2] & (halite_ships > 0)) + 3*(
    #     observation['rewards_bases_ships'][0][1])
    # Average the vulnerable map over time so that the center of mass is more
    # stable
    # my_time_averaged_vulnerable_score = 0.8*history[
    #   'my_time_averaged_vulnerable_score'] + 0.2*my_vulnerable_score
    # history['my_time_averaged_vulnerable_score'] = (
    #   my_time_averaged_vulnerable_score)
    # smoothed_vulnerable_score = smooth2d(my_time_averaged_vulnerable_score)
    # center_of_vulnerable_mass = np.unravel_index(
    #   smoothed_vulnerable_score.argmax(), smoothed_vulnerable_score.shape)
    # print(observation['step'], center_of_vulnerable_mass)
        
    # FUTURE WORK: don't break ranks when hoarding
    stacked_bases = np.stack(
      [rbs[1] for rbs in observation['rewards_bases_ships']])
    unavailable_hunt_positions = (stacked_bases[1:].sum(0) > 0) | (
      override_move_squares_taken)
    for i in range(num_hunting_ships):
      if hunting_ships_available[i]:
        row = hunting_ships_pos[0][i]
        col = hunting_ships_pos[1][i]
        
        ship_k = my_ship_pos_to_k[row*grid_size+col]
        potential_target_distances_ship = all_target_distances[i]
        potential_target_distances_ship_weighted = (
          potential_target_distances_ship)
        # potential_target_distances_ship_weighted = (
        #   potential_target_distances_ship) + DISTANCES[
        #     center_of_vulnerable_mass][potential_targets_pos]/3
        if preferred_victim is not None and (
            potential_target_distances_ship.min() > 2):
          preferred_victim_ship_ids = player_ids[
            potential_targets_pos] == preferred_victim
          potential_target_distances_ship_weighted[
            preferred_victim_ship_ids] -= 10
        selected_target_id = np.argmin(
          potential_target_distances_ship_weighted)
        target_distance = potential_target_distances_ship[selected_target_id]
        
        if target_distance < 3 or obs_halite[row, col] < 100 or (
            safe_collect_margin[row, col] <= 0):
          target_row = potential_targets_rows[selected_target_id]
          target_col = potential_targets_cols[selected_target_id]
          hunting_bonus = 1e5*get_mask_between_exclude_ends(
            row, col, target_row, target_col, grid_size)
          if target_distance == 1 and not unavailable_hunt_positions[
              target_row, target_col]:
            hunting_bonus[target_row, target_col] = 1e5
          elif target_distance == 1:
            # Move in one of the at most 2 likely opponent next action
            # direction if that direction is still available. 
            # This means I have another hunter/boxer at distance one which has
            # already claimed the target square
            sensible_opponent_dirs = opponent_ships_sensible_actions[
              (target_row, target_col)]
            for d in NOT_NONE_DIRECTIONS:
              if RELATIVE_DIR_MAPPING[d] in sensible_opponent_dirs:
                move_row, move_col = move_ship_row_col(
                    row, col, d, grid_size)
                if not unavailable_hunt_positions[move_row, move_col]:
                  hunting_bonus[move_row, move_col] = 1e5
            
          # Prefer to move in the same direction as the target when I am
          # tracking the target closely
          opponent_ship_k = ship_pos_to_key[
            target_row*grid_size+target_col]
          if opponent_ship_k in prev_step_opponent_ship_moves and (
              target_distance <= 2):
            target_prev_move = prev_step_opponent_ship_moves[opponent_ship_k]
            bonus_rel_dir = RELATIVE_DIR_MAPPING[target_prev_move]
            bonus_rows = np.mod(row + bonus_rel_dir[0]*(
              1+np.arange(half_distance_mask_dim)), grid_size)
            bonus_cols = np.mod(col + bonus_rel_dir[1]*(
              1+np.arange(half_distance_mask_dim)), grid_size)
            hunting_bonus[(bonus_rows, bonus_cols)] *= 1.5
            
          if (hunting_bonus > 0).sum() > 1:
            # Prefer to move to low halite squares in order to avoid conflicts
            # with collect ships
            hunting_bonus[hunting_bonus > 0] -= 10*(np.minimum(
              1000, 10*obs_halite[hunting_bonus > 0]) + obs_halite[
                hunting_bonus > 0]/10)
            # Give a small penalty to same rows or columns in order to allow
            # more move options downstream
            hunting_bonus[row] -= 1
            hunting_bonus[:, col] -= 1
        else:
          # Override the pack hunt and collect at the current square
          hunting_bonus = np.zeros((grid_size, grid_size))
          if ship_k in history['temporary_hoarding_collect_ships']:
            print(observation['step'], row, col, ship_k,
                  "INVESTIGATE: ship in hoarding collect but also hoarding??")
          else:
            history['temporary_hoarding_collect_ships'].append(ship_k)
            # print(observation['step'], row, col, ship_k,
            #       "Temporarily collecting")
            hunting_bonus[row, col] = 11e4 # Consider 
            # import pdb; pd.set_trace()
            # x=1
        
        all_ship_scores[ship_k][0][:] += hunting_bonus
  
  # print(standard_ships, available_pack_hunt_ships.sum(),
  #       stacked_ships[0].sum())
  # print(observation['step'], history['temporary_hoarding_collect_ships'])
  history['hunting_season_standard_ships'] = standard_ships
  history['hunting_season_started'] = True
  history['prev_step_hoarded_one_step_opponent_keys'] = (
      hoarded_one_step_opponent_keys)
  
  # if observation['step'] == 192:
  #   import pdb; pdb.set_trace()
  
  return all_ship_scores, history, override_move_squares_taken

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

def get_my_guaranteed_safe_collect_squares(
    opponent_ships, grid_size, my_bases, obs_halite, collect_rate,
    halite_ships, observation, halite_on_board_mult=1e-6):
  opp_ship_locations = np.where(opponent_ships)
  nearest_opponent_stacked_distances = [
            99*np.ones((grid_size, grid_size))]
  for i in range(opponent_ships.sum()):
    opponent_row = opp_ship_locations[0][i]
    opponent_col = opp_ship_locations[1][i]
    opponent_ship_halite = max(0, halite_ships[opponent_row, opponent_col])
    opponent_distances = DISTANCES[opponent_row, opponent_col]
    nearest_opponent_stacked_distances.append(
      opponent_distances + halite_on_board_mult*opponent_ship_halite)
  nearest_opponent_distances = np.stack(
    nearest_opponent_stacked_distances).min(0)
  my_base_locations = np.where(my_bases)
  my_nearest_base_distances = [DISTANCES[
        my_base_locations[0][i], my_base_locations[1][i]] for i in range(
          my_bases.sum())]
  safe_to_collect = np.zeros((grid_size, grid_size), dtype=np.bool)
  safe_to_collect_margin = -1*np.ones((grid_size, grid_size), dtype=np.int)
  safe_to_return_halites = -1/halite_on_board_mult*np.ones(
    (grid_size, grid_size), dtype=np.int)
  safe_to_return_base_halites = []
  for i in range(my_bases.sum()):
    considered_base = my_base_locations[0][i], my_base_locations[1][i]
    margin = np.floor((
        nearest_opponent_distances[considered_base]-1) - (
            my_nearest_base_distances[i] + halite_on_board_mult*(
              np.maximum(0, halite_ships)+(
                collect_rate*obs_halite).astype(np.int))+1e-12)).astype(np.int)
    safe_base_reach = (my_nearest_base_distances[i] + halite_on_board_mult*(
      np.maximum(0, halite_ships)+(
        collect_rate*obs_halite).astype(np.int))) < (
        nearest_opponent_distances[considered_base]-1)
    safe_to_collect |= safe_base_reach
    safe_to_collect_margin[safe_base_reach] = np.maximum(
      safe_to_collect_margin[safe_base_reach], margin[safe_base_reach]+1)
    base_safe_return_thresholds = 1/halite_on_board_mult*(
        nearest_opponent_distances[considered_base] - (
            my_nearest_base_distances[i]))
    safe_to_return_halites = np.maximum(
      safe_to_return_halites, base_safe_return_thresholds)
    safe_to_return_base_halites.append(
      (base_safe_return_thresholds, considered_base))
                                        
  # if observation['step'] == 78:
  #   import pdb; pdb.set_trace()
    
  # nearest_opponent_stacked_distances_old = [DISTANCES[
  #       opp_ship_locations[0][i], opp_ship_locations[1][i]] for i in range(
  #         opponent_ships.sum())] + [99*np.ones((grid_size, grid_size))]
  # nearest_opponent_distances_old = np.stack(
  #   nearest_opponent_stacked_distances_old).min(0)
  # my_nearest_base_distances_old = np.stack(my_nearest_base_distances + [
  #           99*np.ones((grid_size, grid_size))]).min(0)
  # safe_to_collect_old = my_nearest_base_distances_old <= (
  #   nearest_opponent_distances_old-2)
      
  return (safe_to_collect, safe_to_collect_margin, safe_to_return_halites,
          safe_to_return_base_halites)

def get_ignored_convert_positions(
     likely_convert_opponent_positions, main_base_distances, stacked_ships,
     abs_rel_opponent_scores, observation, my_base_distances, opponent_bases,
     boxed_in_attack_squares):
  ignore_convert_positions = []
  for (row, col) in likely_convert_opponent_positions:
    main_base_distance = main_base_distances[row, col]
    opponent_id = np.where(stacked_ships[:, row, col])[0][0]
    if (abs_rel_opponent_scores[opponent_id-1] == 0) and (
        main_base_distance >= 9-(observation['relative_step']*6)) and (
          my_base_distances[:, row, col].min() >= 5-(
            observation['relative_step']*3)):
      ignore_convert_positions.append((row, col))
      opponent_bases[row, col] = True
      boxed_in_attack_squares[ROW_COL_MAX_DISTANCE_MASKS[row, col, 1]] = 0
      
  # if observation['step'] == 84:
  #   import pdb; pdb.set_trace()
      
  return ignore_convert_positions, opponent_bases, boxed_in_attack_squares

def get_avoid_attack_squares(
    boxed_in_attack_squares, approximate_score_diff, currently_winning,
    abs_rel_opponent_scores, my_zero_halite_ships, opponent_ships,
    influence_map, influence_map_unweighted, my_base_distances,
    boxed_in_opponent_ids, observation):
  grid_size = opponent_ships.shape[0]
  avoid_attack_squares_zero_halite = np.zeros(
    (grid_size, grid_size), dtype=np.bool)
  # Decide what opponent to attack regardless of the risk of ship loss
  # Policy: I am a close second or I am winning and attacking the second
  always_attack_opponent_id = None
  best_opponent_id = 1+np.argmin(approximate_score_diff)
  if np.all(currently_winning) or (
      (~currently_winning).sum() == 1 and abs_rel_opponent_scores[
        best_opponent_id-1] > 0):
    always_attack_opponent_id = best_opponent_id
  if np.any(boxed_in_attack_squares):
    # Count nearby zero halite and opponent ships
    all_boxed_squares = np.where(boxed_in_attack_squares)
    for i in range(all_boxed_squares[0].size):
      boxed_row = all_boxed_squares[0][i]
      boxed_col = all_boxed_squares[1][i]
      num_my_nearby_zero_halite = my_zero_halite_ships[
        ROW_COL_MAX_DISTANCE_MASKS[boxed_row, boxed_col, 3]].sum()
      num_opponent_nearby = opponent_ships[
        ROW_COL_MAX_DISTANCE_MASKS[boxed_row, boxed_col, 5]].sum()
      
      if ((influence_map[boxed_row, boxed_col] < 0.5) and (
          influence_map_unweighted[boxed_row, boxed_col] < -2) and (
            num_my_nearby_zero_halite == 1) and (
              num_opponent_nearby > 4) and (
                my_base_distances[:, boxed_row, boxed_col].min() >= 5)) and (
                  always_attack_opponent_id is None or (
                    boxed_in_opponent_ids[boxed_row, boxed_col] != (
                      always_attack_opponent_id))):
        # Flag the square as bad if I don't have a likely escape path
        can_escape = False
        avoid_attack_escape_distance = 4
        for d in NOT_NONE_DIRECTIONS:
          if d == NORTH:
            considered_row = (boxed_row - avoid_attack_escape_distance) % (
              grid_size)
            considered_col = boxed_col
          elif d == SOUTH:
            considered_row = (boxed_row + avoid_attack_escape_distance) % (
              grid_size)
            considered_col = boxed_col
          elif d == EAST:
            considered_row = boxed_row
            considered_col = (boxed_col + avoid_attack_escape_distance) % (
              grid_size) 
          elif d == WEST:
            considered_row = boxed_row
            considered_col = (boxed_col - avoid_attack_escape_distance) % (
              grid_size) 
          if influence_map[considered_row, considered_col] > 0.5:
            can_escape = True
            break
        if not can_escape:
          avoid_attack_squares_zero_halite[boxed_row, boxed_col] = 1
      
    # if np.any(avoid_attack_squares_zero_halite):
    #   print(observation['step'], np.where(avoid_attack_squares_zero_halite))
    # import pdb; pdb.set_trace()
    # x=1
          
  return avoid_attack_squares_zero_halite, always_attack_opponent_id

def override_initial_collect(
    config, all_ship_scores, obs_halite, halite_ships, stacked_ships,
    stacked_bases, player_influence_maps, player_obs, observation,
    history):
  # ORPHANED LOGIC, this was not going anywhere
  # Initial collect override logic. Ships should initially aim for halite
  # squares at the boundaries of their influence sphere - that way opponents
  # don't get to mine it and I can then later focus on halite near my base
  grid_size = stacked_ships.shape[1]
  
  # Stack the collect scores for all my ships
  ship_rows = []
  ship_cols = []
  ship_keys = []
  for ship_k in player_obs[2]:
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    ship_rows.append(row)
    ship_cols.append(col)
    ship_keys.append(ship_k)
  my_num_ships = len(ship_rows)
  if my_num_ships > 0:
    my_ship_positions = (np.array(ship_rows), np.array(ship_cols))
    opponent_ship_positions = np.where(stacked_ships[1:].sum(0) > 0)
    my_ship_halites = halite_ships[my_ship_positions]
    
    # Obtain the nearest distances for my and opponent ships
    my_nearest_ship_distances = 99*np.ones((grid_size, grid_size))
    for ship_id in range(my_num_ships):
      # FUTURE WORK: Should I exclude my returning to base ships? Probably not.
      row = my_ship_positions[0][ship_id]
      col = my_ship_positions[1][ship_id]
      my_nearest_ship_distances = np.minimum(
        my_nearest_ship_distances, DISTANCES[row, col])
      
    my_nearest_base_distances = 99*np.ones((grid_size, grid_size))
    my_num_bases = stacked_bases[0].sum()
    my_base_positions = np.where(stacked_bases[0])
    for base_id in range(my_num_bases):
      row = my_base_positions[0][base_id]
      col = my_base_positions[1][base_id]
      my_nearest_base_distances = np.minimum(
        my_nearest_base_distances, DISTANCES[row, col])
      
    opponent_nearest_ship_distances = 99*np.ones((grid_size, grid_size))
    opponent_ships = stacked_ships[1:].sum(0) > 0
    num_opponent_ships = opponent_ships.sum()
    for ship_id in range(num_opponent_ships):
      row = opponent_ship_positions[0][ship_id]
      col = opponent_ship_positions[1][ship_id]
      opponent_nearest_ship_distances = np.minimum(
        opponent_nearest_ship_distances, DISTANCES[row, col])
      
    opponent_nearest_base_distances = 99*np.ones((grid_size, grid_size))
    opponent_bases = stacked_bases[1:].sum(0) > 0
    num_opponent_bases = opponent_bases.sum()
    opponent_base_positions = np.where(opponent_bases)
    for base_id in range(num_opponent_bases):
      row = opponent_base_positions[0][base_id]
      col = opponent_base_positions[1][base_id]
      opponent_nearest_base_distances = np.minimum(
        opponent_nearest_base_distances, DISTANCES[row, col])
    
    nearest_ship_distance_difference = (
      my_nearest_ship_distances - opponent_nearest_ship_distances)
    
    if observation['step'] == 0:
      original_position_multiplier =  1.5**(3-np.abs(np.maximum(
        -3, nearest_ship_distance_difference)))
      history['original_position_multiplier'] = original_position_multiplier
    else:
      original_position_multiplier = history['original_position_multiplier']
    
    # Use the opponent influence to determine the value of gather squares -
    # squares where the nearest distance to one of my ships is equal have the
    # highest value since these will likely be where the competition happens
    smooth_multiplier = smooth2d(obs_halite)
    smooth_multiplier /= smooth_multiplier.mean()
    collect_values = np.copy(obs_halite) * smooth_multiplier
    collect_values *= original_position_multiplier
    smooth_collect_values = smooth2d(collect_values, 5)
    
    # Compute ship specific collect scores for all considered collect values
    # This factors in the distance to each square and the halite on board
    # FUTURE WORK: factor in halite on board
    all_ship_collect_scores = np.zeros((my_num_ships, grid_size, grid_size))
    my_assigned_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
    ships_assigned = np.zeros(my_num_ships, dtype=np.bool)
    assigned_id = 0
    initial_collect_zero_halite_targets = history[
      'initial_collect_zero_halite_targets']
    for ship_id, ship_k in enumerate(ship_keys):
      row = my_ship_positions[0][ship_id]
      col = my_ship_positions[1][ship_id]
      valid_considered_mask = np.ones((grid_size, grid_size), dtype=np.bool)
      bad_directions = list(set(MOVE_DIRECTIONS) - set(all_ship_scores[ship_k][6]))
      for d in bad_directions:
        if d is None:
          valid_considered_mask[row, col] = 0
        else:
          valid_considered_mask[HALF_PLANES_CATCH[row, col][d]] = 0
      dm = DISTANCE_MASKS[(row, col)]
      ship_collect_scores = dm*collect_values*valid_considered_mask
      ship_collect_scores[row, col] *= int(my_num_bases > 0)*(
        config['relative_stand_still_collect_boost'])
      all_ship_collect_scores[ship_id] = ship_collect_scores
      
      if my_ship_halites[ship_id] > 0 and not None in bad_directions and (
          ship_collect_scores[row, col] == ship_collect_scores.max()):
        # import pdb; pdb.set_trace()
        my_assigned_squares[row, col] = True
        all_ship_scores[ship_k][0][row, col] = 1e5 - assigned_id
        ships_assigned[ship_id] = True
        assigned_id += 1
      elif my_ship_halites[ship_id] == 0 and ship_k in (
          initial_collect_zero_halite_targets):
        # Prefer consistent target selection - only reevaluate after reaching the
        # target
        all_ship_collect_scores[ship_id][initial_collect_zero_halite_targets[
          ship_k]] *= 2
        
    # if observation['step'] == 13:
    #   import pdb; pdb.set_trace()
        
    lowest_collect_score = all_ship_collect_scores.min()
    # original_ship_collect_scores = np.copy(all_ship_collect_scores)
    all_ship_collect_scores[:, my_assigned_squares] = lowest_collect_score
    all_ship_collect_scores[ships_assigned] = lowest_collect_score
    
    # First assign the zero halite ships - Ideally, they should spread out and
    # target high value halite squares at the boundary of the influence sphere
    num_zero_halite_ships = (my_ship_halites == 0).sum()
    zero_halite_ids = np.where(my_ship_halites == 0)
    
    zero_halite_collect_scores = all_ship_collect_scores[zero_halite_ids]
    zero_halite_targets = {}
    for _ in range(num_zero_halite_ships):
      (best_ship_id, best_row, best_col) = np.unravel_index(
        zero_halite_collect_scores.argmax(), zero_halite_collect_scores.shape)
      # import pdb; pdb.set_trace()
      ship_k = ship_keys[best_ship_id]
      my_assigned_squares[best_row, best_col] = True
      
      # Create a mask between the current and target positions, where we
      # encourage the ships to prefer squares with a higher smooth collect value
      row = my_ship_positions[0][best_ship_id]
      col = my_ship_positions[1][best_ship_id]
      collect_ship_score_mask = get_mask_between_exclude_ends(
        row, col, best_row, best_col, grid_size)
      collect_ship_score_mask[best_row, best_col] = 1
      collect_ship_scores = (1e5-assigned_id)*collect_ship_score_mask-(
        10*obs_halite)
      to_target_dirs = get_dir_from_target(
          row, col, best_row, best_col, grid_size)
      if len(to_target_dirs) == 2:
        first_move_pos = move_ship_row_col(
          row, col, to_target_dirs[0], grid_size)
        second_move_pos = move_ship_row_col(
          row, col, to_target_dirs[1], grid_size)
        if smooth_collect_values[first_move_pos] > smooth_collect_values[
            second_move_pos]:
          avoid_dir = to_target_dirs[1]
        else:
          avoid_dir = to_target_dirs[0]
        collect_ship_scores[HALF_PLANES_CATCH[row, col][avoid_dir]] -= 1
      # import pdb; pdb.set_trace()
      all_ship_scores[ship_k][0][:] = collect_ship_scores
      zero_halite_targets[ship_k] = (best_row, best_col)
      
      assigned_id += 1
      ships_assigned[best_ship_id] = True
      all_ship_collect_scores[:, my_assigned_squares]
      zero_halite_collect_scores[:, best_row, best_col] = lowest_collect_score
      zero_halite_collect_scores[best_ship_id] = lowest_collect_score
    history['initial_collect_zero_halite_targets'] = zero_halite_targets 
    print(observation['step'], zero_halite_targets)
    
  return all_ship_scores, history

def get_ship_scores(config, observation, player_obs, env_config, np_rng,
                    ignore_bad_attack_directions, history,
                    env_obs_ids, env_observation, verbose):
  ship_scores_start_time = time.time()
  convert_cost = env_config.convertCost
  spawn_cost = env_config.spawnCost
  stacked_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships']])
  all_my_bases = copy.copy(stacked_bases[0])
  my_bases = stacked_bases[0]
  # Exclude bases that are persistently camped by opponents
  num_my_bases_with_excluded = my_bases.sum()
  base_locations_with_excluded = np.where(my_bases)
  excluded_base_distances = []
  for base_pos in history['my_base_not_attacked_positions']:
    # Note: stacking ensures we are working on a copy of the original base 
    # observation!
    my_bases[base_pos] = 0
    excluded_base_distances.append(DISTANCES[base_pos])
  obs_halite = np.maximum(0, observation['halite'])
  # Clip obs_halite to zero when gathering it doesn't add to the score
  # code: delta_halite = int(cell.halite * configuration.collect_rate)
  collect_rate = env_config.collectRate
  obs_halite[obs_halite < 1/collect_rate] = 0
  obs_halite_sum = obs_halite.sum()
  my_ship_count = len(player_obs[2])
  num_my_bases = my_bases.sum()
  first_base = my_ship_count == 1 and num_my_bases == 0 and observation[
    'step'] <= 10
  max_ships = config['max_initial_ships']
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
  can_deposit_halite = num_my_bases > 0
  stacked_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']])
  my_ships = stacked_ships[0]
  opponent_ships = stacked_ships[1:].sum(0) > 0
  all_ship_count = opponent_ships.sum() + my_ship_count
  my_ship_fraction = my_ship_count/(1e-9+all_ship_count)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  my_zero_halite_ships = my_ships & (halite_ships == 0)
  last_ship_standing_no_collect = observation[
    'relative_step'] > 1/4 and (
      stacked_ships[0] & (halite_ships == 0)).sum() == 1
  opponent_bases = stacked_bases[1:].sum(0)
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
  camping_ships_strategy = history['camping_ships_strategy']
    
  # Get the distance to the nearest base for all squares
  all_bases = stacked_bases.sum(0) > 0
  base_locations = np.where(all_bases)
  num_bases = all_bases.sum()
  all_base_distances = [DISTANCES[
    base_locations[0][i], base_locations[1][i]] for i in range(num_bases)] + [
        99*np.ones((grid_size, grid_size))]
  nearest_base_distances = np.stack(all_base_distances).min(0)
  if num_my_bases_with_excluded > 0:
    all_base_distances_with_excluded = np.stack([DISTANCES[
    base_locations_with_excluded[0][i],
    base_locations_with_excluded[1][i]] for i in range(
      num_my_bases_with_excluded)])
    nearest_base_distances_with_my_excluded = (
      all_base_distances_with_excluded.min(0))
  else:
    all_base_distances_with_excluded = np.zeros((0, grid_size, grid_size))
    nearest_base_distances_with_my_excluded = 99*np.ones(
      (grid_size, grid_size), dtype=np.int)
  
  # Flag to indicate I should not occupy/flood my base with early ships
  my_halite = observation['rewards_bases_ships'][0][0]
  avoid_base_early_game = my_halite >= spawn_cost and (
    observation['step'] < 20) and num_my_bases == 1 and (
      my_halite % spawn_cost) == 0 and my_ship_count < 9

  # if observation['step'] in [160, 242]:
  #   import pdb; pdb.set_trace()
      
  # Distance to nearest base mask - gathering closer to my base is better
  (base_nearest_distance_scores, my_base_distances,
   my_nearest_base_distances) = get_nearest_base_distances(
     grid_size, history['my_base_not_attacked_positions'], observation)

  # Get opponent ship actions that avoid collisions with less halite ships
  (opponent_ships_sensible_actions, opponent_ships_sensible_actions_no_risk,
   boxed_in_attack_squares, boxed_in_opponent_ids,
   boxed_in_zero_halite_opponents, likely_convert_opponent_positions,
   possible_convert_opponent_positions) = get_valid_opponent_ship_actions(
    config, observation['rewards_bases_ships'], halite_ships, grid_size,
    history, nearest_base_distances_with_my_excluded, observation,
    env_config)
  
  # Get the weighted base mask
  (weighted_base_mask, main_base_distances, non_abandoned_base_distances,
   ship_diff_smoothed) = get_weighted_base_mask(
    stacked_bases, stacked_ships, observation, history)
  
  # Get the influence map
  (influence_map, influence_map_unweighted, player_influence_maps,
   priority_scores, ship_priority_weights,
   escape_influence_probs) = get_influence_map(
    config, stacked_bases, stacked_ships, halite_ships, observation,
    player_obs)
     
  # Scale the opponent bases as a function of attack desirability
  (opponent_bases_scaled, opponent_ships_scaled, abs_rel_opponent_scores,
   currently_winning, approximate_score_diff, history,
   ballistic_attack_base_targets) = scale_attack_scores_bases_ships(
     config, observation, player_obs, spawn_cost, non_abandoned_base_distances,
     weighted_base_mask, steps_remaining, obs_halite, halite_ships, history,
     smoothed_halite, player_influence_maps,
     nearest_base_distances_with_my_excluded, player_ids)
  ignore_bad_attack_directions = ignore_bad_attack_directions or len(
    ballistic_attack_base_targets) > 0

  # Decide what converting ships to let convert peacefully
  (ignore_convert_positions, opponent_bases,
   boxed_in_attack_squares) = get_ignored_convert_positions(
     likely_convert_opponent_positions, main_base_distances, stacked_ships,
     abs_rel_opponent_scores, observation, my_base_distances, opponent_bases,
     boxed_in_attack_squares)

  # Decide what boxed in escape squares to avoid - if I use a lonely zero
  # halite ship to destroy an opponent's ship, I am likely to lose my ship in
  # one of the subsequent turns
  (avoid_attack_squares_zero_halite,
   always_attack_opponent_id) = get_avoid_attack_squares(
    boxed_in_attack_squares, approximate_score_diff, currently_winning,
    abs_rel_opponent_scores, my_zero_halite_ships, opponent_ships,
    influence_map, influence_map_unweighted, my_base_distances,
    boxed_in_opponent_ids, observation)
     
  # Get the squares that have no zero halite neighbors - this makes it hard
  # to successfully camp out next to the base
  no_zero_halite_neighbors = get_no_zero_halite_neighbors(
    observation['halite'])
  
  # Only conditionally attack the bases where I have a camper that is active
  my_prev_step_base_attacker_ships = history[
    'my_prev_step_base_attacker_ships']
  camp_attack_mask = np.ones((grid_size, grid_size), dtype=np.bool)
  for ship_k in camping_ships_strategy:
    base_location = camping_ships_strategy[ship_k][5]
    consider_base_attack = camping_ships_strategy[ship_k][4]
    camp_attack_mask[base_location] = consider_base_attack
    
  # Attack opponent ships that camp out next to my base
  attack_opponent_campers = history['attack_opponent_campers']
  
  # Don't worry about collecting if I have a base at distance <= d and the
  # nearest opponent is at a distance of at least d+2
  (safe_to_collect, safe_to_collect_margin, safe_to_return_halites,
   safe_to_return_base_halites) = get_my_guaranteed_safe_collect_squares(
     opponent_ships, grid_size, all_my_bases, obs_halite, collect_rate,
     halite_ships, observation)
  
  # Early on, the collect boost is high as the distance from the nearest base
  # grows. This effect rapidly decays once the hunting season starts
  stand_still_collect_boost = config['relative_stand_still_collect_boost']
     
  # print(observation['step'], my_ship_count, (stacked_ships[0] & (
  #   halite_ships == 0)).sum())
  
  n_step_avoid_min_die_prob_cutoff = config[
    'n_step_avoid_min_die_prob_cutoff']
  if history['num_destroyed_ships'] == 0:
    low_risk_limit = 0.1
    early_game_risk_limit = max(
      low_risk_limit, config['max_risk_n_step_risky'] - (
        config['max_risk_n_step_risky']-low_risk_limit)*observation[
          'step']/config['max_steps_n_step_risky'])
    n_step_avoid_min_die_prob_cutoff = max(
      early_game_risk_limit, n_step_avoid_min_die_prob_cutoff)
  
  all_ship_scores = {}
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
    ship_influence_priority_multipliers = (
      1+config['influence_weights_additional_multiplier']*(
        ship_priority_weights[ship_k])**config[
          'influence_weights_exponent']) ** priority_scores
    collect_grid_scores = dm*(
      smoothed_halite * config['collect_smoothed_multiplier'] + 
      obs_halite * config['collect_actual_multiplier']) * (
        config['collect_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) * (
            base_nearest_distance_scores ** config[
              'collect_base_nearest_distance_exponent'])*(
                ship_influence_priority_multipliers)
    if observation['step'] < 50:
      # import pdb; pdb.set_trace()
      collect_grid_scores *= (history['original_position_multiplier']**(
        observation['step']/50))
    base_distance = nearest_base_distances_with_my_excluded[row, col]
    collect_grid_scores[row, col] *= int(num_my_bases_with_excluded > 0)*(
      stand_still_collect_boost * (1+config[
        'initial_collect_boost_away_from_base'] * max(0, base_distance-5)/7)**(
          (1-observation['relative_step'])**14))
    if ship_k in history['initial_not_collect_near_base_ships']:
      collect_grid_scores[history['initial_not_collect_near_base_mask']] = 0
      
    # if observation['step'] == 233:
    #   import pdb; pdb.set_trace()
                    
    if last_ship_standing_no_collect and ship_halite == 0:
      collect_grid_scores[row, col] = -1e13
                    
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
      
    # At the end of a game, disincintivize all collect squares that are too far
    # away from the nearest base to be able to return before the game is over
    if steps_remaining < grid_size and nearest_base_distances.min() == 0:
      trajectory_lengths = DISTANCES[row, col] + my_nearest_base_distances
      collect_grid_scores[trajectory_lengths > (steps_remaining-1-int(
        steps_remaining > grid_size//2))] = 0
      
    # Scores 2: returning to any of my bases - delay the return when it is
    # safe to collect
    safe_collect_ship_margin = safe_to_collect_margin[row, col]
    if ship_k in history['returning_to_base_ships'] or (
        safe_collect_ship_margin <= 0) or num_my_bases < 2:
      delay_return_divisor = 1
    else:
      # delay_return_divisor = 1
      delay_return_divisor = 2**(safe_collect_ship_margin/2)
    # Always use the maximum value for a base return if I can safely move there
    # (regardless of my influence in that area)
    weighted_base_mask_ship_return = np.copy(weighted_base_mask)
    for base_safe_return_halite, base_location in safe_to_return_base_halites:
      if ship_halite < base_safe_return_halite[row, col]:
        weighted_base_mask_ship_return[base_location] = max(
          1.0, weighted_base_mask_ship_return[base_location])
        
    base_return_grid_multiplier = dm*min(ship_halite, 2*convert_cost)/(
      delay_return_divisor)*(config['return_base_multiplier'] * (
        config['return_base_less_halite_ships_multiplier_base'] ** (
          opponent_smoother_less_halite_ships)) + early_game_return_boost)*(
            weighted_base_mask_ship_return)
    # Further incentivize moving onto a base after a return has started when I
    # am close to a base since that means I can count on some of the best
    # collect score for future steps
    if ship_k in history['returning_to_base_ships'] and observation[
        'relative_step'] > config['start_hunting_season_relative_step']:
      base_return_grid_multiplier += (dm**2)*collect_grid_scores.max()/1.5
    chase_details = history['chase_counter'][0].get(ship_k, None)
    if chase_details is not None:
      # Keep the relative order using the minimum in case the return to base
      # pull is big
      base_return_grid_multiplier = np.minimum(
        base_return_grid_multiplier+5e4, base_return_grid_multiplier*(config[
        'chase_return_base_exponential_bonus']**chase_details[1]))
            
    # Force returning to a base when the episode is almost over and I
    # have halite on board
    if ship_halite > 0 and steps_remaining < grid_size:
      base_return_grid_multiplier, end_game_base_return = (
        force_return_base_end_episode(
          my_bases, base_return_grid_multiplier, main_base_distances, row, col,
          steps_remaining, opponent_less_halite_ships, weighted_base_mask,
          safe_to_collect))
    else:
      end_game_base_return = False
    
    # Override the return base score to 0 to avoid blocking the base early on
    # in the game.
    if avoid_base_early_game:
      base_return_grid_multiplier = override_early_return_base_scores(
        base_return_grid_multiplier, my_bases, row, col, my_ship_count)
      
    # if observation['step'] == 247 and row == 15 and col == 4:
    #   import pdb; pdb.set_trace()
    
    # Scores 3: establish a new base
    first_base_or_can_spawn = my_ship_count == 1 and num_my_bases == 0 and (
      observation['step'] <= 10 or (player_obs[0]+ship_halite) >= (
        2*spawn_cost))
    establish_base_scores = dm**(config['establish_base_dm_exponent'])*(
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
                      no_zero_halite_neighbors)) - 1e5*int(not (
                        first_base_or_can_spawn))
                  
    # if observation['step'] == 391 and ship_k == '58-1':
    #   import pdb; pdb.set_trace()
                        
    # Scores 4: attack an opponent base at row, col
    attack_step_multiplier = min(5, max(1, 1/(
      2*(1-observation['relative_step']+1e-9))))
    if ship_k in my_prev_step_base_attacker_ships:
      # Encourage the base attack of a ship to be persistent
      attack_step_multiplier *= 5
    attack_base_scores = dm*np.minimum(15e5, camp_attack_mask*(
      attack_step_multiplier)*config['attack_base_multiplier']*(
        opponent_bases_scaled)*(config[
          'attack_base_less_halite_ships_multiplier_base'] ** (
            opponent_smoother_less_halite_ships))) - (config[
              'attack_base_halite_sum_multiplier'] * obs_halite_sum**0.8 / (
                all_ship_count))*int(my_ship_fraction < 0.5) - 1e12*(
                  ship_halite > 0)
    # Keep the preference order in ballistic mode without abandoning recue or
    # base defense ships
    attack_base_scores = np.minimum(15e5, attack_base_scores) + 1e-10*(
      attack_base_scores) * (attack_base_scores > 15e5)
                
    # Update the scores as a function of nearby opponent ships to avoid
    # collisions with opposing ships that carry less halite and encourage 
    # collisions with opponent ships that carry less halite
    # Also incorporate the camping score override behavior here
    camping_override_strategy = camping_ships_strategy.get(ship_k, ())
    attack_campers_override_strategy = attack_opponent_campers.get(ship_k, ())
    (collect_grid_scores, base_return_grid_multiplier, establish_base_scores,
     attack_base_scores, preferred_directions, valid_directions,
     agent_surrounded, two_step_bad_directions, n_step_bad_directions,
     one_step_valid_directions, n_step_bad_directions_die_probs,
     original_n_step_bad_directions,
     original_n_step_bad_directions_die_probs) = update_scores_opponent_ships(
       config, collect_grid_scores, base_return_grid_multiplier,
       establish_base_scores, attack_base_scores, opponent_ships,
       opponent_bases, halite_ships, row, col, grid_size, spawn_cost,
       drop_None_valid, obs_halite, collect_rate, np_rng,
       opponent_ships_sensible_actions,
       opponent_ships_sensible_actions_no_risk, ignore_bad_attack_directions,
       observation, ship_k, all_my_bases, my_ships, steps_remaining, history,
       escape_influence_probs, player_ids, env_obs_ids, env_observation,
       main_base_distances, nearest_base_distances, end_game_base_return,
       camping_override_strategy, attack_campers_override_strategy,
       boxed_in_attack_squares, safe_to_collect,
       boxed_in_zero_halite_opponents, ignore_convert_positions,
       avoid_attack_squares_zero_halite, n_step_avoid_min_die_prob_cutoff,
       safe_to_return_halites, safe_to_return_base_halites,
       my_nearest_base_distances)
       
    # if observation['step'] == 169 and ship_k == '65-2':
    #   import pdb; pdb.set_trace()
       
    # Update the scores as a function of blocking opponent bases and my early
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
      last_episode_step_convert = ship_halite >= convert_cost
      if last_episode_step_convert and num_my_bases_with_excluded > 0:
        # Don't convert if I can safely move to a base next to my square.
        min_base_distance = all_base_distances_with_excluded[:, row, col].min()
        if min_base_distance == 1:
          if opponent_less_halite_ships.sum() == 0:
            last_episode_step_convert = False
          else:
            for base_id in range(num_my_bases_with_excluded):
              base_row = base_locations_with_excluded[0][base_id]
              base_col = base_locations_with_excluded[1][base_id]
              if all_base_distances_with_excluded[base_id, row, col] == 1:
                if DISTANCES[base_row, base_col][
                    opponent_less_halite_ships].min() > 1:
                  last_episode_step_convert = False
                  break
                
      if last_episode_step_convert:
        establish_base_scores[row, col] = 1e12
        base_locations_with_excluded = (
          np.append(base_locations_with_excluded[0], row),
          np.append(base_locations_with_excluded[1], col))
        all_base_distances_with_excluded = np.concatenate(
          [all_base_distances_with_excluded,
           np.expand_dims(DISTANCES[row, col], 0)])
        num_my_bases_with_excluded += 1
      elif ship_halite > 0:
        base_return_grid_multiplier[DISTANCES[row, col] == 1] += 1e5
        end_game_base_return = True
    else:
      last_episode_step_convert = False
      
    all_ship_scores[ship_k] = (
      collect_grid_scores, base_return_grid_multiplier, establish_base_scores,
      attack_base_scores, preferred_directions, agent_surrounded,
      valid_directions, two_step_bad_directions, n_step_bad_directions,
      one_step_valid_directions, opponent_base_directions, 0,
      end_game_base_return, last_episode_step_convert,
      n_step_bad_directions_die_probs, opponent_smoother_less_halite_ships,
      ship_influence_priority_multipliers, original_n_step_bad_directions,
      original_n_step_bad_directions_die_probs)
    
  # if observation['relative_step'] < config[
  #     'initial_collect_override_relative_step']:
  #   all_ship_scores, history = override_initial_collect(
  #     config, all_ship_scores, obs_halite, halite_ships, stacked_ships,
  #     stacked_bases, player_influence_maps, player_obs, observation,
  #     history)
    
  ship_scores_duration = time.time() - ship_scores_start_time
  return (all_ship_scores, opponent_ships_sensible_actions,
          opponent_ships_sensible_actions_no_risk, weighted_base_mask,
          opponent_ships_scaled, main_base_distances, ship_scores_duration,
          halite_ships, player_influence_maps, boxed_in_zero_halite_opponents,
          ignore_convert_positions, ship_diff_smoothed,
          ballistic_attack_base_targets, safe_to_return_halites,
          safe_to_collect_margin, always_attack_opponent_id,
          likely_convert_opponent_positions,
          possible_convert_opponent_positions, my_base_distances,
          nearest_base_distances, history)

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
    history, max_considered_attackers=3, halite_on_board_mult=1e-6):
  obs_halite = np.maximum(0, observation['halite'])
  grid_size = obs_halite.shape[0]
  collect_rate = env_config.collectRate
  obs_halite[obs_halite < 1/collect_rate] = 0
  my_ships = observation['rewards_bases_ships'][0][2]
  my_ship_count = my_ships.sum()
  my_bases = observation['rewards_bases_ships'][0][1]
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
  last_ship_standing_no_collect = observation[
    'relative_step'] > 1/4 and (
      stacked_ships[0] & (halite_ships == 0)).sum() == 1
  halite_density = smooth2d(obs_halite, smooth_kernel_dim=10)
  my_base_density = smooth2d(my_bases, smooth_kernel_dim=10)
  
  can_deposit_halite = expected_payoff_conversion > convert_cost
  restored_base_pos = None
  can_defend_converted = False
  if can_deposit_halite:
    # Decide what ship to convert - it should be relatively central, have high
    # ship halite on board and be far away from opponent ships and bases
    # Also, don't build a new base next to a base where there is a camper
    # that I am not attacking
    # Preferably restore a base close to other halite and close to my other
    # bases
    next_to_my_camped_not_attacked = np.zeros(
      (grid_size, grid_size), dtype=np.bool)
    for base_pos in history['my_base_not_attacked_positions']:
      next_to_my_camped_not_attacked[ROW_COL_BOX_MAX_DISTANCE_MASKS[
        base_pos[0], base_pos[1], 2]] = 1
    
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
      if num_considered_distances >= 1 and can_defend:
        can_defend = my_ship_distances[0] < opponent_ship_distances[0]
      
      can_afford = (halite_ships[row, col] + player_obs[0]) >= convert_cost*(
        1+1e10*int(last_ship_standing_no_collect and ship_halite == 0))
      
      convert_priority_scores[i] = ship_halite + halite_density[
        row, col]/10 + (500*(
          max(0, 0.3-np.abs(0.3-my_base_density[row, col])))) - 100*(
        my_ship_density-opponent_ship_density)[row, col] - 200*(
          opponent_base_density[row, col]) - 1e12*int(
            not can_defend or not can_afford or next_to_my_camped_not_attacked[
              row, col])
          
    can_defend_converted = convert_priority_scores.max() > -1e11
    
  # if observation['step'] == 153:
  #   import pdb; pdb.set_trace()
    
  if can_defend_converted:
    convert_k = list(player_obs[2].keys())[np.argmax(convert_priority_scores)]
    convert_row, convert_col = row_col_from_square_grid_pos(
        player_obs[2][convert_k][0], grid_size)
    restored_base_pos = (convert_row, convert_col)
    all_ship_scores[convert_k][2][convert_row, convert_col] = 1e12
    
    # Send the closest ship towards the base
    convert_distances = DISTANCES[convert_row, convert_col][my_ships]
    convert_distances[convert_distances == 0] = 100
    closest_id = np.argmin(convert_distances)
    my_ship_positions = np.where(my_ships)
    ship_to_base_row = my_ship_positions[0][closest_id]
    ship_to_base_col = my_ship_positions[1][closest_id]
    my_ship_pos_to_k = {v[0]: k for k, v in player_obs[2].items()}
    ship_to_base_k = my_ship_pos_to_k[
      ship_to_base_row*grid_size + ship_to_base_col]
    to_base_directions = get_dir_from_target(
        ship_to_base_row, ship_to_base_col, convert_row, convert_col,
        grid_size)
    for d in to_base_directions:
      if not d in all_ship_scores[ship_to_base_k][6]:
        all_ship_scores[ship_to_base_k][6].append(d)
    print(observation['step'], "Moving ship to restored base",
          ship_to_base_row, ship_to_base_col, convert_row, convert_col)
    all_ship_scores[ship_to_base_k][1][convert_row, convert_col] = 1e9
  else:
    # Don't gather
    # Add some small positive noise to the establish base score, away from
    # the current square - this ensures ships keep moving around when I don't
    # plan on restoring my last destroyed base
    # Move ships closer to each other if we want to convert a base but they are
    # not able to defend it
    # Send all to the least dense opponent point that is still close to halite
    # and my other bases
    opponent_density = smooth2d(opponent_ships+opponent_bases,
                                smooth_kernel_dim=5)
    desirability_score = 500*np.maximum(
      0, 0.3-np.abs(0.3-my_base_density))+(halite_density/10)-100*(
        opponent_density)
    best_gather_locations = np.where(
      desirability_score == desirability_score.max())
    gather_row = best_gather_locations[0][0]
    gather_col = best_gather_locations[1][0]
    # lowest_densities = np.where(opponent_density == opponent_density.min())
    # halite_density = smooth2d(obs_halite)
    # target_id = np.argmax(halite_density[lowest_densities])
    # gather_row = lowest_densities[0][target_id]
    # gather_col = lowest_densities[1][target_id]
    
    num_zero_halite_ships = ((halite_ships == 0) & my_ships).sum()
    for ship_k in player_obs[2]:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      all_ship_scores[ship_k][0][:] *= 0
      if can_deposit_halite:
        # Gather with some low probability since we may not have enough halite
        # to convert a ship (except when it is the last remaining zero halite
        # ship)
        if obs_halite[row, col] > 0 and np_rng.uniform() < 0.2 and (
            halite_ships[row, col] > 0 or num_zero_halite_ships > 1):
          all_ship_scores[ship_k][0][row, col] = 2e6
        ensure_move_mask = 1e6*DISTANCE_MASKS[(gather_row, gather_col)]
      else:
        ensure_move_mask = np_rng.uniform(1e5, 1e9, (grid_size, grid_size))
        ensure_move_mask[row, col] = 0
      all_ship_scores[ship_k][2][:] += ensure_move_mask
      
    can_deposit_halite = False
    
  return all_ship_scores, can_deposit_halite, restored_base_pos

def edge_aware_3_avg(a, b, c, grid_size):
  # Decide if we should be edge aware near the high or low end
  vals = np.array([a, b, c])
  num_lower_half = (vals <= grid_size//2).sum()
  if num_lower_half == 1:
    near_low_edge = vals <= grid_size//4
    if np.any(near_low_edge):
      low_edge_id = np.where(near_low_edge)[0][0]
      vals[low_edge_id] += grid_size
  elif num_lower_half == 2:
    near_high_edge = vals >= 3*grid_size//4
    if np.any(near_high_edge):
      high_edge_id = np.where(near_high_edge)[0][0]
      vals[high_edge_id] -= grid_size
  return int(np.round(vals.sum())/3) % grid_size

def consider_adding_strategic_bases(
    config, observation, env_config, all_ship_scores, player_obs, convert_cost,
    np_rng, history, player_influence_maps, obs_halite,
    non_abandoned_base_pos, all_base_pos, halite_ships,
    my_nearest_ship_distances, my_nearest_ship_distances_raw,
    opponent_nearest_ship_distances, evaluation_add_interval=15):
  
  opponent_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships']][1:]).sum(0) > 0
  my_ships = observation['rewards_bases_ships'][0][2]
  my_ship_halite_on_board = halite_ships[my_ships]
  my_ship_positions = np.where(my_ships)
  my_ship_pos_to_k = {v[0]: k for k, v in player_obs[2].items()}
  
  num_bases = all_base_pos[0].size
  num_non_abandoned_bases = non_abandoned_base_pos[0].size
  num_abandoned_bases = num_bases - num_non_abandoned_bases
  grid_size = opponent_bases.shape[0]
  convert_unavailable_positions = np.zeros(
    (grid_size, grid_size), dtype=np.bool)
  target_strategic_base_distance = config['target_strategic_base_distance']
  my_stacked_ship_distances = np.stack(my_nearest_ship_distances_raw)
  spawn_cost = env_config.spawnCost
  base_added = False
  added_base_pos = None
  
  # Decide *WHERE* to construct a new strategic base
  
  # Determine if a converted square can be defended by the second closest of
  # my ships (the closest would be converted)
  second_closest_ids = np.argsort(my_stacked_ship_distances, 0)[1].flatten()
  subset_ids = (second_closest_ids, np.repeat(np.arange(grid_size), grid_size),
                np.tile(np.arange(grid_size), grid_size))
  my_second_closest_ship_distances = my_stacked_ship_distances[
    subset_ids].reshape((grid_size, grid_size))
  can_defend_desirability = my_second_closest_ship_distances < (
    opponent_nearest_ship_distances)
  
  # Compute the desirability for each square to establish a new base.
  # We want bases that are far away from opponent bases (critical), close to
  # our other bases (but not too close), close to current and future potential
  # halite, far away from opponent ships and boxing in a large number of
  # future farming halite.
  influence_desirability = player_influence_maps[0]-player_influence_maps[
    1:].sum(0)
  opponent_near_base_desirability = -smooth2d(
    opponent_bases, smooth_kernel_dim=6)
  opponent_distant_base_desirability = -smooth2d(
    opponent_bases, smooth_kernel_dim=10)
  near_halite_desirability = smooth2d(obs_halite)-obs_halite
  near_halite_desirability /= max(1e3, near_halite_desirability.max())
  near_potential_halite_desirability = smooth2d(observation['halite'] > 0)-(
    observation['halite'] > 0)
  
  independent_base_distance_desirability = np.zeros((grid_size, grid_size))
  independent_base_distance_desirabilities = []
  for base_id in range(num_non_abandoned_bases):
    base_row = non_abandoned_base_pos[0][base_id]
    base_col = non_abandoned_base_pos[1][base_id]
    target_distance_scores = (1-np.abs(
      target_strategic_base_distance - DISTANCES[base_row, base_col])/(
        target_strategic_base_distance))**2
    independent_base_distance_desirabilities.append(target_distance_scores)
    if num_non_abandoned_bases == 1:
      independent_base_distance_desirability += target_distance_scores
    
  near_base_desirability = np.zeros((grid_size, grid_size))
  for base_id in range(num_bases):
    base_row = all_base_pos[0][base_id]
    base_col = all_base_pos[1][base_id]
    near_base_desirability[ROW_COL_MAX_DISTANCE_MASKS[
      base_row, base_col, 6]] -= 1
        
  triangle_base_distance_desirability = np.zeros((grid_size, grid_size))
  if num_non_abandoned_bases > 1:
    # For each potential triangle (consider all pairs of bases): factor in the
    # symmetry and amount of potential enclosed halite
    triangle_desirabilities = []
    for i in range(num_non_abandoned_bases-1):
      first_row, first_col = (non_abandoned_base_pos[0][i],
                              non_abandoned_base_pos[1][i])
      for j in range(i+1, num_non_abandoned_bases):
        second_row, second_col = (non_abandoned_base_pos[0][j],
                                  non_abandoned_base_pos[1][j])
        col_diff = second_col-first_col
        row_diff = second_row-first_row
        col_distance = min(np.abs(col_diff), grid_size - np.abs(col_diff))
        row_distance = min(np.abs(row_diff), grid_size - np.abs(row_diff))
        distance = col_distance+row_distance
        target_height = np.sqrt(target_strategic_base_distance**2-(
          (target_strategic_base_distance/2)**2))*(np.sqrt(2)/2)
        combined_distance_scores = independent_base_distance_desirabilities[
          i]*independent_base_distance_desirabilities[j]
        
        # Additionally, aim for triangles with equal angles - boost for min
        # distance from the two optimal locations.
        col_base_diff = np.abs(col_diff) if (
          np.abs(col_diff) <= grid_size//2) else (grid_size-np.abs(col_diff))
        if (col_diff < 0) != (np.abs(col_diff) <= grid_size//2):
          left_vertex = (first_row, first_col)
          if first_row < second_row:
            if row_diff <= grid_size//2:
              row_base_diff = row_diff
            else:
              row_base_diff = row_diff-grid_size
          else:
            # This only happens with equal rows because of the np.where row
            # order
            if (-row_diff) <= grid_size//2:
              row_base_diff = row_diff
            else:
              row_base_diff = row_diff+grid_size
        else:
          left_vertex = (second_row, second_col)
          if second_row < first_row :
            # This never happens because of the np.where row order
            if (-row_diff) <= grid_size//2:
              row_base_diff = (-row_diff)
            else:
              row_base_diff = -row_diff-grid_size
          else:
            if row_diff <= grid_size//2:
              row_base_diff = -row_diff
            else:
              row_base_diff = -row_diff+grid_size
        base_vector = np.array([row_base_diff, col_base_diff])
        found_it = False
        correction = 0
        correction_range = [-5, 5]
        iteration = 0
        # Use binary search to get the optimal triangle points
        while not found_it and iteration < 10:
          orthogonal_vector = np.array([col_base_diff, -row_base_diff])
          orthogonal_vector = orthogonal_vector/np.linalg.norm(
            orthogonal_vector)
          dot_product = np.dot(orthogonal_vector, np.array([1, 0]))
          angle = np.arccos(dot_product)
          height_divisor = max([
            np.cos(np.abs(angle)), np.sin(angle), np.sin(angle+np.pi)])
          orthogonal_vector = orthogonal_vector*(
            target_height/height_divisor+correction)
          mid_point = (left_vertex[0]+base_vector[0]/2,
                       left_vertex[1]+base_vector[1]/2)
          first_optimal = (
            int(np.round(mid_point[0]+orthogonal_vector[0]) % grid_size),
            int(np.round(mid_point[1]+orthogonal_vector[1]) % grid_size))
          second_optimal = (
            int(np.round(mid_point[0]-orthogonal_vector[0]) % grid_size),
            int(np.round(mid_point[1]-orthogonal_vector[1]) % grid_size))
          distance_to_first = DISTANCES[first_optimal][first_row, first_col]
          # print(observation['step'], distance_to_first, correction, iteration)
          if distance_to_first > target_strategic_base_distance:
            correction_range[1] = correction
          elif distance_to_first < target_strategic_base_distance:
            correction_range[0] = correction
          else:
            found_it = True
          iteration += 1
          correction = (correction_range[0]+correction_range[1])/2
        
        first_optimal_center = (
          edge_aware_3_avg(first_optimal[0], first_row, second_row, grid_size),
          edge_aware_3_avg(first_optimal[1], first_col, second_col, grid_size))
        second_optimal_center = (
          edge_aware_3_avg(
            second_optimal[0], first_row, second_row, grid_size),
          edge_aware_3_avg(
            second_optimal[1], first_col, second_col, grid_size))
        
        # Give a penalty to the first or second optimal point as a function of
        # the current halite and halite potential of the centres of the
        # triangles
        first_center_score = near_halite_desirability[first_optimal_center] + (
          near_potential_halite_desirability[first_optimal_center])/15*(
            1-observation['relative_step'])+10*near_base_desirability[
              first_optimal]
        second_center_score = near_halite_desirability[second_optimal_center] + (
          near_potential_halite_desirability[second_optimal_center])/15*(
            1-observation['relative_step'])+10*near_base_desirability[
              second_optimal]
        first_optimal_addition = 5*min(0.2, max(
          0, second_center_score-first_center_score))
        second_optimal_addition = 5*min(0.2, max(
          0, first_center_score-second_center_score))
        
        optimal_min_distances = np.minimum(
          DISTANCES[first_optimal]+first_optimal_addition,
          DISTANCES[second_optimal]+second_optimal_addition)
        optimal_mask_scores = np.exp(-optimal_min_distances)
        triangle_desirability = combined_distance_scores*optimal_mask_scores
        triangle_desirabilities.append((
          first_row, first_col, second_row, second_col, distance,
          triangle_desirability))
        
        # if observation['step'] == 47:
        #   import pdb; pdb.set_trace()
        #   x=1
        
        if distance <= target_strategic_base_distance+2:
          triangle_base_distance_desirability += triangle_desirability
  
  new_base_desirability = 100*near_base_desirability + config[
    'target_strategic_influence_desirability_multiplier']*(
      influence_desirability) + 2*opponent_near_base_desirability + (
        opponent_distant_base_desirability) + near_halite_desirability + (
          near_potential_halite_desirability/config[
            'target_strategic_potential_divisor']*(
              1-observation['relative_step'])) + (config[
                'target_strategic_independent_base_distance_multiplier']*(
                  independent_base_distance_desirability) + config[
                    'target_strategic_triangle_weight']*(
                      triangle_base_distance_desirability))*(
                        opponent_near_base_desirability > -0.2) + (  # Min D 5
                          0.5*can_defend_desirability)
  bad_positions = (near_base_desirability < 0) | (
    opponent_near_base_desirability < -0.2)
  any_valid_locations = np.any(~bad_positions)
  if any_valid_locations:
    new_base_desirability[bad_positions] = new_base_desirability[
      ~bad_positions].min()
  else:
    # print(observation['step'],
    #       "Not constructing a new base since there are no valid locations")
    new_base_desirability *= 0
         
  # if observation['step'] == 211:
  #   import pdb; pdb.set_trace()
  #   x=1
  
  # if triangle_base_distance_desirability.max() > 0:
  #   import pdb; pdb.set_trace()
  #   x=1
                        
  # Give a bonus to the preceding best square if we decided to create a base on
  # that location
  if history['construct_strategic_base_position']:
    new_base_desirability[history['construct_strategic_base_position']] += 1
            
  # Decide *IF* we should add a strategic base
  # Alway add a strategic base early on in the game
  # Later on, only decide to add a base at fixed intervals if
  # - There is significant halite left to be mined AND
  # - I have plenty of ships relative to my number of bases - give a boost to
  #   the target number of bases if I am currently winning
  num_my_ships = observation['rewards_bases_ships'][0][2].sum()
  current_scores = history['current_scores']
  ahead_in_score = (current_scores[0] == current_scores.max()) and np.all((
    current_scores[0]-3*spawn_cost) >= current_scores[1:])
  current_halite_sum = history['current_halite_sum']
  winning_clearly = ahead_in_score and (
    current_halite_sum[0] >= (current_halite_sum.max()-2*spawn_cost))
  game_almost_over = observation['relative_step'] >= 0.8
  
  strategic_additional_bases = num_my_ships/config[
    'target_strategic_num_bases_ship_divisor'] - 0.5*history[
      'num_destroyed_bases']
  if not ahead_in_score and strategic_additional_bases > 2/0.8:
    # Don't create too many bases if I am not winning clearly (relates to
    # fourth bases and beyond in the case of no base losses)
    strategic_additional_bases *= 0.8
    
  my_target_num_non_abandoned_bases = int(1+strategic_additional_bases)
    
  if winning_clearly and not game_almost_over:
    if obs_halite.sum() > config['min_halite_to_stop_early_hunt']:
      my_target_num_non_abandoned_bases += 1
  else:
    # Don't reconstruct strategic bases towards the end of the game as they
    # get destroyed
    if observation['relative_step'] >= 0.8:
      my_target_num_non_abandoned_bases -= 1
    if observation['relative_step'] >= 0.9:
      my_target_num_non_abandoned_bases -= 1
    if observation['relative_step'] >= 0.95:
      my_target_num_non_abandoned_bases -= 1
  if observation['step'] % evaluation_add_interval == 0:
    # print(observation['step'], my_target_num_non_abandoned_bases,
    #       num_non_abandoned_bases)
    add_base = my_target_num_non_abandoned_bases > (
      num_non_abandoned_bases + 0.5*(1+num_abandoned_bases))
    history['add_strategic_base'] = add_base
  else:
    consider_adding_base = my_target_num_non_abandoned_bases > (
      num_non_abandoned_bases)
    add_base = history['add_strategic_base']
    add_base = add_base and consider_adding_base
    history['add_strategic_base'] = add_base
    
  # if observation['step'] == 97:
  #   import pdb; pdb.set_trace()
    
  # Decide *HOW* to add a strategic base
  should_spawn_base_next_step = None
  if add_base and any_valid_locations:
    best_positions = np.where(
      new_base_desirability == new_base_desirability.max())
    add_strategic_base_position = (
      best_positions[0][0], best_positions[1][0])
    history['construct_strategic_base_position'] = (
      add_strategic_base_position)
    # print(observation['step'], add_strategic_base_position)
    
    # If we can afford to create and defend: go ahead and do so!
    # Otherwise, start saving up
    near_new_base_ship_distances = DISTANCES[add_strategic_base_position][
      my_ships]
    base_position_is_defended = can_defend_desirability[
      add_strategic_base_position]
    convert_mission_cost = int(
      not(base_position_is_defended))*spawn_cost + convert_cost
    near_ship_scores = 100*near_new_base_ship_distances-np.maximum(
      my_ship_halite_on_board, max(0, convert_mission_cost-player_obs[0]))
    nearest_ship_id = np.argmin(near_ship_scores)
    near_ship_position = (my_ship_positions[0][nearest_ship_id],
                          my_ship_positions[1][nearest_ship_id])
    near_ship_halite = halite_ships[near_ship_position]
    required_halite_to_convert = convert_mission_cost - near_ship_halite
    
    requested_save_conversion_budget = max(0, required_halite_to_convert)
    distance_to_conversion_square = DISTANCES[add_strategic_base_position][
        near_ship_position]
    conversion_ship_pos = near_ship_position[0]*grid_size+near_ship_position[
        1]
    conversion_ship_k = my_ship_pos_to_k[conversion_ship_pos]
    if required_halite_to_convert <= player_obs[0] and not (
        distance_to_conversion_square == 0 and not None in all_ship_scores[
          conversion_ship_k][9]):
      # Issue the conversion ship with a conversion objective
      all_ship_scores[conversion_ship_k][2][add_strategic_base_position] = 1e12
      convert_unavailable_positions[near_ship_position] = 1
      
      # Decide if the second closest ship should be used to defend the future
      # base
      second_closest_id = np.argmin(
        near_new_base_ship_distances + 100*(
          np.arange(num_my_ships) == nearest_ship_id))
      second_closest_distance = near_new_base_ship_distances[
        second_closest_id]
      # import pdb; pdb.set_trace()
      move_second_closest_to_base = second_closest_distance + 2 > int(
        opponent_nearest_ship_distances[add_strategic_base_position])
      if move_second_closest_to_base:
        second_closest_row = my_ship_positions[0][second_closest_id]
        second_closest_col = my_ship_positions[1][second_closest_id]
        towards_base_mask = get_mask_between_exclude_ends(
              second_closest_row, second_closest_col,
              add_strategic_base_position[0], add_strategic_base_position[1],
              grid_size)
        second_closest_ship_pos = grid_size*second_closest_row+(
          second_closest_col)
        second_closest_ship_k = my_ship_pos_to_k[second_closest_ship_pos]
        all_ship_scores[second_closest_ship_k][0][towards_base_mask] += 3e6
        convert_unavailable_positions[
          second_closest_row, second_closest_col] = 1
      
      # if observation['step'] == 169:
      #   import pdb; pdb.set_trace()
      #   x=1
      
      # Return the conversion square when I am converting this step
      if distance_to_conversion_square == 0:
        proceed_conversion = True
        if not(base_position_is_defended):
          # Make sure the base considers a spawn in the next step so it can be
          # defended
          # import pdb; pdb.set_trace()
          proceed_conversion = opponent_nearest_ship_distances[
            add_strategic_base_position] > 2
          if proceed_conversion:
            should_spawn_base_next_step = add_strategic_base_position
        if proceed_conversion:
          base_added = True
          added_base_pos = add_strategic_base_position
          history['add_strategic_base'] = False
          requested_save_conversion_budget = max(
            0, requested_save_conversion_budget-convert_cost)
          if move_second_closest_to_base:
            for score_id in range(2):
              all_ship_scores[second_closest_ship_k][score_id][
                add_strategic_base_position] += 3e6
            # Move onto the base if the ship can safely do so
            second_to_base_distance = DISTANCES[add_strategic_base_position][
              second_closest_row, second_closest_col]
            if second_to_base_distance == 1:
              to_base_dir = get_dir_from_target(
                second_closest_row, second_closest_col,
                add_strategic_base_position[0], add_strategic_base_position[1],
                grid_size)[0]
              if to_base_dir in all_ship_scores[second_closest_ship_k][9] and (
                  not to_base_dir in all_ship_scores[
                    second_closest_ship_k][6]):
                all_ship_scores[second_closest_ship_k][6].append(to_base_dir)
    else:
      # If I have a ship that can move towards the desired convert position and
      # the target square is currently not defended and we would otherwise
      # proceed with the conversion: move the ship closer
      if not base_position_is_defended and (
          required_halite_to_convert-player_obs[0]) <= spawn_cost:
        # import pdb; pdb.set_trace()
        second_closest_id = np.argmin(
          near_new_base_ship_distances + 100*(
            np.arange(num_my_ships) == nearest_ship_id))
        second_closest_row = my_ship_positions[0][second_closest_id]
        second_closest_col = my_ship_positions[1][second_closest_id]
        towards_base_mask = get_mask_between_exclude_ends(
              second_closest_row, second_closest_col,
              add_strategic_base_position[0], add_strategic_base_position[1],
              grid_size)
        second_closest_ship_pos = grid_size*second_closest_row+(
          second_closest_col)
        second_closest_ship_k = my_ship_pos_to_k[second_closest_ship_pos]
        all_ship_scores[second_closest_ship_k][0][towards_base_mask] += 3e6
        convert_unavailable_positions[
          second_closest_row, second_closest_col] = 1
  else:
    history['construct_strategic_base_position'] = None
    requested_save_conversion_budget = 0
    
  # if observation['step'] == 275:
  #   import pdb; pdb.set_trace()
  #   x=1
  
  return (all_ship_scores, base_added, added_base_pos,
          requested_save_conversion_budget, convert_unavailable_positions,
          should_spawn_base_next_step)

def protect_base(observation, env_config, all_ship_scores, player_obs,
                 defend_base_pos, history, base_override_move_positions,
                 ignore_defender_positions, max_considered_attackers=3,
                 halite_on_board_mult=1e-6):
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  defend_base_ignore_collision_key = None
  base_protected = True
  my_defend_base_ship_positions = np.zeros_like(opponent_ships)
  ignore_base_collision_ship_keys = []
  base_defense_keys = []
  if opponent_ships.sum():
    opponent_ship_count = opponent_ships.sum()
    grid_size = opponent_ships.shape[0]
    obs_halite = np.maximum(0, observation['halite'])
    collect_rate = env_config.collectRate
    obs_halite[obs_halite < 1/collect_rate] = 0
    my_ship_count = len(player_obs[2])
    base_row, base_col = defend_base_pos
    stacked_ships = np.stack(
      [rbs[2] for rbs in observation['rewards_bases_ships']])
    halite_ships = np.stack([
      rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
    halite_ships[stacked_ships.sum(0) == 0] = -1e-9
    opponent_ship_distances = DISTANCES[(base_row, base_col)][opponent_ships]+(
      halite_on_board_mult*halite_ships[opponent_ships])
    sorted_opp_distance = np.sort(opponent_ship_distances)
    ship_keys = list(player_obs[2].keys())
    
    ship_base_distances = np.zeros((my_ship_count, 8))
    # Go over all my ships and approximately compute how far they are expected
    # to be from the base !with no halite on board! by the end of the next turn
    # Approximate since returning ships are expected to always move towards the
    # base and other ships are assumed to be moving away.
    attack_opponent_campers = history['attack_opponent_campers']
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
      rather_not_move_to_base = len(
        set(to_base_directions) & set(ship_scores[9])) == 0
      can_not_move_to_base = int(
        (ship_halite != 0) and rather_not_move_to_base) or (
          ignore_defender_positions[row, col])
      prev_step_box_ship_target_distance = 10-history['prev_step'][
        'ships_on_box_mission'].get(ship_k, 10)
      
      # Exclude ships that attack opponent campers from the base defense logic
      if ship_k in attack_opponent_campers:
        current_distance = 1e2
        ship_halite = 1e6
        can_not_move_to_base = True
      
      ship_base_distances[i, 0] = current_distance
      ship_base_distances[i, 1] = current_distance + 1 - int(2*is_returning)
      ship_base_distances[i, 2] = ship_halite
      ship_base_distances[i, 3] = row
      ship_base_distances[i, 4] = col
      ship_base_distances[i, 5] = can_not_move_to_base
      ship_base_distances[i, 6] = int(rather_not_move_to_base)
      ship_base_distances[i, 7] = prev_step_box_ship_target_distance
    
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
        
    base_protected = worst_case_opponent_distances[0] > 0
    
    # if observation['step'] == 186:
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
          # FUTURE WORK: remove, this looks like a special case of the next
          # elif
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
      
      # Resolve ties by picking the ships that can safely move towards the
      # base
      defend_distances += 1e-9*ship_base_distances[:, 6]
      
      # Resolve ties by picking the ships that were not on a box in mission
      # in the past step
      defend_distances += 1e-10*ship_base_distances[:, 7]
      
      # Summon the closest K agents towards or onto the base to protect it.
      # When the ship halite is zero, we should aggressively attack base
      # raiders
      num_attackers = 1+np.where(opponent_can_attack_sorted)[0][-1]
      defend_distances_ids = np.argsort(defend_distances)
      
      # if observation['step'] == 178:
      #   import pdb; pdb.set_trace()
      
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
          base_defense_keys.append(ship_k)
          ship_scores = list(all_ship_scores[ship_k])
          ship_halite = int(ship_base_distances[ship_id, 2])
          row = int(ship_base_distances[ship_id, 3])
          col = int(ship_base_distances[ship_id, 4])
          if distance_to_base > 0 or i == 0:
            if distance_to_base <= 1:
              # Stay or move to the base; or stay 1 step away
              if i == 0:
                ship_scores[1][base_row, base_col] += 1e6*(
                  3+max_considered_attackers-i)
              elif obs_halite[row, col] == 0 or (
                  worst_case_opponent_distances[i] > distance_to_base):
                base_override_move_positions[row, col] = 1
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
              # Set the base as the target and override the base return
              # synchronization
              towards_base_mask = get_mask_between_exclude_ends(
                row, col, base_row, base_col, grid_size)
              ship_scores[0][towards_base_mask] += 9e5*(
                  1+max_considered_attackers-i)
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
                ship_scores[6] = list(
                  set(ship_scores[6] + not_bad_defend_dirs))
              
            my_defend_base_ship_positions[row, col] = 1
            all_ship_scores[ship_k] = tuple(ship_scores)
            
  return (all_ship_scores, defend_base_ignore_collision_key,
          base_protected, ignore_base_collision_ship_keys,
          my_defend_base_ship_positions, base_override_move_positions,
          base_defense_keys)

def update_occupied_count(
    row, col, occupied_target_squares, occupied_squares_count):
  occupied_target_squares[row, col] = 1
  occupied_squares_count[row, col] += 1
    
def update_scores_rescue_missions(
    config, all_ship_scores, stacked_ships, observation, halite_ships,
    steps_remaining, player_obs, obs_halite, history,
    opponent_ships_sensible_actions, weighted_base_mask, my_bases, np_rng,
    main_base_distances, my_defend_base_ship_positions, safe_to_return_halites,
    player_influence_maps, nearest_base_distances, max_box_distance=5,
    max_1_step_rescue_risk=0.02):
  grid_size = stacked_ships.shape[1]
  opponent_ships = stacked_ships[1:].sum(0) > 0
  my_zero_halite_ships = stacked_ships[0] & (halite_ships == 0) & (
    ~my_defend_base_ship_positions)
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
  
  # Exclude my campers that are not available for rescuing
  camping_ships_strategy = history['camping_ships_strategy']
  for ship_k in camping_ships_strategy:
    if not camping_ships_strategy[ship_k][3]:
      camping_row, camping_col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      my_zero_halite_ships[camping_row, camping_col] = 0 
  
  opponent_zero_halite_ships = opponent_ships & (halite_ships == 0)
  opponent_zero_halite_ship_density = smooth2d(
    opponent_zero_halite_ships, smooth_kernel_dim=4)
  zero_halite_pos = np.where(my_zero_halite_ships)
  on_rescue_mission = np.zeros((grid_size, grid_size), dtype=np.bool)
  rescue_move_positions_taken = np.zeros((grid_size, grid_size), dtype=np.bool)
  pos_to_k = {v[0]: k for k, v in player_obs[2].items()}
  _, base_distances, _ = get_nearest_base_distances(
    grid_size, history['my_base_not_attacked_positions'], observation)
  my_ships = observation['rewards_bases_ships'][0][2]
  my_bases = np.copy(observation['rewards_bases_ships'][0][1])
  for base_pos in history['my_base_not_attacked_positions']:
    # Note: Make sure to work on a copy of the original base observation!
    my_bases[base_pos] = 0 
  my_base_locations = np.where(my_bases)
  # main_base_location = np.where(main_base_distances == 0)
  
  # Identify the squares that are surrounded by an opponent by computing
  # the minimum halite of each square in each box in direction for each
  # opponent
  num_players = stacked_ships.shape[0]
  opponent_min_halite_box_dirs = 1e6*np.ones(
    (num_players, 4, grid_size, grid_size))
  opponents_num_nearby = np.zeros(
    (num_players, grid_size, grid_size))
  for i in range(1, num_players):
    opponent_ship_pos = np.where(stacked_ships[i])
    for j in range(stacked_ships[i].sum()):
      row = opponent_ship_pos[0][j]
      col = opponent_ship_pos[1][j]
      opponents_num_nearby[i][DISTANCES[row, col] <= 7] += 1
      opponent_ship_halite = halite_ships[row, col]
      for dir_id, d in enumerate(NOT_NONE_DIRECTIONS):
        mask = ROW_COL_BOX_DIR_MAX_DISTANCE_MASKS[row, col, d]
        opponent_min_halite_box_dirs[i, dir_id][mask] = np.minimum(
          opponent_min_halite_box_dirs[i, dir_id][mask],
          opponent_ship_halite)
  
  opponent_min_halite_box_all_dirs = np.max(opponent_min_halite_box_dirs, 1)
  opponent_min_halite_box_all_dirs[opponents_num_nearby < 4] = 1e6
  any_opponent_min_halite_box_all_dirs = np.min(
    opponent_min_halite_box_all_dirs, 0)
  my_boxed_ships = my_ships & (
    halite_ships > any_opponent_min_halite_box_all_dirs)
  boxed_ships = []
  if np.any(my_boxed_ships):
    my_boxed_pos = np.where(my_boxed_ships)
    for box_id in range(my_boxed_pos[0].size):
      row = my_boxed_pos[0][box_id]
      col = my_boxed_pos[1][box_id]
      ship_k = pos_to_k[row*grid_size + col]
      boxed_ships.append(ship_k)
    
    # if observation['step'] == 36:
    #   import pdb; pdb.set_trace()
    # print("Boxed in ships", observation['step'], my_boxed_pos)
  
  # Consider chased or boxed in ships
  chased_ships = list(history['chase_counter'][0].keys())
  chased_or_boxed = list(set(chased_ships+boxed_ships))
  
  # Put the ships that are on the escort to base list first
  escort_to_base_ships = list(set([
    e[0] for e in history['escort_to_base_list'] if (e[0] in player_obs[2])]))
  if len(escort_to_base_ships):
    chased_or_boxed = list(escort_to_base_ships + list(set(
      chased_or_boxed)-set(escort_to_base_ships)))
  already_escorted_ships = []
  num_considered_bases = my_base_locations[0].size
  for ship_k in chased_or_boxed:
    recompute_pos = False
    row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
    ship_scores = all_ship_scores[ship_k]
    valid_directions = copy.copy(ship_scores[6])
    
    should_rescue_is_chased = ship_k in history['chase_counter'] and len(
      set(valid_directions) - set(ship_scores[7]+ship_scores[8])) == 0 and (
        history['chase_counter'][0][ship_k][1] > 3)
      # Only call for help when the considered ship is nearly boxed in in all
      # directions and has been chased for a while
    
    # Flag the ship for rescuing if there is no safe path to my nearest base
    base_distances_ship = base_distances[:, row, col]
    nearest_base_id = np.argmin(base_distances_ship)
    # main_base_row = main_base_location[0][0]
    # main_base_col = main_base_location[1][0]
    # return_main_base_distance = main_base_distances[row, col]
    
    # Returning to a base that is not the main base is also supported
    # Select the target base as a function of the distance and influence score
    # on the square halfway between my current square and the considered base
    base_return_scores = np.zeros(num_considered_bases)
    for base_id in range(num_considered_bases):
      base_row = my_base_locations[0][base_id]
      base_col = my_base_locations[1][base_id]
      base_distance = DISTANCES[row, col][base_row, base_col]
      
      relative_pos = get_relative_position(
        row, col, base_row, base_col, grid_size)
      mid_row = int(row + relative_pos[0]/2) % grid_size
      mid_col = int(col + relative_pos[1]/2) % grid_size
      
      return_base_directions = get_dir_from_target(
        row, col, base_row, base_col, grid_size)
      can_move_to_base = False
      for d in return_base_directions:
        if d in valid_directions:
          can_move_to_base = True
          break
      
      base_return_scores[base_id] = player_influence_maps[
        0, mid_row, mid_col] - (player_influence_maps[
          1:, mid_row, mid_col].sum() + base_distance/1.5) + 2*int(
            can_move_to_base)
          
    best_base_id = np.argmax(base_return_scores)
    return_base_row = my_base_locations[0][best_base_id]
    return_base_col = my_base_locations[1][best_base_id]
    
    # if base_distances_ship[nearest_base_id] < (
    #     return_main_base_distance-1):
    #   return_base_row = my_base_locations[0][nearest_base_id]
    #   return_base_col = my_base_locations[1][nearest_base_id]
      
    # else:
    #   return_base_row = main_base_row
    #   return_base_col = main_base_col
      
    # if observation['step'] == 288:
    #   import pdb; pdb.set_trace()
      
    return_base_directions = get_dir_from_target(
      row, col, return_base_row, return_base_col, grid_size)
    one_step_invalid = list(set(NOT_NONE_DIRECTIONS).difference(set(
      all_ship_scores[ship_k][9])))
    not_good_dirs = list(set(all_ship_scores[ship_k][7] + all_ship_scores[
      ship_k][8] + one_step_invalid))
    base_return_not_good_dirs = [d in not_good_dirs for d in (
      return_base_directions)]
    should_rescue_can_not_return_base = (halite_ships[row, col] > 0) and (
      len(set(return_base_directions) & set(
        all_ship_scores[ship_k][6])) == 0 or (
          np.all(np.array(base_return_not_good_dirs)))) and (
            halite_ships[row, col] >= safe_to_return_halites[row, col])
          
    if (should_rescue_is_chased or should_rescue_can_not_return_base) and (
        base_distances_ship[nearest_base_id] > 2):
      # import pdb; pdb.set_trace()
      # print(observation['step'], row, col, return_base_row, return_base_col)
      # if observation['step'] == 56 and row == 18:
      #   import pdb; pdb.set_trace()
      
      # if ship_k in boxed_ships and not ship_k in chased_ships:
      #   import pdb; pdb.set_trace()
      #   x=1
      
      nearly_boxed_in = True
      if should_rescue_is_chased and not should_rescue_can_not_return_base:
        valid_directions = valid_directions if len(ship_scores[8]) == 0 else (
            ship_scores[17]) # Also consider bad N-step directions
        threat_opponents = opponent_ships & (halite_ships < halite_ships[
          row, col])
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
        
        # if None in valid_directions and None in ship_scores[8] and len(
        #     valid_directions) == 1 and len(ship_scores[17]) > 1:
        #   # Edge case - we should consider all n step bad directions rather
        #   # than only considering staying still when rescuing a ship
        #   valid_directions = ship_scores[17]
        #   this_ship_scores = list(all_ship_scores[ship_k])
        #   this_ship_scores[6] = copy.copy(valid_directions)
        #   this_ship_scores[8] = copy.copy(valid_directions)
        #   this_ship_scores[14] = this_ship_scores[18]
        #   all_ship_scores[ship_k] = tuple(this_ship_scores)
        
        # if num_considered_bases > 1:
        #   # import pdb; pdb.set_trace()
        # print(observation['step'], row, col, return_base_row,
        #       return_base_col)
        
        # Compute weights for all d1 and d2 friendly zero halite rescuers
        num_d1_friends = (friendly_zero_halite_distances == 1).sum()
        num_d2_friends = (friendly_zero_halite_distances == 2).sum()
        any_good_d1_rescuers = False
        any_good_d2_rescuers = False
        if num_d1_friends > 0:
          # import pdb; pdb.set_trace()
          d1_rescuer_ids = np.where(friendly_zero_halite_distances == 1)[0]
          d1_scores = np.zeros(num_d1_friends)
          for rescuer_id in range(num_d1_friends):
            rescuer_row = zero_halite_pos[0][d1_rescuer_ids[rescuer_id]]
            rescuer_col = zero_halite_pos[1][d1_rescuer_ids[rescuer_id]]
            to_rescuer_dir = get_dir_from_target(
              row, col, rescuer_row, rescuer_col, grid_size)[0]
            rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
            ship_k_not_immediate_bad = list(
              set(all_ship_scores[ship_k][6]).union(
                set(all_ship_scores[ship_k][17])))
            aligned_dirs = list(set(ship_k_not_immediate_bad) & set(
                all_ship_scores[rescuer_k][6]) & set(return_base_directions))
            good_rescuer = (to_rescuer_dir in valid_directions) or (
              len(aligned_dirs) > 0)
            d1_scores[rescuer_id] = -50 + 100*int(good_rescuer) - DISTANCES[
              rescuer_row, rescuer_col][return_base_row, return_base_col]/1.5+(
                player_influence_maps[0, rescuer_row, rescuer_col]) - (
                  player_influence_maps[1:, rescuer_row, rescuer_col].sum())
          
          any_good_d1_rescuers = np.any(d1_scores > 0)
          
        if num_d2_friends > 0:
          # import pdb; pdb.set_trace()
          d2_rescuer_ids = np.where(friendly_zero_halite_distances == 2)[0]
          d2_scores = np.zeros(num_d2_friends)
          for rescuer_id in range(num_d2_friends):
            rescuer_row = zero_halite_pos[0][d2_rescuer_ids[rescuer_id]]
            rescuer_col = zero_halite_pos[1][d2_rescuer_ids[rescuer_id]]
            to_rescuer_dir = get_dir_from_target(
              row, col, rescuer_row, rescuer_col, grid_size)
            any_rescuer_dir_ind_valid_dir = False
            for d in to_rescuer_dir:
              any_rescuer_dir_ind_valid_dir = any_rescuer_dir_ind_valid_dir or(
                d in valid_directions)
            rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
            ship_k_not_immediate_bad = list(
              set(all_ship_scores[ship_k][6]).union(
                set(all_ship_scores[ship_k][17])))
            aligned_dirs = list(set(ship_k_not_immediate_bad) & set(
                all_ship_scores[rescuer_k][6]) & set(return_base_directions))
            good_rescuer = (any_rescuer_dir_ind_valid_dir and None in (
              all_ship_scores[rescuer_k][6])) or (len(aligned_dirs) > 0)
            d2_scores[rescuer_id] = -50 + 100*int(good_rescuer) - DISTANCES[
              rescuer_row, rescuer_col][return_base_row, return_base_col]/1.5+(
                player_influence_maps[0, rescuer_row, rescuer_col]) - (
                  player_influence_maps[1:, rescuer_row, rescuer_col].sum())
          
          any_good_d2_rescuers = np.any(d2_scores > 0)
        
        friendly_zero_halite_distance_d1_scores = np.copy(
            friendly_zero_halite_distances).astype(np.float)
        if any_good_d1_rescuers:
          # If there are multiple 1-distance ships: pick the rescuer that is
          # closest to the target base which I can move to safely
          # import pdb; pdb.set_trace()
          friendly_zero_halite_distance_d1_scores[d1_rescuer_ids[
            np.argmax(d1_scores)]] -= 1e-6
        elif min_halite_distance == 1:
          # Go for a 2-distance ship if I can move to it safely and I can't
          # move safely to the 1-distance ship.
          if any_good_d2_rescuers:
            # Use a d2 rescuer instead
            # import pdb; pdb.set_trace()
            min_halite_distance = 2
          else:
            # Pick the best bad d1 rescuer
            friendly_zero_halite_distance_d1_scores[d1_rescuer_ids[
              np.argmax(d1_scores)]] -= 1e-6
        
        # if observation['step'] >= 154:
        #   import pdb; pdb.set_trace()
        
        # Follow the nearest zero halite ship home if it is at a distance one
        # of the ship
        is_protected = False
        if min_halite_distance == 1:
          # Jointly move to the nearest weighted base and assume that the
          # former zero halite position won't be attacked
          # dm = DISTANCE_MASKS[(row, col)]
          # base_scores = dm*weighted_base_mask*my_bases
          # target_base = np.where(base_scores == base_scores.max())
          # target_base = (target_base[0][0], target_base[1][0])
          target_base = (return_base_row, return_base_col)
          
          nearest_halite_id = np.argmin(
            friendly_zero_halite_distance_d1_scores)
          rescuer_row = zero_halite_pos[0][nearest_halite_id]
          rescuer_col = zero_halite_pos[1][nearest_halite_id]
          on_rescue_mission[row, col] = 1
          on_rescue_mission[rescuer_row, rescuer_col] = 1
          recompute_pos = True
          to_rescuer_dir = get_dir_from_target(
            row, col, rescuer_row, rescuer_col, grid_size)[0]
          rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
          ship_k_not_immediate_bad = list(
            set(all_ship_scores[ship_k][6]).union(
              set(all_ship_scores[ship_k][17])))
          aligned_dirs = list(set(ship_k_not_immediate_bad) & set(
              all_ship_scores[rescuer_k][6]) & set(return_base_directions))
          
          # Don't risk moving onto the previous step zero halite square if a
          # potential opponent has a considerable probability of engaging in
          # 1-step risky actions
          if len(aligned_dirs) == 0:
            risk_score = 0
            for d in NOT_NONE_DIRECTIONS:
              neighbor_rescue_row, neighbor_rescue_col = move_ship_row_col(
                rescuer_row, rescuer_col, d, grid_size)
              if opponent_zero_halite_ships[
                  neighbor_rescue_row, neighbor_rescue_col]:
                opponent_id = np.where(stacked_ships[
                  :, neighbor_rescue_row, neighbor_rescue_col])[0][0]
                risk_score += history['zero_halite_move_behavior'][opponent_id][
                  'False_0']
            shared_escape_dirs = list(set(ship_k_not_immediate_bad) & set(
              all_ship_scores[rescuer_k][6]))
            if None in shared_escape_dirs:
              shared_escape_dirs.remove(None)
            if risk_score > max_1_step_rescue_risk and len(
                shared_escape_dirs) > 0 and len(aligned_dirs) == 0:
              # print("Avoiding risky rescue move", observation['step'], row,
              #       col, risk_score)
              aligned_dirs = shared_escape_dirs
          # elif risk_score > max_1_step_rescue_risk and OPPOSITE_MAPPING[
          #     to_rescuer_dir] in all_ship_scores[rescuer_k][6] and len(
          #       aligned_dirs) == 0 and (
          #         len(all_ship_scores[ship_k][6]) > 0 and (
          #           not None in all_ship_scores[ship_k][6])):
          #   # Also consider having the zero halite ship take the position of
          #   # the rescued ship if that is safe. This branch is likely
          #   # impossible
          #   import pdb; pdb.set_trace()
          #   x=1
            
          override_override_mask = False
          if (to_rescuer_dir not in ship_k_not_immediate_bad) or (
              len(aligned_dirs) > 0):
            # It is better to take a safe step with both ships if that option
            # is available
            # import pdb; pdb.set_trace()
            if len(aligned_dirs) == 0 and not to_rescuer_dir in (
                all_ship_scores[ship_k][6]):
              all_ship_scores[ship_k][6].append(to_rescuer_dir)
            else:
              if len(aligned_dirs) > 1:
                str_aligned_dirs = [d if d is not None else "None" for d in (
                  aligned_dirs)]
                str_aligned_dirs.sort()  # Set intersect is flaky
                aligned_dirs = [d if d != "None" else None for d in (
                  str_aligned_dirs)]
              override_override_mask = True
            
          if override_override_mask:
            aligned_dir = np_rng.choice(aligned_dirs)
            rescuer_move_row, rescuer_move_col = move_ship_row_col(
              rescuer_row, rescuer_col, aligned_dir, grid_size)
            rescue_move_positions_taken[rescuer_move_row, rescuer_move_col] = 1
            for score_id in range(3):
              all_ship_scores[rescuer_k][score_id][
                rescuer_move_row, rescuer_move_col] += 1e4
            if not aligned_dir in all_ship_scores[ship_k][6]:
              all_ship_scores[ship_k][6].append(aligned_dir)
            escape_row, escape_col = move_ship_row_col(
              row, col, aligned_dir, grid_size)
          else:
            # Consider escaping along the opposite side of the board if one
            # side is much more congested
            relative_pos = get_relative_position(
              rescuer_row, rescuer_col, target_base[0], target_base[1],
              grid_size)
            mid_path_row_short = int(
              rescuer_row + relative_pos[0]/2) % grid_size
            mid_path_col_short = int(
              rescuer_col + relative_pos[1]/2) % grid_size
            mid_path_row_long = int(
              rescuer_row - (grid_size - relative_pos[0])/2) % grid_size
            mid_path_col_long = int(
              rescuer_col - (grid_size - relative_pos[1])/2) % grid_size
            start_rescuer_row = rescuer_row
            if np.abs(relative_pos[0]) > 7:
              # Determine start_rescuer_row
              mid_density_short_row = player_influence_maps[
                0, mid_path_row_short, mid_path_col_short] - (
                  player_influence_maps[
                    1:, mid_path_row_short, mid_path_col_short]).sum()
              mid_density_long_row = player_influence_maps[
                0, mid_path_row_long, mid_path_col_short] - (
                  player_influence_maps[
                    1:, mid_path_row_long, mid_path_col_short]).sum()
                  
              distance_diff_short = grid_size-2*np.abs(relative_pos[0])
              if mid_density_long_row - mid_density_short_row > (
                  distance_diff_short/2):
                # import pdb; pdb.set_trace()
                start_rescuer_row = mid_path_row_long
              
            start_rescuer_col = rescuer_col
            if np.abs(relative_pos[1]) > 7:
              # Determine start_rescuer_col
              mid_density_short_col = player_influence_maps[
                0, mid_path_row_short, mid_path_col_short] - (
                  player_influence_maps[
                    1:, mid_path_row_short, mid_path_col_short]).sum()
              mid_density_long_col = player_influence_maps[
                0, mid_path_row_short, mid_path_col_long] - (
                  player_influence_maps[
                    1:, mid_path_row_short, mid_path_col_long]).sum()
              distance_diff_short = grid_size-2*np.abs(relative_pos[1])
              if mid_density_long_col - mid_density_short_col > (
                  distance_diff_short):
                # import pdb; pdb.set_trace()
                start_rescuer_col = mid_path_col_long
            
            increase_mask = get_mask_between_exclude_ends(
              target_base[0], target_base[1], start_rescuer_row,
              start_rescuer_col, grid_size)
            # Avoid the rescued ship half plane if that leaves the rescuer with
            # options
            to_base_dirs = get_dir_from_target(
              start_rescuer_row, start_rescuer_col, target_base[0],
              target_base[1], grid_size)
            valid_to_base_dirs = list(set(to_base_dirs) & set(
              all_ship_scores[rescuer_k][6]))
            if len(valid_to_base_dirs) > 1:
              to_rescued_dir = OPPOSITE_MAPPING[to_rescuer_dir]
              if to_rescued_dir in all_ship_scores[rescuer_k][6]:
                # import pdb; pdb.set_trace()
                increase_mask[HALF_PLANES_CATCH[
                  start_rescuer_row, start_rescuer_col][to_rescued_dir]] = 0
            if len(all_ship_scores[rescuer_k][6]) == 1 and (
                all_ship_scores[rescuer_k][6][0] is None):
              # Take a move risk rather than staying still - otherwise the
              # non zero halite ship is likely to get orphaned
              # import pdb; pdb.set_trace()
              risk_scores = np.zeros(len(to_base_dirs))
              for risk_id, d in enumerate(to_base_dirs):
                move_row, move_col = move_ship_row_col(
                  rescuer_row, rescuer_col, d, grid_size)
                for potential_threat_dir in MOVE_DIRECTIONS:
                  threat_row, threat_col = move_ship_row_col(
                    move_row, move_col, potential_threat_dir, grid_size)
                  if opponent_ships[threat_row, threat_col] and halite_ships[
                      threat_row, threat_col] == 0:
                    opponent_id = player_ids[threat_row, threat_col]
                    is_near_base = nearest_base_distances[
                      threat_row, threat_col] <= config[
                        'log_near_base_distance']
                    distance = int(d is not None) + int(
                      potential_threat_dir is not None)
                    risk_lookup_k = str(is_near_base) + '_' + str(distance)
                    risk_scores[risk_id] = max(
                      risk_scores[risk_id], history[
                        'zero_halite_move_behavior'][opponent_id][
                          risk_lookup_k])
              best_risk_score = risk_scores.min()
              if best_risk_score < 0.15:
                print("Risky escort move with boxed in zero halite ship",
                      observation['step'], row, col)
                # import pdb; pdb.set_trace()
                rescuer_selected_move_dir = to_base_dirs[
                  np.argmin(risk_scores)]
                increase_mask = np.zeros_like(increase_mask)
                rescue_move_position = move_ship_row_col(
                  rescuer_row, rescuer_col, rescuer_selected_move_dir,
                  grid_size)
                all_ship_scores[rescuer_k][0][rescue_move_position] += 3e5
                all_ship_scores[rescuer_k][6].remove(None)
                all_ship_scores[rescuer_k][6].append(rescuer_selected_move_dir)
                
            for score_id in range(3):
              all_ship_scores[rescuer_k][score_id][increase_mask] += 1e4
            escape_row = rescuer_row
            escape_col = rescuer_col
          all_ship_scores[rescuer_k][3][:] -= 1e4 # No attack during rescue
            
          all_ship_scores[ship_k][0][escape_row, escape_col] = 1e8
          rescue_move_positions_taken[escape_row, escape_col] = 1
          
          history['escort_to_base_list'].append( # I will not let you go baby
            (ship_k, rescuer_k, True, 50, 50))
          already_escorted_ships.append(ship_k)
          
          # Add the ship pair to the escort-list for a fixed number of steps
          is_protected = True
          
        elif min_halite_distance == 2:
          friendly_zero_halite_distance_scores = np.copy(
            friendly_zero_halite_distances).astype(np.float)
          # If there is a single threatening ship: prefer distance 2 ships
          # that are less towards that direction and avoid that direction when
          # moving to the base
          threat_directions = list(
            set(NOT_NONE_DIRECTIONS) - set(valid_directions))
          avoid_2_step_direction = None
          if not None in valid_directions and len(threat_directions) == 1:
            single_threat_direction = threat_directions[0]
            avoid_2_step_direction = OPPOSITE_MAPPING[single_threat_direction]
            for zero_2_id, zero_halite_ship_id in enumerate(
                np.argsort(friendly_zero_halite_distances)[:5]):
              rescuer_distance = friendly_zero_halite_distances[
                zero_halite_ship_id]
              if rescuer_distance == 2:
                rescuer_row = zero_halite_pos[0][zero_halite_ship_id]
                rescuer_col = zero_halite_pos[1][zero_halite_ship_id]
                direction_distance = get_directional_distance(
                  row, col, rescuer_row, rescuer_col, grid_size,
                  avoid_2_step_direction)
                friendly_zero_halite_distance_scores[zero_halite_ship_id] +=(
                  1e-6*direction_distance)
          
          considered_zero_2_ship_ids = np.argsort(
            friendly_zero_halite_distance_scores)[:5]
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
              
              # Plan A: Let the non zero halite ship wait
              # (stupid, this never happens - the non zero halite ship can not
              # wait, it is being chased)
              # if None in valid_directions:
              #   rescue_dirs = get_dir_from_target(
              #     rescuer_row, rescuer_col, row, col, grid_size)
              #   safe_rescue_dirs = set(rescue_dirs) & set(
              #     all_ship_scores[rescuer_k][6])
              #   if len(safe_rescue_dirs) > 0:
              #     rescue_dirs = list(safe_rescue_dirs)
              #   rescue_dir = np_rng.choice(rescue_dirs)
              #   rescuer_move_row, rescuer_move_col = move_ship_row_col(
              #     rescuer_row, rescuer_col, rescue_dir, grid_size)
              #   move_row = row
              #   move_col = col
              #   is_protected = True
              
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
                  if len(valid_to_rescuer_dirs) > 1 and (
                      avoid_2_step_direction in valid_to_rescuer_dirs) and (
                        not avoid_2_step_direction is None):
                    valid_to_rescuer_dirs.remove(avoid_2_step_direction)
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
                        d, rescue_r, rescue_c, DISTANCES[rescue_r, rescue_c][
                          return_base_row, return_base_col]))
                
                # Order the zero halite move squares by distance to the target
                # base
                distance_order = np.argsort(np.array([s[3] for s in (
                 safe_zero_halite_squares_dirs)])).tolist()
                safe_zero_halite_squares_dirs_ordered = []
                for ordered_id in distance_order:
                  safe_zero_halite_squares_dirs_ordered.append(
                    safe_zero_halite_squares_dirs[ordered_id])
                
                for rescue_d, rescue_r, rescue_c, _ in (
                    safe_zero_halite_squares_dirs_ordered):
                  # Figure out if the chased ship can move to the new rescuer
                  # square
                  valid_to_rescuer_moved_dirs = []
                  for d in valid_directions:
                    if d is not None:
                      move_row, move_col = move_ship_row_col(
                        row, col, d, grid_size)
                      
                      if HALF_PLANES_RUN[row, col][d][rescue_r, rescue_c] and (
                          not (move_row == rescue_r and move_col == rescue_c)):
                        valid_to_rescuer_moved_dirs.append(d)
                  
                  if valid_to_rescuer_moved_dirs:
                    if len(valid_to_rescuer_moved_dirs) > 1 and (
                        not avoid_2_step_direction is None) and (
                      avoid_2_step_direction in valid_to_rescuer_moved_dirs):
                      valid_to_rescuer_moved_dirs.remove(
                        avoid_2_step_direction)
                    to_rescuer_dir = np_rng.choice(valid_to_rescuer_moved_dirs)
                    move_row, move_col = move_ship_row_col(
                      row, col, to_rescuer_dir, grid_size)
                    rescuer_move_row = rescue_r
                    rescuer_move_col = rescue_c
                    is_protected = True
                    break
                  
              # Plan D: Consider other rescuers up to distance 3 if I
              # can't wait at the current/nearby square
              # This won't be implemented
              
            if is_protected:
              on_rescue_mission[row, col] = 1
              on_rescue_mission[rescuer_row, rescuer_col] = 1
              recompute_pos = True
              rescue_move_positions_taken[
                rescuer_move_row, rescuer_move_col] = 1
              rescue_move_positions_taken[move_row, move_col] = 1
              
              all_ship_scores[ship_k][0][move_row, move_col] = 1e8
              all_ship_scores[rescuer_k][0][
                rescuer_move_row, rescuer_move_col] = 1e8
              already_escorted_ships.append(ship_k)
              break
        
        if not is_protected and len(valid_directions) > 0:
          # Only consider zero halite ships in the directions I can move to
          # or ships that are about as close as the nearest threatening
          # opponent
          valid_rescue_mask = np.zeros_like(my_zero_halite_ships)
          for d in valid_directions:
            valid_rescue_mask[HALF_PLANES_RUN[row, col][d]] = 1
            
          nearest_threat_distance = DISTANCES[row, col][opponent_ships & (
            halite_ships < halite_ships[row, col])].min()
          nearby_mask = DISTANCES[row, col] <= (nearest_threat_distance+1)
          valid_rescue_mask = valid_rescue_mask | nearby_mask
            
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
                else:
                  if valid_move_dirs:
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
                      
                    on_rescue_mission[row, col] = 1
                    on_rescue_mission[rescuer_row, rescuer_col] = 1
                    recompute_pos = True
                    rescue_move_positions_taken[move_row, move_col] = 1
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
              rescue_move_positions_taken[move_row, move_col] = 1
              all_ship_scores[ship_k][0][move_row, move_col] = 1e4
            else:
              move_dir = valid_directions[0]
            
            # Slightly incentivize the nearest zero halite ship to move
            # towards my move square (does not have to be in my valid move
            # direction mask)
            if zero_halite_pos[0].size:
              move_row, move_col = move_ship_row_col(
                row, col, move_dir, grid_size)
              my_zero_halite_ships_move_distances = DISTANCES[
                move_row, move_col][my_zero_halite_ships]
              nearest_halite_id = np.argmin(
                my_zero_halite_ships_move_distances)
              rescuer_row = zero_halite_pos[0][nearest_halite_id]
              rescuer_col = zero_halite_pos[1][nearest_halite_id]
              rescuer_k = pos_to_k[rescuer_row*grid_size+rescuer_col]
              increase_mask = get_mask_between_exclude_ends(
                move_row, move_col, rescuer_row, rescuer_col, grid_size)
              for score_id in range(3):
                all_ship_scores[rescuer_k][score_id][increase_mask] += 1e2
                
    if recompute_pos:
      my_zero_halite_ships &= (~on_rescue_mission)
      zero_halite_pos = np.where(my_zero_halite_ships)
        
      
  # Escort previously chased ships to the base - only stop escorting once I
  # have two valid actions towards the base or have one valid action and am
  # in the same row/col
  new_escort_list = []
  added_escorted_ships = []
  for (ship_k, rescuer_k, rescue_executed, min_escort_steps_remaining,
       max_escort_steps_remaining) in history['escort_to_base_list']:
    if ship_k in player_obs[2] and (rescuer_k in player_obs[2]):
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      
      dm = DISTANCE_MASKS[(row, col)]
      base_scores = dm*weighted_base_mask*my_bases
      # target_base = np.where(base_scores == base_scores.max())
      # target_base = (target_base[0][0], target_base[1][0])
      ship_scores = all_ship_scores[ship_k]
      valid_directions = valid_directions if len(ship_scores[8]) == 0 else (
        ship_scores[17]) # Also consider bad N-step directions
      
      base_return_scores = np.zeros(num_considered_bases)
      for base_id in range(num_considered_bases):
        base_row = my_base_locations[0][base_id]
        base_col = my_base_locations[1][base_id]
        base_distance = base_distances[base_id][row, col]
        relative_pos = get_relative_position(
          row, col, base_row, base_col, grid_size)
        mid_row = int(row + relative_pos[0]/2) % grid_size
        mid_col = int(col + relative_pos[1]/2) % grid_size
        # mid_row = int(np.where(
        #     np.abs(base_row-row) <= (grid_size // 2),
        #     np.round((base_row*(1-1e-9)+row*(1+1e-9))/2),
        #     np.where(base_row*(1-1e-9)+row*(1+1e-9) >= grid_size,
        #              np.round(
        #                (base_row*(1-1e-9)+row*(1+1e-9)-grid_size)/2),
        #              np.mod(np.round(
        #                (base_row*(1-1e-9)+row*(1+1e-9)+grid_size)/2),
        #                grid_size))
        #     ))
        # mid_col = int(np.where(
        #   np.abs(base_col-col) <= (grid_size // 2),
        #   np.round((base_col*(1-1e-9)+col*(1+1e-9))/2),
        #   np.where(base_col*(1-1e-9)+col*(1+1e-9) >= grid_size,
        #            np.round(
        #              (base_col*(1-1e-9)+col*(1+1e-9)-grid_size)/2),
        #            np.mod(np.round(
        #              (base_col*(1-1e-9)+col*(1+1e-9)+grid_size)/2),
        #              grid_size))
        #   ))
        
        return_base_directions = get_dir_from_target(
          row, col, base_row, base_col, grid_size)
        can_move_to_base = False
        for d in return_base_directions:
          if d in valid_directions:
            can_move_to_base = True
            break
        
        # base_return_scores[base_id] = player_influence_maps[
        #   0, mid_row, mid_col] - (player_influence_maps[
        #     1:, mid_row, mid_col].sum() + base_distance/3)
            
        base_return_scores[base_id] = player_influence_maps[
          0, mid_row, mid_col] - (player_influence_maps[
            1:, mid_row, mid_col].sum() + base_distance/1.5) + 2*int(
              can_move_to_base)
          
      best_base_id = np.argmax(base_return_scores)
      return_base_row = my_base_locations[0][best_base_id]
      return_base_col = my_base_locations[1][best_base_id]
      target_base = (return_base_row, return_base_col)
      
      base_distances_ship = base_distances[:, row, col]
      nearest_base_id = np.argmin(base_distances_ship)
      abort_rescue = base_distances_ship[nearest_base_id] <= 2
      # main_base_row = main_base_location[0]
      # main_base_col = main_base_location[1]
      # return_main_base_distance = main_base_distances[row, col]
      
      # # Returning to a base that is not the main base
      # if base_distances_ship[nearest_base_id] < (
      #     return_main_base_distance-1):
      #   return_base_row = my_base_locations[0][nearest_base_id]
      #   return_base_col = my_base_locations[1][nearest_base_id]
      # else:
      #   return_base_row = main_base_row
      #   return_base_col = main_base_col
      
      # if observation['step'] == 278:
      #   import pdb; pdb.set_trace()
        
      return_base_directions = get_dir_from_target(
        row, col, return_base_row, return_base_col, grid_size)
      one_step_invalid = list(set(NOT_NONE_DIRECTIONS).difference(set(
        all_ship_scores[ship_k][9])))
      not_good_dirs = list(set(all_ship_scores[ship_k][7] + all_ship_scores[
        ship_k][8] + one_step_invalid))
      base_return_good_dirs = np.array([not d in not_good_dirs for d in (
        return_base_directions)])
      num_good_base_return_dirs = base_return_good_dirs.sum()
      if not rescue_executed and base_scores.max() > 0 and (
        not ship_k in already_escorted_ships) and not (
          ship_k in added_escorted_ships):
        # print("ENTERING THE ESCORT TO BASE BRANCH")
        # Jointly move to the nearest weighted base and assume that the
        # former zero halite position won't be attacked
        rescuer_row, rescuer_col = row_col_from_square_grid_pos(
          player_obs[2][rescuer_k][0], grid_size)
        
        # Abort the rescue operation when the rescuing ship has gobbled up
        # another non-zero halite ship or if both are no longer next to
        # each other
        abort_rescue = (DISTANCES[row, col][rescuer_row, rescuer_col] > 1) or (
          halite_ships[rescuer_row, rescuer_col] > 0)
        
        # Abort once I have two valid actions towards the base or have one
        # valid action and am in the same row/col
        # print(row, col, min_escort_steps_remaining, observation['step'])
        if min_escort_steps_remaining == 0:
          abort_rescue = abort_rescue or (num_good_base_return_dirs == 2)
          if not abort_rescue and num_good_base_return_dirs == 1:
            # import pdb; pdb.set_trace()
            abort_rescue = (row == return_base_row) or (
              col == return_base_col)
            
        # Abort if the ship can safely return to a base
        abort_rescue = abort_rescue or (
          halite_ships[row, col] < safe_to_return_halites[row, col])
        
        if not abort_rescue:
          already_escorted_ships.append(ship_k)
          on_rescue_mission[row, col] = 1
          on_rescue_mission[rescuer_row, rescuer_col] = 1
          to_rescuer_dir = get_dir_from_target(
            row, col, rescuer_row, rescuer_col, grid_size)[0]
          ship_k_not_immediate_bad = list(
            set(all_ship_scores[ship_k][6]).union(
              set(all_ship_scores[ship_k][17])))
          aligned_dirs = list(set(ship_k_not_immediate_bad) & set(
              all_ship_scores[rescuer_k][6]) & set(return_base_directions))
          
          # if observation['step'] == 231:
          #   import pdb; pdb.set_trace()
          
          if len(aligned_dirs) == 0:
            # Don't risk moving onto the previous step zero halite square if a
            # potential opponent has a considerable probability of engaging in
            # 1-step risky actions
            risk_score = 0
            for d in NOT_NONE_DIRECTIONS:
              neighbor_rescue_row, neighbor_rescue_col = move_ship_row_col(
                rescuer_row, rescuer_col, d, grid_size)
              if opponent_zero_halite_ships[
                  neighbor_rescue_row, neighbor_rescue_col]:
                opponent_id = np.where(stacked_ships[
                  :, neighbor_rescue_row, neighbor_rescue_col])[0][0]
                risk_score += history['zero_halite_move_behavior'][
                  opponent_id]['False_0']
            shared_escape_dirs = list(set(ship_k_not_immediate_bad) & set(
              all_ship_scores[rescuer_k][6]))
            if risk_score > max_1_step_rescue_risk and len(
                shared_escape_dirs) > 0 and len(aligned_dirs) == 0:
              # print("Avoiding risky rescue move", observation['step'], row,
              #       col)
              aligned_dirs = shared_escape_dirs
            
          override_override_mask = False
          if (to_rescuer_dir not in ship_k_not_immediate_bad) or (
              len(aligned_dirs) > 0):
            # It is probably worth the risk when I am rescuing a ship in
            # trouble
            # It is better to take a safe step with both ships if that option
            # is available
            # import pdb; pdb.set_trace()
            if len(aligned_dirs) == 0 and not to_rescuer_dir in (
                all_ship_scores[ship_k][6]):
              all_ship_scores[ship_k][6].append(to_rescuer_dir)
            else:
              if len(aligned_dirs) > 1:
                str_aligned_dirs = [d if d is not None else "None" for d in (
                  aligned_dirs)]
                str_aligned_dirs.sort()  # Set intersect is flaky
                aligned_dirs = [d if d != "None" else None for d in (
                  str_aligned_dirs)]
              override_override_mask = True
            
          if override_override_mask:
            aligned_dir = aligned_dirs[0]
            rescuer_move_row, rescuer_move_col = move_ship_row_col(
              rescuer_row, rescuer_col, aligned_dir, grid_size)
            rescue_move_positions_taken[rescuer_move_row, rescuer_move_col] = 1
            for score_id in range(3):
              all_ship_scores[rescuer_k][score_id][
                rescuer_move_row, rescuer_move_col] += 1e4
            if not aligned_dir in all_ship_scores[ship_k][6]:
              all_ship_scores[ship_k][6].append(aligned_dir)
            escape_row, escape_col = move_ship_row_col(
              row, col, aligned_dir, grid_size)
          else:
            # Consider escaping along the opposite side of the board if one
            # side is much more congested
            relative_pos = get_relative_position(
              rescuer_row, rescuer_col, target_base[0], target_base[1],
              grid_size)
            mid_path_row_short = int(
              rescuer_row + relative_pos[0]/2) % grid_size
            mid_path_col_short = int(
              rescuer_col + relative_pos[1]/2) % grid_size
            mid_path_row_long = int(
              rescuer_row - (grid_size - relative_pos[0])/2) % grid_size
            mid_path_col_long = int(
              rescuer_col - (grid_size - relative_pos[1])/2) % grid_size
            start_rescuer_row = rescuer_row
            if np.abs(relative_pos[0]) > 7:
              # Determine start_rescuer_row
              mid_density_short_row = player_influence_maps[
                0, mid_path_row_short, mid_path_col_short] - (
                  player_influence_maps[
                    1:, mid_path_row_short, mid_path_col_short]).sum()
              mid_density_long_row = player_influence_maps[
                0, mid_path_row_long, mid_path_col_short] - (
                  player_influence_maps[
                    1:, mid_path_row_long, mid_path_col_short]).sum()
                  
              distance_diff_short = grid_size-2*np.abs(relative_pos[0])
              if mid_density_long_row - mid_density_short_row > (
                  distance_diff_short/2):
                # import pdb; pdb.set_trace()
                start_rescuer_row = mid_path_row_long
              
            start_rescuer_col = rescuer_col
            if np.abs(relative_pos[1]) > 7:
              # Determine start_rescuer_col
              mid_density_short_col = player_influence_maps[
                0, mid_path_row_short, mid_path_col_short] - (
                  player_influence_maps[
                    1:, mid_path_row_short, mid_path_col_short]).sum()
              mid_density_long_col = player_influence_maps[
                0, mid_path_row_short, mid_path_col_long] - (
                  player_influence_maps[
                    1:, mid_path_row_short, mid_path_col_long]).sum()
              distance_diff_short = grid_size-2*np.abs(relative_pos[1])
              if mid_density_long_col - mid_density_short_col > (
                  distance_diff_short):
                # import pdb; pdb.set_trace()
                start_rescuer_col = mid_path_col_long
            
            increase_mask = get_mask_between_exclude_ends(
              target_base[0], target_base[1], start_rescuer_row,
              start_rescuer_col, grid_size)
            # Avoid the rescued ship half plane if that leaves the rescuer with
            # options (that way the rescue mission makes progress)
            to_base_dirs = get_dir_from_target(
              start_rescuer_row, start_rescuer_col, target_base[0],
              target_base[1], grid_size)
            valid_to_base_dirs = list(set(to_base_dirs) & set(
              all_ship_scores[rescuer_k][6]))
            if len(valid_to_base_dirs) > 1:
              to_rescued_dir = OPPOSITE_MAPPING[to_rescuer_dir]
              if to_rescued_dir in all_ship_scores[rescuer_k][6]:
                # import pdb; pdb.set_trace()
                increase_mask[HALF_PLANES_CATCH[
                  start_rescuer_row, start_rescuer_col][to_rescued_dir]] = 0
            if len(all_ship_scores[rescuer_k][6]) == 1 and (
                all_ship_scores[rescuer_k][6][0] is None):
              # Take a move risk rather than staying still - otherwise the
              # non zero halite ship is likely to get orphaned
              # import pdb; pdb.set_trace()
              risk_scores = np.zeros(len(to_base_dirs))
              for risk_id, d in enumerate(to_base_dirs):
                move_row, move_col = move_ship_row_col(
                  rescuer_row, rescuer_col, d, grid_size)
                for potential_threat_dir in MOVE_DIRECTIONS:
                  threat_row, threat_col = move_ship_row_col(
                    move_row, move_col, potential_threat_dir, grid_size)
                  if opponent_ships[threat_row, threat_col] and halite_ships[
                      threat_row, threat_col] == 0:
                    opponent_id = player_ids[threat_row, threat_col]
                    is_near_base = nearest_base_distances[
                      threat_row, threat_col] <= config[
                        'log_near_base_distance']
                    distance = int(d is not None) + int(
                      potential_threat_dir is not None)
                    risk_lookup_k = str(is_near_base) + '_' + str(distance)
                    risk_scores[risk_id] = max(
                      risk_scores[risk_id], history[
                        'zero_halite_move_behavior'][opponent_id][
                          risk_lookup_k])
              best_risk_score = risk_scores.min()
              if best_risk_score < 0.15:
                print("Risky escort move with boxed in zero halite ship",
                      observation['step'], row, col)
                # import pdb; pdb.set_trace()
                rescuer_selected_move_dir = to_base_dirs[
                  np.argmin(risk_scores)]
                increase_mask = np.zeros_like(increase_mask)
                rescue_move_position = move_ship_row_col(
                  rescuer_row, rescuer_col, rescuer_selected_move_dir,
                  grid_size)
                all_ship_scores[rescuer_k][0][rescue_move_position] += 3e5
                all_ship_scores[rescuer_k][6].remove(None)
                all_ship_scores[rescuer_k][6].append(rescuer_selected_move_dir)
            
            for score_id in range(3):
              all_ship_scores[rescuer_k][score_id][increase_mask] += 1e4
            escape_row = rescuer_row
            escape_col = rescuer_col
          all_ship_scores[rescuer_k][3][:] -= 1e4 # No attack during rescue
            
          all_ship_scores[ship_k][0][escape_row, escape_col] = 1e8
          rescue_move_positions_taken[escape_row, escape_col] = 1
          
      if not abort_rescue:
        if min_escort_steps_remaining > 1:
          if not ship_k in added_escorted_ships:
            added_escorted_ships.append(ship_k)
            new_escort_list.append((ship_k, rescuer_k, False,
                                    min_escort_steps_remaining-1,
                                    max_escort_steps_remaining-1))
        elif (max_escort_steps_remaining > 1) and not (
            ship_k in added_escorted_ships):
          # can_not_return_base = (halite_ships[row, col] > 0) and (
          #   len(set(return_base_directions) & set(
          #     all_ship_scores[ship_k][6])) == 0 or (
          #       len(set(return_base_directions) & set(
          #         all_ship_scores[ship_k][7])) == num_return_directions) or (
          #           len(set(return_base_directions) & set(
          #             all_ship_scores[ship_k][8])) == num_return_directions))
          added_escorted_ships.append(ship_k)
          # can_not_return_base = (halite_ships[row, col] > 0) and (
          #   len(set(return_base_directions) & set(
          #     all_ship_scores[ship_k][6])) == 0 or (
          #       np.all(np.array(base_return_not_good_dirs))))
          
          # Decide on the next step if we should abort escorting the ship
          new_escort_list.append(
            (ship_k, rescuer_k, False, 0, max_escort_steps_remaining-1))
  history['escort_to_base_list'] = new_escort_list
  # print(observation['step'], history['escort_to_base_list'])
  
  return (all_ship_scores, on_rescue_mission, rescue_move_positions_taken,
          history)

def update_scores_victory_formation(
    all_ship_scores, config, env_config, stacked_ships, observation,
    halite_ships, steps_remaining, obs_halite, player_obs,
    main_base_distances):
  # Compute the current approximate player score
  scores = np.array(
    [rbs[0] for rbs in observation['rewards_bases_ships']])
  halite_cargos = np.array(
    [rbs[3].sum() for rbs in observation['rewards_bases_ships']])
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  my_bases = observation['rewards_bases_ships'][0][1]
  grid_size = halite_ships.shape[0]
  ship_counts = stacked_ships.sum((1, 2))
  all_ship_count = ship_counts.sum()
  obs_halite_sum = observation['halite'].sum()
  ship_value = min(env_config.spawnCost,
                   steps_remaining*obs_halite_sum**0.6/(
                     all_ship_count+1e-9))
  current_scores = scores+halite_cargos+ship_value*ship_counts
  min_advantage = current_scores[0] - current_scores[1:].max()
  
  # Obtain the TOM pattern location 
  top_row = 8
  left_col = 1
  target_pattern = np.zeros((grid_size, grid_size), dtype=np.bool)
  # T of TOM
  target_pattern[top_row, left_col:(left_col+5)] = 1
  target_pattern[top_row:(top_row+4), left_col+2] = 1
  # O of TOM
  target_pattern[top_row, (left_col+7):(left_col+9)] = 1
  target_pattern[top_row+3, (left_col+7):(left_col+9)] = 1
  target_pattern[(top_row+1):(top_row+3), left_col+6] = 1
  target_pattern[(top_row+1):(top_row+3), left_col+9] = 1
  # M of TOM
  target_pattern[top_row+3, left_col+11] = 1
  target_pattern[top_row+2, left_col+12] = 1
  target_pattern[top_row+1, left_col+13] = 1
  target_pattern[top_row+2, left_col+14] = 1
  target_pattern[top_row+1, left_col+15] = 1
  target_pattern[top_row+2, left_col+16] = 1
  target_pattern[top_row+3, left_col+17] = 1
  
  # First compute if we can afford a victory formation
  uncovered_pattern = target_pattern & ~(my_bases)
  num_uncovered_squares = uncovered_pattern.sum()
  min_required_for_formation = (
    steps_remaining*0.2+10+num_uncovered_squares)*(
      env_config.spawnCost + ship_value)
  player_ship_counts = stacked_ships.sum((1, 2))
  ship_advantage = player_ship_counts[0] - player_ship_counts[1:].max()
  
  available_ships_formation = np.copy(stacked_ships[0])
  can_form_victory_formation = min_advantage > min_required_for_formation and (
    steps_remaining < 20) and (ship_advantage > num_uncovered_squares) and (
      num_uncovered_squares > 0)
  if can_form_victory_formation:
    my_ship_pos_to_k = {v[0]: k for k, v in player_obs[2].items()}
    uncovered_positions = np.where(uncovered_pattern)
    for uncovered_id in range(num_uncovered_squares):
      pattern_row = uncovered_positions[0][uncovered_id]
      pattern_col = uncovered_positions[1][uncovered_id]
      my_pattern_distances = DISTANCES[pattern_row, pattern_col] + 100*(
        ~available_ships_formation)
      best_target_match = np.where(
        my_pattern_distances == my_pattern_distances.min())
      ship_row = best_target_match[0][0]
      ship_col = best_target_match[1][0]
      ship_k = my_ship_pos_to_k[ship_row*grid_size+ship_col]
      # import pdb; pdb.set_trace()
      all_ship_scores[ship_k][0][pattern_row, pattern_col] += 1e12
      all_ship_scores[ship_k][2][pattern_row, pattern_col] = 1e13
      available_ships_formation[ship_row, ship_col] = 0
      
    # for ship_k in all_ship_scores:
    #   row, col = row_col_from_square_grid_pos(
    #     player_obs[2][ship_k][0], grid_size)
    #   ship_halite = halite_ships[row, col]
        
    #   # Greedily assign ships to convert at each of the uncovered positions in
    #   # the target pattern
    #   for j in [0, 1, 2, 3]:
    #     all_ship_scores[ship_k][j][:] = -1e6
    #   dance_scores = 1e12*(v_pattern*(
    #     DISTANCE_MASKS[(row, col)] ** (1/np.sqrt(steps_remaining+1)))*(
    #       1.08**main_base_distances) - 0.2*(
    #       my_zero_halite_ship_density))
    #   if obs_halite[row, col] > 0:
    #     dance_scores[row, col] = -1e6
    #   best_dance_score = dance_scores.max()
    #   best_squares = np.where(dance_scores == best_dance_score)
    #   best_row = best_squares[0][0]
    #   best_col = best_squares[1][0]
    #   v_pattern[best_row, best_col] = 0
    #   all_ship_scores[ship_k][0][best_row, best_col] = best_dance_score
              
  return all_ship_scores, can_form_victory_formation

def get_ship_plans(
    config, observation, player_obs, env_config, verbose, all_ship_scores,
    np_rng, weighted_base_mask, steps_remaining,
    opponent_ships_sensible_actions, opponent_ships_scaled,
    main_base_distances, history, env_observation, player_influence_maps,
    ignore_convert_positions, ship_diff_smoothed, safe_to_return_halites,
    safe_collect_margin, always_attack_opponent_id,
    likely_convert_opponent_positions, possible_convert_opponent_positions,
    my_current_base_distances, nearest_base_distances,
    convert_first_ship_on_None_action=True, halite_on_board_mult=1e-6):
  ship_plans_start_time = time.time()
  all_my_bases = copy.copy(observation['rewards_bases_ships'][0][1])
  my_considered_bases = copy.copy(observation['rewards_bases_ships'][0][1])
  all_base_pos = np.where(all_my_bases)
  my_abandoned_base_count = len(
    history['my_base_not_attacked_positions'])
  if history['my_base_not_attacked_positions']:
    abandoned_rows, abandoned_cols = zip(*history[
      'my_base_not_attacked_positions'])
    abandoned_base_pos = (np.array(abandoned_rows), np.array(abandoned_cols))
  else:
    abandoned_base_pos = ()
  # Exclude bases that are persistently camped by opponents
  for base_pos in history['my_base_not_attacked_positions']:
    my_considered_bases[base_pos] = 0
  opponent_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  can_deposit_halite = my_considered_bases.sum() > 0
  stacked_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']])
  all_ships = stacked_ships.sum(0) > 0
  my_ships = stacked_ships[0]
  opponent_ships = stacked_ships[1:].sum(0) > 0
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
  spawn_cost = env_config.spawnCost
  num_bases = my_considered_bases.sum()
  non_abandoned_base_pos = np.where(my_considered_bases)
  new_bases = []
  base_attackers = {}
  max_attackers_per_base = config['max_attackers_per_base']
  camping_ships_strategy = history['camping_ships_strategy']
  
  my_nearest_ship_distances_raw = []
  opponent_nearest_ship_distances = [1e6*np.ones((grid_size, grid_size))]
  my_ship_pos = np.where(my_ships)
  opponent_ship_pos = np.where(opponent_ships)
  if my_ships.sum():
    for ship_id in range(my_ship_pos[0].size):
      row = my_ship_pos[0][ship_id]
      col = my_ship_pos[1][ship_id]
      my_nearest_ship_distances_raw.append(DISTANCES[row, col] + (
          halite_on_board_mult*halite_ships[row, col]))
  my_nearest_ship_distances_raw.append(1e6*np.ones((grid_size, grid_size)))
  if opponent_ships.sum():
    for ship_id in range(opponent_ship_pos[0].size):
      row = opponent_ship_pos[0][ship_id]
      col = opponent_ship_pos[1][ship_id]
      opponent_nearest_ship_distances.append(DISTANCES[row, col] + (
          halite_on_board_mult*halite_ships[row, col]))
  my_nearest_ship_distances = np.stack(my_nearest_ship_distances_raw).min(0)
  opponent_nearest_ship_distances = np.stack(
    opponent_nearest_ship_distances).min(0)
  
  # if observation['step'] == 131:
  #   import pdb; pdb.set_trace()
  
  # Update ship scores to make sure that the plan does not contradict with
  # invalid actions when the plan is executed (ship_plans_to_actions)
  for ship_k in all_ship_scores:
    # if observation['step'] == 360 and ship_k == '134-1':
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
        # Don't punish entire half plains when there is only a single bad
        # direction. > 20 hack to avoid blocking my own base early on
        if len(bad_directions) == 1 and observation['step'] > 20:
          # FUTURE WORK: should we do this only for base return? Should we make
          # this even less strict?
          if d in [NORTH, SOUTH]:
            mask_avoid &= COLUMN_MASK[col]
          elif d in [EAST, WEST]:
            mask_avoid &= ROW_MASK[row]
        if d is not None:
          mask_avoid[row, col] = False
        for i in range(3):
          all_ship_scores[ship_k][i][mask_avoid] -= 1e5
        if d is None and num_bases == 0:
          # Don't suppress conversion at the current square for the first or
          # reconstructed base
          all_ship_scores[ship_k][2][mask_avoid] += 1e5
        if d not in all_ship_scores[ship_k][10]:
          move_row, move_col = move_ship_row_col(row, col, d, grid_size)
          if opponent_bases.sum() > 0 and DISTANCES[(move_row, move_col)][
              opponent_bases].min() > 1:
            if not d in all_ship_scores[ship_k][9]:
              all_ship_scores[ship_k][3][mask_avoid] -= 1e5
            
  # if observation['step'] == 131:
  #   import pdb; pdb.set_trace()
            
  # Decide whether to build a new base after my last base has been destroyed.
  # A camped base where I do not consider attacking the campers is also
  # considered destroyed
  should_spawn_base_next_step = None
  if num_bases == 0 and my_ship_count > 1:
    requested_save_conversion_budget = 0
    all_ship_scores, can_deposit_halite, restored_base_pos = (
      consider_restoring_base(
        observation, env_config, all_ship_scores, player_obs, convert_cost,
        np_rng, history))
    num_restored_or_added_bases = int(can_deposit_halite)
    convert_unavailable_positions = np.zeros(
        (grid_size, grid_size), dtype=np.bool)
    if can_deposit_halite:
      non_abandoned_base_pos = (
        np.array([restored_base_pos[0]]), np.array([restored_base_pos[1]]))
      restored_or_added_base_pos = restored_base_pos
      my_considered_bases[restored_base_pos] = 1
  else:
    can_deposit_halite = num_bases > 0
    
    # Strategically add bases
    if num_bases > 0 and my_ship_count > 2:
      (all_ship_scores, base_added, added_base_pos,
       requested_save_conversion_budget, convert_unavailable_positions,
       should_spawn_base_next_step) = consider_adding_strategic_bases(
          config, observation, env_config, all_ship_scores, player_obs,
          convert_cost, np_rng, history, player_influence_maps, obs_halite,
          non_abandoned_base_pos, all_base_pos, halite_ships,
          my_nearest_ship_distances, my_nearest_ship_distances_raw,
          opponent_nearest_ship_distances)
      num_restored_or_added_bases = int(base_added)
      num_bases += num_restored_or_added_bases
      if base_added:
        non_abandoned_base_pos = (
          np.array(non_abandoned_base_pos[0].tolist() + [added_base_pos[0]]),
          np.array(non_abandoned_base_pos[1].tolist() + [added_base_pos[1]])
          )
        restored_or_added_base_pos = added_base_pos
    else:
      num_restored_or_added_bases = 0
      requested_save_conversion_budget = 0
      convert_unavailable_positions = np.zeros(
        (grid_size, grid_size), dtype=np.bool)
    
  # if observation['step'] == 360:
  #   import pdb; pdb.set_trace()
    
  # Decide to redirect ships to the base to avoid the main and potential other
  # strategic bases being destroyed by opposing ships
  defend_base_ignore_collision_key = None
  ignore_base_collision_ship_keys = []
  should_defend = (my_ship_count-num_restored_or_added_bases) > min(
    4, 2 + steps_remaining/5)
  remaining_defend_base_budget = max(0, min([7, int((my_ship_count-2)/2.5)]))
  base_override_move_positions = history['base_camping_override_positions']
  my_defended_abandoned_bases = np.zeros((grid_size, grid_size), dtype=np.bool)
  defend_base_ignore_collision_keys = []
  bases_protected = {}
  all_ignore_base_collision_ship_keys = []
  this_step_base_defense_keys = []
  my_defend_base_ship_positions = np.zeros(
    (grid_size, grid_size), dtype=np.bool)
  my_considered_bases_rescue_mission = np.copy(my_considered_bases)
  my_considered_bases_rescue_mission &= (
    my_nearest_ship_distances <= opponent_nearest_ship_distances)
  if num_restored_or_added_bases > 0:
    # The converted ship can not be used to defend the newly created base
    my_defend_base_ship_positions[restored_or_added_base_pos] = 1
  
  early_hunting_season = observation['relative_step'] >= config[
      'start_hunting_season_relative_step'] and observation[
        'relative_step'] <= config[
          'early_hunting_season_less_collect_relative_step'] and (
            obs_halite.sum() < config['min_halite_to_stop_early_hunt']) and (
              not history['early_hunting_season_ended'])
  late_hunting_season = observation['relative_step'] > config[
          'late_hunting_season_more_collect_relative_step']
  if (num_bases >= 1 or my_abandoned_base_count > 0) and should_defend:
    if abandoned_base_pos:
      can_defend_abandoned = my_nearest_ship_distances[abandoned_base_pos] <= (
        opponent_nearest_ship_distances[abandoned_base_pos])
      num_can_defend_abandoned = can_defend_abandoned.sum()
    else:
      num_can_defend_abandoned = 0
    defend_desirabilities = player_influence_maps[0] - player_influence_maps[
      1:].max(0)
    abandoned_defend_desirabilities = defend_desirabilities[abandoned_base_pos]
    
    # First, consider non abandoned bases - prefer bases where I have a
    # relatively high number of ships (same logic to compute the main base)
    can_defend_not_abandoned = my_nearest_ship_distances[
      non_abandoned_base_pos] <= opponent_nearest_ship_distances[
        non_abandoned_base_pos]
    non_abandoned_defend_desirabilities = ship_diff_smoothed[
      non_abandoned_base_pos] + 100*can_defend_not_abandoned
    defend_priority_ids = np.argsort(-non_abandoned_defend_desirabilities)
    base_locations_defense_budget = []
    num_non_abandoned = defend_priority_ids.size
    if num_non_abandoned > 0:
      non_abandoned_min_defenders = min(3, (
        remaining_defend_base_budget-num_can_defend_abandoned)//(
          num_non_abandoned))
      if (num_non_abandoned + num_can_defend_abandoned) == 1:
        main_base_defense_budget = 3
      else:
        if remaining_defend_base_budget <= (
            num_non_abandoned + num_can_defend_abandoned):
          main_base_defense_budget = 1
        else:
          main_base_defense_budget = 2
          
      if not early_hunting_season and observation['relative_step'] >= config[
          'start_hunting_season_relative_step']:
        # Once we trigger the condition once, the early hunting season remains
        # over
        history['early_hunting_season_ended'] = True
        
      # Defend with less passion if my bases were not destroyed before
      if history['num_destroyed_bases'] == 0:
        non_abandoned_min_defenders = 1
        main_base_defense_budget = min(
          main_base_defense_budget, 1 + int(
            num_non_abandoned == 1 or early_hunting_season))
      elif history['num_destroyed_bases'] == 1:
        non_abandoned_min_defenders = min(non_abandoned_min_defenders, 2)
        main_base_defense_budget = min(main_base_defense_budget, 2)
        
      for defend_id, defend_priority_id in enumerate(defend_priority_ids):
        base_max_defenders = main_base_defense_budget if defend_id == 0 else 1
        num_defenders = max(non_abandoned_min_defenders, min(
          remaining_defend_base_budget, base_max_defenders))
        if num_defenders > 0 and (
            defend_id == 0 or can_defend_not_abandoned[defend_priority_id]):
          remaining_defend_base_budget -= num_defenders
          base_locations_defense_budget.append((
            non_abandoned_base_pos[0][defend_priority_id],
            non_abandoned_base_pos[1][defend_priority_id],
            num_defenders))
    # First consider the non main, non abandoned bases, then the main base and
    # finally the abandoned bases
    base_locations_defense_budget = base_locations_defense_budget[::-1]
    
    if abandoned_base_pos and can_defend_abandoned.sum() and (
        remaining_defend_base_budget > 0):
      # Minimally (1 ship) defend at most N of the abandoned bases - prefer
      # squares where I have a relatively high influence
      abandoned_defense_scores = -50+100*can_defend_abandoned + (
        abandoned_defend_desirabilities)
      abandoned_defend_priority_ids = np.argsort(-abandoned_defense_scores)
      max_abandoned_defenses = 3
      for abandoned_defend_priority_id in abandoned_defend_priority_ids:
        num_defenders = min(
          remaining_defend_base_budget,
          int(can_defend_abandoned[abandoned_defend_priority_id] and (
            max_abandoned_defenses > 0)))
        if num_defenders > 0:
          abandoned_base_row = abandoned_base_pos[
            0][abandoned_defend_priority_id]
          abandoned_base_col = abandoned_base_pos[
            1][abandoned_defend_priority_id]
          my_defended_abandoned_bases[
            abandoned_base_row, abandoned_base_col] = 1
          max_abandoned_defenses -= 1
          remaining_defend_base_budget -= num_defenders
          base_locations_defense_budget.append(
            (abandoned_base_row, abandoned_base_col, num_defenders))
    
    # print(observation['step'], base_locations_defense_budget)
    # if observation['step'] == 360:
    #   import pdb; pdb.set_trace()
    
    for base_row, base_col, num_defenders in base_locations_defense_budget:
      defend_base_pos = (base_row, base_col)
      (all_ship_scores, defend_base_ignore_collision_key, base_protected,
       ignore_base_collision_ship_keys, defend_base_ship_positions_base,
       base_override_move_positions, base_defense_keys) = protect_base(
         observation, env_config, all_ship_scores, player_obs, defend_base_pos,
         history, base_override_move_positions, my_defend_base_ship_positions,
         max_considered_attackers=num_defenders)
      defend_base_ignore_collision_keys.append(
        defend_base_ignore_collision_key)
      bases_protected[defend_base_pos] = base_protected
      all_ignore_base_collision_ship_keys += ignore_base_collision_ship_keys
      this_step_base_defense_keys += base_defense_keys
      my_defend_base_ship_positions |= defend_base_ship_positions_base
  else:
    # main_base_protected = True
    my_defend_base_ship_positions = np.zeros(
      (grid_size, grid_size), dtype=np.bool)
    num_non_abandoned = 0
    
  # if observation['step'] == 360:
  #   import pdb; pdb.set_trace()
    
  # Decide on redirecting ships to friendly ships that are boxed in/chased and
  # can not return to any of my bases
  if (num_bases - num_restored_or_added_bases) > 0:
    (all_ship_scores, on_rescue_mission, rescue_move_positions_taken,
     history) = update_scores_rescue_missions(
        config, all_ship_scores, stacked_ships, observation, halite_ships,
        steps_remaining, player_obs, obs_halite, history,
        opponent_ships_sensible_actions, weighted_base_mask,
        my_considered_bases_rescue_mission, np_rng, main_base_distances,
        my_defend_base_ship_positions, safe_to_return_halites,
        player_influence_maps, nearest_base_distances)
  else:
    on_rescue_mission = np.zeros((grid_size, grid_size), dtype=np.bool)
    rescue_move_positions_taken = np.zeros(
      (grid_size, grid_size), dtype=np.bool)
  override_move_squares_taken = rescue_move_positions_taken | (
    base_override_move_positions)
    
  # if observation['step'] == 188:
  #   import pdb; pdb.set_trace()
    
  # Coordinate box in actions of opponent more halite ships
  box_start_time = time.time()
  if main_base_distances.max() > 0:
    (all_ship_scores, boxing_in_mission, box_opponent_targets,
     override_move_squares_taken,
     ships_on_box_mission) = update_scores_opponent_boxing_in(
      all_ship_scores, stacked_ships, observation, env_config,
      opponent_ships_sensible_actions, halite_ships, steps_remaining,
      player_obs, np_rng, opponent_ships_scaled, collect_rate, obs_halite,
      main_base_distances, history, on_rescue_mission,
      my_defend_base_ship_positions, env_observation, player_influence_maps,
      override_move_squares_taken, ignore_convert_positions,
      convert_unavailable_positions, always_attack_opponent_id,
      num_non_abandoned, likely_convert_opponent_positions,
      possible_convert_opponent_positions, my_current_base_distances)
  else:
    boxing_in_mission = np.zeros((grid_size, grid_size), dtype=np.bool)
    box_opponent_targets = []
    ships_on_box_mission = {}
  box_in_duration = time.time() - box_start_time
  
  # if observation['step'] == 188:
  #   import pdb; pdb.set_trace()
  
  # Coordinated pack hunting (hoard in fixed directions with zero halite ships)
  # Send all non zero halite ships to a base so we can hunt safely
  if observation['relative_step'] >= config[
      'start_hunting_season_relative_step'] and observation[
        'relative_step'] <= config['end_hunting_season_relative_step']:
    if not history['limit_ships_timeout']:
      (all_ship_scores, history,
       override_move_squares_taken) = update_scores_pack_hunt(
        all_ship_scores, config, stacked_ships, observation,
        opponent_ships_sensible_actions, halite_ships, steps_remaining,
        player_obs, np_rng, opponent_ships_scaled, collect_rate, obs_halite,
        main_base_distances, history, on_rescue_mission, boxing_in_mission,
        my_defend_base_ship_positions, env_observation, box_opponent_targets,
        override_move_squares_taken, player_influence_maps,
        ignore_convert_positions, convert_unavailable_positions,
        early_hunting_season, late_hunting_season, safe_collect_margin,
        spawn_cost)
       
  # if observation['step'] == 188:
  #   import pdb; pdb.set_trace()
       
  # Go into a victory formation when I have won
  # This can be perceived as arrogant (because it is)
  if observation['relative_step'] >= 0.85:
    all_ship_scores, victory_formation = update_scores_victory_formation(
      all_ship_scores, config, env_config, stacked_ships, observation,
      halite_ships, steps_remaining, obs_halite, player_obs,
      main_base_distances)
  else:
    victory_formation = False
       
  # Lower the collect scores for non high priority ships for the squares where
  # a high priority ships has claimed the move position.
  # High priority actions:
  # - Rescue
  # - Base defense
  # - Base attack
  # - Boxing in
  # - Opponent hoarding
  # - Victory dance
  if override_move_squares_taken.sum() > 0:
    for ship_k in all_ship_scores:
      best_collect_score = all_ship_scores[ship_k][0].max()
      if best_collect_score <= 5e4:
        # Very likely not a high priority collect override ship - lower the
        # collect scores for the claimed high priority ships to avoid conflicts
        # downstream
        # override_move_squares_taken = 1
        all_ship_scores[ship_k][0][override_move_squares_taken] *= 0.1
  
  # First, process the convert actions
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
    one_step_valid_directions = ship_scores[9]
    almost_boxed_in = not None in one_step_valid_directions and (len(
      one_step_valid_directions) == 1 or set(one_step_valid_directions) in [
        set([NORTH, SOUTH]), set([EAST, WEST])])
        
    can_return_safely = all_my_bases & (
      opponent_nearest_ship_distances >= (
        DISTANCES[row, col] + ship_halite*halite_on_board_mult))
        
    # if np.any(can_return_safely & (~(my_considered_bases | (
    #         my_defended_abandoned_bases)))):
    #   import pdb; pdb.set_trace()
    #   x=1
    
    # if observation['step'] == 129 and row == 11 and col == 8:
    #   import pdb; pdb.set_trace()
    
    # if observation['step'] == 63 and ship_k in ['5-3', '36-1']:
    #   import pdb; pdb.set_trace()
        
    if (has_budget_to_convert and (
        my_ship_count > 1 or observation['step'] < 20 or (
          steps_remaining == 1 and ship_halite >= convert_cost and (
            ship_halite + player_obs[0]) >= 2*convert_cost or ((
              (ship_halite + player_obs[0]) >= convert_cost) and (
                my_ship_count > 3)))) and (
              ship_scores[2].max()) >= max([
          ship_scores[0].max()*can_deposit_halite,
          (ship_scores[1]*(my_considered_bases | (
            my_defended_abandoned_bases) | can_return_safely)).max(),
          ship_scores[3].max(),
          ]) and (not almost_boxed_in)) or convert_surrounded_ship or (
            ship_scores[13]) or victory_formation:
      # Obtain the row and column of the new target base
      target_base = np.where(ship_scores[2] == ship_scores[2].max())
      target_row = target_base[0][0]
      target_col = target_base[1][0]
      my_non_converted_ship_count -= 1
      
      if (target_row == row and target_col == col) or convert_surrounded_ship:
        ship_plans[ship_k] = CONVERT
        new_bases.append((row, col))
        my_considered_bases[row, col] = True
        can_deposit_halite = True
      else:
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False,
                              row, col, -1)
        
  # if observation['step'] == 322:
  #   import pdb; pdb.set_trace()
  #   x=1
        
  # Next, do another pass to coordinate the target squares. This is done in a
  # double pass for now where the selection order is determined based on the 
  # availability of > 1 direction in combination with the initial best score.
  # The priorities are recomputed as target squares are taken by higher
  # priority ships.
  my_prev_step_base_attacker_ships = history[
    'my_prev_step_base_attacker_ships']
  best_ship_scores = {}
  ship_priority_scores = np.zeros(my_ship_count)
  ship_priority_matrix = np.zeros((my_ship_count, 8))
  all_ship_valid_directions = {}
  for i, ship_k in enumerate(player_obs[2]):
    if ship_k in ship_plans:
      # Make sure that already planned ships stay on top of the priority Q
      ship_priority_scores[i] = 1e20
    else:
      ship_scores = all_ship_scores[ship_k]
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      
      # if observation['step'] == 131 and ship_k in ['74-1']:
      #   import pdb; pdb.set_trace()
      
      # Incorporate new bases in the return to base scores
      ship_scores = list(ship_scores)
      ship_halite = player_obs[2][ship_k][1]
      can_return_safely = all_my_bases & (
        opponent_nearest_ship_distances >= (
          DISTANCES[row, col] + ship_halite*halite_on_board_mult))
      ship_scores[1][np.logical_not(
        my_considered_bases | my_defended_abandoned_bases | (
          can_return_safely))] = -1e7
      ship_scores[3][np.logical_not(opponent_bases)] = -1e7
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
      
      prev_base_attacker = int(ship_k in my_prev_step_base_attacker_ships)
      
      all_ship_valid_directions[ship_k] = copy.copy(ship_scores[6])
      ship_priority_scores[i] = best_score + 1e12*(
          (len(ship_scores[6]) == 1)) - 1e6*(
            num_non_immediate_bad_directions) + 1e4*(
              len(ship_scores[8])) + 1e2*(
              num_two_step_neighbors) - 1e5*(
                len(ship_scores[9])) + 1e7*(
                ship_scores[11]) + 3e5*prev_base_attacker
      ship_priority_matrix[i] = np.array([
        len(ship_scores[6]) == 1,  # 1e12
        ship_scores[11],  # 1e7
        num_non_immediate_bad_directions,  # 1e6
        prev_base_attacker,  # 3e5
        len(ship_scores[9]),  #1e5
        len(ship_scores[8]),  #1e4
        num_two_step_neighbors,  #1e2
        best_score,  #no unit
        ])
         
  ship_order = np.argsort(-ship_priority_scores)
  occupied_target_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  occupied_squares_count = np.zeros((grid_size, grid_size), dtype=np.int)
  single_path_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  single_path_max_block_distances = np.ones(
    (grid_size, grid_size), dtype=np.int)
  return_base_distances = []
  chain_conflict_resolution = []
  ship_conflict_resolution = []
  
  # if observation['step'] == 279:
  #   # ship_positions = [
  #   #   row_col_from_square_grid_pos(
  #   #     player_obs[2][ship_ids[o]][0], grid_size) for o in (ship_order)]
  #   # print([ship_ids[o] for o in ship_order])
  #   import pdb; pdb.set_trace()
  
  # List the ships that want to collect and greedily assign conflicted collect
  # squares to the ship with the highest collect score
  all_collect_scores = []
  collect_ship_keys = []
  all_collect_multipliers = []
  taken_collect_squares = np.zeros((grid_size, grid_size), dtype=np.bool)
  for i in range(my_ship_count):
    ship_k = ship_ids[ship_order[i]]
    ship_scores = all_ship_scores[ship_k]
    collect_scores = np.copy(ship_scores[0])
    best_collect_score = collect_scores.max()
    best_return_score = ship_scores[1].max()
    best_establish_score = ship_scores[2].max()
    best_attack_base_score = ship_scores[3].max()
    best_other_score = max([
      best_return_score, best_establish_score, best_attack_base_score])
    best_score = max(best_collect_score, best_other_score)
    
    if best_collect_score < 0:
      # Set the collect scores to a small value on all squares in the valid
      # directions
      row, col = row_col_from_square_grid_pos(
          player_obs[2][ship_k][0], grid_size)
      valid_mask = np.zeros((grid_size, grid_size), dtype=np.bool)
      if len(ship_scores[6]) == 0:
        # Add all squares to the valid mask - everything seems lost anyway
        valid_mask |= np.ones_like(valid_mask)
      else:
        # Add a small positive value for the escape directions - this way
        # the planning is undisturbed and the other ships remain unaffected of
        # the selected escape path
        for d in ship_scores[6]:
          valid_mask |= HALF_PLANES_CATCH[row, col][d]
        
      # Boxed in by zero halite ships on a non zero halite square
      collect_scores[valid_mask] = 1e-10*np.random.uniform(
        size=valid_mask.sum())
    
    if best_score == best_collect_score:
      if best_collect_score < 1e4:
        collect_ship_keys.append(ship_k)
        all_collect_scores.append(collect_scores)
        all_collect_multipliers.append(ship_scores[16])
      else:
        best_collect_pos = np.unravel_index(
          collect_scores.argmax(), collect_scores.shape)
        taken_collect_squares[best_collect_pos] = 1
      
  num_collect_ships = len(collect_ship_keys)
  if num_collect_ships > 0:
    stacked_collect_scores = np.stack(all_collect_scores)
    updated_collect_scores = np.copy(stacked_collect_scores)
    # original_collect_scores = np.copy(stacked_collect_scores)
    stacked_collect_scores /= np.stack(all_collect_multipliers)
    min_collect_score = stacked_collect_scores.min()
    if np.any(taken_collect_squares):
      stacked_collect_scores[:, taken_collect_squares] = min_collect_score
    
    for loop_id in range(num_collect_ships):
      best_index = np.unravel_index(stacked_collect_scores.argmax(),
                                    stacked_collect_scores.shape)
      # print(loop_id, best_index)
      # Reduce the best square for all other ships by an order of magnitude
      other_ships = np.arange(num_collect_ships) != best_index[0]
      updated_collect_scores[
        other_ships, best_index[1], best_index[2]] /= 10
      stacked_collect_scores [
        other_ships, best_index[1], best_index[2]] = min_collect_score
      
      # Exclude the assigned ship for subsequent steps
      stacked_collect_scores[best_index[0]] = min_collect_score
        
    for ship_id, ship_k in enumerate(collect_ship_keys):
      all_ship_scores[ship_k][0][:] = updated_collect_scores[ship_id]
      
  # List the ships that want to attack bases and greedily assign
  # base squares to the ships with the highest attack scores
  if max_attackers_per_base > 0:
    all_attack_details = {}
    for i in range(my_ship_count):
      ship_k = ship_ids[ship_order[i]]
      ship_scores = all_ship_scores[ship_k]
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      attack_base_scores = ship_scores[3]
      best_collect_score = ship_scores[0].max()
      best_return_score = ship_scores[1].max()
      best_establish_score = ship_scores[2].max()
      best_attack_base_score = attack_base_scores.max()
      best_other_score = max([
        best_collect_score, best_return_score, best_establish_score])
      best_score = max(best_attack_base_score, best_other_score)
      
      if best_score == best_attack_base_score:
        best_collect_pos = np.unravel_index(
          attack_base_scores.argmax(), attack_base_scores.shape)
        attack_ship_details = (best_score, ship_k, row, col)
        if best_collect_pos in all_attack_details:
          all_attack_details[best_collect_pos].append(
            attack_ship_details)
        else:
          all_attack_details[best_collect_pos] = [attack_ship_details]
          
    for base_pos in all_attack_details:
      num_attackers_this_base = len(all_attack_details[base_pos])
      if num_attackers_this_base > max_attackers_per_base:
        attack_ship_scores = np.array([d[0] for d in (
          all_attack_details[base_pos])])
        keep_attack_ids = np.argsort(-attack_ship_scores)[
          :max_attackers_per_base].tolist()
        for attacker_id in range(num_attackers_this_base):
          ship_k = all_attack_details[base_pos][attacker_id][1]
          if not attacker_id in keep_attack_ids:
            all_ship_scores[ship_k][3][base_pos] = -1e12
      
  # if observation['step'] == 75:
  #   import pdb; pdb.set_trace()
  #   x=1
      
  inner_loop_start_time = time.time()
  recompute_ship_plan_order_duration = 0
  base_return_ship_keys = []
  for i in range(my_ship_count):
    ship_k = ship_ids[ship_order[i]]
    ship_scores = all_ship_scores[ship_k]
    ship_halite = player_obs[2][ship_k][1]
    if not ship_k in ship_plans:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      valid_directions = ship_scores[6]
      
      # if observation['step'] == 118 and ship_k == '50-1':
      #   import pdb; pdb.set_trace()
      
      after_blocked_valid_dirs = copy.copy(ship_scores[6])
      if single_path_squares.sum() and not ship_scores[12]:
        (s0, s1, s2, s3, after_blocked_valid_dirs, _,
         _) = update_scores_blockers(
           ship_scores[0], ship_scores[1], ship_scores[2], ship_scores[3],
           row, col, grid_size, single_path_squares,
           single_path_max_block_distances, valid_directions,
           ship_scores[9], update_attack_base=True)
        ship_scores = (s0, s1, s2, s3, ship_scores[4], ship_scores[5],
                       ship_scores[6], ship_scores[7], ship_scores[8],
                       ship_scores[9], ship_scores[10], ship_scores[11],
                       ship_scores[12], ship_scores[13], ship_scores[14],
                       ship_scores[15], ship_scores[16], ship_scores[17],
                       ship_scores[18])
          
      if ship_halite == 0 and (len(after_blocked_valid_dirs) == 0 or (len(
          after_blocked_valid_dirs) == 1 and (
          valid_directions[0] is None))) and obs_halite[row, col] > 0:
        # There are no longer any valid directions available for my zero halite
        # ship due to my other ships taking the final escape squares.
        # Typically this means being surrounded by opponent zero halite ships
        # In this situation we have to prefer risky actions over staying still!
        for d in np_rng.permutation(NOT_NONE_DIRECTIONS):
          move_row, move_col = move_ship_row_col(
                row, col, d, grid_size)
          if not single_path_squares[move_row, move_col]:
            ship_scores[0][move_row, move_col] = 5e4
            ship_scores[6].append(d)
            break
      
      best_collect_score = ship_scores[0].max()
      best_return_score = ship_scores[1].max()
      best_establish_score = ship_scores[2].max()
      best_attack_base_score = ship_scores[3].max()
      best_score = max([best_collect_score, best_return_score,
                        best_establish_score, best_attack_base_score])
      
      # if observation['step'] == 247 and ship_k == '52-2':
      #   import pdb; pdb.set_trace()
      
      if best_collect_score == best_score:
        # 1) Gather mode
        target_gather = np.where(ship_scores[0] == ship_scores[0].max())
        target_row = target_gather[0][0]
        target_col = target_gather[1][0]
        
        if target_row == row and target_col == col and my_ship_count == 1 and (
            num_bases == 0 and (
              ship_halite+player_obs[0]) >= 2*convert_cost) and (
                convert_first_ship_on_None_action):
          ship_plans[ship_k] = CONVERT
          my_considered_bases[row, col] = True
          update_occupied_count(
            row, col, occupied_target_squares, occupied_squares_count)
        else:
          # Join the base attack in some base camping phases
          consider_base_attack = False
          if ship_k in camping_ships_strategy:
            base_location = camping_ships_strategy[ship_k][5]
            consider_base_attack = camping_ships_strategy[ship_k][4]
            if consider_base_attack:
              base_distance = DISTANCES[base_location][row, col]
              attack_tuple = (base_distance, ship_halite, ship_k, row, col,
                              True)
              if base_location in base_attackers:
                base_attackers[base_location].append(attack_tuple)
              else:
                base_attackers[base_location] = [attack_tuple]
          
          ship_plans[ship_k] = (target_row, target_col, ship_scores[4],
                                consider_base_attack, row, col, best_score)
          
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
          
      elif best_return_score == best_score:
        # 2) Return base mode
        target_return = np.where(ship_scores[1] == ship_scores[1].max())
        target_row = target_return[0][0]
        target_col = target_return[1][0]
        
        # Element 4 is whether we can ignore collisions when moving onto a base
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4],
                              ship_k in defend_base_ignore_collision_keys and (
                                not bases_protected.get(
                                  (target_row, target_col), True)),
                              row, col, best_score)
        base_distance = DISTANCES[target_row, target_col][row, col]
        base_return_ship_keys.append(ship_k)
        
        # Mark the ship as returning to a base
        if not ship_k in history['returning_to_base_ships']:
          history['returning_to_base_ships'].append(ship_k)
        
        if not bases_protected.get((target_row, target_col), True):
          bases_protected[target_row, target_col] = base_distance==0
        if not ship_k in all_ignore_base_collision_ship_keys:
          return_base_distances.append((target_row, target_col, base_distance))
      elif best_establish_score == best_score:
        # 3) Establish base mode
        target_base = np.where(ship_scores[2] == ship_scores[2].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], False,
                              row, col, best_score)
        update_occupied_count(
            target_row, target_col, occupied_target_squares,
            occupied_squares_count)
      else:
        # 4) Attack base mode
        # print("Attack!", observation['step'], row, col)
        target_base = np.where(ship_scores[3] == ship_scores[3].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        base_distance = DISTANCES[(row, col)][target_row, target_col]
        attack_tuple = (base_distance, ship_halite, ship_k, row, col, False)
        if (target_row, target_col) in base_attackers:
          base_attackers[(target_row, target_col)].append(attack_tuple)
        else:
          base_attackers[(target_row, target_col)] = [attack_tuple]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[4], True,
                              row, col, best_score)
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
    # not to be a performance bottleneck.
    # Just to be sure: don't do this when I have a risk of timing out
    if not history['limit_ships_timeout']:
      reorder_start_time = time.time()
      for j in range(i+1, my_ship_count):
        order_id = ship_order[j]
        ship_k_future = ship_ids[order_id]
        if not ship_k_future in ship_plans and not all_ship_scores[
            ship_k_future][12]:
          future_ship_scores = all_ship_scores[ship_k_future]
          future_row, future_col = row_col_from_square_grid_pos(
            player_obs[2][ship_k_future][0], grid_size)
          
          # if observation['step'] == 122 and ship_k_future == '102-1':
          #   import pdb; pdb.set_trace()
          
          future_ship_scores[0][occupied_target_squares] = -1e7
          future_ship_scores[2][occupied_target_squares] = -1e7
          future_ship_scores[3][
            occupied_squares_count >= max_attackers_per_base] = -1e7
          
          # if np.any(future_ship_scores_c[0] != future_ship_scores[0]) or np.any(
          #     future_ship_scores_c[2] != future_ship_scores[2]) or np.any(
          #       future_ship_scores_c[3] != future_ship_scores[3]):
          #   import pdb; pdb.set_trace()
    
          for (r, c, d) in return_base_distances:
            # This coordinates return to base actions and avoids base blocking
            if DISTANCES[r, c][future_row, future_col] == d:
              future_ship_scores[1][r, c] = -1e7
          
          updated_best_score = max([
            future_ship_scores[0].max(), future_ship_scores[1].max(),
            future_ship_scores[2].max(), future_ship_scores[3].max()])
          
          # Verify if the number of valid directions changed from > 1 to 1 or
          # from 1 to 0 and change the priority accordingly.
          considered_valid_directions = all_ship_valid_directions[ship_k_future]
          valid_considered_valid_directions = []
          for d in considered_valid_directions:
            move_future_row, move_future_col = move_ship_row_col(
              future_row, future_col, d, grid_size)
            if not single_path_squares[move_future_row, move_future_col]:
              valid_considered_valid_directions.append(d)
          all_ship_valid_directions[ship_k_future] = (
            valid_considered_valid_directions)
          
          # if (int(len(
          #     valid_considered_valid_directions) == 1) - int(len(
          #     considered_valid_directions) == 1)) != 0 and observation[
          #       'step'] == 142:
          #   import pdb; pdb.set_trace()
          
          # Update the priority for future ships using the updated ship scores
          # and the updated single valid direction counts
          priority_change = updated_best_score - (
            best_ship_scores[ship_k_future]) + 1e12*(int(len(
              valid_considered_valid_directions) == 1) - int(len(
              considered_valid_directions) == 1))
          # assert priority_change <= 0  # Only relevant for best score changes
          ship_priority_scores[order_id] += priority_change
          best_ship_scores[ship_k_future] = updated_best_score
          
          all_ship_scores[ship_k_future] = future_ship_scores
      
      # if observation['step'] == 142:
      #   import pdb; pdb.set_trace()
      
      # Update the ship order - this works since priorities can only be lowered
      # and we only consider future ships when downgrading priorities
      # Make sure no ships get skipped by the +X hack
      ship_priority_scores[ship_order[:(i+1)]] += 1e30 # Max float 32: 3e38
      ship_order = np.argsort(-ship_priority_scores)
      recompute_ship_plan_order_duration += time.time() - reorder_start_time
    
  # Override of the ship plan: if I have a ship that is returning to a base
  # after the first N steps, and it can safely collect on an interesting
  # halite square: collect rather than return. 
  if observation['relative_step'] > config[
      'collect_on_safe_return_relative_step']:
    for ship_k in base_return_ship_keys:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      base_row = ship_plans[ship_k][0]
      base_col = ship_plans[ship_k][1]
      safe_collect_ship_margin = safe_collect_margin[row, col]
      return_score = ship_plans[ship_k][6]
      this_square_collect_score = all_ship_scores[ship_k][0][row, col]
      distance_to_base = DISTANCES[row, col][base_row, base_col]
      
      # if observation['step'] == 393 and row == 11 and col == 7:
      #   import pdb; pdb.set_trace()
      
      if safe_collect_ship_margin > 0 and obs_halite[row, col] > 100 and (
          not occupied_target_squares[row, col]) and (
            (distance_to_base == (steps_remaining-1)) or (
              return_score < 1e6 and (
                distance_to_base > 5-3*observation['relative_step']))) and (
                  this_square_collect_score < 1e5) and (
                    steps_remaining >= distance_to_base):
        # print("Forced gather", observation['step'], row, col)
        ship_plans[ship_k] = (
          row, col, [], False, row, col, this_square_collect_score)
    
  # Drop the camping ships from the base attackers if there are no non-camper
  # base attacker for the targeted base
  del_keys = []
  for target_base in base_attackers:
    attackers = base_attackers[target_base]
    has_non_camp_attacker = False
    for i in range(len(attackers)):
      if not attackers[i][5]:
        has_non_camp_attacker = True
        break
      
    if not has_non_camp_attacker:
      del_keys.append(target_base)
  for del_key in del_keys:
    del base_attackers[del_key]
    
  my_prev_step_base_attacker_ships = []
  for target_base in base_attackers:
    base_attacker_keys = [attacker[2] for attacker in base_attackers[
      target_base]]
    my_prev_step_base_attacker_ships.extend(base_attacker_keys)
  history['my_prev_step_base_attacker_ships'] = (
    my_prev_step_base_attacker_ships)
  # if observation['step'] == 247:
  #   import pdb; pdb.set_trace()
    
  # Reorder the ship plans by current location for debugging purposed
  ship_plans_reordered = []
  all_reordered_vals = []
  order_scores = []
  for k in ship_plans:
    vals = ship_plans[k]
    if isinstance(vals, str):
      reordered_vals = vals
    else:
      reordered_vals = (vals[4], vals[5], k, vals[0], vals[1], vals[2],
                        vals[3], int(vals[6]))
    all_reordered_vals.append(reordered_vals)
    order_scores.append(vals[4]*grid_size+vals[5])
  for ordered_id in np.argsort(np.array(order_scores)):
    ship_plans_reordered.append(all_reordered_vals[ordered_id])
  
  ship_plans_duration = time.time() - ship_plans_start_time
  inner_loop_ship_plans_duration = time.time() - inner_loop_start_time
  return (ship_plans, my_considered_bases, all_ship_scores, base_attackers,
          box_in_duration, history, ship_plans_duration,
          inner_loop_ship_plans_duration,
          recompute_ship_plan_order_duration, on_rescue_mission,
          ships_on_box_mission, requested_save_conversion_budget,
          non_abandoned_base_pos, this_step_base_defense_keys,
          should_spawn_base_next_step, ship_plans_reordered,
          victory_formation)

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
    all_ship_scores, before_plan_ship_scores, ship_plans, np_rng,
    ignore_bad_attack_directions, base_attackers, steps_remaining,
    opponent_ships_sensible_actions, opponent_ships_sensible_actions_no_risk,
    history, env_obs_ids, opponent_ships_scaled, main_base_distances,
    ignore_convert_positions, ballistic_attack_base_targets,
    player_influence_maps):
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
  my_bases = stacked_bases[0]
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
    
  # When boxed in: avoid squares where lower halite opponents have their only
  # escape square since they will most likely move there
  avoid_squares_boxed_in = np.zeros_like(stacked_ships[0])
  for (row, col) in opponent_ships_sensible_actions:
    escape_directions = opponent_ships_sensible_actions[(row, col)]
    if len(escape_directions) == 1:
      escape_dir = RELATIVE_DIR_TO_DIRECTION_MAPPING[escape_directions[0]]
      move_row, move_col = move_ship_row_col(row, col, escape_dir, grid_size)
      avoid_squares_boxed_in[move_row, move_col] = 1
  
  # For debugging - the order in which actions are planned
  ordered_debug_ship_plans = [[k]+list(v) for k, v in ship_plans.items()]
  ordered_debug_ship_plans = ordered_debug_ship_plans
  
  # if observation['step'] == 394:
  #   import pdb; pdb.set_trace()
  
  base_attack_override_wait = []
  for target_base in base_attackers:
    attackers = base_attackers[target_base]
    # Only consider attackers that can make it onto the base in time
    attackers = [a for a in attackers if a[0] <= steps_remaining]
    
    num_attackers = len(attackers)
    attack_distances = np.array([a[0] for a in attackers])
    if num_attackers > 1 and steps_remaining > 2*attack_distances.max():
      # If the base can not be defended: don't bother synchronizing the attack
      # if observation['step'] == 349:
      #   import pdb; pdb.set_trace()
      #   x=1
        
      can_defend = base_can_be_defended(
        base_attackers, target_base[0], target_base[1], stacked_bases,
        stacked_ships, halite_ships)
      
      # Synchronize the attackers
      try_to_wait_strategy = False
      if can_defend:
        argsort_distances = np.argsort(attack_distances)
        sorted_distances = attack_distances[argsort_distances]
        
        if not np.all(np.diff(sorted_distances) == 1):
          # Pause the attackers that are the closest to the target on squares
          # with no halite
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
              
          # if observation['step'] == 225:
          #   import pdb; pdb.set_trace()
              
          # Verify that all ships that should wait can wait
          can_all_wait = True
          should_wait_ids = np.where(should_wait)[0]
          for should_wait_id in should_wait_ids:
            row = attackers[should_wait_id][3]
            col = attackers[should_wait_id][4]
            # Don't wait at one of my bases or a non-zero halite square
            if obs_halite[row, col] > 0 or my_bases[row, col]:
              can_all_wait = False
              break
          if can_all_wait:
            for should_wait_id in should_wait_ids:
              ship_k = attackers[should_wait_id][2]
              row = attackers[should_wait_id][3]
              col = attackers[should_wait_id][4]
              ship_plans[ship_k] = (row, col, [], False, row, col, -2)
              if not None in all_ship_scores[ship_k][6]:
                # print("PERFORMING A RISKY BASE ATTACK OVERRIDE SHIP WAIT",
                #       observation['step'], row, col)
                all_ship_scores[ship_k][6].append(None)
          else:
            # Consider waiting with one of the two closest attackers to make
            # the two closest attackers aligned
            try_to_wait_strategy = True
      else:
        try_to_wait_strategy = True
        
      if try_to_wait_strategy:
        # Maybe override the attack actions - wait on zero halite squares so
        # we can take the base down in one of the subsequent steps
        attack_distances = np.array([a[0] for a in attackers])
        min_distance = attack_distances.min()
        consider_wait_ids = None
        if (attack_distances == min_distance).sum() > 1:
          consider_wait_ids = np.where(attack_distances == min_distance)[0]
        elif (attack_distances == (min_distance+1)).sum() > 1:
          consider_wait_ids = np.where(attack_distances == (min_distance+1))[0]
        if consider_wait_ids is not None:
          for wait_id in range(consider_wait_ids.size):
            consider_wait_id = consider_wait_ids[wait_id]
            considered_ship_k = attackers[consider_wait_id][2]
            considered_row = attackers[consider_wait_id][3]
            considered_col = attackers[consider_wait_id][4]
            if obs_halite[considered_row, considered_col] == 0:
              base_attack_override_wait.append(considered_ship_k)
              break
  
  # List all positions you definitely don't want to move to. Initially this
  # only contains opponent bases and eventually also earlier ships.
  opponent_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  bad_positions = np.copy(opponent_bases)
  
  # Add current likely converted bases to the bad positions
  for (ignore_convert_row, ignore_convert_col) in ignore_convert_positions:
    bad_positions[ignore_convert_row, ignore_convert_col] = 1
  
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
       _, _, _) = ship_plans[ship_k]
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                             grid_size)
      
      # Filter out bad positions from the shortest actions
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        if not bad_positions[move_row, move_col] or (
            ignore_base_collision and (
              move_row == target_row and move_col == target_col)) or (
                not opponent_bases[move_row, move_col] and all_ship_scores[
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
    #   all_ship_scores[ship_k][6] + all_ship_scores[ship_k][8]))
    
    # Just keep the order from the planning - this is cleaner and works better!
    ship_priority_scores[i] = -i
    
    # ship_priority_scores[i] = -1e6*num_non_immediate_bad_directions -1e3*len(
    #   valid_actions) + 1e4*len(all_ship_scores[ship_k][8]) - 1e5*len(
    #     before_plan_ship_scores[ship_k][9]) - i + 1e7*(
    #       all_ship_scores[ship_k][11])
  
  ship_order = np.argsort(-ship_priority_scores)
  ordered_ship_plans = [ship_key_plans[o] for o in ship_order]
  
  # Keep track of all my ship positions and rearrange the action planning when
  # one of my ships only has one remaining option that does not self destruct.
  ship_non_self_destructive_actions = {}
  for ship_k in ordered_ship_plans:
    ship_non_self_destructive_actions[ship_k] = copy.copy(MOVE_DIRECTIONS)
  
  num_ships = len(ordered_ship_plans)
  action_overrides = np.zeros((7))
  camping_ships_strategy = history['camping_ships_strategy']
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
              all_ship_scores[ship_k][2][row, col] < 1e6):
        # Override the convert logic - it's better to lose some ships than to
        # convert too often (good candidate for stateful logic)
        # We enter this path when the only remaining base is reconstructed
        # (rare!)
        if not all_ship_scores[ship_k][6]:
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
              move_row, move_col] and all_ship_scores[ship_k][12]):
            path_lookup_k = (move_row, move_col)
            if not path_lookup_k in shortest_path_count:
              shortest_path_count[path_lookup_k] = 0
        ship_plans[ship_k] = (target_row, target_col, [], False, row, col, -3)
      else:
        ship_actions[ship_k] = ship_plans[ship_k]
        obs_halite[row, col] = 0
        remaining_budget -= convert_cost
        has_selected_action = True
        del ship_non_self_destructive_actions[ship_k]
        
    # if observation['step'] == 78 and ship_k in ['30-2', '35-2']:
    #   import pdb; pdb.set_trace()
        
    if not has_selected_action:
      (target_row, target_col, preferred_directions, ignore_base_collision,
       _, _, _) = ship_plans[ship_k]
      
      # Override the target row and column if this ship is an aggressive base
      # camper and the base can not be defended
      if ship_k in camping_ships_strategy:
        base_location = camping_ships_strategy[ship_k][5]
        consider_base_attack = camping_ships_strategy[ship_k][4]
        if consider_base_attack and (
            base_location[0], base_location[1]) in base_attackers:
          base_distance = DISTANCES[base_location][row, col]
          
          can_defend = base_can_be_defended(
            base_attackers, base_location[0], base_location[1], stacked_bases,
            stacked_ships, halite_ships)
          
          if base_distance == 1 and not can_defend:
            target_row = base_location[0]
            target_col = base_location[1]
            shortest_path_count[base_location] = 1.0
          
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                             grid_size)
      
      if ship_k in base_attack_override_wait:
        shortest_actions = [None]
        path_lookup_k = (row, col)
        if not path_lookup_k in shortest_path_count:
          shortest_path_count[path_lookup_k] = 1.0
      
      ignore_bad_attack_directions_this_ship = ignore_bad_attack_directions
      if ignore_base_collision and not ignore_bad_attack_directions and (
          (target_row, target_col) in base_attackers):
        
        can_defend = base_can_be_defended(
          base_attackers, target_row, target_col, stacked_bases, stacked_ships,
          halite_ships)
          
        if can_defend:
          ignore_base_collision = False
        else:
          ignore_bad_attack_directions_this_ship = True
          
      # You shall have no fear in ballistic mode
      if (target_row, target_col) in ballistic_attack_base_targets:
        ignore_bad_attack_directions_this_ship = True
        ignore_base_collision = True
        
      # if observation['step'] == 106 and row == 20 and col == 7:
      #   import pdb; pdb.set_trace()
        
      # Remove own ships from the shortest action bad positions when the ships
      # are returning to the base at the end of the game
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        if bad_positions[move_row, move_col] and not opponent_bases[
            move_row, move_col] and all_ship_scores[ship_k][12]:
          bad_positions[move_row, move_col] = False
        
      # Filter out bad positions from the shortest actions
      valid_actions = []
      valid_move_positions = []
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(row, col, a, grid_size)
        if (not bad_positions[move_row, move_col] and (
            a in all_ship_scores[ship_k][6])) or (ignore_base_collision and ((
              move_row == target_row and move_col == target_col) or (
                ignore_bad_attack_directions_this_ship and not bad_positions[
                  move_row, move_col])) and not my_next_ships[
                    move_row, move_col]) or (
                      all_ship_scores[ship_k][12] and steps_remaining == 1):
          valid_actions.append(a)
          valid_move_positions.append((move_row, move_col))
          path_lookup_k = (move_row, move_col)
          if not path_lookup_k in shortest_path_count:
            # import pdb; pdb.set_trace()
            print("Path key lookup fail step", observation['step'], row, col,
                  ship_k, i, move_row, move_col)
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
        
        # Don't take an action that blocks my path to the target with an
        # opponent base
        if len(valid_actions) > 1:
          row_distance = DISTANCES[row, 0][target_row, 0]
          col_distance = DISTANCES[0, col][0, target_col]
          single_path_square = None
          if row_distance == 1 and col_distance > 1:
            single_path_square = (target_row, col)
            single_path_dir = NORTH if NORTH in valid_actions else SOUTH
          elif col_distance == 1 and row_distance > 1:
            single_path_square = (row, target_col)
            single_path_dir = EAST if EAST in valid_actions else WEST
            
          if single_path_square is not None:
            should_avoid = np.any(opponent_bases[get_mask_between_exclude_ends(
              single_path_square[0], single_path_square[1], target_row,
              target_col, grid_size)])
            if should_avoid and single_path_dir in valid_actions:
              valid_actions.remove(single_path_dir)
          
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
            
        # Move onto one of my friendly bases if I have halite on board and it
        # is one of the available options
        if len(valid_actions) > 1 and halite_ships[row, col] > 0:
          move_onto_base_action = None
          for a in valid_actions:
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            if my_bases[move_row, move_col]:
              move_onto_base_action = a
              break
          if move_onto_base_action is not None:
            valid_actions = [move_onto_base_action]
            
        # When all actions are equally valid, move to a square with lower non
        # zero halite ship density in order to avoid double box-ins.
        # Only do this for non zero halite ships
        if len(valid_actions) > 1 and halite_ships[row, col] > 0:
          considered_densities = np.zeros(len(valid_actions))
          for a_id, a in enumerate(valid_actions):
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            considered_densities[a_id] = my_non_zero_halite_ship_density[
              move_row, move_col]
          min_density = considered_densities.min()
          valid_actions = [a for (a_id, a) in enumerate(valid_actions) if (
            considered_densities[a_id] == min_density)]
            
        if len(valid_actions) > 1 and halite_ships[row, col] > 0:
          # When all actions are equally valid: prefer a square where I have
          # more influence
          considered_influences = np.zeros(len(valid_actions))
          for a_id, a in enumerate(valid_actions):
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            considered_influences[a_id] = player_influence_maps[
              0, move_row, move_col] - player_influence_maps[
                1:, move_row, move_col].sum()
          max_influence = considered_influences.max()
          valid_actions = [a for (a_id, a) in enumerate(valid_actions) if (
            considered_influences[a_id] == max_influence)]
          
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
          if a in all_ship_scores[ship_k][6]:
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
         
          if len(self_escape_actions) == 0:
            # Consider non risky opponent actions to plan my escape
            self_escape_actions = get_opponent_blocked_escape_dir(
              bad_positions, opponent_ships_sensible_actions_no_risk, row, col,
              np_rng, grid_size, observation, ship_k)
          
          if self_escape_actions:
            if halite_ships[row, col] == 0 and len(
                self_escape_actions) > 1 and None in self_escape_actions and (
                  obs_halite[row, col] > 0) and (
                    main_base_distances[row, col] < 0 or main_base_distances[
                      row, col] > 2):
              # Filter out the stay still action for zero halite ships on a non
              # zero halite square if that leaves us with options
              self_escape_actions.remove(None)
            
            if before_plan_ship_scores[ship_k][9]:
              # Filter out 1-step bad actions if that leaves us with options
              self_escape_actions_not_1_step_bad = list(
                set(self_escape_actions) & set(
                  before_plan_ship_scores[ship_k][9]))
              if self_escape_actions_not_1_step_bad:
                self_escape_actions = self_escape_actions_not_1_step_bad
            
            if all_ship_scores[ship_k][7]:
              # Filter out 2-step bad actions if that leaves us with options
              self_escape_actions_not_2_step_bad = list(
                set(self_escape_actions) - set(all_ship_scores[ship_k][7]))
              if self_escape_actions_not_2_step_bad:
                self_escape_actions = self_escape_actions_not_2_step_bad
                
            # Filter out n-step bad actions if that leaves us with options
            if all_ship_scores[ship_k][8]:
              self_escape_actions_not_n_step_bad = list(
                set(self_escape_actions) - set(all_ship_scores[ship_k][8]))
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
            if len(all_ship_scores[ship_k][8]) > 1 and len(
                self_escape_actions) > 1:
              if np.all(
                  [a in all_ship_scores[ship_k][8] for a in (
                    self_escape_actions)]):
                missing_keys = [a for a in (self_escape_actions) if not a in (
                  all_ship_scores[ship_k][14])]
                if len(missing_keys) > 0:
                  print("MISSING N-STEP BAD KEYS:", observation['step'],
                        row, col, missing_keys)
                die_probs = np.array([all_ship_scores[ship_k][14].get(
                  a, 0) for a in self_escape_actions])
                self_escape_actions = [
                  self_escape_actions[np.argmin(die_probs)]]
               
            valid_not_bad_actions = self_escape_actions
              
        # There is no valid direction available; consider n step bad actions
        # This check is most probably obsolete since it would appear in
        # self_escape_actions
        if not valid_not_bad_actions:
          action_overrides[2] += 1
          for a in np_rng.permutation(MOVE_DIRECTIONS):
            if a in all_ship_scores[ship_k][8]:
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
            if a in all_ship_scores[ship_k][7]:
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
                opponent_ships_sensible_actions_no_risk[
                  chaser_row, chaser_col])
              
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
                      attack_base_bonus) - 0.5*all_ship_scores[ship_k][15][
                        move_row, move_col] - 0.5*(
                          my_non_zero_halite_ship_density[
                            move_row, move_col]) + my_zero_halite_ship_density[
                              move_row, move_col] -1e2*(
                                avoid_squares_boxed_in[move_row, move_col])
                
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
           _, _, _) = ship_plans[future_ship_k]
          shortest_actions = get_dir_from_target(
            row, col, target_row, target_col, grid_size)
          
          # Filter out bad positions from the shortest actions
          valid_actions = []
          for a in shortest_actions:
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            if not bad_positions[move_row, move_col] or (
                ignore_base_collision and (
                  move_row == target_row and move_col == target_col)) or (
                not opponent_bases[move_row, move_col] and all_ship_scores[
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
      if not all_ship_scores[ship_k][12]:
        for j in range(i+1, num_ships):
          other_ship_k = ordered_ship_plans[j]
          if ship_plans[other_ship_k] != CONVERT:
            other_row = ship_plans[other_ship_k][4]
            other_col = ship_plans[other_ship_k][5]
          
            # If I move to a distance of <= 1 of the other ship: update valid
            # non self destruct actions
            # Exception: end of episode base return
            distance = DISTANCES[new_row, new_col][other_row, other_col]
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
    weighted_base_mask, history, requested_save_conversion_budget,
    victory_formation):

  spawn_cost = env_config.spawnCost
  convert_cost = env_config.convertCost
  my_ship_count = my_next_ships.sum()
  
  # Start saving for an alternative base when my base is spammed by opponents
  # print(observation['step'], history['my_base_flooded_counter'])
  if history['my_base_flooded_counter']:
    min_flood_counter = np.array(
      list(history['my_base_flooded_counter'].values())).min()
    save_base_flood_fraction = min(1, (min_flood_counter/5))**0.5
    save_restore_budget = save_base_flood_fraction*convert_cost
  else:
    save_restore_budget = 0
  
  max_spawns = int((remaining_budget-save_restore_budget-(
    requested_save_conversion_budget))/spawn_cost)
  relative_step = observation['relative_step']
  max_allowed_ships = config['max_initial_ships'] - relative_step*(
    config['max_initial_ships'] - config['max_final_ships'])
  # print(observation['step'], max_allowed_ships, my_ship_count)
  if history['limit_ships_timeout']:
    # In order to avoid weird timeouts
    max_allowed_ships -= 15
  total_ship_count = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships']]).sum()
  max_spawns = min(max_spawns, int(max_allowed_ships - my_ship_count))
  
  # if observation['step'] == 60:
  #   import pdb; pdb.set_trace()
  
  # Restrict the number of spawns if there is little halite remaining on the
  # map when I am not winning or the game is almost over
  current_scores = history['current_scores']
  current_halite_sum = history['current_halite_sum']
  not_winning = (current_scores[0] < current_scores.max()) or (
    current_halite_sum[0] < (current_halite_sum[1:].max()+(3-10*(
      0.8-max(0.8, observation['relative_step'])))*spawn_cost))
  halite_sum_spawn_advantage = (current_halite_sum[0] - (
    current_halite_sum[1:].max()))/spawn_cost
  massively_ahead_in_halite = halite_sum_spawn_advantage > 8
  game_almost_over = observation['relative_step'] >= 0.8
  steps_remaining = env_config.episodeSteps-1-observation['step']
  if not_winning or (game_almost_over and not massively_ahead_in_halite):
    max_spawns = min(max_spawns, int(
      min(3000, (obs_halite.sum())**0.8)/min(
        total_ship_count+1e-9, (my_ship_count+1e-9)*2)/spawn_cost*(
        1-relative_step)*(env_config.episodeSteps-2)/config[
          'max_spawn_relative_step_divisor']))
  elif game_almost_over and massively_ahead_in_halite and (
      steps_remaining >= 20 or not victory_formation):
    max_spawns = min(max_spawns, int(halite_sum_spawn_advantage-8))
  last_episode_turn = observation['relative_step'] == 1

  # if observation['step'] == 176:
  #   import pdb; pdb.set_trace()

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
    # # Don't spawn if it is not the main base - OBSOLETE
    # spawn_scores[i] -= 1e12*int(weighted_base_mask[row, col] < 1)
    
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
          updated_distance = DISTANCES[row, col][updated_ship_pos[k]]
          if updated_distance == 1:
            spawn_scores[i] -= 1e6
            break
          
    # Spawn when the base is instructed to do so in the previous step (newly)
    # created
    if history['prev_step']['should_spawn_base_next_step'] == (row, col):
      # import pdb; pdb.set_trace()
      print(observation['step'], "NEW BASE SPAWN", (row, col))
      spawn_scores[i] += 1e6
    
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
  num_above_threshold = (spawn_scores > config['spawn_score_threshold']).sum()
  spawn_ids = np.argsort(-spawn_scores)[:min(
    num_above_threshold, max_spawns)]
  
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
                np.array([prev_pos_opponent]), prev_count+1, row, col,
                prev_row, prev_col)
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
  convert_cost = env_config.convertCost
  
  if observation['step'] in [0, env_config.episodeSteps-2]:
    history['raw_box_data'] = [[] for _ in range(num_players)]
    history['inferred_boxed_in_conv_threshold'] = [[
      convert_cost/2, convert_cost] for _ in range(num_players)]
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
          prev_halite_score = prev_player_obs[0]
          if (prev_halite + prev_halite_score) >= convert_cost:
            did_convert = k in converted_ships
            history['raw_box_data'][player_id].append(
              (prev_halite, did_convert))
      
      # Infer the halite convert threshold when being boxed in
      if history['raw_box_data'][player_id]:
        box_data = np.array(history['raw_box_data'][player_id])
        current_thresholds = history['inferred_boxed_in_conv_threshold'][
          player_id]
        
        # The high estimate is the lowest halite value at which a ship was
        # ever converted - defaults to convert_cost
        if np.any(box_data[:, 1] == 0):
          highest_not_converted = box_data[box_data[:, 1] == 0, 0].max()
        else:
          highest_not_converted = 0
        if np.any(box_data[:, 1] == 1):
          lowest_converted = box_data[box_data[:, 1] == 1, 0].min()
        else:
          lowest_converted = current_thresholds[1]
          
        # This can happen if the opponent does not take the halite on board
        # into account (non consistent convert threshold)
        highest_not_converted = min(highest_not_converted, lowest_converted)
        
        # The low estimate is the average between the high estimate and the
        # highest halite at which a ship was *not* converted (defaults) to 0
        history['inferred_boxed_in_conv_threshold'][player_id] = [
          (lowest_converted+highest_not_converted)/2, lowest_converted]
        
  return history

def update_zero_halite_ship_behavior(
    config, history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config):
  grid_size = stacked_ships.shape[1]
  num_players = stacked_ships.shape[0]
  near_base_distance = config['log_near_base_distance']
  max_recent_considered_relevant = config[
    'max_recent_considered_relevant_zero_move_count']
  
  # Minimum number of required examples to be able to estimate the opponent's
  # zero halite ship behavior. Format ('nearbase_shipdistance')
  min_considered_types = {
    'False_0': 0,
    'False_1': 8,
    'False_2': 15,
    'True_0': 8,
    'True_1': 8,
    'True_2': config['near_base_2_step_risky_min_count'],
    }
  
  if observation['step'] == 0:
    history['raw_zero_halite_move_data'] = [[] for _ in range(num_players)]
    history['zero_halite_move_behavior'] = [{} for _ in range(num_players)]
    history['my_zero_lost_ships_opponents'] = {}
    
    initial_aggressive_behavior = {}
    for is_near_base in [False, True]:
      for considered_distance in [0, 1, 2]:
        dict_k = str(is_near_base) + '_' + str(considered_distance)
        dict_k_always_careful = dict_k + '_always_careful'
        dict_k_real_count = dict_k + '_real_count'
        dict_k_ever_risky = dict_k + '_ever_risky'
        if min_considered_types[dict_k] == 0:
          initial_aggressive_behavior[dict_k] = 0.0
          initial_aggressive_behavior[dict_k_always_careful] = True
          initial_aggressive_behavior[dict_k_real_count] = 0
          initial_aggressive_behavior[dict_k_ever_risky] = False
        else:
            initial_aggressive_behavior[dict_k] = 1.0
            initial_aggressive_behavior[dict_k_always_careful] = False
            initial_aggressive_behavior[dict_k_real_count] = 0
            initial_aggressive_behavior[dict_k_ever_risky] = False
    for player_id in range(1, num_players):
      history['zero_halite_move_behavior'][player_id] = (
        copy.copy(initial_aggressive_behavior))
    
  else:
    prev_stacked_bases = history['prev_step']['stacked_bases']
    all_prev_bases = prev_stacked_bases.sum(0) > 0
    prev_stacked_ships = history['prev_step']['stacked_ships']
    all_prev_ships = np.sum(prev_stacked_ships, 0) > 0
    prev_base_locations = np.where(all_prev_bases)
    prev_boxed_in_zero_halite_opponents = history['prev_step'][
      'boxed_in_zero_halite_opponents']
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
      prev_halite_ships = history['prev_step']['halite_ships']
      prev_opponent_sensible_actions_no_risk = history['prev_step'][
        'opponent_ships_sensible_actions_no_risk']
      my_prev_ship_pos_to_key = {
        v[0]: k for k, v in history['prev_step']['env_observation'].players[
          env_obs_ids[0]][2].items()}
      all_prev_ship_pos_to_key = {}
      all_ship_keys = []
      for player_id in range(num_players):
        env_obs_id = env_obs_ids[player_id]
        all_prev_ship_pos_to_key.update({
          v[0]: k for k, v in history['prev_step']['env_observation'].players[
            env_obs_id][2].items()})
        all_ship_keys.extend(list(env_observation.players[
          env_obs_id][2].keys()))
        
      history['my_zero_lost_ships_opponents'] = {}        
      for player_id in range(1, num_players):
        # Consider all zero halite ships and infer the action that each player
        # took
        env_obs_id = env_obs_ids[player_id]
        player_obs = env_observation.players[env_obs_id]
        prev_player_obs = history['prev_step']['env_observation'].players[
            env_obs_id]
        
        for k in prev_player_obs[2]:
          prev_row, prev_col = row_col_from_square_grid_pos(
            prev_player_obs[2][k][0], grid_size)
          ignore_boxed_zero_halite = (prev_row, prev_col) in (
            prev_boxed_in_zero_halite_opponents)
          if k in player_obs[2] and prev_player_obs[2][k][1] == 0 and (
              not ignore_boxed_zero_halite):
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
            
            if len(prev_opponent_sensible_actions_no_risk[
                prev_row, prev_col]) < len(MOVE_DIRECTIONS):
              # Loop over all zero halite opponent ships at a distance of max 2
              # and log the distance, None action count, move towards count and
              # move away count as well as the distance to the nearest base.
              # Also record whether the nearest base is friendly or not.
              considered_threat_data = []
              for row_shift, col_shift, distance in (
                  D2_ROW_COL_SHIFTS_DISTANCES):
                considered_row = (prev_row + row_shift) % grid_size
                considered_col = (prev_col + col_shift) % grid_size
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
                    friendly_prev_nearest_base, observation['step'], True))
                      
              # Aggregate the per-ship behavior - only consider the nearest
              # opponent threats
              num_considered_threats = len(considered_threat_data)
              if num_considered_threats == 1:
                history['raw_zero_halite_move_data'][player_id].append(
                  considered_threat_data[0])
              elif num_considered_threats > 0:
                threat_data = np.array(considered_threat_data)
                min_distance = threat_data[:, 0].min()
                for row_id in range(num_considered_threats):
                  if threat_data[row_id, 0] == min_distance:
                    history['raw_zero_halite_move_data'][player_id].append(
                      considered_threat_data[row_id])
                    
          elif k not in player_obs[2] and prev_player_obs[2][k][1] == 0 and (
              not ignore_boxed_zero_halite):
            # The opponent lost their zero halite ship - infer what happened
            
            # Investigate if there is no new base at the position where there
            # used to be a ship (this would explain why the ship disappeared)
            prev_pos = prev_player_obs[2][k][0]
            if not (prev_pos in player_obs[1].values() and (
                not prev_pos in prev_player_obs[1].values())):
              prev_row, prev_col = row_col_from_square_grid_pos(
                prev_pos, grid_size)
              nearest_prev_base_distance = nearest_prev_base_distances[
                prev_row, prev_col]
              
              base_destroyed = False
              if nearest_prev_base_distance == 1:
                # Determine if a base at a distance of 1 from the previous
                # position was destroyed
                for destroyed_dir in NOT_NONE_DIRECTIONS:
                  base_near_row, base_near_col = move_ship_row_col(
                    prev_row, prev_col, destroyed_dir, grid_size)
                  base_near_pos = base_near_row*grid_size+base_near_col
                  if all_prev_bases[base_near_row, base_near_col]:
                    base_destroyed = True
                    for p_id in range(num_players):
                      if base_near_pos in env_observation.players[
                          p_id][1].values():
                        base_destroyed = False
                        break
                      
                  if base_destroyed:
                    break
              
              nearest_prev_base_id = np.argmin(stacked_prev_base_distances[
                :, prev_row, prev_col])
              nearest_prev_base_row = prev_base_locations[0][
                nearest_prev_base_id]
              nearest_prev_base_col = prev_base_locations[1][
                nearest_prev_base_id]
              nearest_base_player = prev_base_player_ids[
                nearest_prev_base_row, nearest_prev_base_col]
              friendly_prev_nearest_base = (nearest_base_player == player_id)
              
              # Consider all potential move squares for the destroyed ship and
              # all potential non friendly ship movements. For my ships: only
              # consider the actions I actually selected.
              potential_ship_collisions = []
              for d1 in MOVE_DIRECTIONS:
                # D1 are the potential actions of the opponent ship
                row_d1, col_d1 = move_ship_row_col(
                  prev_row, prev_col, d1, grid_size)
                for d2 in MOVE_DIRECTIONS:
                  # D2 are the potential collision directions, relative to the
                  # moved position of the opponent ship
                  row_d2, col_d2 = move_ship_row_col(
                    row_d1, col_d1, d2, grid_size)
                  if all_prev_ships[row_d2, col_d2] and (
                      prev_halite_ships[row_d2, col_d2] == 0) and (
                        prev_ship_player_ids[row_d2, col_d2] != player_id):
                    d2_pos = row_d2*grid_size+col_d2
                    my_ship_position = d2_pos in my_prev_ship_pos_to_key
                    
                    consider_action = not my_ship_position
                    if not consider_action:
                      # Only consider the actions I actually selected for my
                      # ships
                      my_action = history['prev_step']['my_ship_actions'][
                        my_prev_ship_pos_to_key[d2_pos]]
                      consider_action = my_action is not CONVERT and (
                        my_action == OPPOSITE_MAPPING[d2])
                    if consider_action:
                      # Final condition: the ship at the considered location
                      # no longer exists in the current step
                      other_ship_k = all_prev_ship_pos_to_key[
                        row_d2*grid_size+col_d2]
                      if other_ship_k not in all_ship_keys:
                        distance = DISTANCES[prev_row, prev_col][
                          row_d2, col_d2]
                        threat_distance = DISTANCES[row_d2, col_d2][
                          row_d1, col_d1]
                        # Use the distance of the opponent ship to the base
                        # rather than the collision distance since it is
                        # ambiguous where the ships collided
                        potential_ship_collisions.append((
                          distance, threat_distance,
                          nearest_prev_base_distance,
                          friendly_prev_nearest_base, observation['step'],
                          my_ship_position, row_d2, col_d2))
                    
              # if observation['step'] == 99:
              #   import pdb; pdb.set_trace()
                    
              # Potential collisions at distance 2 come in pairs
              # of items in potential_ship_collisions must be even.
              # Investigate if it is 0 without a ship collisions or >= 4
              num_potential_collisions = len(potential_ship_collisions)
              to_add_data = []
              if num_potential_collisions == 0:
                if not base_destroyed and observation['relative_step'] < 0.9:
                  pass
                  # Unexplained ship loss - likely due to an opponent self
                  # collision
              elif num_potential_collisions <= 2:
                # Either a single d1/d2 potential collisions or two potential
                # d2 collisions. Either way: the collision data would be
                # identical
                to_add_data = potential_ship_collisions[0][:-3]
              else:
                # In case of disagreement in the distance of the collision:
                # pick the lowest distance one
                collision_distances = np.array([
                  pc[0] for pc in potential_ship_collisions])
                to_add_data = potential_ship_collisions[
                  np.argmin(collision_distances)][:-3]
              certain_data = False
              if num_potential_collisions == 1:
                ship_collision = potential_ship_collisions[0]
                certain_data = ship_collision[5]
                if ship_collision[5]:
                  history['my_zero_lost_ships_opponents'][
                    (ship_collision[6], ship_collision[7])] = player_id
              if to_add_data:
                history['raw_zero_halite_move_data'][player_id].append(tuple(
                  list(to_add_data) + [certain_data]))
        
        # Infer the zero halite behavior as a function of distance to opponent
        # base and distance to other zero halite ships
        if history['raw_zero_halite_move_data'][player_id]:
          zero_halite_data = np.array(history['raw_zero_halite_move_data'][
            player_id])
          aggregate_data = {}
          for is_near_base in [False, True]:
            for considered_distance in [0, 1, 2]:
              relevant_rows = (zero_halite_data[:, 0] == max(
                1, considered_distance))
              if is_near_base:
                relevant_rows &= (zero_halite_data[:, 2] <= near_base_distance)
              else:
                relevant_rows &= (zero_halite_data[:, 2] > near_base_distance)
              if considered_distance < 2:
                relevant_rows &= (zero_halite_data[:, 5] == 1)
              num_relevant = relevant_rows.sum()
              if num_relevant > max_recent_considered_relevant:
                num_relevant = max_recent_considered_relevant
                relevant_ids = np.where(relevant_rows)[0]
                relevant_rows[
                  relevant_ids[:-(max_recent_considered_relevant)]] = 0
              aggressive_relevant_count = (
                relevant_rows & (zero_halite_data[:, 1] <= min(
                  1, considered_distance))).sum()
              
              dict_k = str(is_near_base) + '_' + str(considered_distance)
              dict_k_always_careful = dict_k + '_always_careful'
              dict_k_real_count = dict_k + '_real_count'
              dict_k_ever_risky = dict_k + '_ever_risky'
              min_considered = min_considered_types[dict_k]
              num_aggressive_added = max(0, min_considered-num_relevant)
              if num_relevant == 0 and num_aggressive_added == 0:
                num_relevant = 1
              elif num_aggressive_added > 0:
                num_aggressive_added = min_considered-num_relevant
                num_relevant += num_aggressive_added
                aggressive_relevant_count += num_aggressive_added
                
              # if player_id == 3 and considered_distance == 0 and is_near_base:
              #   print(observation['step'], num_relevant, num_aggressive_added,
              #         aggressive_relevant_count)
              #   # if observation['step'] == 72:
              #   #   import pdb; pdb.set_trace()
              #   #   x=1
                
              aggregate_data[dict_k] = aggressive_relevant_count/max(
                1e-9, num_relevant)
              aggregate_data[dict_k_always_careful] = (
                aggressive_relevant_count == 0)
              aggregate_data[dict_k_real_count] = (
                num_relevant - num_aggressive_added)
              aggregate_data[dict_k_ever_risky] = (
                aggressive_relevant_count > num_aggressive_added)
              
              if is_near_base:
                # If an opponent is aggressive away from the base, they likely
                # are near the base too
                # Approach: assume the most aggressive of near and away from
                # the base behavior when the considered ship is near the base
                dict_k_away_base = str(False) + '_' + str(considered_distance)
                dict_k_always_careful_away_base = dict_k_away_base + (
                  '_always_careful')
                dict_k_ever_risky_away_base = dict_k_away_base + '_ever_risky'
                
                aggregate_data[dict_k] = max(
                  aggregate_data[dict_k], aggregate_data[dict_k_away_base])
                aggregate_data[dict_k_always_careful] = (
                  aggregate_data[dict_k_always_careful]) and aggregate_data[
                    dict_k_always_careful_away_base]
                aggregate_data[dict_k_ever_risky] = (
                  aggregate_data[dict_k_ever_risky]) or aggregate_data[
                    dict_k_ever_risky_away_base]
                    
          history['zero_halite_move_behavior'][player_id] = aggregate_data
          
  return history

def update_base_camping_strategy(
    config, history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config, np_rng, continued_camping_bonus=0.2, corner_camping_patience=3,
    other_camping_patience=5, max_non_unique_campers=2,
    max_campers_per_base=2, play_safe_aggression_limit=1,
    my_base_flooded_patience=5, flood_patience_buffer=2,
    min_ships_to_consider_camping=5, camping_risk_phase_2_7_multiplier=0.00):
  grid_size = stacked_ships.shape[1]
  num_players = stacked_ships.shape[0]
  my_ships_obs = env_observation.players[env_obs_ids[0]][2]
  stacked_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships']])
  # flood_base_convert_threshold = my_base_flooded_patience + (
  #   min(3, stacked_bases[0].sum()))**2 - 1
  halite_ships = np.stack([
        rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  all_bases = stacked_bases.sum(0) > 0
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
    player_ids[stacked_bases[i]] = i
    
  # if observation['step'] == 116:
  #   import pdb; pdb.set_trace()
    
  if observation['step'] == 0:
    history['camping_ships_strategy'] = {}
    history['camping_ships_targets'] = {}
    history['remaining_camping_budget'] = config['max_camper_ship_budget']
    history['aggression_stage_opponents_camping'] = [
      0 for _ in range(num_players)]
    history['aggression_opponents_camping_counter'] = [
      0 for _ in range(num_players)]
    history['camping_phase_opponents'] = [{} for _ in range(num_players)]
    history['camping_attack_opponent_budget'] = [2 for _ in range(num_players)]
    history['camping_phase_2_details_opponents'] = [(0, 0) for _ in range(
      num_players)]
    history['camping_phase_3_4_ignore_threats_counter'] = [0 for _ in range(
      num_players)]
    history['base_deposit_data'] = [[] for _ in range(num_players)]
    history['obs_base_camping_behavior'] = {}
    history['attack_opponent_campers'] = {}
    history['my_base_not_attacked_positions'] = []
    history['my_camped_base_not_attacked_positions'] = []
    history['base_camping_override_positions'] = np.zeros((
      grid_size, grid_size), dtype=np.bool)
    history['my_base_flooded_counter'] = {}
    history['current_scores'] = np.zeros(num_players)
    history['current_halite_sum'] = np.zeros(num_players)
  else:
    # Compute the current approximate player score
    scores = np.array(
      [rbs[0] for rbs in observation['rewards_bases_ships']])
    my_score = scores[0]
    num_my_ships = stacked_ships[0].sum()
    prev_stacked_bases = np.stack(
      [rbs[1] for rbs in history['prev_step']['observation'][
        'rewards_bases_ships']])
    halite_cargos = np.array(
      [rbs[3].sum() for rbs in observation['rewards_bases_ships']])
    obs_halite = np.maximum(0, observation['halite'])
    collect_rate = env_config.collectRate
    obs_halite[obs_halite < 1/collect_rate] = 0
    opponent_bases = stacked_bases[1:].sum(0) > 0
    opponent_base_counts = stacked_bases[1:].sum((1, 2))
    num_all_opponent_bases = opponent_base_counts.sum()
    base_counts = stacked_bases.sum((1, 2))
    convert_cost = env_config.convertCost
    spawn_cost = env_config.spawnCost
    steps_remaining = env_config.episodeSteps-1-observation['step']
    
    if steps_remaining < grid_size:
      # Our ships may be returning - aggressively attack nearby campers
      corner_camping_patience = 1
      other_camping_patience = 1
    
    obs_halite_sum = observation['halite'].sum()
    base_value = min(convert_cost+spawn_cost,
                     steps_remaining*obs_halite_sum**0.6/(
                       10*base_counts.sum()+1e-9))
    ship_counts = stacked_ships.sum((1, 2))
    all_ship_count = ship_counts.sum()
    ship_value = max(min(1, 2*(1-observation['relative_step']))*(
      env_config.spawnCost), min(
        env_config.spawnCost, (steps_remaining-1)*obs_halite_sum**0.6/(
          all_ship_count+1e-9)))
    # A base is only valuable if there are ships to return to them
    base_score_counts = np.minimum(base_counts, 1+ship_counts/5)
    current_scores = scores+halite_cargos+ship_value*ship_counts+(
      base_value*np.sqrt(np.maximum(0, base_score_counts-1))) + (
        convert_cost+ship_value)*(base_score_counts > 0)*min(
          1, (steps_remaining-1)/20)
    # print(observation['step'], ship_value, base_value)
    history['current_scores'] = current_scores
    history['current_halite_sum'] = scores+halite_cargos
    base_camping_override_positions = np.zeros((
      grid_size, grid_size), dtype=np.bool)
    
    # Keep track of the deposit location of all opponents (used to decide
    # what bases to attack)
    for player_id in range(1, num_players):
      env_obs_id = env_obs_ids[player_id]
      player_obs = env_observation.players[env_obs_id]
      prev_player_obs = history['prev_step']['env_observation'].players[
          env_obs_id]
      # For all ships that are at a base location and had halite in the
      # previous turn: add it to the memory
      base_locations = list(player_obs[1].values())
      player_base_pos_to_key = {v: k for k, v in player_obs[1].items()}
      for ship_k in player_obs[2]:
        ship_position = player_obs[2][ship_k][0]
        if ship_position in base_locations and ship_k in prev_player_obs[2]:
          halite_deposited = prev_player_obs[2][ship_k][1]
          if halite_deposited > 0:
            history['base_deposit_data'][player_id].append((
              player_base_pos_to_key[ship_position], observation['step']))
            
    # Keep track of the non friendly zero halite behavior near all bases
    base_locations = np.where(all_bases)
    prev_base_camping_behavior = history['obs_base_camping_behavior']
    obs_base_camping_behavior = {}
    num_bases = base_locations[0].size
    for base_id in range(num_bases):
      base_row = base_locations[0][base_id]
      base_col = base_locations[1][base_id]
      base_k = (base_row, base_col)
      base_player_id = player_ids[base_k]
      not_my_base = base_player_id > 0
      around_base_mask = ROW_COL_BOX_MAX_DISTANCE_MASKS[base_row, base_col, 1]
      zero_halite_near_base_mask = edge_aware_square_subset_mask(
        (player_ids != base_player_id) & (halite_ships == 0), base_row,
        base_col, window=1, box=around_base_mask, grid_size=grid_size)
      zero_halite_near_base_players = edge_aware_square_subset_mask(
        player_ids, base_row, base_col, window=1, box=around_base_mask,
        grid_size=grid_size)
      
      if zero_halite_near_base_mask.sum() > 0:
        if base_k in prev_base_camping_behavior:
          prev_camping = prev_base_camping_behavior[base_k]
          prev_zero_halite_near_base_mask = prev_camping[3]
          prev_corner_camping_counter = prev_camping[4]
          prev_other_camping_counter = prev_camping[5]
          prev_zero_halite_near_base_players = prev_camping[6]
        else:
          prev_zero_halite_near_base_mask = np.zeros_like(
            zero_halite_near_base_mask)
          prev_corner_camping_counter = np.zeros(9)
          prev_other_camping_counter = np.zeros(9)
          prev_zero_halite_near_base_players = -1*np.ones(9)
        
        # Count the number of zero halite opponents that stay at the same
        # location (switching zero halite ships within the same team would also
        # qualify here)
        stay_near_base = prev_zero_halite_near_base_mask & (
          zero_halite_near_base_mask) & (
            prev_zero_halite_near_base_players == (
              zero_halite_near_base_players ))
        num_stay_near_base = stay_near_base.sum()
        num_my_stay_near_base = (
          stay_near_base & (zero_halite_near_base_players == 0)).sum()
        
        # Inspect if there is any ship that statically remains at a base corner
        stay_base_corner = (np.mod(np.arange(9), 2) == 1) & (
          stay_near_base)
        num_my_stay_near_base_corner = (
          stay_base_corner & (zero_halite_near_base_players == 0)).sum()
        
        other_camping_counter = prev_other_camping_counter
        other_camping_counter[stay_near_base] += 1
        other_camping_counter[~stay_near_base] = 0
        
        if np.any(stay_base_corner):
          corner_camping_counter = prev_corner_camping_counter
          corner_camping_counter[stay_base_corner] += 1
          corner_camping_counter[~stay_base_corner] = 0
          attack_corner_camper = np.any(
            corner_camping_counter >= corner_camping_patience)
          
          # LEGEND:
          # obs_base_camping_behavior[k] = (consider_for_camping_target,
          # attack_corner_camper, attack_non_corner_camper,
          # zero_halite_near_base_mask, corner_camping_counter,
          # zero_halite_near_base_players)
          
          obs_base_camping_behavior[base_k] = (
            not_my_base and num_my_stay_near_base_corner > 0,
            attack_corner_camper, False, zero_halite_near_base_mask,
            corner_camping_counter, other_camping_counter,
            zero_halite_near_base_players)
        else:
          attack_other_camper = np.any(
            other_camping_counter >= other_camping_patience)
          obs_base_camping_behavior[base_k] = (
            not_my_base and ((num_stay_near_base-num_my_stay_near_base) < 2),
            False, attack_other_camper, zero_halite_near_base_mask,
            np.zeros(9), other_camping_counter,
            zero_halite_near_base_players)
      else:
        obs_base_camping_behavior[base_k] = (
          not_my_base, False, False, zero_halite_near_base_mask, np.zeros(9),
          np.zeros(9), -1*np.ones(9))
        
    history['obs_base_camping_behavior'] = obs_base_camping_behavior
    
    
    # Update the opponent camper attack budget based on my planned opponent
    # camper ship attacks
    for k in history['attack_opponent_campers']:
      if k in env_observation.players[env_obs_ids[0]][2]:
        # My attacking ship is still alive - add it back to the attack budget
        opponent_id = history['attack_opponent_campers'][k][4]
        history['camping_attack_opponent_budget'][opponent_id] += 1
    history['attack_opponent_campers'] = {}
    
    # Decide on which campers I should attack or if I should create a new base
    num_my_bases = stacked_bases[0].sum()
    my_zero_halite_excluded_from_camping = np.zeros_like(stacked_ships[0])
    # Exclude ships that were used for base defense in the previous step and
    # ships located at any of my bases
    for ship_k in list(set(
        history['prev_step']['base_defense_keys'] + history[
          'prev_step']['prev_base_defense_keys'])):
      if ship_k in my_ships_obs:
        ship_position = my_ships_obs[ship_k][0]
        row, col = row_col_from_square_grid_pos(ship_position, grid_size)
        my_zero_halite_excluded_from_camping[row, col] = 1
    my_zero_halite_ships_pos = np.where(
        (halite_ships == 0) & stacked_ships[0] & (
          ~my_zero_halite_excluded_from_camping) & (~stacked_bases[0]))
    my_num_zero_halite_ships = my_zero_halite_ships_pos[0].size
    history['my_base_not_attacked_positions'] = []
    history['my_camped_base_not_attacked_positions'] = []
    
    #######################
    ### DEFENSIVE LOGIC ###
    #######################
    opponent_ships = stacked_ships[1:].sum(0) > 0
    num_opponent_zero_halite_ships = (
      opponent_ships & (halite_ships == 0)).sum()
    opponent_zero_halite_ships_pos = np.where(
      opponent_ships & (halite_ships == 0))
    
    # if observation['step'] == 315:
    #   import pdb; pdb.set_trace()
    
    if num_my_bases > 0:
      ship_pos_to_key = {}
      for i in range(num_players):
        ship_pos_to_key.update({
          v[0]: k for k, v in env_observation.players[
            env_obs_ids[i]][2].items()})
      for base_k in obs_base_camping_behavior:
        if stacked_bases[0, base_k[0], base_k[1]]:
          # Only attack campers around my previous step main base
          opp_camping_behavior = obs_base_camping_behavior[base_k]
          
          if num_my_bases == 1 or (
              base_k in history['prev_step']['non_abandoned_base_locations']):
            if opp_camping_behavior[1] or opp_camping_behavior[2]:
              my_score_rank = (current_scores >= current_scores[0]).sum()
              
              # Loop over the opponent camping ships which have to be punished
              # It is not a camper if the ship sits at an opponent base
              offending_ship_flat_pos = np.where((
                opp_camping_behavior[4] >= corner_camping_patience) | (
                  opp_camping_behavior[5] >= other_camping_patience))[0]
              offending_ship_rows = np.mod(
                base_k[0] + (offending_ship_flat_pos//3) - 1, grid_size)
              offending_ship_cols = np.mod(
                base_k[1] + np.mod(offending_ship_flat_pos, 3) - 1, grid_size)
              for i in range(offending_ship_flat_pos.size):
                opponent_row = offending_ship_rows[i]
                opponent_col = offending_ship_cols[i]
                opponent_id = player_ids[opponent_row, opponent_col]
                opponent_score_rank = (
                  current_scores >= current_scores[opponent_id]).sum()
                # import pdb; pdb.set_trace()
                if ((my_score < convert_cost and num_my_ships > 4) or (
                    opponent_score_rank+my_score_rank <= 3) or (
                    my_score_rank == 1) or (
                      opponent_score_rank+my_score_rank == 4 and (
                        observation['relative_step'] < 0.4)) or (
                    history['camping_attack_opponent_budget'][
                      opponent_id] > 0)) and num_my_ships > 4 and (
                        not opponent_bases[opponent_row, opponent_col]):
                          
                  # Attack the opponent if there is a zero halite ship nearby
                  my_zero_halite_distances = DISTANCES[
                    opponent_row, opponent_col][my_zero_halite_ships_pos]
                  my_zero_halite_at_base = (DISTANCES[base_k][
                    my_zero_halite_ships_pos] == 0)
                  
                  if my_num_zero_halite_ships != 0:
                    # Attack the ship if I have a non base ship at distance 
                    # <= 3 with the non base ship.
                    # Otherwise: attack the ship with some probability with the
                    # base ship
                    safe_attack_ships = np.where((
                      my_zero_halite_distances <= 3) & (
                        ~my_zero_halite_at_base))[0]
                    defender_id = -1
                    
                    # If it takes too long for a safe attack: attack from the
                    # base anyway
                    edge_attack_from_base_prob = (
                      opp_camping_behavior[4].max()-(
                        corner_camping_patience+2))/4
                    if safe_attack_ships.size > 0 and (
                        edge_attack_from_base_prob) < np_rng.uniform():
                      defender_id = safe_attack_ships[
                        np.where(my_zero_halite_distances[
                          safe_attack_ships] == (my_zero_halite_distances[
                            safe_attack_ships].min()))[0][0]]
                    else:
                      # Attack the camper with a probability so that that it is
                      # hard to model (losing the base is always worse).
                      if np_rng.uniform() < 0.5:
                        defender_id = np.where(my_zero_halite_distances == (
                            my_zero_halite_distances.min()))[0][0]
                        
                    if defender_id >= 0:
                      defender_row = my_zero_halite_ships_pos[0][defender_id]
                      defender_col = my_zero_halite_ships_pos[1][defender_id]
                      defender_k = ship_pos_to_key[
                        defender_row*grid_size + defender_col]
                      opponent_k = ship_pos_to_key[
                        opponent_row*grid_size + opponent_col]
                      opponent_distance = my_zero_halite_distances[defender_id]
                      base_camping_override_positions[
                        opponent_row, opponent_col] = 1
                      attack_risk_threshold = 0 if opponent_distance > 2 else (
                        0.5 if opponent_distance == 1 else 0.05)
                      history['attack_opponent_campers'][defender_k] = (
                        opponent_row, opponent_col, 1e10, opponent_k,
                        opponent_id, opponent_distance, attack_risk_threshold)
                      my_zero_halite_excluded_from_camping[
                        defender_row, defender_col] = True
                      history['camping_attack_opponent_budget'][
                        opponent_id] -= 1
                else:
                  history['my_base_not_attacked_positions'].append(base_k)
                  history['my_camped_base_not_attacked_positions'].append(
                    base_k)
                      
          else:
            if opp_camping_behavior[1] or opp_camping_behavior[2]:
              # Flag the base as bad (there is a camper present), and don't
              # consider it when returning to a base
              # import pdb; pdb.set_trace()
              history['my_base_not_attacked_positions'].append(base_k)
              history['my_camped_base_not_attacked_positions'].append(base_k)
              
              
          # Identify if a base is jammed by opponent ships making it hard for
          # me to return to the base
          # Only consider zero halite opponents
          base_row, base_col = base_k
          if (opponent_ships & (halite_ships == 0)).sum() > 2:
            potential_threat_rows = opponent_zero_halite_ships_pos[0]
            potential_threat_cols = opponent_zero_halite_ships_pos[1]
            south_dist = np.where(
              potential_threat_rows >= base_row,
              potential_threat_rows-base_row,
              potential_threat_rows-base_row+grid_size)
            vert_dist = np.where(south_dist <= grid_size//2, south_dist,
                                 grid_size-south_dist)
            east_dist = np.where(
              potential_threat_cols >= base_col,
              potential_threat_cols-base_col,
              potential_threat_cols-base_col+grid_size)
            horiz_dist = np.where(east_dist <= grid_size//2, east_dist,
                                  grid_size-east_dist)
            dist = horiz_dist+vert_dist
            considered_distance_ids = dist <= 4 # 12 considered squares
            
            if considered_distance_ids.sum() > 1:
              # Check each quadrant for threats
              north_threat_ids = (south_dist[
                considered_distance_ids] > grid_size//2) & (
                  vert_dist[considered_distance_ids] >= horiz_dist[
                    considered_distance_ids])
              north_threat_score = (
                1/dist[considered_distance_ids][north_threat_ids]).sum()
              south_threat_ids = (south_dist[
                considered_distance_ids] < grid_size//2) & (
                  vert_dist[considered_distance_ids] >= horiz_dist[
                    considered_distance_ids])
              south_threat_score = (1/dist[
                considered_distance_ids][south_threat_ids]).sum()
              east_threat_ids = (east_dist[
                considered_distance_ids] < grid_size//2) & (
                  vert_dist[considered_distance_ids] <= horiz_dist[
                    considered_distance_ids])
              east_threat_score = (1/dist[
                considered_distance_ids][east_threat_ids]).sum()
              west_threat_ids = (east_dist[
                considered_distance_ids] > grid_size//2) & (
                  vert_dist[considered_distance_ids] <= horiz_dist[
                    considered_distance_ids])
              west_threat_score = (1/dist[
                considered_distance_ids][west_threat_ids]).sum()
                  
              threat_scores = np.array([
                north_threat_score, south_threat_score, east_threat_score,
                west_threat_score])
              min_threat_score = threat_scores.min()
            else:
              min_threat_score = 0
              
            current_flood_counter = history[
              'my_base_flooded_counter'].get(base_k, 0)
            
            # Linear model on the expected min threat score of a sim study
            expected_min_threat_score = 3.412e-03 - 1.047e-03*(
              num_opponent_zero_halite_ships) + 8.706e-05*(
                num_opponent_zero_halite_ships**2) - 2.878e-07*(
                  num_opponent_zero_halite_ships**3)
            # print(num_opponent_zero_halite_ships, expected_min_threat_score)
            current_flood_counter = max(0, min(
              my_base_flooded_patience+flood_patience_buffer,
              current_flood_counter+min_threat_score-expected_min_threat_score)
              )
            history['my_base_flooded_counter'][base_k] = (
              current_flood_counter)
            # print(observation['step'], threat_counts, current_flood_counter)
            # import pdb; pdb.set_trace()
            # x=1
            
            if current_flood_counter >= my_base_flooded_patience and not (
                base_k in history['my_base_not_attacked_positions']):
              history['my_base_not_attacked_positions'].append(base_k)
  
    # Delete the base flooded counter for destroyed bases
    destroyed_bases = ~stacked_bases[0] & (
      history['prev_step']['stacked_bases'][0])
    if np.any(destroyed_bases):
      destroyed_base_pos = np.where(destroyed_bases)
      for destroyed_base_id in range(destroyed_base_pos[0].size):
        destroyed_base_row = destroyed_base_pos[0][destroyed_base_id]
        destroyed_base_col = destroyed_base_pos[1][destroyed_base_id]
        destroyed_base_k = (destroyed_base_row, destroyed_base_col)
        if destroyed_base_k in history['my_base_flooded_counter']:
          del history['my_base_flooded_counter'][destroyed_base_k]
          
  # Reset the 'my_camped_base_not_attacked_positions' for the bases that are
  # flooded if all bases are flooded or camped and when I have a large number
  # of bases to avoid creating too many bases
  if (len(history['my_base_not_attacked_positions']) - len(
      history['my_camped_base_not_attacked_positions'])) > 2 and len(
        history['my_base_not_attacked_positions']) > 3:
    history['my_base_not_attacked_positions'] = copy.copy(
      history['my_camped_base_not_attacked_positions'])
    
  ########################
  ### AGGRESSIVE LOGIC ###
  ########################
  if (observation['relative_step'] >= config['relative_step_start_camping']):
    remaining_camping_budget = history['remaining_camping_budget']
    prev_camping_ships_targets = history['camping_ships_targets']
    number_already_camping = len(prev_camping_ships_targets)
    camping_ships_strategy = {}
    camping_ships_targets = {}
    aggression_stage_opponents = copy.copy(
      history['aggression_stage_opponents_camping'])
    aggression_camping_counter = copy.copy(
      history['aggression_opponents_camping_counter'])
    camping_phase_opponents = copy.copy(history['camping_phase_opponents'])
    prev_camping_phase_opponents = copy.copy(
      history['camping_phase_opponents'])
    prev_opponent_bases = history['prev_step']['stacked_bases'][1:].sum(0) > 0
    my_zero_lost_ships_opponents = history['my_zero_lost_ships_opponents']
    total_opponent_bases_count = stacked_bases.sum((1, 2))[1:]
    max_camping_budget_this_step = int(observation['relative_step']*(
      config['max_camper_ship_budget']*2))
    if remaining_camping_budget >= 1 or (number_already_camping > 0):
      # Aim higher than the current ranking: being behind is only counted half
      # as much as being ahead to determine who to attack
      score_diffs = current_scores[0]-current_scores[1:]
      win_preferred_score_diff = np.abs(score_diffs)
      win_preferred_score_diff[score_diffs < 0] /= 2
      
      opponent_scores_scaled = 1-win_preferred_score_diff/max(
        100, steps_remaining)/15-1e2*(
          (scores[1:] < env_config.spawnCost) & (ship_counts[1:] == 0))
    
      # Always keep targeting the number two when I am the number one
      # Also keep targeting the number one if I am the number two and the 
      # number three is far behind
      my_score_rank = (current_scores >= current_scores[0]).sum()
      play_safe_aggression_limits = np.ones(num_players)*(
        play_safe_aggression_limit)
      if my_score_rank == 1 or (my_score_rank == 2 and np.all(
          opponent_scores_scaled <= 0)):
        argmax_id = np.argmax(opponent_scores_scaled)
        opponent_scores_scaled[argmax_id] = max(
          opponent_scores_scaled[argmax_id], 1e-9)
        play_safe_aggression_limits[argmax_id+1] += 50
        
      # Don't camp at an opponent that has more bases than the camping budget
      # for this step
      opponent_scores_scaled[
        total_opponent_bases_count > min(
          max_camping_budget_this_step, remaining_camping_budget)] = -100
          
      if num_all_opponent_bases > 0:
        # Compute target scores for each of the opponent bases
        base_camping_scores = {}
        for player_id in range(1, num_players):
          opponent_bases = stacked_bases[player_id]
          env_obs_id = env_obs_ids[player_id]
          player_obs = env_observation.players[env_obs_id]
          current_base_keys = list(player_obs[1].keys())
          if opponent_bases.sum():
            num_opponent_bases = opponent_bases.sum()
            opponent_base_positions = np.where(opponent_bases)
            deposit_data = np.array(history['base_deposit_data'][player_id])
            # Delete old bases from the deposit data and consider at most the
            # most recent 20 deposits
            if deposit_data.size > 0:
              actual_base_rows = np.array([
                d in current_base_keys for d in deposit_data[:, 0]])
              deposit_data = deposit_data[actual_base_rows][-20:]
            player_base_pos_to_key = {v: k for k, v in player_obs[1].items()}
            for base_id in range(num_opponent_bases):
              base_row = opponent_base_positions[0][base_id]
              base_col = opponent_base_positions[1][base_id]
              base_key = player_base_pos_to_key[base_row*grid_size+base_col]
              if deposit_data.size == 0:
                relative_deposit_base_score = 1
              else:
                if opponent_scores_scaled[player_id-1] < 0:
                  relative_deposit_base_score = 1
                else:
                  relative_deposit_base_score = min(1, (deposit_data[:, 0] == (
                    base_key)).mean() + 1e-2)
              should_consider_camp_penalty = -1*int(
                not obs_base_camping_behavior[(base_row, base_col)][0])
              base_camping_score = opponent_scores_scaled[player_id-1]*(
                relative_deposit_base_score)+should_consider_camp_penalty
              base_camping_scores[(base_row, base_col)] = base_camping_score
        
        # print(observation['step'], base_camping_scores)
        all_opponent_bases = list(base_camping_scores.keys())
        
        # Increment the score for the bases where we are already camping to
        # avoid switching targets too often
        prev_env_observation = history['prev_step']['env_observation']
        my_prev_ships_obs = prev_env_observation.players[env_obs_ids[0]][2]
        delete_keys = []
        for ship_k in prev_camping_ships_targets:
          ship_still_alive = ship_k in my_ships_obs
          
          # Incorporate the past step opponent behavior to infer what phase of
          # the camping each opponent is in (see details on the phases below)
          base_target = prev_camping_ships_targets[ship_k]
          prev_pos = my_prev_ships_obs[ship_k][0]
          prev_row, prev_col = row_col_from_square_grid_pos(
            prev_pos, grid_size)
          prev_base_distance = DISTANCES[prev_row, prev_col][base_target]
          opponent_id = np.where(prev_stacked_bases[
            :, base_target[0], base_target[1]])[0][0]
          opponent_prev_camping_phase = prev_camping_phase_opponents[
            opponent_id][base_target]
          
          aggression_already_added = False
          if not ship_still_alive:
            if (prev_row, prev_col) in my_zero_lost_ships_opponents:
              aggression_occurred = my_zero_lost_ships_opponents[
                (prev_row, prev_col)] == opponent_id
              if aggression_occurred:
                aggression_already_added = True
                aggression_camping_counter[opponent_id] += 1
                if aggression_camping_counter[opponent_id] >= (
                    play_safe_aggression_limits[opponent_id]):
                  aggression_stage_opponents[opponent_id] = 2
                  camping_phase_opponents[opponent_id][base_target] = 7
          
          if prev_base_distance <= 2:
            # Possible transitions (on a per-opponent level):
            #   - 2 -> 3: My ship does not get attacked at least M times and
            #             the opponent has returned at least N ships to the 
            #             base and there is a zero halite square right next to
            #             the base
            #   - 3 -> 4: My ship is not attacked when there are > 1 opponent
            #             zero halite ships that can get to me
            #   - 3 -> 5: The opponent is ignoring my camper and stil returns
            #             to the base
            #   - 4 -> 5: The opponent is ignoring my camper and stil returns
            #             to the base
            #   - 2 -> 6: My camping ship is not aggressively attacked but
            #             there is no zero halite square at distance 1 of the
            #             base
            #   - 6 -> 7: My camping ship is aggressively attacked
            #   - 2 -> 7: My camping ship is aggressively attacked
            if opponent_prev_camping_phase == 2 and (
                camping_phase_opponents[opponent_id][base_target] == 2):
              (num_halite_ships_returned, non_aggression_counter) = history[
                'camping_phase_2_details_opponents'][opponent_id]
              # Update the number of opponent non zero halite ships returned
              # to the base
              opponent_pos_to_ship = {v[0]: k for k, v in (
                env_observation.players[env_obs_ids[opponent_id]][2]).items()}
              opponent_prev_ships_obs = prev_env_observation.players[
                env_obs_ids[opponent_id]][2]
              target_base_pos = base_target[0]*grid_size + base_target[1]
              if target_base_pos in opponent_pos_to_ship:
                at_base_opponent_ship_k = opponent_pos_to_ship[target_base_pos]
                if at_base_opponent_ship_k in opponent_prev_ships_obs:
                  num_halite_ships_returned += int(opponent_prev_ships_obs[
                    at_base_opponent_ship_k][1] > 0)
              
              # Update the non aggression counter.
              aggression_occurred = (prev_pos in opponent_pos_to_ship) and (
                opponent_prev_ships_obs[
                  opponent_pos_to_ship[prev_pos]][1] == 0)
              
              # If I lost my ship, it was likely due to an aggression
              if not ship_still_alive and not aggression_occurred and (
                  prev_row, prev_col) in my_zero_lost_ships_opponents:
                aggression_occurred = my_zero_lost_ships_opponents[
                  (prev_row, prev_col)] == opponent_id
              
              # If an aggression occurred: move to phase 7.
              if aggression_occurred:
                if not aggression_already_added:
                  aggression_camping_counter[opponent_id] += 1
                if aggression_camping_counter[opponent_id] >= (
                    play_safe_aggression_limits[opponent_id]):
                  aggression_stage_opponents[opponent_id] = 2
                  camping_phase_opponents[opponent_id][base_target] = 7
              else:
                non_aggression_counter += 1
                # If the no aggression and ship return thresholds get exceeded:
                # move to phase 3 or 6
                if non_aggression_counter >= 10 and (
                    num_halite_ships_returned >= 5):
                  # Figure out if there is a non-zero halite square to camp at
                  # right next to the base to decide on the next camping phase
                  dist_1_zero_halite = (obs_halite == 0) & (DISTANCES[
                    base_target] == 1)
                  if aggression_stage_opponents[opponent_id] != 2:
                    aggression_stage_opponents[opponent_id] = 1
                  if dist_1_zero_halite.sum() > 0:
                    camping_phase_opponents[opponent_id][base_target] = 3
                  else:
                    camping_phase_opponents[opponent_id][base_target] = 6
              
              history['camping_phase_2_details_opponents'][opponent_id] = (
                num_halite_ships_returned, non_aggression_counter)
              
            elif opponent_prev_camping_phase == 6:
              # Some nice code duplication
              # Update the number of opponent non zero halite ships returned
              # to the base
              opponent_pos_to_ship = {v[0]: k for k, v in (
                env_observation.players[env_obs_ids[opponent_id]][2]).items()}
              opponent_prev_ships_obs = prev_env_observation.players[
                env_obs_ids[opponent_id]][2]
              aggression_occurred = (prev_pos in opponent_pos_to_ship) and (
                opponent_prev_ships_obs[
                  opponent_pos_to_ship[prev_pos]][1] == 0)
              
              # If I lost my ship, it was likely due to an aggression
              if not ship_still_alive and not aggression_occurred and (
                  prev_row, prev_col) in my_zero_lost_ships_opponents:
                aggression_occurred = my_zero_lost_ships_opponents[
                  (prev_row, prev_col)] == opponent_id
              
              # If an aggression occurred: move to phase 7.
              if aggression_occurred:
                # import pdb; pdb.set_trace()
                camping_phase_opponents[opponent_id][base_target] = 7
              
            elif opponent_prev_camping_phase in [3, 4]:
              # If I remain at a zero halite square at a distance of 1 of the
              # target and the opponent repeatedly ignores my threat: go to
              # phase 5
              ignore_camping_threats_counter = history[
                'camping_phase_3_4_ignore_threats_counter'][opponent_id]
              if ship_still_alive:
                ship_position = my_ships_obs[ship_k][0]
                row, col = row_col_from_square_grid_pos(
                  ship_position, grid_size)
                if prev_row == row and prev_col == col and obs_halite[
                    row, col] == 0 and (prev_base_distance == 1):
                  opponent_pos_to_ship = {v[0]: k for k, v in (
                    env_observation.players[
                      env_obs_ids[opponent_id]][2]).items()}
                  opponent_prev_ships_obs = prev_env_observation.players[
                    env_obs_ids[opponent_id]][2]
                  target_base_pos = base_target[0]*grid_size + base_target[1]
                  if target_base_pos in opponent_pos_to_ship:
                    at_base_opponent_ship_k = opponent_pos_to_ship[
                      target_base_pos]
                    if at_base_opponent_ship_k in opponent_prev_ships_obs:
                      if opponent_prev_ships_obs[
                          at_base_opponent_ship_k][1] > 0:
                        ignore_camping_threats_counter += 1
                        if ignore_camping_threats_counter >= 3*0:
                          camping_phase_opponents[opponent_id][base_target] = 5
              else:
                # If a successful aggression occurred: move to phase 7.
                camping_phase_opponents[opponent_id][base_target] = 7
              history['camping_phase_3_4_ignore_threats_counter'][
                opponent_id] = ignore_camping_threats_counter
          
          if ship_still_alive:
            if base_target in all_opponent_bases:
              base_camping_scores[base_target] += continued_camping_bonus
          else:
            # Delete the camping ship from the dict if it was destroyed
            # Subtract half a ship from the budget if I ended up attacking a
            # base
            # Subtract a full ship if I used the ship to construct a new base
            delete_keys.append(ship_k)
            number_already_camping -= 1
            my_prev_move = history['prev_step']['my_ship_actions'][ship_k]
            my_prev_row, my_prev_col = row_col_from_square_grid_pos(
              history['prev_step']['env_observation'].players[
                env_obs_ids[0]][2][ship_k][0], grid_size)
            if my_prev_move == CONVERT:
              remaining_camping_budget -= 1
            else:
              my_moved_row, my_moved_col = move_ship_row_col(
                my_prev_row, my_prev_col, my_prev_move, grid_size)
              if not prev_opponent_bases[my_moved_row, my_moved_col]:
                remaining_camping_budget -= 1/2
              
          # Move to stage 7 after exceeding the allowable aggression count
          if aggression_camping_counter[opponent_id] >= (
              play_safe_aggression_limits[opponent_id]):
            camping_phase_opponents[opponent_id][base_target] = 7
        
        for del_ship_k in delete_keys:
          del prev_camping_ships_targets[del_ship_k]
        
        # Camping ships participate in opponent hunts depending on the phase
        max_campers_assigned = min([
          max_camping_budget_this_step,
          np.floor(remaining_camping_budget),
          100*int(opponent_scores_scaled.max() > 0),
          number_already_camping+100*int(
            config['start_camp_if_not_winning'] or np.all(
              (current_scores[1:] <= (current_scores[0] - 2*spawn_cost)))),
          ])
        num_interesting_bases = (
          np.array(list(base_camping_scores.values())) > 0).sum()
        num_campers_assigned = int(min([
          num_all_opponent_bases + max_non_unique_campers,
          max(number_already_camping, max_campers_assigned),
          my_num_zero_halite_ships-my_zero_halite_excluded_from_camping.sum(),
          num_interesting_bases + max_non_unique_campers,
          100*int(num_my_ships >= min_ships_to_consider_camping),
          ]))
        
        # Assign the campers to the top identified opponent bases
        camping_ships_targets = {}
        if num_campers_assigned > 0 and num_all_opponent_bases > 0 and (
            num_interesting_bases > 0):
          my_ship_pos_to_key = {v[0]: k for k, v in my_ships_obs.items()}
          my_zero_halite_ship_positions = np.where(
            stacked_ships[0] & (halite_ships == 0) & (
              ~my_zero_halite_excluded_from_camping))
          target_pos_scores = np.array([list(k) + [v] for k, v in (
            base_camping_scores.items())])
          target_pos_scores = np.concatenate([np.zeros(
            (target_pos_scores.shape[0], 1)), target_pos_scores], 1)
          for score_id in range(target_pos_scores.shape[0]):
            base_row = int(target_pos_scores[score_id, 1])
            base_col = int(target_pos_scores[score_id, 2])
            opponent_id = np.where(stacked_bases[
              :, base_row, base_col])[0][0]
            if not (base_row, base_col) in camping_phase_opponents[
                opponent_id]:
              initial_phase = 2 if (
                aggression_stage_opponents[opponent_id] <= 1) else 7
              camping_phase_opponents[opponent_id][(base_row, base_col)] = (
                initial_phase)
            
            target_pos_scores[score_id, 0] = camping_phase_opponents[
              opponent_id][(base_row, base_col)]
          
          target_rows = np.argsort(-target_pos_scores[:, -1])[
            :num_campers_assigned]
          
          consider_non_unique_base_attack = np.any(
            target_pos_scores[target_rows, -1] <= 0) or target_rows.size < (
              num_campers_assigned)
          if consider_non_unique_base_attack:
            # Only continue if there is a target opponent which is in camping
            # phase 6 or 7, and the number of assigned campers is strictly
            # greater than the number of positive base camping scores.
            # Update num_campers_assigned to the number of positive base
            # camping scores if I don't find such a base
            target_rows = target_rows[target_pos_scores[target_rows, -1] > 0]
            num_valid_targeted_bases = target_rows.size
            num_unassigned_campers = num_campers_assigned-(
              num_valid_targeted_bases)
            bases_ids_that_can_take_more_campers = target_rows[
              target_pos_scores[target_rows, 0] >= 6]
            num_can_add_bases_multiple_campers = (
              bases_ids_that_can_take_more_campers.size)
            
            if num_can_add_bases_multiple_campers > 0:
              max_add_iterations = max_campers_per_base-1
              add_iteration = 0
              while num_unassigned_campers > 0 and (
                  add_iteration < max_add_iterations):
                num_added = min(num_unassigned_campers,
                                num_can_add_bases_multiple_campers)
                target_rows = np.concatenate([
                  target_rows, bases_ids_that_can_take_more_campers[
                    :num_added]])
                num_unassigned_campers -= num_added
                add_iteration += 1
              num_campers_assigned = target_rows.size
            else:
              num_campers_assigned = num_valid_targeted_bases
          
          # First handle the bases where we already have a camper to avoid
          # releasing the pressure temporarily
          if num_campers_assigned > 1:
            all_prev_camping_bases = list(prev_camping_ships_targets.values())
            already_camping = np.zeros(num_campers_assigned, dtype=np.bool)
            for i in range(num_campers_assigned):
              considered_row = target_rows[i]
              base_row = int(target_pos_scores[considered_row, 1])
              base_col = int(target_pos_scores[considered_row, 2])
              camp_prev_step = (base_row, base_col) in all_prev_camping_bases
              near_my_zh_ship = ((DISTANCES[base_row, base_col] <= 2) & (
                halite_ships == 0) & (stacked_ships[0])).sum()
              already_camping[i] = camp_prev_step and (near_my_zh_ship > 0)
            target_rows = np.concatenate([target_rows[already_camping],
                                          target_rows[~already_camping]])
          
          camping_ships_targets_positions = {}
          for target_id in range(num_campers_assigned):
            target_row = target_rows[target_id]
            # Compute the distance to all my zero halite ships and pick the
            # closest to camp out
            base_row = int(target_pos_scores[target_row, 1])
            base_col = int(target_pos_scores[target_row, 2])
            zero_halite_base_distances = DISTANCES[base_row, base_col][
              my_zero_halite_ship_positions]
            best_ship_id = np.argmin(zero_halite_base_distances)
            camper_ship_row = my_zero_halite_ship_positions[0][best_ship_id]
            camper_ship_col = my_zero_halite_ship_positions[1][best_ship_id]
            ship_camper_k = my_ship_pos_to_key[
              camper_ship_row*grid_size+camper_ship_col]
            camping_ships_targets[ship_camper_k] = (base_row, base_col)
            camping_ships_targets_positions[(base_row, base_col)] = (
              camping_ships_targets_positions.get((base_row, base_col), [])) +(
                [(camper_ship_row, camper_ship_col)])
            
            # Delete the selected ship from my_zero_halite_ship_positions so
            # that it does not get assigned twice
            remaining_rows = np.delete(my_zero_halite_ship_positions[0],
                                       best_ship_id)
            remaining_cols = np.delete(my_zero_halite_ship_positions[1],
                                       best_ship_id)
            my_zero_halite_ship_positions = (remaining_rows, remaining_cols)
        
        # Camping strategy - always risky so losing a ship is fine
        # Phase 1: Aggressively navigate to the proximity of the base
        # Navigate cautiously once I am at a distance 2 of the base, since the
        # target opponent may start to get aggressive here.
        # The subsequent phases are computed on a per-opponent basis. 
          
        # Phase 2: Aim at a static zero halite corner or circle around a base
        # as long as not many opponent returns to the base. Shy away when a
        # zero halite opponent ship threatens me.
        # Phase 3: Aim at a zero halite square right next to the base but move
        # away when > 1 zero halite ships threaten me.
        # Phase 4: aim at a zero halite square right next to the base and do
        # not move away.
        # Phase 5: If my camper gets ignored: attack the base when it can not
        # be protected.
        # Phase 6: Aggressively circle around the base
        # Phase 7: Keep circling annoyingly around the base but do so in a safe
        # way.
        for ship_k in camping_ships_targets:
          target_base_row, target_base_col = camping_ships_targets[ship_k]
          target_base_loc_tuple = (target_base_row, target_base_col)
          ship_position = my_ships_obs[ship_k][0]
          row, col = row_col_from_square_grid_pos(ship_position, grid_size)
          current_base_distance = DISTANCES[row, col][
            target_base_row, target_base_col]
          opponent_id = np.where(stacked_bases[
            :, target_base_row, target_base_col])[0][0]
          camping_phase = camping_phase_opponents[opponent_id][(
            target_base_row, target_base_col)]
          zeros_grid_mask = np.zeros((grid_size, grid_size))
          
          if current_base_distance > 2:
            # Phase 1
            # Aim towards the base neighborhood and prefer squares that are
            # closer to my current square
            collect_override_addition = 2e4/(DISTANCES[
              target_base_row, target_base_col]+1)
            collect_override_addition[target_base_row, target_base_col] = -1e5
            collect_override_addition[row, col] = -1e5
            # Increase the 0.0 to encourage more risky behavior when navigating
            # towards an opponent base
            camping_ships_strategy[ship_k] = (
              0.01, collect_override_addition, zeros_grid_mask,
              current_base_distance > 3, False, target_base_loc_tuple)
          else:
            dist_1_zero_halite = (obs_halite == 0) & (DISTANCES[
              target_base_row, target_base_col] == 1)
            if dist_1_zero_halite.sum() == 0 and camping_phase in [
                3, 4, 5]:
              camping_phase = 6
            
            if camping_phase in [2, 6, 7]:
              # Passively select targets around the base in phase 2 and 7.
              # Set the targets aggressively in phase 6
              # Avoid my other zero halite ships next to the base
              target_box = ROW_COL_BOX_MAX_DISTANCE_MASKS[
                target_base_row, target_base_col, 1]
              collect_override_addition = 2e4*target_box
              collect_override_addition[target_base_row, target_base_col] = (
                -1e5)
              if obs_halite[row, col] > 0:
                collect_override_addition[row, col] = -1e5

              # Prefer box corners (top left, top righ, bottom left, bottom
              # right) that have an opposite corner with no halite
              target_box_pos = np.where(target_box)
              top_left = (target_box_pos[0][0], target_box_pos[1][0])
              top_left_zero = obs_halite[top_left] == 0
              top_right = (target_box_pos[0][2], target_box_pos[1][2])
              top_right_zero = obs_halite[top_right] == 0
              bottom_left = (target_box_pos[0][6], target_box_pos[1][6])
              bottom_left_zero = obs_halite[bottom_left] == 0
              bottom_right = (target_box_pos[0][8], target_box_pos[1][8])
              bottom_right_zero = obs_halite[bottom_right] == 0
              if top_left_zero and bottom_right_zero:
                collect_override_addition[top_left] += 7e4
                collect_override_addition[bottom_right] += 7e4
              if top_right_zero and bottom_left_zero:
                collect_override_addition[top_right] += 7e4
                collect_override_addition[bottom_left] += 7e4

              # Get the nearby mask of other zero halite ships so that my
              # ships that camp at the same base stay out of each other's way
              subtract_mask = np.zeros_like(collect_override_addition)
              num_this_base_campers = len(camping_ships_targets_positions[(
                  target_base_row, target_base_col)])
              my_ship_at_opposite_edge = False
              for other_row, other_col in camping_ships_targets_positions[(
                  target_base_row, target_base_col)]:
                if other_row != row or other_col != col:
                  base_row_diff = other_row-target_base_row
                  base_col_diff = other_col-target_base_col
                  ship_row_dir = np.sign(base_row_diff)
                  ship_col_dir = np.sign(base_col_diff)
                  row_central_mask = np.mod(target_base_row + 
                    3*ship_row_dir, grid_size)
                  col_central_mask = np.mod(target_base_col +
                    3*ship_col_dir, grid_size)
                  other_row_central_mask = np.mod(target_base_row - 
                    3*ship_row_dir, grid_size)
                  other_col_central_mask = np.mod(target_base_col -
                    3*ship_col_dir, grid_size)
                  mask_ship_dir = ROW_COL_BOX_MAX_DISTANCE_MASKS[
                    row_central_mask, col_central_mask, 3]
                  mask_other_ship_dir = ROW_COL_BOX_MAX_DISTANCE_MASKS[
                    other_row_central_mask, other_col_central_mask, 3]
                  subtract_mask += (2e3*(mask_ship_dir * DISTANCE_MASKS[
                    other_row, other_col] + ROW_COL_BOX_MAX_DISTANCE_MASKS[
                      other_row, other_col, 2]) - 1e4*mask_other_ship_dir)*(
                      DISTANCE_MASKS[row, col]**0.1)
                      
                  # Very high bonus for camping out at opposite corners or a
                  # zero halite square next to the base
                  if np.abs(base_row_diff) == 1 and np.abs(base_col_diff) == 1:
                    opposite_corner_row = np.mod(
                      target_base_row - base_row_diff, grid_size)
                    opposite_corner_col = np.mod(
                      target_base_col - base_col_diff, grid_size)
                    # When I block an opponent with two static ships at zero
                    # halite corners: keep them there!
                    my_ship_at_opposite_edge = my_ship_at_opposite_edge or (
                      row == opposite_corner_row and (
                        col == opposite_corner_col))
                    subtract_mask[opposite_corner_row, opposite_corner_col] -=(
                      1e6)
                      
              collect_override_addition -= subtract_mask
                           
              # if observation['step'] == 107:
              #   import pdb; pdb.set_trace()
                
              # Take more risky actions when I have > 1 campers
              risk_threshold = camping_risk_phase_2_7_multiplier*(
                num_this_base_campers-1) if (camping_phase in [2, 7]) else 0.1
              consider_ship_other_tactics = not my_ship_at_opposite_edge
              camping_ships_strategy[ship_k] = (
                risk_threshold, collect_override_addition, zeros_grid_mask,
                consider_ship_other_tactics, consider_ship_other_tactics,
                target_base_loc_tuple)
            elif camping_phase in [3, 4, 5]:
              # Select a zero halite target right next to the base to camp
              # The aggression level depends on the number of nearby zero
              # halite ships and the camping phase
              # Only attack the base when it is not protected in phase 5
              base_protected = (DISTANCES[row, col][
                target_base_row, target_base_col] > 1) or ((stacked_ships[
                  opponent_id]) & (halite_ships == 0) & (
                    DISTANCES[target_base_row, target_base_col] <= 1)
                    ).sum() > 0
              
              collect_override_addition = np.zeros((grid_size, grid_size))
              if base_protected or camping_phase != 5:
                target_camp_positions = np.where(dist_1_zero_halite)
                for target_camp_id in range(target_camp_positions[0].size):
                  target_camp_row = target_camp_positions[0][target_camp_id]
                  target_camp_col = target_camp_positions[1][target_camp_id]
                  collect_override_addition += 2e4/((DISTANCES[
                    target_camp_row, target_camp_col]+1)**2)
                  
                collect_override_addition *= (DISTANCE_MASKS[row, col]**0.2)
                
                collect_override_addition[
                  target_base_row, target_base_col] = -1e5
                if obs_halite[row, col] > 0:
                  collect_override_addition[row, col] = -1e5
                num_zero_halite_threats = ((stacked_ships[1:].sum(0) > 0) & (
                  halite_ships == 0) & (DISTANCES[row, col] == 1)).sum()
                risk_threshold = 0.0 if (
                  camping_phase == 3 and num_zero_halite_threats > 1) else 0.1
                camping_ships_strategy[ship_k] = (
                  risk_threshold, collect_override_addition, zeros_grid_mask,
                  False, camping_phase == 5, target_base_loc_tuple)
              else:
                # Successfully attack the base
                print("Aggressively attacking base", target_base_row,
                      target_base_col, observation['step'])
                collect_attack_addition = 1e5*(
                  DISTANCES[target_base_row, target_base_col] == 0)
                camping_ships_strategy[ship_k] = (
                  0.0, collect_override_addition,
                  collect_attack_addition, False, True, target_base_loc_tuple)
        
    history['camping_ships_targets'] = camping_ships_targets
    history['camping_ships_strategy'] = camping_ships_strategy
    history['remaining_camping_budget'] = remaining_camping_budget
    history['aggression_stage_opponents_camping'] = aggression_stage_opponents
    history['aggression_opponents_camping_counter'] = (
      aggression_camping_counter)
    history['camping_phase_opponents'] = camping_phase_opponents
    history['base_camping_override_positions'] = (
      base_camping_override_positions)
  return history

def update_opponent_ships_move_directions(
    history, observation, env_observation, env_obs_ids):
  prev_step_opponent_ship_moves = {}
  if observation['step'] > 0:
    prev_env_players = history['prev_step']['env_observation'].players
    num_players = len(prev_env_players)
    grid_size = observation['halite'].shape[0]
    for player_id in range(1, num_players):
      env_obs_id = env_obs_ids[player_id]
      player_ships = env_observation.players[env_obs_id][2]
      prev_player_ships = prev_env_players[env_obs_id][2]
      for ship_k in player_ships:
        if ship_k in prev_player_ships:
          prev_row, prev_col = row_col_from_square_grid_pos(
            prev_player_ships[ship_k][0], grid_size)
          row, col = row_col_from_square_grid_pos(
            player_ships[ship_k][0], grid_size)
          prev_action = get_dir_from_target(
            prev_row, prev_col, row, col, grid_size)[0]
          if prev_action is not None:
            prev_step_opponent_ship_moves[ship_k] = prev_action
  
  history['prev_step_opponent_ship_moves'] = prev_step_opponent_ship_moves
  
  return history

def update_cycle_counters(
    config, history, observation, player_obs, env_observation, env_obs_ids):
  num_players = len(observation['rewards_bases_ships'])
  stacked_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships']])
  all_bases = stacked_bases.sum(0) > 0
  grid_size = all_bases.shape[0]
    
  if observation['step'] == 0:
    history['ship_action_cycle_counter'] = {}
    history['avoid_cycle_actions'] = {}
    history['opponent_cycle_counters'] = [{} for _ in range(num_players-1)]
    history['empty_or_cycled_positions'] = np.zeros(
      (grid_size, grid_size), dtype=np.bool)
    history['cycled_position_counts'] = -1*np.ones((grid_size, grid_size))
    history['opponent_ship_pos_to_key'] = {}
  elif config['avoid_cycles'] > 0:
    cycle_counters = history['ship_action_cycle_counter']
    avoid_cycle_actions = {}
    prev_ship_actions = history['prev_step']['my_ship_actions']
    prev_step_rescue_ships = history['prev_step']['ships_on_rescue_mission']
    opponent_ship_pos_to_key = {}
    
    # Update the cycles and according cycle counters for all my ships that
    # are still alive
    for ship_k in prev_ship_actions:
      if not ship_k in player_obs[2]:
        # The ship died or was converted - delete it from the cycle data
        if ship_k in cycle_counters:
          del cycle_counters[ship_k]
      else:
        # If the ship is new: add the action and default counter to the cycle
        # counters dict
        prev_action = prev_ship_actions[ship_k]
        if not ship_k in cycle_counters:
          cycle_counters[ship_k] = (prev_action, -1, 0)
        else:
          prev_a_min_1, prev_a_min_2, cycle_count = cycle_counters[ship_k]
          if cycle_count > 0 and (prev_a_min_2 != prev_action):
            cycle_count = 0
              
          cycle_counters[ship_k] = (prev_action, prev_a_min_1, cycle_count+1)
          
          ship_halite = player_obs[2][ship_k][1]
          cycle_limit = 12 if ship_halite > 0 else 20
          if cycle_count > cycle_limit:
            # Avoid the action if the action is not a rescue action and the
            # ship is not near a base
            row, col = row_col_from_square_grid_pos(
              player_obs[2][ship_k][0], grid_size)
            near_base = np.any(all_bases[ROW_COL_MAX_DISTANCE_MASKS[
              row, col, 2]])
            if not ship_k in prev_step_rescue_ships and not near_base:
              avoid_cycle_actions[ship_k] = prev_a_min_1
          
    history['ship_action_cycle_counter'] = cycle_counters
    history['avoid_cycle_actions'] = avoid_cycle_actions
    
  if observation['step'] > 0:
    # Update the opponent cycle counters
    prev_env_players = history['prev_step']['env_observation'].players
    for player_id in range(1, num_players):
      cycle_counters = history['opponent_cycle_counters'][player_id-1]
      env_obs_id = env_obs_ids[player_id]
      
      # Update the cycles and according cycle counters for all opponnent ships
      # that are still alive
      player_ships = env_observation.players[env_obs_id][2]
      prev_player_ships = prev_env_players[env_obs_id][2]
      grid_size = observation['halite'].shape[0]
      
      for ship_k in prev_player_ships:
        if not ship_k in player_ships:
          # The ship died or was converted - delete it from the cycle data
          if ship_k in cycle_counters:
            del cycle_counters[ship_k]
        else:
          # If the ship is new: add the action and default counter to the cycle
          # counters dict
          prev_row, prev_col = row_col_from_square_grid_pos(
            prev_player_ships[ship_k][0], grid_size)
          row, col = row_col_from_square_grid_pos(
            player_ships[ship_k][0], grid_size)
          opponent_ship_pos_to_key[(row, col)] = ship_k
          prev_action =  get_dir_from_target(
            prev_row, prev_col, row, col, grid_size)
          if not ship_k in cycle_counters:
            cycle_counters[ship_k] = (prev_action, -1, 0)
          else:
            prev_a_min_1, prev_a_min_2, cycle_count = cycle_counters[ship_k]
            if cycle_count > 0 and (prev_a_min_2 != prev_action):
              cycle_count = 0
                
            cycle_counters[ship_k] = (prev_action, prev_a_min_1, cycle_count+1)
            
      history['opponent_cycle_counters'][player_id-1] = cycle_counters
      history['opponent_ship_pos_to_key'] = opponent_ship_pos_to_key
      
    # Loop over all ships and mark their corresponding squares as empty or
    # cycled
    prev_stacked_bases = history['prev_step']['stacked_bases']
    all_prev_bases = prev_stacked_bases.sum(0) > 0
    base_positions_equal = all_prev_bases == all_bases
    empty_or_cycled_positions = np.copy(base_positions_equal)
    empty_or_extended_cycled_positions = np.copy(base_positions_equal)
    cycled_position_counts = np.copy(base_positions_equal).astype(np.int)*-1
    for player_id in range(num_players):
      env_obs_id = env_obs_ids[player_id]
      player_ships = env_observation.players[env_obs_id][2]
      prev_player_ships = prev_env_players[env_obs_id][2]
      if player_id == 0:
        cycle_counters = history['ship_action_cycle_counter']
      else:
        cycle_counters = history['opponent_cycle_counters'][player_id-1]
        
      for ship_k in player_ships:
        row, col = row_col_from_square_grid_pos(
          player_ships[ship_k][0], grid_size)
        if not ship_k in prev_player_ships:
          empty_or_cycled_positions[row, col] = 0
          empty_or_extended_cycled_positions[row, col] = 0
          cycled_position_counts[row, col] = 0
        else:
          cycle_duration = cycle_counters[ship_k][2]
          ship_in_repeat_cycle = (cycle_duration >= config[
            'surrounding_ships_cycle_extrapolate_step_count']) and ((
              cycle_counters[ship_k][0] != cycle_counters[ship_k][1]) or (
                cycle_counters[ship_k][0] is None))
          ship_in_extended_repeat_cycle = (cycle_duration >= config[
            'surrounding_ships_extended_cycle_extrapolate_step_count']) and ((
              cycle_counters[ship_k][0] != cycle_counters[ship_k][1]) or (
                cycle_counters[ship_k][0] is None))
          
          cycled_position_counts[row, col] = cycle_duration*int(
            ship_in_repeat_cycle)
          empty_or_cycled_positions[row, col] = ship_in_repeat_cycle
          empty_or_extended_cycled_positions[row, col] = (
            ship_in_extended_repeat_cycle)
          
    history['empty_or_cycled_positions'] = empty_or_cycled_positions
    history['empty_or_extended_cycled_positions'] = (
      empty_or_extended_cycled_positions)
    history['cycled_position_counts'] = cycled_position_counts
    
  return history

def update_destoyed_base_ship_count(history, observation, player_obs):
  if observation['step'] == 0:
    history['num_destroyed_bases'] = 0
    history['num_destroyed_ships'] = 0
  else:
    grid_size = observation['rewards_bases_ships'][0][1].shape[0]
    prev_bases = history[
      'prev_step']['observation']['rewards_bases_ships'][0][1]
    current_bases = observation['rewards_bases_ships'][0][1]
    destroyed_base_count_this_step = (prev_bases& (~current_bases )).sum()
    history['num_destroyed_bases'] += destroyed_base_count_this_step
    
    prev_player_obs = history['prev_step']['player_obs']
    current_ship_keys = list(player_obs[2].keys())
    destroyed_ship_count_this_step = 0
    for ship_k in prev_player_obs[2]:
      if not ship_k in current_ship_keys:
        row, col = row_col_from_square_grid_pos(
          prev_player_obs[2][ship_k][0], grid_size)
        if not current_bases[row, col]:
          # import pdb; pdb.set_trace()
          destroyed_ship_count_this_step += 1 
    history['num_destroyed_ships'] += destroyed_ship_count_this_step
    
  return history

def update_returning_to_base_ships(history, observation, player_obs):
  if observation['step'] == 0:
    history['returning_to_base_ships'] = []
    history['temporary_hoarding_collect_ships'] = []
  else:
    current_bases = observation['rewards_bases_ships'][0][1]
    grid_size = current_bases.shape[0]
    delete_keys = []
    delete_temp_hoard_keys = []
    # Temporary collect ships that don't collect should join the hoarding
    for ship_k in history['temporary_hoarding_collect_ships']:
      if not ship_k in player_obs[2]:
        # My ship was destroyed
        delete_temp_hoard_keys.append(ship_k)
      else:
        row, col = row_col_from_square_grid_pos(
          player_obs[2][ship_k][0], grid_size)
        ship_halite = player_obs[2][ship_k][1]
        if ship_halite == 0:
          delete_temp_hoard_keys.append(ship_k)
    for ship_k in history['returning_to_base_ships']:
      if not ship_k in player_obs[2]:
        # My ship was destroyed
        delete_keys.append(ship_k)
      else:
        row, col = row_col_from_square_grid_pos(
          player_obs[2][ship_k][0], grid_size)
        ship_halite = player_obs[2][ship_k][1]
        if ship_halite == 0:
          delete_keys.append(ship_k)
          
    for ship_k in delete_keys:
      history['returning_to_base_ships'].remove(ship_k)
      if ship_k in history['temporary_hoarding_collect_ships'] and ship_k in (
          player_obs[2]):
        # Join the pack hunt if the base has a nearby opponent or if there
        # is little halite to be mined nearby
        row, col = row_col_from_square_grid_pos(
          player_obs[2][ship_k][0], grid_size)
        if history['prev_step']['safe_to_collect_margin'][row, col] < 4 or (
            observation['halite'][ROW_COL_MAX_DISTANCE_MASKS[
              row, col, 5]].sum() < 1000):
          delete_temp_hoard_keys.append(ship_k)
        # else:
        #   print(observation['step'], row, col,
        #         "Not joining the pack hunt quite yet")
    for ship_k in delete_temp_hoard_keys:
      if ship_k in history['temporary_hoarding_collect_ships']:
        history['temporary_hoarding_collect_ships'].remove(ship_k)
    
  return history

def update_early_best_opponent(config, history, observation):
  if observation['step'] == 0:
    history['ballistic_early_best_target_mode'] = False
    history['ballistic_early_best_targets_sorted'] = None
  elif history['ballistic_early_best_targets_sorted'] is None:
    if observation['relative_step'] >= config[
        'early_best_opponent_relative_step']:
      current_scores = history['current_scores']
      # import pdb; pdb.set_trace()
      history['ballistic_early_best_targets_sorted'] = np.argsort(
        -current_scores[1:])
  
  # print(observation['step'], history['ballistic_early_best_targets_sorted'])
  return history
    
def update_initial_collect_boost(history, observation, player_obs):
  if observation['step'] == 0:
    stacked_ships = np.stack([rbs[2] for rbs in observation[
      'rewards_bases_ships']])
    grid_size = stacked_ships.shape[1]
  
    # Stack the collect scores for all my ships
    ship_rows = []
    ship_cols = []
    ship_keys = []
    for ship_k in player_obs[2]:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      ship_rows.append(row)
      ship_cols.append(col)
      ship_keys.append(ship_k)
      
    my_num_ships = len(ship_rows)
    my_ship_positions = (np.array(ship_rows), np.array(ship_cols))
    opponent_ship_positions = np.where(stacked_ships[1:].sum(0) > 0)
    
    # Obtain the nearest distances for my and opponent ships
    my_nearest_ship_distances = 99*np.ones((grid_size, grid_size))
    for ship_id in range(my_num_ships):
      row = my_ship_positions[0][ship_id]
      col = my_ship_positions[1][ship_id]
      my_nearest_ship_distances = np.minimum(
        my_nearest_ship_distances, DISTANCES[row, col])
      
    opponent_nearest_ship_distances = 99*np.ones((grid_size, grid_size))
    opponent_ships = stacked_ships[1:].sum(0) > 0
    num_opponent_ships = opponent_ships.sum()
    for ship_id in range(num_opponent_ships):
      row = opponent_ship_positions[0][ship_id]
      col = opponent_ship_positions[1][ship_id]
      opponent_nearest_ship_distances = np.minimum(
        opponent_nearest_ship_distances, DISTANCES[row, col])
      
    nearest_ship_distance_difference = (
      my_nearest_ship_distances - opponent_nearest_ship_distances)
    
    original_position_multiplier =  (1.5**(np.minimum(
      3, nearest_ship_distance_difference)))**0.5
    history['original_position_multiplier'] = original_position_multiplier
    
  return history

def update_initial_ship_strategy(
    history, observation, player_obs, not_collect_first_ships_near_base=6,
    initial_near_base_collect_excluded_steps=20, initial_exclude_box_size=3):
  grid_size = observation['halite'].shape[0]
  if observation['step'] == 0:
    history['first_base_position'] = None
    history['initial_not_collect_near_base_ships'] = []
    history['initial_not_collect_near_base_mask'] = np.zeros(
      (grid_size, grid_size), dtype=np.bool)
  else:
    if history['first_base_position'] is None and len(player_obs[1]) > 0:
      base_row, base_col = row_col_from_square_grid_pos(
        list(player_obs[1].values())[0], grid_size)
      history['first_base_position'] = (base_row, base_col)
      history['initial_not_collect_near_base_mask'] = (
        ROW_COL_BOX_MAX_DISTANCE_MASKS[
          base_row, base_col, initial_exclude_box_size])
    
    if history['first_base_position'] is not None:
      # The first "not_collect_first_ships_near_base" are added to the excluded
      # near base collect list. They are removed from the list after returning
      # to a base after "initial_near_base_collect_excluded_steps" steps
      
      # Detect new ships
      if len(history['initial_not_collect_near_base_ships']) < (
          not_collect_first_ships_near_base) and observation['step'] <= (
            initial_near_base_collect_excluded_steps):
        current_excluded = set(history['initial_not_collect_near_base_ships'])
        exclusion_budget = not_collect_first_ships_near_base - len(
          current_excluded)
        current_ships = list(player_obs[2].keys())
        new_ships = list(set(current_ships) - current_excluded)
        if len(new_ships) > 0:
          new_excluded = new_ships[:exclusion_budget]
          history['initial_not_collect_near_base_ships'] += new_excluded
          
      elif (
          observation['step'] > initial_near_base_collect_excluded_steps) and (
            len(history['initial_not_collect_near_base_ships']) > 0):
        # Detect ships that have returned to a base
        delete_keys = []
        base_positions = list(player_obs[1].values())
        for ship_k in history['initial_not_collect_near_base_ships']:
          if not ship_k in player_obs[2]:
            # The ship was destroyed
            delete_keys.append(ship_k)
          else:
            ship_position = player_obs[2][ship_k][0]
            if ship_position in base_positions:
              delete_keys.append(ship_k)
        
        for ship_k in delete_keys:
          history['initial_not_collect_near_base_ships'].remove(ship_k)
  
  # print(observation['step'], history['initial_not_collect_near_base_ships'])
  return history

def update_history_start_step(
    config, history, observation, env_observation, env_obs_ids, env_config,
    np_rng):
  history_start_time = time.time()
  
  stacked_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])
  opponent_ships = stacked_ships[1:].sum(0) > 0
  other_halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']])[1:].sum(0)
  other_halite_ships[~opponent_ships] = 1e9
  grid_size = opponent_ships.shape[0]
  player_obs = env_observation.players[env_obs_ids[0]]
  player_ids = -1*np.ones((grid_size, grid_size), dtype=np.int)
  for i in range(stacked_ships.shape[0]):
    player_ids[stacked_ships[i]] = i
  
  if observation['step'] == 0:
    history['hunting_season_started'] = False
    history['hunting_season_standard_ships'] = []
    history['prev_step_boxing_in_ships'] = []
    history['prev_step_hoarded_one_step_opponent_keys'] = []
    history['my_prev_step_base_attacker_ships'] = []
    history['request_increment_num_standard_hunting'] = 0
    history['request_decrement_num_standard_hunting'] = 0
    history['add_strategic_base'] = False
    history['construct_strategic_base_position'] = None
    history['ballistic_mode'] = False
    history['my_time_averaged_vulnerable_score'] = np.zeros(
      (grid_size, grid_size))
    history['early_hunting_season_ended'] = False
    history['targeted_hoard_mode'] = False
    history['limit_ships_timeout'] = False
    history['initial_collect_zero_halite_targets'] = {}
    history['prev_ballistic_target_override'] = None
  
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
    config, history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config)
  
  # Update the data that keeps track of camping behavior
  history = update_base_camping_strategy(
    config, history, observation, env_observation, stacked_ships, env_obs_ids,
    env_config, np_rng)
  
  # Update the data that keeps track of the early game best opponent (targeted
  # at the end if I am winning massively)
  history = update_early_best_opponent(config, history, observation)
  
  # Update the move directions of all opponent ships
  history = update_opponent_ships_move_directions(
    history, observation, env_observation, env_obs_ids)
  
  # Update the counters that keep track of my repetitive actions - avoid
  # cycling in a cycle of max length 2 for more than X steps when I can afford
  # other actions
  history = update_cycle_counters(
    config, history, observation, player_obs, env_observation, env_obs_ids)
  
  # Update the count of my destroyed bases and ships
  history = update_destoyed_base_ship_count(history, observation, player_obs)
  
  # Update returning to base ships
  history = update_returning_to_base_ships(history, observation, player_obs)
    
  # Update the initial collect boost to make sure we focus on halite near the
  # boundary of our influence sphere
  history = update_initial_collect_boost(history, observation, player_obs)
  
  # Update the logic to make sure that my first ships don't gather near my
  # first base in the first N steps
  history = update_initial_ship_strategy(history, observation, player_obs)
  
  return history, (time.time()-history_start_time)

def update_history_end_step(
    history, observation, ship_actions, opponent_ships_sensible_actions,
    opponent_ships_sensible_actions_no_risk, ship_plans, player_obs,
    env_observation, main_base_distances, on_rescue_mission,
    boxed_in_zero_halite_opponents, ships_on_box_mission,
    non_abandoned_base_pos, this_step_base_defense_keys,
    should_spawn_base_next_step, ballistic_attack_base_targets,
    safe_to_collect_margin, get_actions_start_time):
  none_included_ship_actions = {k: (ship_actions[k] if (
    k in ship_actions) else None) for k in player_obs[2]}
  stacked_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])
  stacked_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  halite_ships[stacked_ships.sum(0) == 0] = -1e-9
  grid_size = halite_ships.shape[0]
  if main_base_distances.max() > 0:
    base_zero_dist_locations = np.where(main_base_distances == 0)
    my_main_base_location = (base_zero_dist_locations[0][0],
                             base_zero_dist_locations[1][0])
  else:
    my_main_base_location = (-1, -1)
  ships_on_rescue_mission = []
  rescue_positions = np.where(on_rescue_mission)
  ship_pos_to_key = {v[0]: k for k, v in player_obs[2].items()}
  if np.any(on_rescue_mission):
    for i in range(rescue_positions[0].size):
      position = grid_size*rescue_positions[0][i] + rescue_positions[1][i]
      ships_on_rescue_mission.append(ship_pos_to_key[position])
  
  non_abandoned_base_locations = []
  if non_abandoned_base_pos:
    for base_id in range(non_abandoned_base_pos[0].size):
      non_abandoned_base_locations.append((
        non_abandoned_base_pos[0][base_id],
        non_abandoned_base_pos[1][base_id]))
  
  if 'prev_step' in history:
    prev_base_defense_keys = history['prev_step']['base_defense_keys']
  else:
    prev_base_defense_keys = []
  
  # Reduce the max ship count when there is a risk of timing out (
  # < 6 for local debugging) and no longer perform pack hunting
  get_actions_duration = time.time() - get_actions_start_time
  history['limit_ships_timeout'] = history['limit_ships_timeout'] or (
    get_actions_duration > 2 and get_actions_duration < 6)
  
  history['prev_step'] = {
    'my_ship_actions': none_included_ship_actions,
    'opponent_ships_sensible_actions': opponent_ships_sensible_actions,
    'opponent_ships_sensible_actions_no_risk': (
      opponent_ships_sensible_actions_no_risk),
    'boxed_in_zero_halite_opponents': boxed_in_zero_halite_opponents,
    'ship_plans': ship_plans,
    'env_observation': env_observation,
    'stacked_bases': stacked_bases,
    'stacked_ships': stacked_ships,
    'halite_ships': halite_ships,
    'observation': observation,
    'my_main_base_location': my_main_base_location,
    'non_abandoned_base_locations': non_abandoned_base_locations,
    'ships_on_rescue_mission': ships_on_rescue_mission,
    'ships_on_box_mission': ships_on_box_mission,
    'base_defense_keys': this_step_base_defense_keys,
    'prev_base_defense_keys': prev_base_defense_keys,
    'should_spawn_base_next_step': should_spawn_base_next_step,
    'player_obs': player_obs,
    'ballistic_attack_base_targets': ballistic_attack_base_targets,
    'safe_to_collect_margin': safe_to_collect_margin,
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
  get_actions_start_time = time.time()
  
  # if observation['step'] in [242] and (
  #     observation['rewards_bases_ships'][0][1].sum()) == 1:
  #   import pdb; pdb.set_trace()
  
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
    config, history, observation, env_observation, env_obs_ids, env_config,
    np_rng)
  
  # Compute the ship scores for all high level actions
  (all_ship_scores, opponent_ships_sensible_actions,
   opponent_ships_sensible_actions_no_risk, weighted_base_mask,
   opponent_ships_scaled, main_base_distances, ship_scores_duration,
   halite_ships, player_influence_maps, boxed_in_zero_halite_opponents,
   ignore_convert_positions, ship_diff_smoothed,
   ballistic_attack_base_targets, safe_to_return_halites,
   safe_to_collect_margin, always_attack_opponent_id,
   likely_convert_opponent_positions,
   possible_convert_opponent_positions, my_base_distances,
   nearest_base_distances, history) = get_ship_scores(
    config, observation, player_obs, env_config, np_rng,
    ignore_bad_attack_directions, history, env_obs_ids, env_observation,
    verbose)
     
  # if observation['step'] in [242] and (
  #     observation['rewards_bases_ships'][0][1].sum()) == 1:
  #   import pdb; pdb.set_trace()
  
  # Compute the coordinated high level ship plan
  (ship_plans, my_next_bases, plan_ship_scores, base_attackers,
   box_in_duration, history, ship_plans_duration,
   inner_loop_ship_plans_duration, recompute_ship_plan_order_duration,
   on_rescue_mission, ships_on_box_mission,
   requested_save_conversion_budget, non_abandoned_base_pos,
   this_step_base_defense_keys, should_spawn_base_next_step,
   ship_plans_reordered, victory_formation) = get_ship_plans(
    config, observation, player_obs, env_config, verbose,
    copy.deepcopy(all_ship_scores), np_rng, weighted_base_mask,
    steps_remaining, opponent_ships_sensible_actions, opponent_ships_scaled,
    main_base_distances, history, env_observation, player_influence_maps,
    ignore_convert_positions, ship_diff_smoothed, safe_to_return_halites,
    safe_to_collect_margin, always_attack_opponent_id,
    likely_convert_opponent_positions, possible_convert_opponent_positions,
    my_base_distances, nearest_base_distances)
  
  # Translate the ship high level plans to basic move/convert actions
  (mapped_actions, remaining_budget, my_next_ships, my_next_halite,
   updated_ship_pos, action_overrides,
   ship_map_duration) = map_ship_plans_to_actions(
     config, observation, player_obs, env_observation, env_config, verbose,
     plan_ship_scores, all_ship_scores, ship_plans, np_rng,
     ignore_bad_attack_directions, base_attackers, steps_remaining,
     opponent_ships_sensible_actions, opponent_ships_sensible_actions_no_risk,
     history, env_obs_ids, opponent_ships_scaled, main_base_distances,
     ignore_convert_positions, ballistic_attack_base_targets,
     player_influence_maps)
  ship_actions = copy.copy(mapped_actions)
  
  # Decide for all bases whether to spawn or keep the base available
  base_actions, remaining_budget = decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships,
    my_next_halite, env_config, remaining_budget, verbose, ship_plans,
    updated_ship_pos, weighted_base_mask, history,
    requested_save_conversion_budget, victory_formation)
  
  # Add data to my history so I can update it appropriately at the beginning of
  # the next step.
  history = update_history_end_step(
    history, observation, ship_actions, opponent_ships_sensible_actions,
    opponent_ships_sensible_actions_no_risk, ship_plans, player_obs,
    env_observation, main_base_distances, on_rescue_mission,
    boxed_in_zero_halite_opponents, ships_on_box_mission,
    non_abandoned_base_pos, this_step_base_defense_keys,
    should_spawn_base_next_step, ballistic_attack_base_targets,
    safe_to_collect_margin, get_actions_start_time)
  
  mapped_actions.update(base_actions)
  
  return mapped_actions, history, ship_plans

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
  
  mapped_actions, HISTORY, ship_plans = get_config_actions(
    CONFIG, current_observation, player_obs, observation, env_config, HISTORY,
    rng_action_seed)
     
  if LOCAL_MODE:
    # This is to allow for debugging of the history outside of the agent
    return mapped_actions, copy.deepcopy(HISTORY)
  else:
    print(ship_plans)
    return mapped_actions
