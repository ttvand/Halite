from collections import OrderedDict
import copy
import numpy as np
from scipy import signal


NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
CONVERT = "CONVERT"
SPAWN = "SPAWN"
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
      
    for dist in range(2, 5):
      dist_mask_dim = dist*2+1
      row_pos = np.tile(np.expand_dims(np.arange(dist_mask_dim), 1),
                        [1, dist_mask_dim])
      col_pos = np.tile(np.arange(dist_mask_dim), [dist_mask_dim, 1])
      for direction in MOVE_DIRECTIONS[1:]:
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
    observation, ship_k, my_bases, my_ships, steps_remaining, min_dist=2):
  direction_halite_diff_distance_raw = {
    NORTH: [], SOUTH: [], EAST: [], WEST: []}
  my_bases_or_ships = np.logical_or(my_bases, my_ships)
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
            
        if num_threat_ships_lt > 1 and not ignore_threat:
          lt_catch_prob = {k: [] for k in RELATIVE_DIRECTIONS[:-1]}
          for i in range(num_threat_ships_lt):
            other_row = less_halite_threat_opponents_lt[0][i]
            other_col = less_halite_threat_opponents_lt[1][i]
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
                lt_catch_prob[threat_dir].append((
                  other_dir_abs_offset+dir_offset)*(
                    my_material_defense_multiplier))
            
          # if observation['step'] == 359 and ship_k == '67-1':
          #   import pdb; pdb.set_trace()
                    
          if np.all([len(v) > 0 for v in lt_catch_prob.values()]):
            survive_probs = np.array([
              (np.maximum(1, np.array(lt_catch_prob[k])-1)/np.array(
                lt_catch_prob[k])).prod() for k in lt_catch_prob])
            min_die_prob = 1-survive_probs.max()
            
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
           
  valid_not_preferred_dirs = list(set(
    two_step_bad_directions + n_step_step_bad_directions))
  if valid_non_base_directions and valid_not_preferred_dirs and (
      len(valid_non_base_directions) - len(valid_not_preferred_dirs)) > 0:
    bad_directions.extend(valid_not_preferred_dirs)
    valid_directions = list(
      set(valid_directions) - set(valid_not_preferred_dirs))
              
  # if observation['step'] == 73 and ship_k == '5-1':
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
  for d in MOVE_DIRECTIONS[1:]:
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

def scale_attack_scores_bases(
    config, observation, player_obs, spawn_cost, main_base_distances,
    weighted_base_mask, laplace_smoother_rel_ship_count=4,
    initial_normalize_ship_diff=10, final_normalize_ship_diff=2):
  stacked_bases = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])
  my_bases = stacked_bases[0]
  stacked_opponent_bases = stacked_bases[1:]
  stacked_opponent_ships = np.stack([rbs[2] for rbs in observation[
    'rewards_bases_ships']])[1:]
  ship_halite_per_player = np.stack([rbs[3] for rbs in observation[
    'rewards_bases_ships']]).sum((1, 2))
  scores = np.array([rbs[0] for rbs in observation['rewards_bases_ships']])
  base_counts = stacked_opponent_bases.sum((1, 2))
  my_ship_count = len(player_obs[2])
  ship_counts = stacked_opponent_ships.sum((1, 2))
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
    additive_nearby_main_base = 5/(1.5**main_base_distances)/(
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
  
  return opponent_bases_scaled

def get_influence_map(config, stacked_bases, stacked_ships, halite_ships,
                      observation, player_obs, smooth_kernel_dim=7):
  
  all_ships = stacked_ships.sum(0).astype(np.bool)
  my_ships = stacked_ships[0].astype(np.bool)
  
  if my_ships.sum() == 0:
    return None, None, None
  
  num_players = stacked_ships.shape[0]
  grid_size = my_ships.shape[0]
  ship_range = 1-config['influence_map_min_ship_weight']
  all_ships_halite = halite_ships[all_ships]
  unique_halite_vals = np.sort(np.unique(all_ships_halite)).astype(
    np.int).tolist()
  num_unique = len(unique_halite_vals)
  
  halite_ranks = [np.array(
    [unique_halite_vals.index(hs) for hs in halite_ships[
      stacked_ships[i]]]) for i in range(num_players)]
  ship_weights = [1 - r/(num_unique-1+1e-9)*ship_range for r in halite_ranks]
  
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
  
  return influence_map, priority_scores, ship_priority_weights

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
    
  # Force an emergency return if the best return scores demands an urgent
  # return in order to bring the halite home before the episode is over
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
    halite_ships, steps_remaining, player_obs, np_rng, box_in_window=3):
  # Loop over the opponent ships and derive if I can box them in
  # For now this is just greedy. We should probably consider decoupling finding
  # targets from actually boxing in.
  opponent_positions = np.where(stacked_ships[1:].sum(0) > 0)
  num_opponent_ships = opponent_positions[0].size
  dist_mask_dim = 2*box_in_window+1
  nearby_rows = np.tile(np.expand_dims(np.arange(dist_mask_dim), 1),
                [1, dist_mask_dim])
  nearby_cols = np.tile(np.arange(dist_mask_dim), [dist_mask_dim, 1])
  ships_available = np.copy(stacked_ships[0])
  grid_size = stacked_ships.shape[1]
  ship_pos_to_key = {v[0]: k for k, v in player_obs[2].items()}
  
  for i in range(num_opponent_ships):
    row = opponent_positions[0][i]
    col = opponent_positions[1][i]
    my_less_halite_mask = np.logical_and(
      halite_ships < halite_ships[row, col], ships_available)
    # Only consider zero halite ships towards the end of a game
    my_less_halite_mask = np.logical_and(
      my_less_halite_mask, np.logical_or(
        halite_ships == 0, steps_remaining > 20))
    box_pos = ROW_COL_BOX_MAX_DISTANCE_MASKS[row, col, box_in_window]
    lowest_opponent_halite = halite_ships[
      (~stacked_ships[0]) & (halite_ships >= 0) & box_pos].min()
    my_less_halite_mask &= (halite_ships <= lowest_opponent_halite)
    my_less_halite_mask_box = edge_aware_square_subset_mask(
      my_less_halite_mask, row, col, np.copy(box_in_window), box_pos,
      grid_size)
    nearby_less_halite_mask = my_less_halite_mask_box.reshape(
        (dist_mask_dim, dist_mask_dim))
    my_num_nearby = nearby_less_halite_mask.sum()
    if my_num_nearby >= 3:
      # Check all directions to make sure I can box the opponent in
      can_box_in = True
      box_in_mask_dirs = np.zeros(
        (4, dist_mask_dim, dist_mask_dim), dtype=np.bool)
      for dim_id, d in enumerate(MOVE_DIRECTIONS[1:]):
        dir_and_ships = BOX_DIRECTION_MASKS[(box_in_window, d)] & (
          nearby_less_halite_mask)
        if not np.any(dir_and_ships):
          can_box_in = False
          break
        else:
          box_in_mask_dirs[dim_id] = dir_and_ships
        
      if can_box_in:
        # Sketch out the escape squares for the target ship
        opponent_distances = np.abs(nearby_rows-box_in_window) + np.abs(
          nearby_cols-box_in_window)
        opponent_euclid_distances = np.sqrt(
          (nearby_rows-box_in_window)**2 + (
          nearby_cols-box_in_window)**2)
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
        escape_squares = opponent_distances < my_nearest_distances.min(0)
        opponent_id = np.where(stacked_ships[:, row, col])[0][0]
        if not np.any(observation['rewards_bases_ships'][opponent_id][1][
            box_pos][escape_squares.flatten()]):
          # Let's box the opponent in!
          # We should move towards the opponent if we can do so without opening
          # up an escape direction
          print(observation['step'], row, col)
          
          # Order the planning by priority of direction and distance to the
          # opponent
          box_in_mask_dirs_sum = box_in_mask_dirs.sum((1, 2))
          ship_priorities = np.zeros(my_num_nearby)
          for j in range(my_num_nearby):
            my_row = nearby_mask_pos[0][j]
            my_col = nearby_mask_pos[1][j]
            box_directions = box_in_mask_dirs[:, my_row, my_col]
            opponent_distance = np.abs(my_row-box_in_window) + np.abs(
              my_col-box_in_window)
            ship_priorities[j] = 20/(
              box_in_mask_dirs_sum[box_directions].prod())-opponent_distance
          
          # DISCERN if we are just chasing or actually attacking the ship in
          # the next move - dummy rule to have at least K neighboring ships
          # for us to attack the position of the targeted ship - this makes it
          # hard to guess the escape direction
          ship_target_1_distances = my_nearest_distances[
            :, box_in_window, box_in_window] == 1
          next_step_attack = len(
            opponent_ships_sensible_actions[row, col]) == 0 and (
              ship_target_1_distances.sum() > 2)
              
          pos_taken = np.zeros_like(dir_and_ships)
          if next_step_attack:
            # If there is a ship that can take the position of my attacker:
            # attack with that ship and replace its position.
            # Otherwise pick a random attacker and keep the others in place.
            # Initial approach: don't move with ships at distance 1.
            ship_target_2_distance_ids = np.where(my_nearest_distances[
              :, box_in_window, box_in_window] == 2)[0].tolist()
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
                my_row, my_col, box_in_window, box_in_window, grid_size=1000)
              
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
                    box_in_window, box_in_window]:
                    move_ids_directions_next_attack[two_step_diff_id] = d
                    # Find the ids of the 1-step ship and make sure that ship
                    # attacks
                    replaced_id = np.where(my_nearest_distances[
                      :, move_row, move_col] == 0)[0][0]
                    one_step_attack_dir = get_dir_from_target(
                      move_row, move_col, box_in_window, box_in_window,
                      grid_size=1000)[0]
                    move_ids_directions_next_attack[replaced_id] = (
                      one_step_attack_dir)
                    pos_taken[box_in_window, box_in_window] = True
              
              
            one_step_diff_ids = np.where(ship_target_1_distances)[0]
            if pos_taken[box_in_window, box_in_window]:
              # Add the remaining one step attackers with stay in place actions
              # TODO: add some randomness so that it is harder to escape.
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
                      my_row, my_col, box_in_window, box_in_window,
                      grid_size=1000)[0]
                else:
                  attack_dir = None
                move_ids_directions_next_attack[one_step_diff_id] = attack_dir
              
          covered_directions = np.zeros(4, dtype=bool)
          ship_order = np.argsort(-ship_priorities)
          box_in_mask_rem_dirs_sum = np.copy(box_in_mask_dirs_sum)
          for j in range(my_num_nearby):
            attack_id = ship_order[j]
            my_row = nearby_mask_pos[0][attack_id]
            my_col = nearby_mask_pos[1][attack_id]
            my_abs_row = (row+my_row-box_in_window) % grid_size
            my_abs_col = (col+my_col-box_in_window) % grid_size
            ship_pos = my_abs_row*grid_size+my_abs_col
            ship_k = ship_pos_to_key[ship_pos]
            box_directions = box_in_mask_dirs[:, my_row, my_col]
            opponent_distance = np.abs(my_row-box_in_window) + np.abs(
              my_col-box_in_window)
            box_in_mask_rem_dirs_sum[box_directions] -= 1
            
            # if observation['step'] == 198:
            #   import pdb; pdb.set_trace()
            
            if next_step_attack:
              # Increase the ship scores for the planned actions
              if attack_id in move_ids_directions_next_attack:
                move_dir = move_ids_directions_next_attack[attack_id]
                move_row, move_col = move_ship_row_col(
                  my_abs_row, my_abs_col, move_dir, grid_size)
                # import pdb; pdb.set_trace()
                ship_scores[ship_k][0][move_row, move_col] = 1e10
            else:
              num_covered_attacker = covered_directions[box_directions]
              attack_dir_id = np.argmin(num_covered_attacker + 0.1*(
                box_in_mask_rem_dirs_sum[box_directions]))
              rel_pos_diff = (my_row-box_in_window, my_col-box_in_window)
              attack_move_id = np.where(box_directions)[0][attack_dir_id]
              attack_cover_dir = np.array(MOVE_DIRECTIONS[1:])[attack_move_id]
              one_hot_cover_dirs = np.zeros(4, dtype=bool)
              one_hot_cover_dirs[attack_move_id] = 1
              other_dirs_covered = one_hot_cover_dirs | covered_directions | (
                box_in_mask_rem_dirs_sum >= 1)
              wait_reinforcements = not np.all(other_dirs_covered) or (
                opponent_distance == 1)
              if wait_reinforcements:
                move_dir = None
              else:
                if covered_directions[attack_move_id]:
                  # Move towards the target on the diagonal (empowerment)
                  move_penalties = 0.001*opponent_euclid_distances**4 + (
                    my_nearest_euclid_distances[attack_id]**4) + 1e3*pos_taken
                  move_penalties[my_row, my_col] += 1e3
                  best_penalty_pos = np.where(
                    move_penalties == move_penalties.min())
                  target_move_row = best_penalty_pos[0][0]
                  target_move_col = best_penalty_pos[1][0]
                  move_dir = get_dir_from_target(
                    my_row, my_col, target_move_row, target_move_col,
                    grid_size=1000)[0]
                if attack_cover_dir == NORTH:
                  if rel_pos_diff[1] == 0:
                    move_dir = SOUTH
                  elif rel_pos_diff[1] < 0:
                    move_dir = EAST
                  else:
                    move_dir = WEST
                elif attack_cover_dir == SOUTH:
                  if rel_pos_diff[1] == 0:
                    move_dir = NORTH
                  elif rel_pos_diff[1] < 0:
                    move_dir = EAST
                  else:
                    move_dir = WEST
                elif attack_cover_dir == EAST:
                  if rel_pos_diff[0] == 0:
                    move_dir = WEST
                  elif rel_pos_diff[0] < 0:
                    move_dir = SOUTH
                  else:
                    move_dir = NORTH
                elif attack_cover_dir == WEST:
                  if rel_pos_diff[0] == 0:
                    move_dir = EAST
                  elif rel_pos_diff[0] < 0:
                    move_dir = SOUTH
                  else:
                    move_dir = NORTH
                    
              # Increase the ship scores for the planned actions
              move_row, move_col = move_ship_row_col(
                my_abs_row, my_abs_col, move_dir, grid_size)
              ship_scores[ship_k][0][move_row, move_col] = 1e10
              
              # Update the covered attack directions
              moved_rel_dir = RELATIVE_DIR_MAPPING[move_dir]
              new_rel_pos = (rel_pos_diff[0] + moved_rel_dir[0],
                             rel_pos_diff[1] + moved_rel_dir[1])
              new_grid_pos = (box_in_window + new_rel_pos[0],
                              box_in_window + new_rel_pos[1])
              
              # print(my_abs_row, my_abs_col, move_dir)
              if not new_rel_pos == (0, 0) and not pos_taken[new_grid_pos]:
                pos_taken[new_grid_pos] = 1
                for threat_dir in RELATIVE_DIRECTIONS[:-1]:
                  nz_dim = int(threat_dir[0] == 0)
                  dir_offset = new_rel_pos[nz_dim]*threat_dir[nz_dim]
                  other_dir_abs_offset = np.abs(new_rel_pos[1-nz_dim])
                  
                  if dir_offset > 0 and other_dir_abs_offset < dir_offset:
                    covered_id = np.where(
                      RELATIVE_DIR_TO_DIRECTION_MAPPING[threat_dir] == (
                        np.array(MOVE_DIRECTIONS[1:])))[0][0]
                    covered_directions[covered_id] = 1
          
          # Flag the boxing in ships as unavailable for other hunts
          ships_available[box_pos & my_less_halite_mask] = 0
  
  return ship_scores


def get_ship_scores(config, observation, player_obs, env_config, np_rng,
                    ignore_bad_attack_directions, verbose):
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
  opponent_bases_scaled = scale_attack_scores_bases(
      config, observation, player_obs, spawn_cost, main_base_distances,
      weighted_base_mask)

  # Get the influence map
  influence_map, priority_scores, ship_priority_weights = get_influence_map(
    config, stacked_bases, stacked_ships, halite_ships, observation,
    player_obs)
  
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
                  config['establish_base_deposit_multiplier'])
                  
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
       observation, ship_k, my_bases, my_ships, steps_remaining)
       
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
      n_step_bad_directions_die_probs)
    
  # Coordinate box in actions of opponent more halite ships
  ship_scores = update_scores_opponent_boxing_in(
    ship_scores, stacked_ships, observation, opponent_ships_sensible_actions,
    halite_ships, steps_remaining, player_obs, np_rng)
    
  return ship_scores, opponent_ships_sensible_actions, weighted_base_mask

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
  expected_payoff_conversion = ship_cargo*0.5 + np.sqrt(max(
    0, steps_remaining-20))*remaining_halite*my_ship_fraction
  
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

def protect_last_base(observation, env_config, all_ship_scores, player_obs,
                      defend_override_base_pos, max_considered_attackers=3,
                      halite_on_board_mult=1e-6):
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  defend_base_ignore_collision_key = None
  last_base_protected = True
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
        
    last_base_protected = worst_case_opponent_distances[0] > 0
    
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
          last_base_protected, ignore_base_collision_ship_keys)

def update_occupied_count(row, col, occupied_target_squares,
                          occupied_squares_count):
  k = (row, col)
  occupied_target_squares.append(k)
  if k in occupied_squares_count:
    occupied_squares_count[k] += 1
  else:
    occupied_squares_count[k] = 1

def get_ship_plans(config, observation, player_obs, env_config, verbose,
                   all_ship_scores, np_rng, weighted_base_mask,
                   steps_remaining, convert_first_ship_on_None_action=True):
  my_bases = observation['rewards_bases_ships'][0][1]
  opponent_bases = np.stack(
    [rbs[1] for rbs in observation['rewards_bases_ships'][1:]]).sum(0) > 0
  can_deposit_halite = my_bases.sum() > 0
  all_ships = np.stack(
    [rbs[2] for rbs in observation['rewards_bases_ships']]).sum(0) > 0
  grid_size = observation['halite'].shape[0]
  ship_ids = list(player_obs[2])
  my_ship_count = len(player_obs[2])
  my_non_converted_ship_count = my_ship_count
  convert_cost = env_config.convertCost
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
     last_base_protected, ignore_base_collision_ship_keys) = protect_last_base(
       observation, env_config, all_ship_scores, player_obs,
       defend_override_base_pos)
  else:
    last_base_protected = True
    
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
      
      # if observation['step'] == 398 and ship_k == '74-1':
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
      
      # if observation['step'] == 398 and ship_k in ['74-1']:
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
                         ship_scores[12], ship_scores[13], ship_scores[14])
      
      best_collect_score = ship_scores[0].max()
      best_return_score = ship_scores[1].max()
      best_establish_score = ship_scores[2].max()
      best_attack_base_score = ship_scores[3].max()
      
      # if observation['step'] == 47 and ship_k == '38-1':
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
                                not last_base_protected),
                              row, col)
        base_distance = grid_distance(target_row, target_col, row, col,
                                      grid_size)
        
        if not last_base_protected:
          last_base_protected = base_distance==0
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
            if square_taken[0]:
              print(observation['step'], row, col)
            else:
              chain_conflict_resolution.append((
                considered_squares[0], considered_squares[1]))
              chain_conflict_resolution.append((
                considered_squares[1], considered_squares[0]))
              
      if deterministic_next_pos is not None:
        det_stack = [deterministic_next_pos]
        while det_stack:
          det_pos = det_stack.pop()
          single_path_squares[det_pos] = 1
          chained_pos = []
          del_pairs = []
          for sq1, sq2 in chain_conflict_resolution:
            if det_pos == sq1:
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
        # TODO: check that the priorities only go down
        priority_change = updated_best_score - best_ship_scores[ship_k_future]
        assert priority_change <= 0 
        ship_priority_scores[order_id] += priority_change
        
        all_ship_scores[ship_k_future] = future_ship_scores
    
    # Update the ship order - this works since priorities can only be lowered
    # and we only consider future ships when downgrading priorities
    # TODO: does this always work??? - Make sure no ships get skipped by a +1
    # hack
    ship_priority_scores[ship_order[:(i+1)]] += 1
    ship_order = np.argsort(-ship_priority_scores)
    
  # if observation['step'] == 269:
  #   import pdb; pdb.set_trace()
      
  return ship_plans, my_bases, all_ship_scores, base_attackers

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
    config, observation, player_obs, env_config, verbose, ship_scores,
    before_plan_ship_scores, ship_plans, np_rng, ignore_bad_attack_directions,
    base_attackers, steps_remaining, opponent_ships_sensible_actions):
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
  my_ship_density = smooth2d(np.logical_and(
    stacked_ships[0], halite_ships > 0), smooth_kernel_dim=3)
  
  # if observation['step'] >= 324:
  #   import pdb; pdb.set_trace()
  #   x=1
  
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
  try:
    ordered_ship_plans = [ship_key_plans[o] for o in ship_order]
  except:
    # !!! TODO: fix the rare bug of skipping ships during planning !!!
    # This may have been fixed already but I am only 99.8% confident
    import pdb; pdb.set_trace()
  
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
        
    # if observation['step'] == 398 and ship_k in ['3-1']:
    #   import pdb; pdb.set_trace()
    #   x=1
        
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
            considered_densities[a_id] = my_ship_density[move_row, move_col]
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
            # print(observation['step'], row, col, self_escape_actions)
            if ship_scores[ship_k][7]:
              # Filter out 2-step bad actions if that leaves us with options
              # import pdb; pdb.set_trace()
              self_escape_actions_not_2_step_bad = list(
                set(self_escape_actions) - set(ship_scores[ship_k][7]))
              if self_escape_actions_not_2_step_bad:
                self_escape_actions = self_escape_actions_not_2_step_bad
                
            # Filter out n-step bad actions if that leaves us with options
            if ship_scores[ship_k][8]:
              self_escape_actions_not_n_step_bad = list(
                set(self_escape_actions) - set(ship_scores[ship_k][8]))
              if self_escape_actions_not_n_step_bad:
                # import pdb; pdb.set_trace()
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
        # if observation['step'] == 258 and True:
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
          # Pick a random, not bad moving action
          shuffled_actions = np_rng.permutation(MOVE_DIRECTIONS[1:])
          found_non_bad = False
          # print("Random escape", observation['step'], row, col)
          for a in shuffled_actions:
            move_row, move_col = move_ship_row_col(row, col, a, grid_size)
            if not bad_positions[move_row, move_col]:
              action = str(a)
              found_non_bad = True
              break
          
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
            
      # Place ships that only have a single non destruct action to the front of
      # the queue.
      if rearrange_self_destruct_ships:
        remaining_ships = [s for s in ordered_ship_plans[(i+1):] if (
          s not in rearrange_self_destruct_ships)]
        
        ordered_ship_plans = ordered_ship_plans[:(i+1)] + (
          rearrange_self_destruct_ships) + remaining_ships
  
  return (ship_actions, remaining_budget, my_next_ships, obs_halite,
          updated_ship_pos, -np.diff(action_overrides))

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
    
  # Add the observation step to the seed so we are less predictable
  step_seed = int(rng_action_seed+observation['step'])
  return np.random.RandomState(step_seed)

def get_config_actions(config, observation, player_obs, env_config,
                       rng_action_seed, verbose=False):
  # Set the random seed
  np_rng = get_numpy_random_generator(
    config, observation, rng_action_seed, print_seed=True)
  
  # Decide how many ships I can have attack bases aggressively
  steps_remaining = env_config.episodeSteps-1-observation['step']
  max_aggressive_attackers = int(len(player_obs[2]) - (3+0.25*steps_remaining))
  ignore_bad_attack_directions = max_aggressive_attackers > 0
  
  # Compute the ship scores for all high level actions
  (ship_scores, opponent_ships_sensible_actions,
   weighted_base_mask) = get_ship_scores(
    config, observation, player_obs, env_config, np_rng,
    ignore_bad_attack_directions, verbose)
  
  # Compute the coordinated high level ship plan
  ship_plans, my_next_bases, plan_ship_scores, base_attackers = get_ship_plans(
    config, observation, player_obs, env_config, verbose,
    copy.deepcopy(ship_scores), np_rng, weighted_base_mask, steps_remaining)
  
  # Translate the ship high level plans to basic move/convert actions
  (mapped_actions, remaining_budget, my_next_ships, my_next_halite,
   updated_ship_pos, action_overrides) = map_ship_plans_to_actions(
     config, observation, player_obs, env_config, verbose, plan_ship_scores,
     ship_scores, ship_plans, np_rng, ignore_bad_attack_directions,
     base_attackers, steps_remaining, opponent_ships_sensible_actions)
  
  # Decide for all bases whether to spawn or keep the base available
  base_actions, remaining_budget = decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships,
    my_next_halite, env_config, remaining_budget, verbose, ship_plans,
    updated_ship_pos, weighted_base_mask)
  
  mapped_actions.update(base_actions)
  halite_spent = player_obs[0]-remaining_budget
  
  step_details = {
    'ship_scores': ship_scores,
    'plan_ship_scores': plan_ship_scores,
    'ship_plans': ship_plans,
    'mapped_actions': mapped_actions,
    'observation': observation,
    'player_obs': player_obs,
    'action_overrides': action_overrides,
    }
  
  # if observation['step'] == 398:
  #   import pdb; pdb.set_trace()
  
  return mapped_actions, halite_spent, step_details