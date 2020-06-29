import numpy as np
from scipy import signal


CONFIG = {
  'halite_config_setting_divisor': 1.0,
  'min_spawns_after_conversions': 0,
  'collect_smoothed_multiplier': 0.0,
  'collect_actual_multiplier': 4.0,
  'return_base_multiplier': 5.0,
  
  'establish_base_smoothed_multiplier': 0.5,
  'establish_base_deposit_multiplier': 1.0,
  'collect_run_enemy_multiplier': 5.0,
  'return_base_run_enemy_multiplier': 1.0,
  'establish_base_run_enemy_multiplier': 4.0,
  
  'collect_catch_enemy_multiplier': 0.5,
  'return_base_catch_enemy_multiplier': 0.5,
  'establish_base_catch_enemy_multiplier': 4.0,
  'ignore_catch_prob': 0.5,
  'max_ships': 20,
  
  'max_spawns_per_step': 2,
  'nearby_ship_halite_spawn_constant': 1.0,
  'nearby_halite_spawn_constant': 50.0,
  'remaining_budget_spawn_constant': 0.1,
  'spawn_score_threshold': 30.0,
  
  'max_spawn_relative_step_divisor': 30,
  }


NORTH = "NORTH"
SOUTH = "SOUTH"
EAST = "EAST"
WEST = "WEST"
CONVERT = "CONVERT"
SHIP_NONE = "SHIP_NONE"
GO_NEAREST_BASE = "GO_NEAREST_BASE"
SPAWN = "SPAWN"

MOVE_DIRECTIONS = [None, NORTH, SOUTH, EAST, WEST]

HALITE_MULTIPLIER_CONFIG_ENTRIES = [
  # V1
  "ship_halite_cargo_conversion_bonus_constant",
  "friendly_ship_halite_conversion_constant",
  "nearby_halite_conversion_constant",
  
  "halite_collect_constant",
  "nearby_halite_move_constant",
  "nearby_onto_halite_move_constant",
  "nearby_base_move_constant",
  "nearby_move_onto_base_constant",
  "adjacent_opponent_ships_move_constant",
  
  "nearby_ship_halite_spawn_constant",
  "nearby_halite_spawn_constant",
          ]

DISTANCE_MASKS = {}
HALF_PLANES_CATCH = {}
HALF_PLANES_RUN = {}
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
    
    for d in MOVE_DIRECTIONS[1:]:
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
  
def row_col_from_square_grid_pos(pos, size):
  col = pos % size
  row = pos // size
  
  return row, col
    
def update_scores_enemy_ships(
    config, collect_grid_scores, return_to_base_scores, establish_base_scores,
    opponent_ships, halite_ships, row, col, grid_size, spawn_cost, min_dist=2):
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
  bad_directions = []
  ignore_catch = np.random.uniform() < config['ignore_catch_prob']
  for direction, halite_diff_dist in direction_halite_diff_distance.items():
    if halite_diff_dist is not None:
      halite_diff = halite_diff_dist[0]
      if halite_diff >= 0:
        # I should avoid a collision
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = HALF_PLANES_RUN[(row, col)][direction]
        if halite_diff_dist[1] == 1:
          mask_collect_return[row, col] = True
        collect_grid_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
          config['collect_run_enemy_multiplier'])*distance_multiplier
        return_to_base_scores -= mask_collect_return*(ship_halite+spawn_cost)*(
          config['return_base_run_enemy_multiplier'])*distance_multiplier
        mask_establish = np.copy(mask_collect_return)
        mask_establish[row, col] = False
        establish_base_scores -= mask_establish*(ship_halite+spawn_cost)*(
          config['establish_base_run_enemy_multiplier'])*distance_multiplier
        
        bad_directions.append(direction)
      elif halite_diff < 0 and not ignore_catch:
        # I would like a collision unless if there is another opponent ship
        # chasing me - risk avoiding policy for now: if there is at least
        # one ship in a direction that has less halite, I should avoid it
        distance_multiplier = 1/halite_diff_dist[1]
        mask_collect_return = HALF_PLANES_CATCH[(row, col)][direction]
        collect_grid_scores -= mask_collect_return*(halite_diff+spawn_cost)*(
          config['collect_catch_enemy_multiplier'])*distance_multiplier
        return_to_base_scores -= mask_collect_return*(halite_diff+spawn_cost)*(
          config['return_base_catch_enemy_multiplier'])*distance_multiplier
        mask_establish = np.copy(mask_collect_return)
        mask_establish[row, col] = False
        establish_base_scores -= mask_establish*(halite_diff+spawn_cost)*(
          config['establish_base_catch_enemy_multiplier'])*distance_multiplier
        
        preferred_directions.append(direction)
        
  return (collect_grid_scores, return_to_base_scores, establish_base_scores,
          preferred_directions, len(bad_directions) == 4)


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


def get_ship_scores(config, observation, player_obs, env_config, verbose):
  convert_cost = env_config.convertCost
  spawn_cost = env_config.spawnCost
  my_bases = observation['rewards_bases_ships'][0][1]
  obs_halite = np.maximum(0, observation['halite'])

  # Override the maximum number of conversions on the last episode turn
  last_episode_turn = observation['relative_step'] == 1

  grid_size = obs_halite.shape[0]
  smoothed_halite = smooth2d(obs_halite)
  can_deposit_halite = my_bases.sum() > 0
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  
  ship_scores = {}
  for i, ship_k in enumerate(player_obs[2]):
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    dm = DISTANCE_MASKS[(row, col)]
    ship_halite = player_obs[2][ship_k][1]
    
    # Scores 1: collecting halite at row, col
    # Multiply the smoothed halite, added with the obs_halite with a distance
    # mask, specific for the current row and column
    collect_grid_scores = dm*(
      smoothed_halite * config['collect_smoothed_multiplier'] + 
      obs_halite * config['collect_actual_multiplier'])
    
    # Scores 2: returning to any of my bases
    return_to_base_scores = my_bases*dm*ship_halite*(
      config['return_base_multiplier'])
    
    # Scores 3: establish a new base
    establish_base_scores = dm*(
      smoothed_halite-obs_halite) * config[
        'establish_base_smoothed_multiplier']*(1-((my_bases*dm).max()))*(
          1-my_bases) - (convert_cost*can_deposit_halite) + ship_halite*(
            config['establish_base_deposit_multiplier'])
            
    # Update the scores as a function of nearby enemy ships to avoid collisions
    # with opposing ships that carry less halite and promote collisions with
    # enemy ships that carry less halite
    (collect_grid_scores, return_to_base_scores, establish_base_scores,
     preferred_directions, agent_surrounded) = update_scores_enemy_ships(
       config, collect_grid_scores, return_to_base_scores,
       establish_base_scores, opponent_ships, halite_ships, row, col,
       grid_size, spawn_cost)
            
    if last_episode_turn:
      establish_base_scores[row, col] = 1e9*(ship_halite > convert_cost)
        
    ship_scores[ship_k] = (collect_grid_scores, return_to_base_scores,
                           establish_base_scores, preferred_directions,
                           agent_surrounded)
    
  return ship_scores

def get_ship_plans(config, observation, player_obs, env_config, verbose,
                   all_ship_scores, convert_first_ship_on_None_action=True):
  my_bases = observation['rewards_bases_ships'][0][1]
  can_deposit_halite = my_bases.sum() > 0
  obs_halite = np.maximum(0, observation['halite'])
  grid_size = obs_halite.shape[0]
  ship_ids = list(player_obs[2])
  num_ships = len(player_obs[2])
  convert_cost = env_config.convertCost
  num_bases = my_bases.sum()
  new_bases = []
  
  # First, process the convert actions
  ship_plans = {}
  for i, ship_k in enumerate(player_obs[2]):
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    
    ship_scores = all_ship_scores[ship_k]
    ship_halite = player_obs[2][ship_k][1]
    convert_surrounded_ship = ship_scores[4] and (
      ship_halite >= (convert_cost/2))
    if ship_scores[2].max() >= max(ship_scores[0].max()*can_deposit_halite,
                                   ship_scores[1].max()) or (
                                     convert_surrounded_ship):
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
        ship_plans[ship_k] = (target_row, target_col, ship_scores[3])
        
  # Next, do another pass to coordinate the target squares. This is done in a
  # single pass for now where the selection order is determined based on the 
  # initial best score
  ship_best_scores = np.zeros(num_ships)
  for i, ship_k in enumerate(player_obs[2]):
    ship_scores = all_ship_scores[ship_k]
    for (r, c) in new_bases:
      ship_scores[0][r, c] = 0
      ship_scores[2][r, c] = 0
    all_ship_scores[ship_k] = ship_scores
    
    ship_best_scores[i] = np.stack([
      ship_scores[0], ship_scores[1], ship_scores[2]]).max()
    
  ship_order = np.argsort(-ship_best_scores)
  occupied_target_squares = []
  for i in range(num_ships):
    ship_k = ship_ids[ship_order[i]]
    if not ship_k in ship_plans:
      row, col = row_col_from_square_grid_pos(
        player_obs[2][ship_k][0], grid_size)
      ship_scores = all_ship_scores[ship_k]
      
      for (r, c) in occupied_target_squares:
        ship_scores[0][r, c] = 0
        ship_scores[2][r, c] = 0
      
      best_collect_score = ship_scores[0].max()
      best_return_score = ship_scores[1].max()
      best_establish_score = ship_scores[2].max()
      
      if best_collect_score >= max(best_return_score, best_establish_score):
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
          ship_plans[ship_k] = (target_row, target_col, ship_scores[3])
          occupied_target_squares.append((target_row, target_col))
      elif best_return_score >= best_establish_score:
        # Return base mode
        # TODO: pick a new established base if that is closer
        target_return = np.where(ship_scores[1] == ship_scores[1].max())
        target_row = target_return[0][0]
        target_col = target_return[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[3])
      else:
        # Establish base mode
        target_base = np.where(ship_scores[2] == ship_scores[2].max())
        target_row = target_base[0][0]
        target_col = target_base[1][0]
        ship_plans[ship_k] = (target_row, target_col, ship_scores[3])
        occupied_target_squares.append((target_row, target_col))
      
  
  return ship_plans, my_bases

def get_dir_from_target(row, col, target_row, target_col, grid_size):
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
      shortest_dir = EAST if (target_col - col) <= half_grid else (
        WEST)
    else:
      shortest_dir = WEST if (col - target_col) <= half_grid else (
        EAST)
    shortest_directions.append(shortest_dir)
  if vert_distance > 0:
    if target_row > row:
      shortest_dir = SOUTH if (target_row - row) <= half_grid else (
        NORTH)
    else:
      shortest_dir = NORTH if (row - target_row) <= half_grid else (
        SOUTH)
    shortest_directions.append(shortest_dir)
    
  return shortest_directions

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

def map_ship_plans_to_actions(config, observation, player_obs, env_config,
                              verbose, ship_scores, ship_plans):
  ship_actions = {}
  remaining_budget = player_obs[0]
  convert_cost = env_config.convertCost
  obs_halite = np.maximum(0, observation['halite'])
  grid_size = obs_halite.shape[0]
  my_next_ships = np.zeros((grid_size, grid_size), dtype=np.bool)
  
  # List all positions you definitely don't want to move to. Initially this
  # only contains enemy bases and eventually also earlier ships.
  bad_positions = np.stack([rbs[1] for rbs in observation[
    'rewards_bases_ships']])[1:].sum(0)
  
  for ship_i, ship_k in enumerate(ship_plans):
    row, col = row_col_from_square_grid_pos(
      player_obs[2][ship_k][0], grid_size)
    if isinstance(ship_plans[ship_k], str):
      ship_actions[ship_k] = ship_plans[ship_k]
      obs_halite[row, col] = 0
      remaining_budget -= convert_cost
    else:
      target_row, target_col, preferred_directions = ship_plans[ship_k]
      shortest_actions = get_dir_from_target(row, col, target_row, target_col,
                                          grid_size)
      
      # Filter out bad positions from the shortest actions
      valid_actions = []
      for a in shortest_actions:
        move_row, move_col = move_ship_row_col(
          row, col, a, grid_size)
        if not bad_positions[move_row, move_col]:
          valid_actions.append(a)
      
      if valid_actions:
        if preferred_directions:
          # TODO: figure out if this is actually helpful (it makes the agent
          # very predictable)
          intersect_directions = list(set(valid_actions) & set(
            preferred_directions))
          if intersect_directions:
            valid_actions = intersect_directions
        action = str(np.random.choice(valid_actions))
      else:
        action = None
        if bad_positions[row, col]:
          # Pick a random, not bad action
          shuffled_actions = np.random.permutation(MOVE_DIRECTIONS[1:])
          for a in shuffled_actions:
            move_row, move_col = move_ship_row_col(
              row, col, a, grid_size)
            if not bad_positions[move_row, move_col]:
              action = str(a)
              break
      
      # Update my_next_ships
      new_row, new_col = move_ship_row_col(
        row, col, action, grid_size)
      my_next_ships[new_row, new_col] = 1
      bad_positions[new_row, new_col] = 1
      if action is not None:
        ship_actions[ship_k] = action
  
  return ship_actions, remaining_budget, my_next_ships, obs_halite

def decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships, obs_halite,
    env_config, remaining_budget, verbose):

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
    obs_halite.sum()/min(total_ship_count+1e-9, (num_ships+1e-9)*2)/spawn_cost*(
      1-relative_step)*398/config['max_spawn_relative_step_divisor']))
  last_episode_turn = observation['relative_step'] == 1

  if max_spawns <= 0 or not player_obs[1] or last_episode_turn:
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
  # Compute the ship scores for all high level actions
  ship_scores = get_ship_scores(config, observation, player_obs, env_config,
                                verbose)
  
  # Compute the coordinated high level ship plan
  ship_plans, my_next_bases = get_ship_plans(
    config, observation, player_obs, env_config, verbose, ship_scores)
  
  # Translate the ship high level plans to basic move/convert actions
  mapped_actions, remaining_budget, my_next_ships, my_next_halite = (
    map_ship_plans_to_actions(config, observation, player_obs, env_config,
                              verbose, ship_scores, ship_plans))
  
  # Decide for all bases whether to spawn or keep the base available
  base_actions, remaining_budget = decide_existing_base_spawns(
    config, observation, player_obs, my_next_bases, my_next_ships,
    my_next_halite, env_config, remaining_budget, verbose)
  
  mapped_actions.update(base_actions)
  halite_spent = player_obs[0]-remaining_budget
  
  return mapped_actions, halite_spent

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
    }

###############################################################################
for k in CONFIG:
  if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
    CONFIG[k] /= CONFIG['halite_config_setting_divisor']

def my_agent(observation, env_config):
  active_id = observation.player
  current_observation = structured_env_obs(env_config, observation, active_id)
  player_obs = observation.players[active_id]
  
  mapped_actions, _ = get_config_actions(
    CONFIG, current_observation, player_obs, env_config)
     
  return mapped_actions
