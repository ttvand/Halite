import json
import numpy as np
import os
import pandas as pd
import rule_utils
import utils

my_submission = [16335046, 16466155][1]
max_analysed = 5

initial_config = {
  'halite_config_setting_divisor': 1.0,
  'min_spawns_after_conversions': 1,
  'collect_smoothed_multiplier': 0.1,
  'collect_actual_multiplier': 9.0,
  'collect_less_halite_ships_multiplier_base': 0.8,

  'collect_base_nearest_distance_exponent': 0.3,
  'return_base_multiplier': 8.0,
  'return_base_less_halite_ships_multiplier_base': 0.9,
  'early_game_return_base_additional_multiplier': 1.0,
  'early_game_return_boost_step': 120,

  'end_game_return_base_additional_multiplier': 5.0,
  'establish_base_smoothed_multiplier': 0.0,
  'establish_first_base_smoothed_multiplier_correction': 1.5,
  'establish_base_deposit_multiplier': 0.8,
  'establish_base_less_halite_ships_multiplier_base': 1.0,
  
  'attack_base_multiplier': 200.0,
  'attack_base_less_halite_ships_multiplier_base': 0.9,
  'attack_base_halite_sum_multiplier': 1.0,
  'attack_base_run_enemy_multiplier': 1.0,
  'attack_base_catch_enemy_multiplier': 1.0,

  'collect_run_enemy_multiplier': 10.0,
  'return_base_run_enemy_multiplier': 2.0,
  'establish_base_run_enemy_multiplier': 2.5,
  'collect_catch_enemy_multiplier': 0.5,
  'return_base_catch_enemy_multiplier': 1.0,
  
  'establish_base_catch_enemy_multiplier': 0.5,
  'two_step_avoid_boxed_enemy_multiplier_base': 0.85,
  'n_step_avoid_boxed_enemy_multiplier_base': 0.5,
  'ignore_catch_prob': 0.5,
  'max_ships': 20,
  
  'max_spawns_per_step': 3,
  'nearby_ship_halite_spawn_constant': 2.0,
  'nearby_halite_spawn_constant': 10.0,
  'remaining_budget_spawn_constant': 0.2,
  'spawn_score_threshold': 50.0,
  'max_spawn_relative_step_divisor': 100.0,
    }

this_folder = os.path.dirname(__file__)
replay_folder = os.path.join(this_folder, '../Rule agents/Leaderboard replays/')
data_folder = os.path.join(replay_folder, str(my_submission))
json_files = np.sort([f for f in os.listdir(data_folder) if f[-4:] == "json"])
json_files = json_files[-max_analysed:]
num_replays = len(json_files)
json_paths = [os.path.join(data_folder, f) for f in json_files]
print(json_files)

json_data = []
for p in json_paths:
  with open(p) as f:
    raw_data = json.load(f)
    json_data.append(json.loads(raw_data))

stable_opponents_folder = os.path.join(
  this_folder, '../Rule agents/Stable opponents pool')
agent_files = [f for f in os.listdir(stable_opponents_folder) if (
      f[-3:] == '.py')]
    
# Return true if the ship at the current position is boxed in by ships with <=
# halite on board
def is_boxed_ship_loss(pos, observation, min_dist=2, grid_size=21):
  opponent_ships = np.stack([
    rbs[2] for rbs in observation['rewards_bases_ships'][1:]]).sum(0)
  halite_ships = np.stack([
    rbs[3] for rbs in observation['rewards_bases_ships']]).sum(0)
  direction_halite_diff_distance={
    utils.NORTH: None,
    utils.SOUTH: None,
    utils.EAST: None,
    utils.WEST: None,
    }
  row, col = utils.row_col_from_square_grid_pos(pos, grid_size)
  for row_shift in range(-min_dist, min_dist+1):
    considered_row = (row + row_shift) % grid_size
    for col_shift in range(-min_dist, min_dist+1):
      considered_col = (col + col_shift) % grid_size
      distance = np.abs(row_shift) + np.abs(col_shift)
      if distance <= min_dist:
        if opponent_ships[considered_row, considered_col]:
          relevant_dirs = []
          relevant_dirs += [] if row_shift >= 0 else [utils.NORTH]
          relevant_dirs += [] if row_shift <= 0 else [utils.SOUTH]
          relevant_dirs += [] if col_shift <= 0 else [utils.EAST]
          relevant_dirs += [] if col_shift >= 0 else [utils.WEST]
          
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
                
  bad_directions = []
  for direction, halite_diff_dist in direction_halite_diff_distance.items():
    if halite_diff_dist is not None:
      halite_diff = halite_diff_dist[0]
      if halite_diff >= 0:
        # I should avoid a collision
        if halite_diff_dist[1] == 1:
          if not None in bad_directions:
            bad_directions.append(None)
        bad_directions.append(direction)
  
  return len(bad_directions) == 5

def get_pos_actions_ships(obs, actions, grid_size=21):
  data = []
  for o in obs:
    row, col = utils.row_col_from_square_grid_pos(obs[o][0], grid_size)
    action = actions[o] if o in actions else 'None'
    data.append({
      'row': row,
      'col': col,
      'key': o,
      'pos': obs[o][0],
      'action': action})
  
  df = pd.DataFrame(data)
  if df.shape[0] > 0:
    df = df.sort_values(["row", "col"])
  
  return df

def get_ship_action(k, actions):
  return actions[k] if k in actions else None

def base_at_collision_pos(prev_pos, ship_action, prev_step, step,
                          grid_size=21):
  row, col = utils.row_col_from_square_grid_pos(prev_pos, grid_size)
  new_row, new_col = rule_utils.move_ship_row_col(
    row, col, ship_action, grid_size)
  new_pos = new_row*grid_size+new_col
  for i in range(len(step)):
    for other_ship in step[i]['action']:
      if step[i]['action'][other_ship] == "CONVERT":
        if prev_step[0]['observation']['players'][i][2][other_ship][0] == (
            new_pos):
          # We ran into a new established shipyard!
          return True
  
  return False

def ship_loss_count_counterfact(actions, prev_units, obs, grid_size=21):
  # Approximately compute how many ships I would have lost at a certain
  # transition with some actions. Approximate because we assume there are
  # no 3-color ship collisions.
  
  # Compute the new position of all ships and bases, ignoring base spawns
  my_bases = np.zeros((grid_size, grid_size), dtype=np.bool)
  my_ships = np.zeros((grid_size, grid_size), dtype=np.bool)
  prev_bases = prev_units[1]
  prev_ships = prev_units[2]
  ship_loss = 0
  for b in prev_bases:
    row, col = utils.row_col_from_square_grid_pos(prev_bases[b], grid_size)
    my_bases[row, col] = True
    
  for ship_k in prev_ships:
    row, col = utils.row_col_from_square_grid_pos(prev_ships[ship_k][0],
                                                  grid_size)
    a = get_ship_action(ship_k, actions)
    if a == "CONVERT":
      my_bases[row, col] = True
    else:
      new_row, new_col = rule_utils.move_ship_row_col(row, col, a, grid_size)
      if my_ships[new_row, new_col]:
        ship_loss += 1
      else:
        my_ships[new_row, new_col] = 1
        ship_halite = prev_ships[ship_k][1]
        for o in obs['rewards_bases_ships'][1:]:
          if o[1][new_row, new_col]:
            ship_loss += 1
            break
          elif o[2][new_row, new_col] and (
              o[3][new_row, new_col] <= ship_halite):
            ship_loss += 1
            break
  
  return ship_loss
  
    
# Count the number of lost ships in each game
# A ship loss is defined as no longer having a ship with the same key in
# subsequent steps and not having a base in place at the original ship position
def get_game_ship_base_loss_count(replay, player_id, game_agent,
                                  process_each_step):
  num_steps = len(replay['steps'])
  prev_units_obs = replay['steps'][0][0]['observation']['players'][player_id]
  destroyed_conversions = 0
  boxed_ship_loss = 0
  shipyard_collision_losses = 0
  ship_loss = 0
  base_loss = 0
  ship_non_boxed_loss_counterfactual = 0
  all_counterfactual_ship_loss = 0
  prev_obs = None
  prev_env_observation = None
  env_configuration = utils.dotdict(replay['configuration'])
  for i in range(num_steps-1):
    current_units_obs = replay['steps'][i+1][0]['observation']['players'][
      player_id]
    env_observation = utils.dotdict(replay['steps'][i+1][0]['observation'])
    env_observation['step'] = i+1
    env_observation.player = player_id
    obs = utils.structured_env_obs(env_configuration, env_observation,
                                   player_id)
    
    prev_actions = replay['steps'][i+1][player_id]['action']
    for k in prev_units_obs[2]:
      if not k in current_units_obs[2]:
        # pos_actions_ships = get_pos_actions_ships(prev_units_obs[2], prev_actions)
        prev_pos = prev_units_obs[2][k][0]
        if not prev_pos in current_units_obs[1].values():
          ship_action = get_ship_action(k, prev_actions)
          if ship_action == "CONVERT":
            destroyed_conversions += 1 
          else:
            boxed_ship_loss += int(is_boxed_ship_loss(prev_pos, prev_obs))
            ship_loss += 1
            if not boxed_ship_loss:
              if ship_action is not None and base_at_collision_pos(
                  prev_pos, ship_action, replay['steps'][i],
                  replay['steps'][i+1]):
                shipyard_collision_losses += 1
              else:
                mapped_actions, _, step_details = (
                  rule_utils.get_config_or_callable_actions(
                    game_agent, prev_obs, prev_units_obs,
                    prev_env_observation, env_configuration))
                # if prev_obs['step'] == 281:
                #   import pdb; pdb.set_trace()
                
                ship_non_boxed_loss_counterfactual += (
                  ship_loss_count_counterfact(mapped_actions, prev_units_obs,
                                              obs))
    
    if process_each_step and prev_obs is not None:
      mapped_actions, _, step_details = (
        rule_utils.get_config_or_callable_actions(
          game_agent, prev_obs, prev_units_obs, prev_env_observation,
          env_configuration))
      # if prev_obs['step'] == 204:
      #   print(mapped_actions)
      #   import pdb; pdb.set_trace()
      
      all_counterfactual_ship_loss += (
        ship_loss_count_counterfact(mapped_actions, prev_units_obs, obs))
    
          
    for k in prev_units_obs[1]:
      if not k in current_units_obs[1]:
        base_loss += 1
        
    prev_env_observation = env_observation
    prev_units_obs = current_units_obs
    prev_obs = obs
    
  return (destroyed_conversions, boxed_ship_loss, shipyard_collision_losses,
          ship_loss, base_loss, ship_non_boxed_loss_counterfactual,
          all_counterfactual_ship_loss)

process_each_step = True
game_agent = [
  rule_utils.sample_from_config_or_path(os.path.join(
  stable_opponents_folder, agent_files[-1]),
    return_callable=True),  # Stable opponents folder
  initial_config][  # Main code
    1]
destroyed_conversion_losses = np.zeros((num_replays, 4))
boxed_ship_losses = np.zeros((num_replays, 4))
shipyard_collision_losses = np.zeros((num_replays, 4))
ship_losses = np.zeros((num_replays, 4))
base_losses = np.zeros((num_replays, 4))
ship_non_boxed_loss_counterfactual = np.zeros((num_replays, 4))
all_ship_loss_counterfactual = np.zeros((num_replays, 4))

for i in range(num_replays):
  print("Processing replay {} of {}".format(i+1, num_replays))
  replay = json_data[i]
  my_id = int(json_files[i][-6])
  other_ids = list(set(range(4)) - set([my_id]))
  
  for j, analysis_id in zip(range(4), [my_id] + other_ids):
    losses = get_game_ship_base_loss_count(replay, analysis_id, game_agent,
                                           process_each_step=process_each_step)
    destroyed_conversion_losses[i, j] = losses[0]
    boxed_ship_losses[i, j] = losses[1]
    shipyard_collision_losses[i, j] = losses[2]
    ship_losses[i, j] = losses[3]
    base_losses[i, j] = losses[4]
    ship_non_boxed_loss_counterfactual[i, j] = losses[5]
    all_ship_loss_counterfactual[i, j] = losses[6]

non_boxed_ship_losses = ship_losses - (
  boxed_ship_losses + shipyard_collision_losses)
print(ship_non_boxed_loss_counterfactual[:, 0].sum())
    