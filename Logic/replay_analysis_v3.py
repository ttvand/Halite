import copy
import json
import numpy as np
import os
import pandas as pd
import rule_utils
import time
import utils

my_submissions = [17114281, 17114329, 17168045, 17170621, 17170654,
                  17170690, 17171012, 17183645, 17187266,
                  17190682, 17190822, 17190934, 17195646]
target_episode = 3279869 # Automatically matches the relevant submission

initial_config = {
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
    'target_strategic_base_distance': 7.0,
    'target_strategic_num_bases_ship_divisor': 9,
    'target_strategic_triangle_weight': 3.0,  # initially: 20
    'target_strategic_independent_base_distance_multiplier': 0.5,  # initially 8.0
    
    'target_strategic_influence_desirability_multiplier': 1.0,  # initially: 1.0
    'target_strategic_potential_divisor': 10.0,  # initially: 15.0
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
    
    'max_standard_ships_early_hunting_season': 5,
    'late_hunting_season_more_collect_relative_step': 0.5,
    'late_hunting_season_collect_max_n_step_risk': 0.3,
    'after_hunting_season_collect_max_n_step_risk': 0.4,
    'late_hunting_season_standard_min_fraction': 0.7,
    
    'max_standard_ships_late_hunting_season': 15,
    'collect_on_safe_return_relative_step': 0.075,
    'min_halite_to_stop_early_hunt': 15000.0,
    'early_best_opponent_relative_step': 0.5,
    'surrounding_ships_cycle_extrapolate_step_count': 5,
    
    'surrounding_ships_extended_cycle_extrapolate_step_count': 7,
    }

this_folder = os.path.dirname(__file__)
replay_folder = os.path.join(
  this_folder, '../Rule agents/Leaderboard replays/')
episode_found = False
for my_submission in my_submissions:
  data_folder = os.path.join(replay_folder, str(my_submission))
  json_files = np.sort(
    [f for f in os.listdir(data_folder) if f[-4:] == "json"])
  episode_match = [str(target_episode) in f for f in json_files]
  if np.any(np.array(episode_match)):
    json_file = [f for (f, m) in zip(json_files, episode_match) if m][0]
    json_path = os.path.join(data_folder, json_file)
    episode_found = True
    print(json_path)
    break

if not episode_found:
  raise ValueError("Load data with scrape_json.py and add to 'my_submissions'")

with open(json_path) as f:
  raw_data = json.load(f)
  json_data = json.loads(raw_data)

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

def ship_loss_count_counterfact(actions, prev_units, obs, grid_size=21,
                                debug=False):
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
    
    # if debug and row == 0:
    #   import pdb; pdb.set_trace()
    
    a = get_ship_action(ship_k, actions)
    if a == "CONVERT":
      my_bases[row, col] = True
    else:
      new_row, new_col = rule_utils.move_ship_row_col(row, col, a, grid_size)
      if my_ships[new_row, new_col]:
        # Self collision
        # import pdb; pdb.set_trace()
        ship_loss += 1
      else:
        my_ships[new_row, new_col] = 1
        ship_halite = prev_ships[ship_k][1]
        for o in obs['rewards_bases_ships'][1:]:
          if o[1][new_row, new_col]:
            # import pdb; pdb.set_trace()
            ship_loss += 1
            break
          elif o[2][new_row, new_col] and (
              o[3][new_row, new_col] <= ship_halite):
            # Collide with less or equal halite ship
            # import pdb; pdb.set_trace()
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
  history = {}
  prev_history = -1
  step_times = []
  my_step_durations = np.zeros((400, 8))
  for i in range(num_steps-1):
    print(i)
    current_units_obs = replay['steps'][i][0]['observation']['players'][
      player_id]
    env_observation = utils.dotdict(replay['steps'][i][0]['observation'])
    env_observation['step'] = i
    env_observation.player = player_id
    obs = utils.structured_env_obs(env_configuration, env_observation,
                                   player_id)
    prev_actions = replay['steps'][i][player_id]['action']
    actions = replay['steps'][i+1][player_id]['action']
    
    # if i == 274:
    #   import pdb; pdb.set_trace()
    
    for k in prev_units_obs[2]:
      if not k in current_units_obs[2]:
        # pos_actions_ships = get_pos_actions_ships(prev_units_obs[2], prev_actions)
        prev_pos = prev_units_obs[2][k][0]
        if not prev_pos in current_units_obs[1].values():
          prev_ship_action = get_ship_action(k, prev_actions)
          if prev_ship_action == "CONVERT":
            destroyed_conversions += 1 
          else:
            boxed_ship_loss += int(is_boxed_ship_loss(prev_pos, prev_obs))
            # import pdb; pdb.set_trace()
            ship_loss += 1
            if not boxed_ship_loss:
              if prev_ship_action is not None and base_at_collision_pos(
                  prev_pos, prev_ship_action, replay['steps'][i],
                  replay['steps'][i+1]):
                # import pdb; pdb.set_trace()
                print(history['prev_step']['observation']['step'],
                      prev_history['prev_step']['observation']['step'])
                shipyard_collision_losses += 1
              else:
                mapped_actions, _, _, _ = (
                  rule_utils.get_config_or_callable_actions(
                    game_agent, prev_obs, prev_units_obs,
                    prev_env_observation, env_configuration, copy.deepcopy(
                      prev_history)))
                # if prev_obs['step'] == 281:
                #   import pdb; pdb.set_trace()
                
                # mapped_actions['6-2'] = 'WEST'
                # import pdb; pdb.set_trace()
                ship_non_boxed_loss_counterfactual += (
                  ship_loss_count_counterfact(mapped_actions, prev_units_obs,
                                              obs, debug=False))
    
    if process_each_step:
      if prev_obs is not None:
        # import pdb; pdb.set_trace()
        mapped_actions, _, _, _ = (
          rule_utils.get_config_or_callable_actions(
            game_agent, prev_obs, prev_units_obs, prev_env_observation,
            env_configuration, copy.deepcopy(prev_history)))
        # if prev_obs['step'] == 204:
        #   print(mapped_actions)
        #   import pdb; pdb.set_trace()
        
        all_counterfactual_ship_loss += (
          ship_loss_count_counterfact(mapped_actions, prev_units_obs, obs))
        
      # import pdb; pdb.set_trace()
      prev_history = copy.deepcopy(history)
      start_time = time.time()
      current_actions, history, _, step_details = (
        rule_utils.get_config_or_callable_actions(
          game_agent, obs, current_units_obs, env_observation,
          env_configuration, history))
      if step_details is not None:
        step_time = time.time()-start_time
        step_times.append(step_time)
        my_step_durations[i] = np.array([
          step_details['get_actions_duration'],
          step_details['ship_scores_duration'],
          step_details['ship_plans_duration'],
          step_details['ship_map_duration'],
          step_details['inner_loop_ship_plans_duration'],
          step_details['recompute_ship_plan_order_duration'],
          step_details['history_start_duration'],
          step_details['box_in_duration'],
          ])
      
      # Overwrite the prev actions in history
      try:
        none_included_ship_actions = {k: (actions[k] if (
          k in actions) else None) for k in current_units_obs[2]}
      except:
        # This happens when my submission times out
        import pdb; pdb.set_trace()
        x=1
      history['prev_step']['my_ship_actions'] = none_included_ship_actions
      
      # print(current_actions, actions)
      # for k in current_actions:
      #   if current_actions[k] != actions[k]:
      #     import pdb; pdb.set_trace()
      #     x=1
    
    for k in prev_units_obs[1]:
      if not k in current_units_obs[1]:
        base_loss += 1
        
    prev_env_observation = env_observation
    prev_units_obs = current_units_obs
    prev_obs = obs
    
    # if i > 1:
    #   print(history['prev_step']['observation']['step'],
    #                     prev_history['prev_step']['observation']['step'])
      
  return ((destroyed_conversions, boxed_ship_loss, shipyard_collision_losses,
           ship_loss, base_loss, ship_non_boxed_loss_counterfactual,
           all_counterfactual_ship_loss), np.array(step_times),
          my_step_durations)

process_each_step = True
game_agent = [
  rule_utils.sample_from_config_or_path(os.path.join(
  stable_opponents_folder, agent_files[-1]),
    return_callable=True),  # Index 0: Stable opponents folder
  initial_config][  # Index 1: Main rule actions code
    1]
num_replays = 1
destroyed_conversion_losses = np.zeros((num_replays, 4))
boxed_ship_losses = np.zeros((num_replays, 4))
shipyard_collision_losses = np.zeros((num_replays, 4))
ship_losses = np.zeros((num_replays, 4))
base_losses = np.zeros((num_replays, 4))
ship_non_boxed_loss_counterfactual = np.zeros((num_replays, 4))
all_ship_loss_counterfactual = np.zeros((num_replays, 4))

for i in range(num_replays):
  print("Processing replay {} of {}".format(i+1, num_replays))
  replay = json_data
  my_id = int(json_file[-6])
  other_ids = list(set(range(4)) - set([my_id]))
  
  for j, analysis_id in zip(range(4), [my_id] + other_ids):
    losses, step_times, step_durations = get_game_ship_base_loss_count(
      replay, analysis_id, game_agent, process_each_step=process_each_step)
    destroyed_conversion_losses[i, j] = losses[0]
    boxed_ship_losses[i, j] = losses[1]
    shipyard_collision_losses[i, j] = losses[2]
    ship_losses[i, j] = losses[3]
    base_losses[i, j] = losses[4]
    ship_non_boxed_loss_counterfactual[i, j] = losses[5]
    all_ship_loss_counterfactual[i, j] = losses[6]
    
    break

non_boxed_ship_losses = ship_losses - (
  boxed_ship_losses + shipyard_collision_losses)
print(ship_non_boxed_loss_counterfactual[:, 0].sum())
print("Plan duration fraction:",
      np.round(step_durations[:,2].sum() / step_durations[:,0].sum(), 2))
    