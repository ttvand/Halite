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
    'initial_standard_ships_hunting_season': 10,
    'minimum_standard_ships_hunting_season': 5,
    'min_standard_ships_fraction_hunting_season': 0.2,
    'max_standard_ships_fraction_hunting_season': 0.6,
    
    'max_standard_ships_low_clip_fraction_hunting_season': 0.4,
    'max_standard_ships_high_clip_fraction_hunting_season': 0.8,
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
    'late_hunting_season_collect_max_n_step_risk': 0.4,
    'late_hunting_season_standard_min_fraction': 0.7,
    'max_standard_ships_late_hunting_season': 15,
    
    'collect_on_safe_return_relative_step': 0.075,
    'min_halite_to_stop_early_hunt': 15000.0,
    'early_best_opponent_relative_step': 0.5,
    'surrounding_ships_cycle_extrapolate_step_count': 5,
    'surrounding_ships_extended_cycle_extrapolate_step_count': 7,
    }


NORTH = "NORTH"

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
