from collections import OrderedDict
import copy
import getpass
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
    'establish_base_deposit_multiplier': 1.0,
    'establish_base_less_halite_ships_multiplier_base': 1.0,
    'max_attackers_per_base': 3,
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
    'max_spawn_relative_step_divisor': 100.0,
    'no_spawn_near_base_ship_limit': 100,
    }


NORTH = "NORTH"

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
    CONFIG, current_observation, player_obs, env_config, HISTORY,
    rng_action_seed)
     
  if LOCAL_MODE:
    # This is to allow for debugging of the history outside of the agent
    return mapped_actions, copy.deepcopy(HISTORY)
  else:
    return mapped_actions
