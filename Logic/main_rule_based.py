from datetime import datetime
import rule_experience
import rule_utils
from shutil import copyfile
import utils

# Make sure the data is deterministic
import numpy as np
import random
np.random.seed(0)
random.seed(0)

NUM_GAMES = 10
config = {
  'max_pool_size': 30, # 1 Means pure self play
  'num_games_previous_pools': NUM_GAMES,
  'num_games_evaluation': NUM_GAMES,
  'num_games_fixed_opponents_pool': NUM_GAMES,
  'max_experience_buffer': 10000,
  'min_new_iteration_win_rate': 0.6,
  'record_videos_new_iteration': True,
  'record_videos_each_main_loop': True,
  'save_experience_data_to_disk': True,
  'use_multiprocessing': True,
  'play_fixed_pool_only': True,
  'fixed_opponents_num_repeat_first_configs': 5,
  
  'num_agents_per_game': 4,
  'pool_name': 'Rule based with evolution II',

  # You need to delete the earlier configs or delete an entire agent pool after
  # making changes to the search ranges
  'initial_config_ranges':{
    'halite_config_setting_divisor': ((1e2, 1e3), "float", 0),
    'max_ship_to_base_ratio': ((2, 6), "float", 0),
    
    'min_spawns_after_conversions': ((0, 3), "int", 0),
    'max_conversions_per_step': ((2, 10), "int", 1),
    'friendly_ship_halite_conversion_constant': ((0.1, 0.5), "float", 0),
    'friendly_bases_conversion_constant': ((0.1, 4), "float", 0),
    'nearby_halite_conversion_constant': ((0.001, 0.3), "float", 0),
    'conversion_score_threshold': ((-1, 1), "float", -float("inf")),
    
    'halite_collect_constant': ((50, 100), "float", 0),
    'nearby_halite_move_constant': ((0.2, 3), "float", 0),
    'nearby_onto_halite_move_constant': ((10, 20), "float", 0),
    'nearby_ships_move_constant': ((0.01, 0.05), "float", 0),
    'nearby_base_move_constant': ((5, 30), "float", 0),
    'nearby_move_onto_base_constant': ((5, 20), "float", 0),
    'halite_dropoff_constant': ((50, 100), "float", 0),
    
    'max_spawns_per_step': ((2, 10), "int", 1),
    'nearby_ship_halite_spawn_constant': ((0.3, 3), "float", 0),
    'nearby_halite_spawn_constant': ((0.3, 3), "float", 0),
    'remaining_budget_spawn_constant': ((0.001, 0.005), "float", 0),
    'spawn_score_threshold': ((-1, 10), "float", -float("inf")),
    }
  }

def main_rule_utils(config):
  rule_utils.store_config_on_first_run(config)
  experience_buffer = utils.ExperienceBuffer(config['max_experience_buffer'])
  while True:
    # Section 1: play games against agents of N previous pools
    if config['num_games_previous_pools'] and not config[
        'play_fixed_pool_only']:
      print('\nPlay vs other rule based agents from the last {} pools'.format(
        config['max_pool_size']))
      (self_play_experience, rules_config_path,
       avg_reward_sp, _) = rule_experience.play_games(
          pool_name=config['pool_name'],
          num_games=config['num_games_previous_pools'],
          max_pool_size=config['max_pool_size'],
          num_agents=config['num_agents_per_game'],
          exclude_current_from_opponents=False,
          record_videos_new_iteration=config['record_videos_new_iteration'],
          initial_config_ranges=config['initial_config_ranges'],
          use_multiprocessing=config['use_multiprocessing'],
          )
      experience_buffer.add(self_play_experience)
    
    # Section 2: play games against agents of the previous pool
    if config['num_games_evaluation'] and not config['play_fixed_pool_only']:
      print('\nPlay vs previous iteration')
      (evaluation_experience, rules_config_path,
       avg_reward_eval, _) = rule_experience.play_games(
          pool_name=config['pool_name'],
          num_games=config['num_games_evaluation'],
          max_pool_size=2,
          num_agents=config['num_agents_per_game'],
          exclude_current_from_opponents=True,
          use_multiprocessing=config['use_multiprocessing'],
          )
      # experience_buffer.add(evaluation_experience)
         
    # Section 3: play games against a fixed opponents pool
    if config['num_games_fixed_opponents_pool']:
      print('\nPlay vs the fixed opponents pool')
      (fixed_opponents_experience, rules_config_path,
       avg_reward_fixed_opponents, opponent_rewards) = (
         rule_experience.play_games(
           pool_name=config['pool_name'],
           num_games=config['num_games_fixed_opponents_pool'],
           max_pool_size=1, # Any positive integer is fine
           num_agents=config['num_agents_per_game'],
           exclude_current_from_opponents=False,
           fixed_opponent_pool=True,
           initial_config_ranges=config['initial_config_ranges'],
           use_multiprocessing=config['use_multiprocessing'],
           num_repeat_first_configs=config[
             'fixed_opponents_num_repeat_first_configs']
           )
         )
      # experience_buffer.add(evaluation_experience)
    
    # Select the values that will be used to determine if a next iteration file
    # will be created
    fixed_pool_mode = config['play_fixed_pool_only']
    serialized_raw_experience = fixed_opponents_experience if (
      fixed_pool_mode) else self_play_experience
         
    # Optionally append the experience of interest to disk
    iteration_config_rewards = (
      rule_utils.serialize_game_experience_for_learning(
        serialized_raw_experience, fixed_pool_mode))
    if config['save_experience_data_to_disk']:
      experience_features_rewards_path = rule_utils.write_experience_data(
        config['pool_name'], iteration_config_rewards)
         
    # Section 4: Update the iteration, store videos and record learning
    # progress.
    if fixed_pool_mode:
      update_config = {'Time stamp': str(datetime.now())}
      for i in range(len(opponent_rewards)):
        update_config['Reward ' + opponent_rewards[i][2]] = np.round(
          opponent_rewards[i][1]/opponent_rewards[i][0], 2)
      rule_utils.update_learning_progress(config['pool_name'], update_config)

      config_override_agents = fixed_opponents_experience[-1].agent_configs
      rule_utils.record_videos(
        rules_config_path, config['num_agents_per_game'],
        extension_override=str(datetime.now())[:19],
        config_override_agents=config_override_agents)
    else:
      # Save a new iteration if it has significantly improved
      data_rules_path = rules_config_path
      if min(avg_reward_sp, avg_reward_eval) >= config[
          'min_new_iteration_win_rate']:
        original_rules_config_path = rules_config_path
        incremented_rules_path = utils.increment_iteration_id(
          rules_config_path, extension='.json')
        copyfile(rules_config_path, incremented_rules_path)
        rules_config_path = incremented_rules_path
        
        if config['record_videos_new_iteration']:
          rule_utils.record_videos(original_rules_config_path,
                                   config['num_agents_per_game'],
                                   )
      elif config['record_videos_each_main_loop']:
        rule_utils.record_videos(rules_config_path,
                                 config['num_agents_per_game'],
                                 str(datetime.now())[:19])
        
      # Record learning progress
      rule_utils.update_learning_progress(config['pool_name'], {
        'Time stamp': str(datetime.now()),
        'Average reward self play': avg_reward_sp,
        'Average evaluation reward': avg_reward_eval,
        'Experience buffer size': experience_buffer.size(),
        'Data rules path': data_rules_path,
        })
    
    # Section 5: Update the latest config range using the data in the
    # experience buffer
    if rules_config_path is not None:
      if fixed_pool_mode:
        import pdb; pdb.set_trace()
        print("TODO: Bayesian hyperparameter optimization")
      else:
        # Evolve the config ranges in a very simple gradient free way.
        rule_utils.evolve_config(
          rules_config_path, iteration_config_rewards,
          config['initial_config_ranges'])
      
      # Create plot(s) of the terminal reward as a function of all serialized
      # features
      if config['save_experience_data_to_disk']:
        rule_utils.plot_reward_versus_features(
          experience_features_rewards_path, iteration_config_rewards,
          plot_name_suffix=str(datetime.now())[:19])
    
main_rule_utils(config)