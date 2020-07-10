from datetime import datetime
import numpy as np
import os
import pandas as pd
import rule_experience
import rule_utils
from shutil import copyfile
from skopt import Optimizer
import utils

# Possibly make the played games deterministic
deterministic_games = True
MAIN_LOOP_INITIAL_SEED = 0 # This allows flexible inspection of replay videos

NUM_GAMES = 50
config = {
  'max_pool_size': 30, # 1 Means pure self play
  'num_games_previous_pools': NUM_GAMES*0,
  'num_games_evaluation': NUM_GAMES*0,
  'num_games_fixed_opponents_pool': NUM_GAMES,
  'max_experience_buffer': 10000,
  'min_new_iteration_win_rate': 0.6,
  'record_videos_new_iteration': True,
  'record_videos_each_main_loop': True,
  'save_experience_data_to_disk': True,
  'use_multiprocessing': True,
  'play_fixed_pool_only': True,
  'play_fixed_pool_fit_prev_data': True,
  'fixed_opponents_num_repeat_first_configs': NUM_GAMES,
  'deterministic_games': deterministic_games,
  
  'num_agents_per_game': 4,
  'pool_name': 'Rule based with evolution VI',

  # You need to delete the earlier configs or delete an entire agent pool after
  # making changes to the search ranges
  # 'initial_config_ranges':{
  #   'halite_config_setting_divisor': ((1.0, 1.0+1e-10), "float", 0),
  #   'min_spawns_after_conversions': ((0, 2), "int", 0),
  #   'collect_smoothed_multiplier': ((0.0, 0.2), "float", 0),
  #   'collect_actual_multiplier': ((6.0, 12.0), "float", 0),
  #   'collect_less_halite_ships_multiplier_base': ((0.5, 0.9), "float", 0),
    
  #   'collect_base_nearest_distance_exponent': ((0.0, 0.5), "float", 0),
  #   'return_base_multiplier': ((6.0, 12.0), "float", 0),
  #   'return_base_less_halite_ships_multiplier_base': ((0.8, 1.0), "float", 0),
  #   'early_game_return_base_additional_multiplier': ((0.0, 2.0), "float", 0),
  #   'early_game_return_boost_step': ((50, 200), "int", 0),
  
  #   'end_game_return_base_additional_multiplier': ((0.0, 10.0), "float", 0),
  #   'establish_base_smoothed_multiplier': ((0.0, 0.1), "float", 0),
  #   'establish_first_base_smoothed_multiplier_correction': ((1.0, 2.5), "float", 0),
  #   'establish_base_deposit_multiplier': ((0.5, 1.0), "float", 0),
  #   'establish_base_less_halite_ships_multiplier_base': ((0.9, 1.0), "float", 0),
  
  #   'attack_base_multiplier': ((0.0, 500.0), "float", 0),
  #   'attack_base_less_halite_ships_multiplier_base': ((0.8, 1.0), "float", 0),
  #   'attack_base_halite_sum_multiplier': ((0.1, 2.0), "float", 0),
  #   'attack_base_run_enemy_multiplier': ((0.1, 2.0), "float", 0),
  #   'attack_base_catch_enemy_multiplier': ((0.0, 2.0), "float", 0),
  
  #   'collect_run_enemy_multiplier': ((5.0, 15.0), "float", 0),
  #   'return_base_run_enemy_multiplier': ((1.0, 3.0), "float", 0),
  #   'establish_base_run_enemy_multiplier': ((0.0, 5.0), "float", 0),
  #   'collect_catch_enemy_multiplier': ((0.0, 2.0), "float", 0),
  #   'return_base_catch_enemy_multiplier': ((0.0, 2.0), "float", 0),
  
  #   'establish_base_catch_enemy_multiplier': ((0.0, 2.0), "float", 0),
  #   'two_step_avoid_boxed_enemy_multiplier': ((0.0, 60.0), "float", 0),
  #   'ignore_catch_prob': ((0.3, 0.5), "float", 0),
  #   'max_ships': ((15, 25), "int", 1),
  #   'max_spawns_per_step': ((1, 4), "int", 1),
  
  #   'nearby_ship_halite_spawn_constant': ((0.5, 3.0), "float", 0),
  #   'nearby_halite_spawn_constant': ((5.0, 20.0), "float", 0),
  #   'remaining_budget_spawn_constant': ((0.1, 0.3), "float", 0),
  #   'spawn_score_threshold': ((0.0, 100.0), "float", -float("inf")),
  #   'max_spawn_relative_step_divisor': ((100.0, 400.0), "float", 1),
  #   }
  
  'initial_config_ranges': {
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
    'two_step_avoid_boxed_enemy_multiplier': 30.0,
    'ignore_catch_prob': 0.5,
    'max_ships': 20,
    'max_spawns_per_step': 3,
    
    'nearby_ship_halite_spawn_constant': 2.0,
    'nearby_halite_spawn_constant': 10.0,
    'remaining_budget_spawn_constant': 0.2,
    'spawn_score_threshold': 50.0,
    'max_spawn_relative_step_divisor': 100.0,
    }
  }
CONFIG_SETTINGS_EXTENSION = "config_settings_scores.csv"

def main_rule_utils(config, main_loop_seed=MAIN_LOOP_INITIAL_SEED):
  rule_utils.store_config_on_first_run(config)
  experience_buffer = utils.ExperienceBuffer(config['max_experience_buffer'])
  config_keys = list(config['initial_config_ranges'].keys())
  
  if deterministic_games:
    utils.set_seed(main_loop_seed)
  
  fixed_pool_mode = config['play_fixed_pool_only']
  if fixed_pool_mode:
    fixed_opp_repeats = config['fixed_opponents_num_repeat_first_configs']
    config_has_range = isinstance(
      list(config['initial_config_ranges'].values())[0], tuple)
    
    # Prepare the Bayesian optimizer
    if config_has_range:
      opt_range = [config['initial_config_ranges'][k][0] for k in config_keys]
      opt = Optimizer(opt_range)
      
      if config['play_fixed_pool_fit_prev_data']:
        fixed_pool_experience_path = rule_utils.get_self_play_experience_path(
          config['pool_name'])
        if os.path.exists(fixed_pool_experience_path):
          print('\nBayesian fit to earlier experiments')
          this_folder = os.path.dirname(__file__)
          agents_folder = os.path.join(
            this_folder, '../Rule agents/' + config['pool_name'])
          config_settings_path = os.path.join(
            agents_folder, CONFIG_SETTINGS_EXTENSION)
          if os.path.exists(config_settings_path):
            config_results = pd.read_csv(config_settings_path)
            suggested = config_results.iloc[:, :-1].values.tolist()
            target_scores = (-config_results.iloc[:, -1].values).tolist()
            opt.tell(suggested, target_scores)
            # import pdb; pdb.set_trace()
            # print(opt.get_result().x, opt.get_result().fun) # WRONG!
    else:
      opt = None
        
    next_fixed_opponent_suggested = None
    iteration_config_rewards = None
    experience_features_rewards_path = None
  
  while True:
    if deterministic_games:
      utils.set_seed(main_loop_seed)
    
    # Section 1: play games against agents of N previous pools
    if config['num_games_previous_pools'] and not fixed_pool_mode:
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
    if config['num_games_evaluation'] and not fixed_pool_mode:
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
         
    if fixed_pool_mode:
      if iteration_config_rewards is not None:
        # Update the optimizer using the most recent fixed opponent pool
        # results
        target_scores = np.reshape(-iteration_config_rewards[
          'episode_reward'].values, [-1, fixed_opp_repeats]).mean(1).tolist()
        if config_has_range:
          opt.tell(next_fixed_opponent_suggested, target_scores)
        
        # Append the tried settings to the settings-scores file
        config_rewards = rule_utils.append_config_scores(
          next_fixed_opponent_suggested, target_scores, config_keys,
          config['pool_name'], CONFIG_SETTINGS_EXTENSION)
        
        # Update the plot of the tried settings and obtained scores
        rule_utils.plot_reward_versus_features(
          experience_features_rewards_path, config_rewards,
          target_col="Average win rate", include_all_targets=True,
          plot_name_suffix="config setting average win rate", all_scatter=True)
      
      # Select the next hyperparameters to try
      try:
        next_fixed_opponent_suggested, next_fixed_opponent_configs = (
          rule_utils.get_next_config_settings(
            opt, config_keys, config['num_games_fixed_opponents_pool'],
            fixed_opp_repeats, config['initial_config_ranges'])
          )
      except:
        import pdb; pdb.set_trace()
         
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
           num_repeat_first_configs=fixed_opp_repeats,
           first_config_overrides=next_fixed_opponent_configs,
           )
         )
      experience_buffer.add(fixed_opponents_experience)
         
    # import pdb; pdb.set_trace()
    # Select the values that will be used to determine if a next iteration file
    # will be created
    serialized_raw_experience = fixed_opponents_experience if (
      fixed_pool_mode) else self_play_experience
         
    # Optionally append the experience of interest to disk
    iteration_config_rewards = (
      rule_utils.serialize_game_experience_for_learning(
        serialized_raw_experience, fixed_pool_mode, config_keys))
    if config['save_experience_data_to_disk']:
      experience_features_rewards_path = rule_utils.write_experience_data(
        config['pool_name'], iteration_config_rewards)
         
    # Section 4: Update the iteration, store videos and record learning
    # progress.
    if fixed_pool_mode:
      update_config = {'Time stamp': str(datetime.now())}
      for i in range(len(opponent_rewards)):
        update_config['Reward ' + opponent_rewards[i][2]] = np.round(
          opponent_rewards[i][1]/(1e-10+opponent_rewards[i][0]), 2)
      rule_utils.update_learning_progress(config['pool_name'], update_config)

      config_override_agents = (
        fixed_opponents_experience[-1].config_game_agents)
      rule_utils.record_videos(
        rules_config_path, config['num_agents_per_game'],
        extension_override=str(datetime.now())[:19],
        config_override_agents=config_override_agents,
        env_seed_deterministic=fixed_opponents_experience[0].env_random_seed,
        rng_action_seeds=fixed_opponents_experience[0].act_random_seeds,
        deterministic_games=config['deterministic_games'],
        deterministic_extension=f" - Seed {main_loop_seed}")
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
          rule_utils.record_videos(
            original_rules_config_path, config['num_agents_per_game'])
      elif config['record_videos_each_main_loop']:
        rule_utils.record_videos(
          rules_config_path, config['num_agents_per_game'],
          str(datetime.now())[:19])
        
      # Record learning progress
      # import pdb; pdb.set_trace()
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
      if not fixed_pool_mode:
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
        
    main_loop_seed += 1
    
main_rule_utils(config)