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

NUM_GAMES = 7
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
  'episode_steps_override': None,  # Also adjusts the relative step
  'early_episode_termination_steps': None,  # Exits environment after N steps
  
  'num_agents_per_game': 4,
  'pool_name': 'Rule based with evolution X',

  # You need to delete the earlier configs or delete an entire agent pool after
  # making changes to the search ranges
  # 'initial_config_ranges': {
  #   'halite_config_setting_divisor': ((1.0, 1.0+1e-10), "float", 0),
  #   'collect_smoothed_multiplier': ((0.0, 0.1), "float", 0),
  #   'collect_actual_multiplier': ((2.0, 8.0), "float", 0),
  #   'collect_less_halite_ships_multiplier_base': ((0.4, 0.7), "float", 0),
  #   'collect_base_nearest_distance_exponent': ((0.0, 0.3), "float", 0),
    
  #   'return_base_multiplier': ((6.0, 12.0), "float", 0),
  #   'return_base_less_halite_ships_multiplier_base': ((0.8, 1.0), "float", 0),
  #   'early_game_return_base_additional_multiplier': ((0.0, 0.5), "float", 0),
  #   'early_game_return_boost_step': ((10, 100), "int", 0),
  #   'establish_base_smoothed_multiplier': ((0.0, 0.1), "float", 0),
  
  #   'establish_first_base_smoothed_multiplier_correction': ((0.0, 4.0), "float", 0),
  #   'establish_base_dm_exponent': ((0.5, 2.0), "float", 0.5),
  #   'first_base_no_4_way_camping_spot_bonus': ((100.0, 1000.0), "float", 0),
  #   'start_camp_if_not_winning': ((0, 1), "int", 0),
  #   'max_camper_ship_budget': ((0, 4), "int", 0),
  
  #   'relative_step_start_camping': ((0.1, 0.4), "float", 0),
  #   'establish_base_deposit_multiplier': ((0.8, 1.0), "float", 0),
  #   'establish_base_less_halite_ships_multiplier_base': ((0.9, 1.0), "float", 0),
  #   'max_attackers_per_base': ((0, 5), "int", -1),
  #   'attack_base_multiplier': ((0.0, 500.0), "float", 0),
  
  #   'attack_base_less_halite_ships_multiplier_base': ((0.8, 1.0), "float", 0),
  #   'attack_base_halite_sum_multiplier': ((1.0, 3.0), "float", 0),
  #   'attack_base_run_opponent_multiplier': ((0.1, 2.0), "float", 0),
  #   'attack_base_catch_opponent_multiplier': ((0.0, 2.0), "float", 0),
  #   'collect_run_opponent_multiplier': ((5.0, 15.0), "float", 0),
  
  #   'return_base_run_opponent_multiplier': ((1.5, 4.0), "float", 0),
  #   'establish_base_run_opponent_multiplier': ((0.0, 5.0), "float", 0),
  #   'collect_catch_opponent_multiplier': ((0.0, 2.0), "float", 0),
  #   'return_base_catch_opponent_multiplier': ((0.0, 2.0), "float", 0),
  #   'establish_base_catch_opponent_multiplier': ((0.0, 2.0), "float", 0),
  
  #   'two_step_avoid_boxed_opponent_multiplier_base': ((0.6, 0.9), "float", 0),
  #   'n_step_avoid_boxed_opponent_multiplier_base': ((0.3, 0.9), "float", 0),
  #   'min_consecutive_chase_extrapolate': ((4, 10), "int", 1),
  #   'chase_return_base_exponential_bonus': ((1.0, 2.0), "float", 0),
  #   'ignore_catch_prob': ((0.1, 0.5), "float", 0),
  
  #   'max_initial_ships': ((30, 70), "int", 1),
  #   'max_final_ships': ((5, 15), "int", 1),
  #   'max_standard_ships_decided_end_pack_hunting': ((0, 5), "int", 1),
  #   'nearby_ship_halite_spawn_constant': ((1.0, 5.0), "float", 0),
  #   'nearby_halite_spawn_constant': ((2.0, 10.0), "float", 0),
  
  #   'remaining_budget_spawn_constant': ((0.2, 0.20001), "float", 0),
  #   'spawn_score_threshold': ((0.0, 100.0), "float", -float("inf")),
  #   'boxed_in_halite_convert_divisor': ((0.5, 2.0), "float", 1),
  #   'n_step_avoid_min_die_prob_cutoff': ((0.0, 0.2), "float", 0),
  #   'n_step_avoid_window_size': ((5, 9), "int", 3),
  
  #   'influence_map_base_weight': ((1.0, 3.0), "float", 0),
  #   'influence_map_min_ship_weight': ((0.0, 0.5), "float", 0),
  #   'influence_weights_additional_multiplier': ((0.0, 5.0), "float", 0),
  #   'influence_weights_exponent': ((3.0, 9.0), "float", 1),
  #   'escape_influence_prob_divisor': ((1.0, 5.0), "float", 1),
  
  #   'rescue_ships_in_trouble': ((0, 1), "int", 0),
  #   'target_strategic_base_distance': ((6.0, 10.0), "float", 1.0),
  #   'target_strategic_num_bases_ship_divisor': ((5, 15), "int", 1.0),
  #   'target_strategic_triangle_weight': ((10.0, 30.0), "float", 1.0),
  #   'target_strategic_independent_base_distance_multiplier': ((0.0, 10.0), "float", 0.0),
  
  #   'target_strategic_influence_desirability_multiplier': ((0.5, 2.0), "float", 1.0),
  #   'target_strategic_potential_divisor': ((5.0, 20.0), "float", 1.0),
  #   'max_spawn_relative_step_divisor': ((5.0, 25.0), "float", 1),
  #   'no_spawn_near_base_ship_limit': ((100, 101), "int", 2),
  #   'avoid_cycles': ((0, 1), "int", 0),
  
  #   'max_risk_n_step_risky': ((0.1, 0.7), "float", 0),
  #   'max_steps_n_step_risky': ((0, 100), "int", 0),
  #   'log_near_base_distance': ((1, 3), "int", 1),
  #   'max_recent_considered_relevant_zero_move_count': ((100, 101), "int", 20),
  #   'near_base_2_step_risky_min_count': ((50, 51), "int", 0),
  
  #   'relative_stand_still_collect_boost': ((1.0, 2.0), "float", 1),
  #   'initial_collect_boost_away_from_base': ((1.0, 3.0), "float", 1),
  #   'start_hunting_season_relative_step': ((0.15, 0.2), "float", 1),
  #   'end_hunting_season_relative_step': ((0.7, 0.8), "float", 0.7),
  #   'early_hunting_season_less_collect_relative_step': ((0.35, 0.4), "float", 0.3),
  
  #   'max_standard_ships_early_hunting_season': ((2, 10), "int", 0),
  #   'late_hunting_season_more_collect_relative_step': ((0.4, 0.6), "float", 0.3),
  #   'late_hunting_season_collect_max_n_step_risk': ((0.1, 0.5), "float", 0),
  #   'after_hunting_season_collect_max_n_step_risk': ((0.2, 0.6), "float", 0),
  #   'late_hunting_season_standard_min_fraction': ((0.5, 0.9), "float", 0),
  
  #   'max_standard_ships_late_hunting_season': ((10, 20), "int", 0),
  #   'collect_on_safe_return_relative_step': ((0.1, 0.4), "float", 0.1),
  #   'min_halite_to_stop_early_hunt': ((10000.0, 15000.0), "float", 0),
  #   'early_best_opponent_relative_step': ((0.1, 0.2), "float", 0.1),
  #   'surrounding_ships_cycle_extrapolate_step_count': ((4, 6), "int", 3),
  
  #   'surrounding_ships_extended_cycle_extrapolate_step_count': ((5, 9), "int", 4),
  #   }
  
  'initial_config_ranges': {
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
          episode_steps_override=config['episode_steps_override'],
          early_episode_termination=config['early_episode_termination_steps'],
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
          episode_steps_override=config['episode_steps_override'],
          early_episode_termination=config['early_episode_termination_steps'],
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
           episode_steps_override=config['episode_steps_override'],
           early_episode_termination=config['early_episode_termination_steps'],
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
        fixed_opponents_experience[0].config_game_agents)
      # import pdb; pdb.set_trace()
      rule_utils.record_videos(
        rules_config_path, config['num_agents_per_game'],
        extension_override=str(datetime.now())[:19],
        config_override_agents=config_override_agents,
        env_seed_deterministic=fixed_opponents_experience[0].env_random_seed,
        rng_action_seeds=fixed_opponents_experience[0].act_random_seeds,
        first_game_recording=fixed_opponents_experience[0].game_recording,
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
