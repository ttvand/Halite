import collect_experience
from datetime import datetime
import learn_from_experience
import models
import numpy as np
from shutil import copyfile
import utils

num_games = 50
config = {
  'max_pool_size': 30, # 1 Means pure self play
  'num_games': num_games,
  'num_games_previous_iterations': 0,
  'num_evaluation_games': num_games,
  'max_experience_buffer': 400,
  'min_new_iteration_win_rate': 0.6,
  'record_videos_new_iteration': True,
  'record_videos_each_main_loop': True,
  
  # For value-based agents
  'boltzman_temperature_range_self_play': [1e-3, 3e-1], # LogUniform sampling
  'boltzman_temperature_range_eval': [1e-3, 1e-1], # LogUniform sampling
  
  'agent_config': {
    'num_agents_per_game': 4,
    
    # 'pool_name': 'Initial pool 4 players U-net',
    # 'model': models.padded_unet,
    # 'unet_start_neurons': 16,
    # 'unet_dropout_ratio': 0.2,
    
    'pool_name': 'Halite reward 4 players - independent ship - base actions',
    'q_output_activation': ['sigmoid', 'none'][1],
    'model': models.convnet_simple,
    'filters_kernels': [
      (64, 3), (32, 3), (32, 3), (32, 3), (32, 3)],
    
    'action_mlp_layers': [32],
    # 'augment_input_version': (True, 'v1'),
    'epsilon_greedy': True,
    'epsilon_range': [0.01, 0.5], # Min, max range
    'num_q_functions': 1,
    },
  
  'learning_config': {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 1,
    'nan_coding_value': -999,
    'symmetric_experience': True,
    'reward_type': ['Terminal', 'Halite change'][1],
    'halite_change_discount': 0.99,
    'max_episodes_per_learning_update_q_learning': 200,
    },
  
  'play_previous_pools': False,
  'previous_pools': [
    ]
  }

config['pool_name'] = config['agent_config']['pool_name']
utils.store_config_on_first_run(config)
utils.add_action_costs_to_config(config)
experience_buffer = utils.ExperienceBuffer(config['max_experience_buffer'])
current_agent_path = None
mean_loss = None
while True:
  # Part 1: improve the agent by updating the network
  if current_agent_path is not None:
    print('\nLearn from experience in the replay buffer')
    mean_loss = learn_from_experience.update_agent(
      experience_buffer.get_data(),
      experience_buffer.episode_ids,
      current_agent_path,
      config['learning_config'],
      agent_config=config['agent_config'],
      )
  
  # Part 2: play against all N previous iterations
  print('\nPlay vs previous {} iteration(s), including the current one'.format(
    config['max_pool_size']))
  config['agent_config']['boltzman_temperature_range'] = config[
    'boltzman_temperature_range_self_play']
  (experience, current_agent_path, avg_reward_sp, num_self_play_opponents,
   avg_final_halite_sp, median_final_halite_sp, avg_reward_per_step_sp) = (
    collect_experience.play_games(
      pool_name=config['pool_name'],
      num_games=config['num_games'],
      max_pool_size=config['max_pool_size'],
      agent_config=config['agent_config'],
      exclude_current_from_opponents=False,
      symmetric_evaluation=config['learning_config']['symmetric_experience'],
      action_costs=config['action_costs'],
      record_videos_new_iteration=config['record_videos_new_iteration'],
      )
    )
  experience_buffer.add(experience)
  episode_lengths = np.array(
    [e.episode_step for e in experience if e.last_episode_action])
  av_self_play_episode_length = 1+episode_lengths.mean()
  
  # Part 3: Optionally play against the best iterations of previous agent pools
  avg_reward_prev = -1
  if config['play_previous_pools']:
    print('\nPlay against the best iterations of previous pools')
    experience, _, avg_reward_prev, _, _, _, _ = collect_experience.play_games(
      pool_name=config['pool_name'],
      num_games=config['num_games_previous_iterations'],
      max_pool_size=config['max_pool_size'],
      agent_config=config['agent_config'],
      exclude_current_from_opponents=False,
      symmetric_evaluation=config['learning_config']['symmetric_experience'],
      action_costs=config['action_costs'],
      best_iteration_opponents=config['previous_pools'],
      )
    experience_buffer.add(experience)
  
  # Part 4: play only against the previous iteration to verify if a new 
  # agent checkpoint is justified.
  print('\nPlay against the previous iteration')
  config['agent_config']['boltzman_temperature_range'] = config[
    'boltzman_temperature_range_eval']
  experience, current_agent_path, avg_reward_eval, _, _, _, _ = (
    collect_experience.play_games(
      pool_name=config['pool_name'],
      num_games=config['num_evaluation_games'],
      max_pool_size=2,
      agent_config=config['agent_config'],
      exclude_current_from_opponents=True,
      symmetric_evaluation=config['learning_config']['symmetric_experience'],
      action_costs=config['action_costs'],
      )
    )
  experience_buffer.add(experience)
  
  # Save a new iteration if it has significantly improved
  data_agent_path = current_agent_path
  if min(avg_reward_sp, avg_reward_eval) >= config[
      'min_new_iteration_win_rate']:
    original_agent_path = current_agent_path
    incremented_agent_path = utils.increment_iteration_id(current_agent_path)
    copyfile(current_agent_path, incremented_agent_path)
    current_agent_path = incremented_agent_path
    
    if config['record_videos_new_iteration']:
      utils.record_videos(original_agent_path,
                          config['agent_config']['num_agents_per_game'])
  elif config['record_videos_each_main_loop']:
    utils.record_videos(current_agent_path,
                        config['agent_config']['num_agents_per_game'],
                        str(datetime.now())[:19])
    
  # Unique state fraction to get an idea of the diversity of experience
  frac_unique_states = np.unique(np.array([
      e.current_obs for e in experience]), axis=0).shape[0]/len(experience)
  print('Fraction of unique evaluation states: {:.3f}'.format(
    frac_unique_states))
  
  utils.update_learning_progress(config['pool_name'], {
    'Average reward self play': avg_reward_sp,
    'Average reward previous iterations': avg_reward_prev,
    'Average evaluation reward': avg_reward_eval,
    'Average final halite self play': int(avg_final_halite_sp),
    'Median final halite self play': median_final_halite_sp,
    'Average halite change per step self play': int(avg_reward_per_step_sp),
    'Fraction of unique states': frac_unique_states,
    'Experience buffer size': experience_buffer.size(),
    'Mean loss': mean_loss,
    'Average self play episode length': av_self_play_episode_length,
    'Data agent path': data_agent_path,
    'Time stamp': str(datetime.now()),
    })
  
  
  
  
  
  
  
  
  
  
  