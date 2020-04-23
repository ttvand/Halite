from kaggle_environments import make as make_environment
from pathlib import Path
from recordtype import recordtype
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import multiprocessing as mp
import numpy as np
import os
import time
import utils

ExperienceStep = recordtype('ExperienceStep', [
  'game_id',
  'current_obs',
  'actions',
  'mapped_actions',
  'valid_actions',
  'network_outputs',
  # 'next_obs',
  'this_agent_action',
  'active_id',
  'episode_step',
  'next_halite',
  'halite_change',
  'num_episode_steps',
  'last_episode_action',
  'episode_reward',
  ])


# Create the first two iterations of an architecture and add it to the pool of
# agents. The first is never trained.
def create_initial_random_agents(agents_folder, agent_config):
  inputs, outputs = agent_config['model'](agent_config)
  model = Model(inputs=inputs, outputs=outputs)
  
  model.compile(loss='mse', optimizer=Adam(lr=0)) # Dummy compilation
  plot_model(model, to_file=os.path.join(agents_folder, 'architecture.png'),
             show_shapes=True)
  
  # # Manual override of weights TO BE COMMENTED!
  # # Motivation: add dropout while using a previous iteration as a startin
  # # point for learning
  # from tensorflow.keras.models import load_model
  # start_weights_folder = os.path.join(agents_folder, 'Start weights')
  # start_weights_path = os.path.join(start_weights_folder, 'iteration_100.h5')
  # start_model = load_model(start_weights_path, custom_objects={
  #   'loss': utils.make_masked_mse(0)})
  # override_weights = start_model.get_weights()
  # model.set_weights(override_weights)
  
  # Save the models to disk
  model_path_0 = os.path.join(agents_folder, 'iteration_0.h5')
  model_path_1 = os.path.join(agents_folder, 'iteration_1.h5')
  model.save(model_path_0)
  model.save(model_path_1)
  
  return [model_path_0, model_path_1]

# Load the current and considered opponent models
def load_pool_models(pool_name, max_pool_size, agent_config,
                     exclude_current_from_opponents,
                     max_pool_size_exclude_current_from_opponent,
                     best_iteration_opponents, record_videos_new_iteration):
  # List all files in the agent pool, create a random agent if there are none
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(this_folder, '../Agents/' + pool_name)
  Path(agents_folder).mkdir(parents=True, exist_ok=True)
  agent_paths = [f for f in os.listdir(agents_folder) if f[-3:]=='.h5']
  if not agent_paths:
    agent_paths = create_initial_random_agents(agents_folder, agent_config)
    if record_videos_new_iteration:
      utils.record_videos(os.path.join(agents_folder, agent_paths[0]),
                          agent_config['num_agents_per_game'],
                          agent_config['num_mirror_dim'])
  
  # Order the agent models on their iteration ids
  agent_paths = utils.decreasing_sort_on_iteration_id(agent_paths)
  agent_paths = agent_paths[:max_pool_size]
  
  # Load all agent models
  agent_full_paths = [os.path.join(agents_folder, p) for p in agent_paths]
  if best_iteration_opponents is None:
    agents = utils.load_models(agent_full_paths)
    this_agent = agents[0]
    if exclude_current_from_opponents or len(
        agents) <= max_pool_size_exclude_current_from_opponent:
      agents = agents[1:]
  else:
    # Load the latest iteration of all opponent pools
    this_agent = utils.load_models(agent_full_paths[:1])[0]
    agents = []
    for opponent_pool in best_iteration_opponents:
      opponent_dir = os.path.join(this_folder, '../Agents/' + opponent_pool)
      opponent_paths = [f for f in os.listdir(opponent_dir) if f[-3:]=='.h5']
      opponent_paths = utils.decreasing_sort_on_iteration_id(opponent_paths)
      opponent_full_path = os.path.join(opponent_dir, opponent_paths[0])
      agents.append(utils.load_models([opponent_full_path])[0])
      
  return this_agent, agents, agent_full_paths

# Select a random opponent agent order based on the starting agent
def get_game_agents(this_agent, opponent_agents, num_agents):
  num_opponent_agents = len(opponent_agents)
  game_agents = [this_agent]
  for _ in range(num_agents-1):
    other_agent_id = np.random.randint(num_opponent_agents)
    other_agent = opponent_agents[other_agent_id]
    game_agents.append(other_agent)
  this_agent_position = 0
    
  return game_agents, this_agent_position, other_agent_id

# Update aggregate and step specific reward statistics
def update_reward(this_episode_reward, episode_rewards, this_game_data,
                  opponent_rewards, opponent_id, num_agents):
  opponent_rewards[opponent_id] = (
    opponent_rewards[opponent_id][0] + 1,
    opponent_rewards[opponent_id][1] + this_episode_reward)
  for i in range(len(this_game_data)):
    active_id = this_game_data[i].active_id
    this_game_data[i].episode_reward = episode_rewards[active_id]

# Sample an exploration parameter
def get_exploration_parameter(agent_config):
  if agent_config['epsilon_greedy']:
    exploration_parameter = np.random.uniform(
      agent_config['epsilon_range'][0],
      agent_config['epsilon_range'][1])
    max_exploration_parameter = agent_config['epsilon_range'][1]
  else:
    exploration_parameter = np.exp(np.random.uniform(
      np.log(agent_config['boltzman_temperature_range'][0]),
      np.log(agent_config['boltzman_temperature_range'][1])))
    max_exploration_parameter = agent_config['boltzman_temperature_range'][1]
    
  return exploration_parameter, max_exploration_parameter

# Naive episode reward computation (but I don't care since it's fast compared
# to the environment): pairwise comparison of all agents.
def get_episode_rewards(halite_scores):
  num_agents = halite_scores.shape[1]
  rewards = np.zeros((num_agents))
  for i in range(num_agents-1):
    for j in range(i+1, num_agents):
      # Pairwise score comparison
      first = np.flip(halite_scores[:, i])
      second = np.flip(halite_scores[:, j])
      unequal_score_ids = np.where(first != second)[0]
      
      if unequal_score_ids.size:
        pairwise_first_score = (
          first[unequal_score_ids[0]] > second[unequal_score_ids[0]]).astype(
            np.float)
      else:
        pairwise_first_score = 0.5
        
      rewards[i] += pairwise_first_score
      rewards[j] += (1-pairwise_first_score)
      
  return rewards/(num_agents-1)

def set_ignored_actions_to_None(
    int_actions, dict_actions, prev_obs, current_obs, previous_observation,
    current_observation):
  grid_size = previous_observation['halite'].shape[0]
  ship_None_action_id = utils.INVERSE_ACTION_MAPPING[utils.SHIP_NONE]
  base_None_action_id = utils.INVERSE_ACTION_MAPPING[utils.BASE_NONE]
  first_convert_spawn = True
  for k in dict_actions:
    # First make sure that the unit is still alive
    if k in current_obs[1] or k in k in current_obs[2]:
      # If the ship is still in the ship dict, it means it was not converted
      if dict_actions[k] == "CONVERT" and k in current_obs[2]:
        row, col = utils.row_col_from_square_grid_pos(
          current_obs[2][k][0], grid_size)
        
        # The ship should still occupy the same square and not be converted
        assert not (current_observation['rewards_bases_ships'][0][1][row, col])
        assert (current_observation['rewards_bases_ships'][0][2][row, col])
        
        import pdb; pdb.set_trace()
        # THIS SHOULD NEVER HAPPEN AFTER THE ENVIRONMENT FIX
        int_actions[row, col, 0] = ship_None_action_id
      
      # Verify if spawn actions created a new ship at the spawn location
      elif dict_actions[k] == "SPAWN":
        base_pos = current_obs[1][k]
        row, col = utils.row_col_from_square_grid_pos(base_pos, grid_size)
        if not current_observation['rewards_bases_ships'][0][2][row, col]:
          
          # Check for ships that were moved to the base location, creating a
          # conflict with the SPAWN command
          ship_at_base = False
          for sk in prev_obs[2]:
            if sk in dict_actions:
              if dict_actions[sk] in ["NORTH", "SOUTH", "EAST", "WEST"]:
                prev_ship_pos = prev_obs[2][sk][0]
                move_ship_pos = utils.move_ship_pos(
                  prev_ship_pos, dict_actions[sk], grid_size)
                if move_ship_pos == base_pos:
                  ship_at_base = True
                  break
            else:
              if prev_obs[2][sk][0] == base_pos:
                ship_at_base = True
                break
              
          if not ship_at_base:
            import pdb; pdb.set_trace()
            # THIS SHOULD NEVER HAPPEN AFTER THE ENVIRONMENT FIX
            int_actions[row, col, 0] = ship_None_action_id
      
      # For move actions: consider if the ship has moved      
      if dict_actions[k] in ["NORTH", "SOUTH", "EAST", "WEST"]:
        if prev_obs[2][k][0] == current_obs[2][k][0]:
          assert not first_convert_spawn
          row, col = utils.row_col_from_square_grid_pos(
            current_obs[2][k][0], grid_size)
          import pdb; pdb.set_trace()
          # THIS SHOULD NEVER HAPPEN AFTER THE ENVIRONMENT FIX
          int_actions[row, col, 2] = base_None_action_id
      
    if dict_actions[k] in ["CONVERT", "SPAWN"]:
      first_convert_spawn = False
  
  return int_actions

def collect_experience_single_game(this_agent, other_agents, num_agents,
                                   agent_config, action_costs, verbose,
                                   game_id):
  episode_start_time = time.time()
  game_agents, this_agent_position, opponent_id = get_game_agents(
    this_agent, other_agents, num_agents)
  
  this_game_data = []
  env = make_environment('halite')
  env.reset(num_agents=num_agents)
  exploration_parameter, max_exploration_parameter = (
    get_exploration_parameter(agent_config))
  max_episode_steps = env.configuration.episodeSteps
  halite_scores = np.full((max_episode_steps, num_agents), np.nan)
  halite_scores[0] = env.state[0].observation.players[0][0]
  episode_step = 0
  
  # Take actions until the game is terminated
  while not env.done:
    env_observation = env.state[0].observation
    player_current_observations = []
    player_current_obs = []
    player_env_obs = []
    player_network_outputs = []
    player_actions = []
    player_mapped_actions = []
    player_valid_actions = []
    store_transition_ids = []
    for active_id in range(num_agents):
      agent_status = env.state[active_id].status
      if agent_status == 'ACTIVE':
        store_transition_ids.append(active_id)
        current_observation = utils.structured_env_obs(
          env.configuration, env_observation, active_id)
        player_obs = env.state[0].observation.players[active_id]
        (current_obs, network_outputs, actions, mapped_actions,
         valid_actions) = utils.get_agent_q_and_a(
            game_agents[active_id], current_observation, player_obs,
            env.configuration, agent_config['epsilon_greedy'],
            exploration_parameter, agent_config['num_mirror_dim'],
            action_costs, pick_first_on_tie=False)
        if verbose:
          print("Player {} obs: {}".format(active_id, player_obs))
          print("Actions: {}\n".format(mapped_actions))
        player_current_observations.append(current_observation)
        player_current_obs.append(current_obs[0][0])
        player_env_obs.append(player_obs)
        player_network_outputs.append(network_outputs)
        player_actions.append(actions)
        player_mapped_actions.append(mapped_actions)
        player_valid_actions.append(valid_actions)
        
      else:
        if agent_status != 'INVALID':
          raise ValueError("Unexpected agent state: {}".format(agent_status))
        player_mapped_actions.append({})
      
    if verbose:
      print("Step: {}; Max halite: {}".format(
        episode_step, current_observation['halite'].max()))
      
    env.step(player_mapped_actions)
    env_observation = env.state[0].observation

    # Store the state transition data
    for i, active_id in enumerate(store_transition_ids):
      next_observation = utils.structured_env_obs(
        env.configuration, env_observation, active_id)
      # next_halite = next_observation['rewards_bases_ships'][0][0]
      # next_obs = utils.state_to_input(next_observation)
      agent_status = env.state[active_id].status
      next_halite = env.state[0].observation.players[active_id][0]
      
      # if next_halite-halite_scores[episode_step, active_id] < -5000:
      #   import pdb; pdb.set_trace()
      
      # Overwrite selected actions to None if the environment did not execute
      # the requested action.
      player_obs = env.state[0].observation.players[active_id]
      player_actions[i] = set_ignored_actions_to_None(
        player_actions[i], player_mapped_actions[active_id], player_env_obs[i],
        player_obs, player_current_observations[i], next_observation)
      
      this_game_data.append(ExperienceStep(
        game_id,
        player_current_obs[i],
        player_actions[i],
        player_mapped_actions[active_id],
        player_valid_actions[i],
        player_network_outputs[i],
        # next_obs, # Dropped out of memory concerns - useful for debugging
        active_id == this_agent_position, # This agent move?
        active_id,
        episode_step,
        next_halite,
        next_halite-halite_scores[episode_step, active_id],
        np.nan, # Number of episode steps, overwritten at the end of episode
        agent_status == 'INVALID', # Last episode action
        np.nan, # Reward, overwritten at the end of the episode
        ))
      
    for i in range(num_agents):
      agent_status = env.state[i].status
      halite_score = -1 if agent_status == 'INVALID' else env.state[
        0].observation.players[i][0]
      halite_scores[episode_step+1, i] = halite_score
    
    episode_step += 1
    
  # Obtain the terminal rewards for all agents
  halite_scores = halite_scores[:episode_step]
  episode_rewards = get_episode_rewards(halite_scores)
  
  # Update statistics which can not be computed before the episode is over.
  for i in range(len(store_transition_ids)):
    this_game_data[-1-i].last_episode_action = True # Last episode action
  for i in range(len(this_game_data)):
    this_game_data[i].num_episode_steps = episode_step
    
  episode_duration = time.time() - episode_start_time
    
  return (this_game_data, episode_rewards, opponent_id, this_agent_position,
          episode_duration)


# Collect experience by playing games of the most recent agent against other
# agents in the pool.
# A fixed number of games is played in each iteration and the opponent and
# starting player is randomly selected in each game.
def play_games(pool_name, num_games, max_pool_size, agent_config,
               exclude_current_from_opponents, symmetric_evaluation,
               action_costs, best_iteration_opponents=None, verbose=False,
               record_videos_new_iteration=False,
               max_pool_size_exclude_current_from_opponent=5,
               use_multiprocessing=False):
  this_agent, other_agents, agent_full_paths = load_pool_models(
    pool_name, max_pool_size, agent_config, exclude_current_from_opponents,
    max_pool_size_exclude_current_from_opponent, best_iteration_opponents,
    record_videos_new_iteration)
  num_opponent_agents = len(other_agents)
  
  
  # Generate experience for a fixed number of games
  experience = []
  reward_sum = 0
  opponent_id_rewards = [(0, 0) for _ in range(num_opponent_agents)]
  num_agents = agent_config['num_agents_per_game']
  if use_multiprocessing:
    # Verify keras models are picklable - see utils.make_keras_picklable
    # import pickle
    # pickle.dumps(this_agent)
    
    # BUGGY FOR NOW - FIX ME (?). Likely issue: multiprocessing and keras (GPU)
    # causing a deadlock
    pool = mp.Pool(processes=mp.cpu_count()-1)
    results = [pool.apply_async(
                collect_experience_single_game, args=(
                  this_agent, other_agents, num_agents, agent_config,
                  action_costs, verbose, g,)) for g in np.arange(num_games)]
    game_outputs = [p.get() for p in results]
  else:
    game_outputs = []
    for game_id in range(num_games):
      single_game_outputs = collect_experience_single_game(
          this_agent, other_agents, num_agents, agent_config, action_costs,
          verbose, game_id)
      game_outputs.append(single_game_outputs)
    
  (n_this_game_data, n_episode_rewards, n_opponent_id, n_this_agent_position,
   n_episode_duration) = list(zip(*[p for p in game_outputs]))
  for game_id in range(num_games):
    # Obtain the terminal rewards for all agents
    this_game_data = n_this_game_data[game_id]
    episode_rewards = n_episode_rewards[game_id]
    opponent_id = n_opponent_id[game_id]
    this_agent_position = n_this_agent_position[game_id]
    
    this_episode_reward = episode_rewards[this_agent_position]
    reward_sum += this_episode_reward
    update_reward(this_episode_reward, episode_rewards, this_game_data,
                  opponent_id_rewards, opponent_id, num_agents)
    
    experience.extend(this_game_data)
    # print("Episode duration: {:.2f}s".format(n_episode_duration[game_id])) 
    
  # Compute the average reward for this agent and the average terminal halite
  # count
  first_final_halite = np.array([e.next_halite for e in experience if (
    e.active_id == 0 and e.last_episode_action)])
  first_final_halite_mean = first_final_halite.mean()
  first_final_halite_median = np.median(first_final_halite)
  first_avg_reward_per_step = np.array(
    [e.halite_change for e in experience if (e.active_id == 0)]).mean()
  actions = np.stack([e.actions for e in experience if (
    e.active_id == 0)])
  act_ids, counts = np.unique(actions[actions>=0], return_counts=True)
  mean_action_frac = np.zeros_like(action_costs)
  mean_action_frac[act_ids] = counts/counts.sum()
  
  print('Summed reward stats in {} games: {:.2f}'.format(
    num_games, reward_sum))
  if best_iteration_opponents is not None:
    print('Rewards versus opponent pool: {}'.format(opponent_id_rewards))
  
  return (experience, agent_full_paths[0], reward_sum/num_games,
          num_opponent_agents, first_final_halite_mean,
          first_final_halite_median,  first_avg_reward_per_step,
          mean_action_frac)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  

  
