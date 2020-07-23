from kaggle_environments import make as make_environment
from kaggle_environments import utils as environment_utils
from pathlib import Path
from recordtype import recordtype
import multiprocessing as mp
import numpy as np
import os
import rule_utils
import time
import utils
ExperienceGame = recordtype('ExperienceGame', [
  'game_id',
  'game_agent_paths',
  'config_game_agents',
  'initial_halite_setup',
  'initial_agents_setup',
  'halite_scores',
  'action_delays',
  'num_episode_steps',
  'episode_rewards',
  'terminal_num_bases',
  'terminal_num_ships',
  'terminal_halite',
  'total_halite_spent',
  'opponent_names',
  'env_random_seed',
  'act_random_seeds',
  # 'first_agent_step_details',
  ])


# Create the first two iterations of the config ranges.
# The first is never evolved.
def create_initial_configs(agents_folder, initial_config_ranges):
  # Save the configs to disk
  agent_path_0 = os.path.join(agents_folder, 'iteration_0.json')
  agent_path_1 = os.path.join(agents_folder, 'iteration_1.json')
  rule_utils.save_config(initial_config_ranges, agent_path_0)
  rule_utils.save_config(initial_config_ranges, agent_path_1)
  
  return [agent_path_0, agent_path_1]

# Load the current and considered opponent configs
def load_pool_agents(pool_name, max_pool_size, exclude_current_from_opponents,
                     max_pool_size_exclude_current_from_opponent,
                     fixed_opponent_pool, record_videos_new_iteration,
                     initial_config_ranges, num_agents):
  # List all files in the agent pool, create a random agent if there are none
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(this_folder, '../Rule agents/' + pool_name)
  Path(agents_folder).mkdir(parents=True, exist_ok=True)
  agent_paths = [f for f in os.listdir(agents_folder) if (
    f[-5:] == '.json' and f != 'initial_config.json')]
  if not agent_paths:
    agent_paths = create_initial_configs(agents_folder, initial_config_ranges)
    if record_videos_new_iteration:
      rule_utils.record_videos(
        os.path.join(agents_folder, agent_paths[0]), num_agents)
  
  # Order the agent models on their iteration ids
  agent_paths = utils.decreasing_sort_on_iteration_id(agent_paths)
  agent_paths = agent_paths[:max_pool_size]
  
  # Load all agent models
  agent_full_paths = [os.path.join(agents_folder, p) for p in agent_paths]
  if not fixed_opponent_pool:
    agents = rule_utils.load_configs(agent_full_paths)
    opponent_names = [p[:-5] for p in agent_paths]
    this_agent = agents[0]
    if exclude_current_from_opponents or len(
        agents) <= max_pool_size_exclude_current_from_opponent:
      agents = agents[1:]
      opponent_names = opponent_names[1:]
  else:
    # Load the latest iteration of all opponent pools
    this_agent = rule_utils.load_configs(agent_full_paths[:1])[0]
    fixed_opponents_folder = os.path.join(
      this_folder, '../Rule agents/Stable opponents pool')
    opponent_files = [f for f in os.listdir(fixed_opponents_folder) if (
      f[-5:] == '.json') or f[-3:] == '.py']
    opponent_names = [e.partition('.')[0] for e in opponent_files]
    opponent_paths = [os.path.join(fixed_opponents_folder, e) for e in (
      opponent_files)]
    agents = rule_utils.load_paths_or_configs(opponent_paths)
      
  return this_agent, agents, agent_full_paths, opponent_names

# Select a random opponent agent order based on the starting agent
def get_game_agents(this_agent, opponent_agents, opponent_names, num_agents,
                    override_opponent_ids):
  num_opponent_agents = len(opponent_agents)
  game_agent_paths = [this_agent]
  game_agents = [rule_utils.sample_from_config(this_agent)]
  other_agent_ids = []
  other_agent_id_names = []
  
  for i in range(num_agents-1):
    if override_opponent_ids is None:
      other_agent_id = np.random.randint(num_opponent_agents)
    else:
      other_agent_id = override_opponent_ids[i]
    other_agent_ids.append(other_agent_id)
    other_agent_id_names.append(opponent_names[other_agent_id])
    other_agent = opponent_agents[other_agent_id]
    game_agent_paths.append(other_agent)
    game_agents.append(rule_utils.sample_from_config_or_path(
      other_agent, return_callable=False))
    
  return game_agent_paths, game_agents, other_agent_ids, other_agent_id_names

# Update aggregate and step specific reward statistics
def update_reward(episode_rewards, opponent_rewards, opponent_ids, num_agents):
  if opponent_ids is not None:
    for i, opponent_id in enumerate(opponent_ids):
      if episode_rewards[i+1] == episode_rewards[0]:
        reward_against_opponent = 0.5
      else: 
        reward_against_opponent = int(
          episode_rewards[0] > episode_rewards[i+1])
      opponent_rewards[opponent_id] = (
        opponent_rewards[opponent_id][0] + 1,
        opponent_rewards[opponent_id][1] + reward_against_opponent,
        opponent_rewards[opponent_id][2])

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
      
  return rewards/(max(2, num_agents)-1)

def get_base_and_ship_counts(env):
  terminal_obs = utils.structured_env_obs(
    env.configuration, env.state[0].observation, 0)
  terminal_base_counts = []
  terminal_ship_counts = []
  for i, (_, bases, ships, _) in enumerate(
      terminal_obs['rewards_bases_ships']):
    terminal_base_counts.append(bases.sum())
    terminal_ship_counts.append(ships.sum())
  
  return terminal_base_counts, terminal_ship_counts

def update_terminal_halite_scores(num_agents, halite_scores, episode_step,
                                  max_episode_steps, env):
  for i in range(num_agents):
    agent_status = env.state[i].status
    valid_agent = (halite_scores[episode_step-1, i] >= 0) and (
      episode_step == (max_episode_steps-1) or (
        agent_status not in ['INVALID', 'TIMEOUT']))
    if valid_agent:
      halite_scores[episode_step, i] = env.state[0].observation.players[i][0]
    else:
      halite_scores[episode_step, i] = -1
  
  halite_scores = halite_scores[:(episode_step+1)]
  
  return halite_scores

def collect_experience_single_game(game_agent_paths, game_agents, num_agents,
                                   verbose, game_id, env_random_seed,
                                   act_random_seeds):
  episode_start_time = time.time()
  
  # Generate reproducible data for better debugging
  utils.set_seed(env_random_seed)
  
  game_agents = [a if isinstance(a, dict) else (
    environment_utils.get_last_callable(a)) for a in game_agents]
  config_game_agents = [a if isinstance(a, dict) else "text_agent" for a in (
    game_agents)]
  
  env = make_environment('halite',
                          configuration = {"randomSeed": env_random_seed}
                         )
  env.reset(num_agents=num_agents)
  max_episode_steps = env.configuration.episodeSteps
  halite_scores = np.full((max_episode_steps, num_agents), np.nan)
  action_delays = np.full((max_episode_steps-1, num_agents), np.nan)
  halite_scores[0] = env.state[0].observation.players[0][0]
  total_halite_spent = np.zeros(num_agents).tolist()
  
  initial_obs = utils.structured_env_obs(
    env.configuration, env.state[0].observation, 0)
  initial_halite_setup = initial_obs['halite']
  initial_agents_setup = np.zeros_like(initial_halite_setup)
  for i, (_, _, ships, _) in enumerate(initial_obs['rewards_bases_ships']):
    initial_agents_setup = initial_agents_setup + (i+1)*ships
  
  # Take actions until the game is terminated
  episode_step = 0
  first_agent_step_details = []
  first_agent_ship_counts = np.zeros(max_episode_steps-1)
  ship_counts = np.full((max_episode_steps, num_agents), np.nan)
  while not env.done:
    env_observation = env.state[0].observation
    player_mapped_actions = []
    for active_id in range(num_agents):
      agent_status = env.state[active_id].status
      if agent_status == 'ACTIVE':
        current_observation = utils.structured_env_obs(
          env.configuration, env_observation, active_id)
        player_obs = env.state[0].observation.players[active_id]
        env_observation.player = active_id
        step_start_time = time.time()
        mapped_actions, halite_spent, step_details = (
          rule_utils.get_config_or_callable_actions(
            game_agents[active_id], current_observation, player_obs,
            env_observation, env.configuration, act_random_seeds[active_id]))
        ship_counts[current_observation['step'], active_id] = len(player_obs[2])
        if active_id == 0:
          first_agent_step_details.append(step_details)
          first_agent_ship_counts[current_observation['step']] = len(
            player_obs[2])
        step_delay = time.time() - step_start_time
        action_delays[episode_step, active_id] = step_delay
        total_halite_spent[active_id] += halite_spent
        if verbose:
          print("Player {} obs: {}".format(active_id, player_obs))
          print("Actions: {}\n".format(mapped_actions))
        player_mapped_actions.append(mapped_actions)
      else:
        player_mapped_actions.append({})
      
    env.step(player_mapped_actions)
    
    for i in range(num_agents):
      agent_status = env.state[i].status
      halite_score = -1 if agent_status in ['INVALID', 'DONE'] else env.state[
        0].observation.players[i][0]
      halite_scores[episode_step+1, i] = halite_score
    
    episode_step += 1
    
  # Write the terminal halite scores
  halite_scores = update_terminal_halite_scores(
    num_agents, halite_scores, episode_step, max_episode_steps, env)
    
  # Evaluate why the game evolved as it did
  import pdb; pdb.set_trace()
  action_override_counts = np.array([first_agent_step_details[i][
    'action_overrides'] for i in range(len(first_agent_step_details))])
  
  # import pdb; pdb.set_trace()
  print("action_override_count:", action_override_counts.sum(0))
    
  # Obtain the terminal rewards for all agents
  episode_rewards = get_episode_rewards(halite_scores)
  
  # Obtain the terminal number of ships and bases for all agents
  terminal_num_bases, terminal_num_ships = get_base_and_ship_counts(env)
  terminal_halite = halite_scores[-1].tolist()
  
  # Store the game data
  this_game_data = ExperienceGame(
        game_id,
        config_game_agents,
        game_agent_paths,
        initial_halite_setup,
        initial_agents_setup,
        halite_scores,
        action_delays,
        episode_step,
        episode_rewards,
        terminal_num_bases,
        terminal_num_ships,
        terminal_halite,
        total_halite_spent,
        None, # Opponent names added outside of this function
        env_random_seed,
        act_random_seeds,
        # first_agent_step_details,
        )
  
  episode_duration = time.time() - episode_start_time
  
  return (this_game_data, episode_duration)

# Collect experience by playing games of the most recent agent against other
# agents in the pool.
# A fixed number of games is played in each iteration and the opponent and
# starting player is randomly selected in each game.
def play_games(pool_name, num_games, max_pool_size, num_agents,
               exclude_current_from_opponents, fixed_opponent_pool=False,
               initial_config_ranges=None, verbose=False,
               record_videos_new_iteration=False,
               max_pool_size_exclude_current_from_opponent=5,
               use_multiprocessing=False, num_repeat_first_configs=1,
               first_config_overrides=None):
  (this_agent, other_agents, agents_full_paths,
   opponent_names) = load_pool_agents(
    pool_name, max_pool_size, exclude_current_from_opponents,
    max_pool_size_exclude_current_from_opponent, fixed_opponent_pool,
    record_videos_new_iteration, initial_config_ranges, num_agents)
  
  # Generate experience for a fixed number of games
  experience = []
  reward_sum = 0
  opponent_id_rewards = [(0, 0, n) for n in opponent_names]
  
  # First generate the random ids in order to obtain seeds that are independent
  # from the number of games
  env_random_seeds = np.zeros((num_games), dtype=np.int)
  act_random_seeds = np.zeros((num_games, 4), dtype=np.int)
  for i in range(num_games):
    env_random_seeds[i] = np.random.randint(0, int(1e9))
    act_random_seeds[i] = np.random.randint(0, int(1e9), size=4)
  
  # Precompute other agent ids in order to obtain stratified opponents
  if fixed_opponent_pool:
    other_sample_probs = rule_utils.fixed_pool_sample_probs(opponent_names)
    num_other_agents = num_games*(num_agents-1)
    other_ids = []
    for i, p in enumerate(other_sample_probs):
      other_ids += [i for _ in range(int(np.ceil(num_other_agents*p)))]
    other_sample_ids = np.random.permutation(other_ids)[
      :num_other_agents].reshape((num_games, -1))
  else:
    other_sample_ids = [None for _ in range(num_games)]
  
  # Set up the game agents for all games
  # A config is repeated n times to reduce the effect of the halite map
  # variation.
  n_game_agent_paths = []
  n_game_agents = []
  n_opponent_ids = []
  n_opponent_id_names = []
  n_env_random_seeds = []
  n_act_random_seeds = []
  for i in range(num_games):
    game_agent_paths, game_agents, opponent_ids, opponent_id_names = (
      get_game_agents(this_agent, other_agents, opponent_names, num_agents,
                      other_sample_ids[i]))
    if i % num_repeat_first_configs == 0:
      if first_config_overrides is not None:
        override_id = i // num_repeat_first_configs
        game_agent_paths[0] = first_config_overrides[override_id]
        game_agents[0] = first_config_overrides[override_id]
    else:
      prev_first_agent_id = i - (i % num_repeat_first_configs)
      game_agent_paths[0] = n_game_agent_paths[prev_first_agent_id][0]
      game_agents[0] = n_game_agents[prev_first_agent_id][0]
    n_game_agent_paths.append(game_agent_paths)
    n_game_agents.append(game_agents)
    n_opponent_ids.append(opponent_ids)
    n_opponent_id_names.append(opponent_id_names)
    n_env_random_seeds.append(int(env_random_seeds[i]))
    n_act_random_seeds.append(act_random_seeds[i])
  
  if use_multiprocessing:
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      results = [pool.apply_async(
                  collect_experience_single_game, args=(
                    gap, ga, num_agents, verbose, g, ers,
                    ars)) for (gap, ga, g, ers, ars) in zip(
                      n_game_agent_paths, n_game_agents, np.arange(num_games),
                      n_env_random_seeds, n_act_random_seeds)]
      game_outputs = [p.get() for p in results]
  else:
    game_outputs = []
    for game_id in range(num_games):
      single_game_outputs = collect_experience_single_game(
          n_game_agent_paths[game_id], n_game_agents[game_id], num_agents,
          verbose, game_id, n_env_random_seeds[game_id],
          n_act_random_seeds[game_id])
      game_outputs.append(single_game_outputs)
    
  (n_this_game_data, n_episode_duration) = list(
    zip(*[p for p in game_outputs]))
  for game_id in range(num_games):
    # Obtain the terminal rewards for all agents
    this_game_data = n_this_game_data[game_id]
    episode_rewards = this_game_data.episode_rewards
    opponent_ids = n_opponent_ids[game_id]
    this_game_data.opponent_names = n_opponent_id_names[game_id]
    
    this_episode_reward = episode_rewards[0]
    reward_sum += this_episode_reward
    update_reward(episode_rewards, opponent_id_rewards, opponent_ids,
                  num_agents)
    
    experience.append(this_game_data)
    # print("Episode duration: {:.2f}s".format(n_episode_duration[game_id])) 
    
  # # Compute the average reward for this agent and the average terminal halite
  # # count
  # first_final_halite = np.array([e.next_halite for e in experience if (
  #   e.active_id == 0 and e.last_episode_action)])
  # first_final_halite_mean = first_final_halite.mean()
  # first_final_halite_median = np.median(first_final_halite)
  # first_avg_reward_per_step = np.array(
  #   [e.halite_change for e in experience if (e.active_id == 0)]).mean()
  # actions = np.stack([e.actions for e in experience if (
  #   e.active_id == 0)])
  # act_ids, counts = np.unique(actions[actions>=0], return_counts=True)
  # mean_action_frac = np.zeros_like(action_costs)
  # mean_action_frac[act_ids] = counts/counts.sum()
  
  print('Summed reward stats in {} games: {:.2f}'.format(
    num_games, reward_sum))
  if fixed_opponent_pool:
    print('Rewards versus fixed opponent pool: {}'.format(opponent_id_rewards))
  
  # import pdb; pdb.set_trace()
  return (experience, agents_full_paths[0], reward_sum/num_games,
          opponent_id_rewards)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  

  
