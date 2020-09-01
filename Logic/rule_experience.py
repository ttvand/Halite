from kaggle_environments import make as make_environment
from kaggle_environments import agent as kaggle_agent
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
  'all_first_durations',
  'action_delays',
  'first_get_actions_durations',
  'first_box_in_durations',
  'first_history_durations',
  'first_ship_scores_durations',
  'first_ship_plans_durations',
  'first_ship_map_durations',
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
  'game_recording',
  'num_lost_ships',
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
def update_reward(episode_rewards, opponent_rewards, opponent_ids, num_agents,
                  rule_actions_id):
  if opponent_ids is not None:
    env_obs_ids = [i for i in range(num_agents)]
    env_obs_ids.remove(rule_actions_id)
    
    for i, opponent_id in enumerate(opponent_ids):
      other_id = env_obs_ids[i]
      if episode_rewards[other_id] == episode_rewards[rule_actions_id]:
        reward_against_opponent = 0.5
      else: 
        reward_against_opponent = int(
          episode_rewards[rule_actions_id] > episode_rewards[other_id])
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

def get_lost_ships_count(player_mapped_actions, prev_players, current_players,
                         prev_observation, verbose_id=-1):
  num_players = len(current_players)
  num_lost_ships = np.zeros(num_players)
  
  prev_bases = np.stack([rbs[1] for rbs in prev_observation[
    'rewards_bases_ships']]).sum(0) > 0
  grid_size = prev_bases.shape[0]
  
  new_convert_positions = np.zeros_like(prev_bases)
  for i in range(num_players):
    prev_player = prev_players[i]
    prev_actions = player_mapped_actions[i]
    for ship_k in prev_actions:
      if prev_actions[ship_k] == "CONVERT":
        new_convert_positions[utils.row_col_from_square_grid_pos(
          prev_player[2][ship_k][0], grid_size)] = 1
  
  for i in range(num_players):
    prev_player = prev_players[i]
    current_player = current_players[i]
    prev_actions = player_mapped_actions[i]
    was_alive = len(prev_player[2]) > 0 or (
      len(prev_player[1]) > 0 and prev_player[0] >= 500)
    
    if was_alive:
      # Loop over all ships and figure out if a ship was lost unintentionally
      for ship_k in prev_player[2]:
        if not ship_k in current_player[2]:
          if (not ship_k in prev_actions) or (
              ship_k in prev_actions and prev_actions[ship_k] != "CONVERT"): 
            row, col = utils.row_col_from_square_grid_pos(
              prev_player[2][ship_k][0], grid_size)
            ship_action = None if not ship_k in prev_actions else prev_actions[
              ship_k]
            move_row, move_col = rule_utils.move_ship_row_col(
              row, col, ship_action, grid_size)
            if not prev_bases[move_row, move_col]:
              # Don't count base attack/defense ship loss or self collision
              # towards the end of the game
              if (not move_row*grid_size + move_col in [
                  s[0] for s in current_player[2].values()]) and (
                    prev_observation['relative_step'] < 0.975):
                # Don't count self collisions or collisions with a new base
                if not new_convert_positions[move_row, move_col]:
                  if i == verbose_id:
                    # import pdb; pdb.set_trace()
                    print("Lost ship at step", prev_observation['step']+1)
                num_lost_ships[i] += 1
          elif prev_actions[ship_k] == "CONVERT" and (
              prev_observation['relative_step'] >= 0.025) and (
                prev_observation['relative_step'] <= 0.975):
            # The ship most likely got boxed in and was forced to convert. 
            # Note that this also counts lost ships due to losing the base.
            if i == verbose_id:
              print("Lost ship at step", prev_observation['step']+1)
            num_lost_ships[i] += 1
      
  return num_lost_ships

def collect_experience_single_game(
    game_agent_paths, game_agents, num_agents, verbose, game_id,
    env_random_seed, act_random_seeds, record_game, episode_steps_override,
    rule_actions_id):
  episode_start_time = time.time()
  
  # Generate reproducible data for better debugging
  utils.set_seed(env_random_seed)
  
  game_agents = [a if isinstance(a, dict) else (
    kaggle_agent.get_last_callable(a)) for a in game_agents]
  config_game_agents = [a if isinstance(a, dict) else "text_agent" for a in (
    game_agents)]
  
  # Add option to shuffle the location of the main agent - for now this serves
  # for testing the stateful history logic.
  first_rule_agent = game_agents.pop(0)
  game_agents.insert(rule_actions_id, first_rule_agent)
  
  env_config = {"randomSeed": env_random_seed}
  if episode_steps_override is not None:
    env_config["episodeSteps"] = episode_steps_override
  env = make_environment('halite', configuration=env_config)
  env.reset(num_agents=num_agents)
  max_episode_steps = env.configuration.episodeSteps
  halite_scores = np.full((max_episode_steps, num_agents), np.nan)
  action_delays = np.full((max_episode_steps-1, num_agents), np.nan)
  first_get_actions_durations = np.full(max_episode_steps-1, np.nan)
  first_box_in_durations = np.full(max_episode_steps-1, np.nan)
  first_history_durations = np.full(max_episode_steps-1, np.nan)
  first_ship_scores_durations = np.full(max_episode_steps-1, np.nan)
  first_ship_plans_durations = np.full(max_episode_steps-1, np.nan)
  first_ship_map_durations = np.full(max_episode_steps-1, np.nan)
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
  num_lost_ships = np.zeros((max_episode_steps-1, num_agents), dtype=np.int)
  first_agent_step_details = []
  first_agent_ship_counts = np.zeros(max_episode_steps-1)
  ship_counts = np.full((max_episode_steps-1, num_agents), np.nan)
  histories = [{} for i in range(num_agents)]
  while not env.done:
    env_observation = env.state[0].observation
    player_mapped_actions = []
    for active_id in range(num_agents):
      agent_status = env.state[active_id].status
      players = env.state[0].observation.players
      if agent_status == 'ACTIVE':
        current_observation = utils.structured_env_obs(
          env.configuration, env_observation, active_id)
        player_obs = players[active_id]
        env_observation.player = active_id
        step_start_time = time.time()
        mapped_actions, updated_history, halite_spent, step_details = (
          rule_utils.get_config_or_callable_actions(
            game_agents[active_id], current_observation, player_obs,
            env_observation, env.configuration, histories[active_id],
            act_random_seeds[active_id]))
        histories[active_id] = updated_history
        ship_counts[current_observation['step'], active_id] = len(
          player_obs[2])
        if active_id == rule_actions_id:
          first_agent_step_details.append(step_details)
          first_get_actions_durations[episode_step] = step_details[
            'get_actions_duration']
          first_box_in_durations[episode_step] = step_details[
            'box_in_duration']
          first_history_durations[episode_step] = step_details[
            'history_start_duration']
          first_ship_scores_durations[episode_step] = step_details[
            'ship_scores_duration']
          first_ship_plans_durations[episode_step] = step_details[
            'ship_plans_duration']
          first_ship_map_durations[episode_step] = step_details[
            'ship_map_duration']
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
    
    num_lost_ships[episode_step] = get_lost_ships_count(
      player_mapped_actions, players, env.state[0].observation.players,
      current_observation, verbose_id=rule_actions_id+0.5)
    
    episode_step += 1
    
  # Write the terminal halite scores
  halite_scores = update_terminal_halite_scores(
    num_agents, halite_scores, episode_step, max_episode_steps, env)
    
  # Evaluate why the game evolved as it did
  # import pdb; pdb.set_trace()
  action_override_counts = np.array([first_agent_step_details[i][
    'action_overrides'] for i in range(len(first_agent_step_details))])
  
  print("Action override counts:", action_override_counts.sum(0))
  print("Num lost ships:", num_lost_ships.sum(0))
    
  # Obtain the terminal rewards for all agents
  episode_rewards = get_episode_rewards(halite_scores)
  
  # Obtain the terminal number of ships and bases for all agents
  terminal_num_bases, terminal_num_ships = get_base_and_ship_counts(env)
  terminal_halite = halite_scores[-1].tolist()
  print("Terminal halite:", terminal_halite)
  
  # Generate the episode recording if requested
  if record_game:
    game_recording = env.render(mode="html", width=800, height=600)
  else:
    game_recording = None
  
  # Combine the different first player durations into a matrix for better
  # analysis
  all_first_durations = np.stack([
    action_delays[:, rule_actions_id],
    first_get_actions_durations,
    first_box_in_durations,
    first_history_durations,
    first_ship_scores_durations,
    first_ship_plans_durations,
    first_ship_map_durations,
    ], -1)
  
  # Store the game data
  this_game_data = ExperienceGame(
        game_id,
        config_game_agents,
        game_agent_paths,
        initial_halite_setup,
        initial_agents_setup,
        halite_scores,
        all_first_durations,
        action_delays,
        first_get_actions_durations,
        first_box_in_durations,
        first_history_durations,
        first_ship_scores_durations,
        first_ship_plans_durations,
        first_ship_map_durations,
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
        game_recording,
        num_lost_ships,
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
               first_config_overrides=None, episode_steps_override=None):
  num_skipped = 0 # ONLY USE THIS FOR DEBUGGING
  rule_actions_id = 2 # USED TO PLAY THE MAIN AGENTS AS DIFFERENT IDS
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
  n_record_games = []
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
    n_record_games.append(i == num_skipped)
  
  if use_multiprocessing:
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      results = [pool.apply_async(
                  collect_experience_single_game, args=(
                    gap, ga, num_agents, verbose, g, ers,
                    ars, rg, episode_steps_override, rule_actions_id)) for (
                      gap, ga, g, ers, ars, rg) in zip(
                        n_game_agent_paths[num_skipped:],
                        n_game_agents[num_skipped:],
                        np.arange(num_games)[num_skipped:],
                        n_env_random_seeds[num_skipped:],
                        n_act_random_seeds[num_skipped:],
                        n_record_games[num_skipped:])]
      game_outputs = [p.get() for p in results]
      
  else:
    game_outputs = []
    for game_id in range(num_skipped, num_games):
      single_game_outputs = collect_experience_single_game(
          n_game_agent_paths[game_id], n_game_agents[game_id], num_agents,
          verbose, game_id, n_env_random_seeds[game_id],
          n_act_random_seeds[game_id], n_record_games[game_id],
          episode_steps_override, rule_actions_id)
      game_outputs.append(single_game_outputs)
    
  (n_this_game_data, n_episode_duration) = list(
    zip(*[p for p in game_outputs]))
  for game_id in range(num_skipped, num_games):
    # Obtain the terminal rewards for all agents
    this_game_data = n_this_game_data[game_id-num_skipped]
    episode_rewards = this_game_data.episode_rewards
    opponent_ids = n_opponent_ids[game_id]
    this_game_data.opponent_names = n_opponent_id_names[game_id]
    
    this_episode_reward = episode_rewards[rule_actions_id]
    reward_sum += this_episode_reward
    update_reward(episode_rewards, opponent_id_rewards, opponent_ids,
                  num_agents, rule_actions_id)
    
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
  
  import pdb; pdb.set_trace()
  print('Summed reward stats in {} games: {:.2f}'.format(
    num_games, reward_sum))
  if fixed_opponent_pool:
    print('Rewards versus fixed opponent pool: {}'.format(opponent_id_rewards))
  
  # import pdb; pdb.set_trace()
  return (experience, agents_full_paths[0], reward_sum/num_games,
          opponent_id_rewards)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  

  
