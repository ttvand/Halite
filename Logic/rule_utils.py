# import copy
import json
from kaggle_environments import make as make_environment
from kaggle_environments import utils as environment_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy import stats
import seaborn as sns
import utils
import rule_actions_v1
import rule_actions_v2



CONVERT = "CONVERT"
SPAWN = "SPAWN"

MOVE_DIRECTIONS = [None, "NORTH", "SOUTH", "EAST", "WEST"]

HALITE_MULTIPLIER_CONFIG_ENTRIES = [
  # V1
  "ship_halite_cargo_conversion_bonus_constant",
  "friendly_ship_halite_conversion_constant",
  "nearby_halite_conversion_constant",
  
  "halite_collect_constant",
  "nearby_halite_move_constant",
  "nearby_onto_halite_move_constant",
  "nearby_base_move_constant",
  "nearby_move_onto_base_constant",
  "adjacent_opponent_ships_move_constant",
  
  "nearby_ship_halite_spawn_constant",
  "nearby_halite_spawn_constant",
          ]

GAUSSIAN_2D_KERNELS = {}
for dim in range(3, 16, 2):
  # Modified from https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
  center_distance = np.floor(np.abs(np.arange(dim) - (dim-1)/2))
  horiz_distance = np.tile(center_distance, [dim, 1])
  vert_distance = np.tile(np.expand_dims(center_distance, 1), [1, dim])
  manh_distance = horiz_distance + vert_distance
  kernel = np.exp(-manh_distance/(dim/4))
  kernel[manh_distance > dim/2] = 0
  
  GAUSSIAN_2D_KERNELS[dim] = kernel
  
FIXED_POOL_AGENT_WEIGHTS = {
    'Single base no spawns': 1,
    'Swarm intelligence': 3,
    'Stochastic swarm intelligence': 3,
    'Self play rule_actions_v2 optimum 1': 2,
    'Self play rule_actions_v2 optimum 2': 2,
    'Self play rule_actions_v2 optimum 2 additional rules 3': 2,
    'Self play rule_actions_v2 optimum 2 additional rules 3 stochastic': 3,
    'Self play rule_actions_v2 optimum 3': 2,
    'Self play rule_actions_v2 optimum 3 additional rules 2': 2,
    'Self play rule_actions_v2 optimum 3 additional rules 3': 2,
    'Self play rule_actions_v2 optimum 3 additional rules 4': 2,
    'Self play rule_actions_v2 optimum 3 additional rules 5': 2,
    'Self play rule_actions_v2 optimum 3 additional rules 5 stochastic': 5,
    # 'Greedy - many spawns and conversions': 2,
    # 'Run yard one ship': 1,
    # 'Self play optimum 1': 2,
    # 'Stable opponents pool optimum 1': 2
    }

def get_action_costs():
  costs = utils.get_action_costs().tolist()
  actions = utils.INVERSE_ACTION_MAPPING.keys()
  
  return dict(zip(actions, costs))

def add_action_costs_to_config(config):
   config['action_costs'] = get_action_costs()

def store_config_on_first_run(config):
  pool_name = config['pool_name']
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(this_folder, '../Rule agents/' + pool_name)
  Path(agents_folder).mkdir(parents=True, exist_ok=True)
  f = os.path.join(agents_folder, 'initial_config.json')
  if not os.path.exists(f):
    with open(f, 'w') as outfile:
      outfile.write(json.dumps(config, indent=4))
      
def update_learning_progress(experiment_name, data_vals): 
  # Append a line to the learning progress line if the file exists. Otherwise:
  # create it
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Rule agents/' + experiment_name)
  progress_path = os.path.join(agents_folder, 'learning_progress.csv')
  
  if os.path.exists(progress_path):
    progress = pd.read_csv(progress_path)
    try:
      progress.loc[progress.shape[0]] = data_vals
    except:
      # This can happen if the stable opponents pool changes throughout a
      # single agent pool iteration.
      import pdb; pdb.set_trace()
  else:
    progress = pd.DataFrame(data_vals, index=[0])
  
  progress.to_csv(progress_path, index=False)
  
def append_config_scores(config_settings, scores, config_keys,
                         experiment_name, save_extension):
  # Append lines to the config results if the file exists. Otherwise: create it
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Rule agents/' + experiment_name)
  settings_path = os.path.join(agents_folder, save_extension)
  
  if os.path.exists(settings_path):
    results = pd.read_csv(settings_path)
    try:
      for i in range(len(scores)):
        results.loc[results.shape[0]] = config_settings[i] + [-scores[i]]
    except:
      import pdb; pdb.set_trace()
  else:
    results = pd.DataFrame(config_settings, columns=config_keys)
    results['Average win rate'] = [-s for s in scores]
  
  results.to_csv(settings_path, index=False)
  
  return results
  
def serialize_game_experience_for_learning(
    experience, only_store_first, config_keys):
  # Create a row for each config - result pair
  list_of_dicts_x = [c for d in experience for c in d.config_game_agents]
  default_config = dict(zip(config_keys, [0 for _ in config_keys]))
  list_of_dicts_x = [c if isinstance(c, dict) else default_config for c in (
    list_of_dicts_x)]
  dict_of_lists_x = {
    k: [dic[k] for dic in list_of_dicts_x] for k in list_of_dicts_x[0]}
  
  list_of_dicts_sh = [{'terminal_num_ships': sh} for d in experience for sh in(
    d.terminal_num_ships)]
  dict_of_lists_sh = {
    k: [dic[k] for dic in list_of_dicts_sh] for k in list_of_dicts_sh[0]}
  
  list_of_dicts_b = [{'terminal_num_bases': b} for d in experience for b in (
    d.terminal_num_bases)]
  dict_of_lists_b = {
    k: [dic[k] for dic in list_of_dicts_b] for k in list_of_dicts_b[0]}
  
  list_of_dicts_sc = [{'episode_reward': sc} for d in experience for sc in (
    d.episode_rewards)]
  dict_of_lists_sc = {
    k: [dic[k] for dic in list_of_dicts_sc] for k in list_of_dicts_sc[0]}
  
  list_of_dicts_h = [{'terminal_halite': h} for d in experience for h in (
    d.terminal_halite)]
  dict_of_lists_h = {
    k: [dic[k] for dic in list_of_dicts_h] for k in list_of_dicts_h[0]}
  
  combined = {}
  for d in [dict_of_lists_x, dict_of_lists_sh, dict_of_lists_b,
            dict_of_lists_sc, dict_of_lists_h]:
    combined.update(d)
    
  combined_df = pd.DataFrame.from_dict(combined)
  if only_store_first:
    num_agents = len(experience[0].terminal_halite)
    combined_df = combined_df.iloc[::num_agents]
    
  for k in config_keys:
    if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
      combined_df[k] *= combined_df['halite_config_setting_divisor']
    
  return combined_df

def clip_ranges(values, ranges):
  for i in range(len(values)):
    values[i] = np.maximum(np.minimum(np.array(values[i]),
                                      ranges[i][1]), ranges[i][0]).tolist()
    
  return values

def get_self_play_experience_path(experiment_name):
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(
    this_folder, '../Rule agents/' + experiment_name)
  data_path = os.path.join(agents_folder, 'self_play_experience.csv')
  
  return data_path
  
def write_experience_data(experiment_name, data):
  # Append to the learning progress line if the file exists.
  # Otherwise: create it
  data_path = get_self_play_experience_path(experiment_name)
  
  if os.path.exists(data_path):
    old_data = pd.read_csv(data_path)
    save_data = pd.concat([old_data, data], axis=0)
  else:
    save_data = data
  
  save_data.to_csv(data_path, index=False)
  
  return data_path

def plot_reward_versus_features(
    features_rewards_path, data, plot_name_suffix,
    target_col="episode_reward", include_all_targets=False, all_scatter=False):
  # data = pd.read_csv(features_rewards_path)
  folder, _ = tuple(features_rewards_path.rsplit('/', 1))
  plots_folder = os.path.join(folder, 'Plots')
  Path(plots_folder).mkdir(parents=True, exist_ok=True)
  
  other_columns = data.columns.tolist()
  other_columns.remove(target_col)
  targets = data[target_col].values
  targets_rounded = np.round(targets, 3)
  
  num_plots = len(other_columns)
  grid_size = int(np.ceil(np.sqrt(num_plots)))
  fig = plt.figure(figsize=[3*6.4, 3*4.8])
  plt.subplots_adjust(hspace=0.8, wspace=0.4)
  
  for i, c in enumerate(other_columns):
    ax = fig.add_subplot(grid_size, grid_size, i+1)
    x_vals = data[c].values
    if not all_scatter and np.abs(
        x_vals - x_vals.astype(np.int)).mean() < 1e-8:
      # Drop data points corresponding with tied results to make the
      # categorical distinction more apparent
      if include_all_targets:
        plot_ids = np.where(np.abs(np.mod(targets*3, 1)) < 1e-8)[0]
      else:
        plot_ids = np.ones_like(targets, dtype=np.bool)
      try:
        sns.violinplot(x=targets_rounded[plot_ids], y=x_vals[plot_ids],
                       ax=ax).set(title=c)
      except:
        import pdb; pdb.set_trace()
        x=1
    else:
      sns.regplot(x=x_vals, y=targets, ax=ax).set(title=c)
      
  fig = ax.get_figure()
  fig.savefig(os.path.join(
    plots_folder, 'combined ' + plot_name_suffix + '.png'))
  plt.clf()
  plt.close()
  
def save_config(config, path):
  with open(path, 'w') as outfile:
    outfile.write(json.dumps(config, indent=4))
    
# Load all json configs for the given paths
def load_configs(paths):
  configs = []
  for p in paths:
    with open(p) as f:
      configs.append(json.load(f))
    
  return configs

# Load all json configs for the given paths
def load_paths_or_configs(paths):
  configs = []
  for p in paths:
    if p[-5:] == ".json":
      with open(p) as f:
        configs.append(json.load(f))
    else:
      configs.append(p)
    
  return configs

def evolve_config(config_path, data, initial_config_ranges,
                  target_col="episode_reward", min_slope_cutoff=0.2,
                  significant_range_reduction=0.95, relative_shift=0.2):
  config = load_configs([config_path])[0]
  y_vals = data[target_col].values
  for k in config:
    x_vals = data[k].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(
      x_vals, y_vals)
    
    if np.abs(r_value) < min_slope_cutoff:
      # Don't adjust the range if the slope is not significant
      pass
    else:
      current_range = config[k][0][1]-config[k][0][0]
      next_range = current_range*significant_range_reduction
      if r_value > 0:
        # Shift the config up and slightly shrink the search range
        min_val = config[k][0][0]+relative_shift*current_range
      else:
        min_val = max(config[k][2],
                      config[k][0][0]-relative_shift*current_range)
      config[k][0][0] = min_val
      config[k][0][1] = config[k][0][0]+next_range

  save_config(config, config_path)

def mirror_edges(observation, num_mirror_dim):
  if num_mirror_dim > 0:
    # observation = np.arange(225).reshape((15,15)) # Debugging test
    assert len(observation.shape) == 2
    grid_size = observation.shape[0]
    new_grid_size = grid_size + 2*num_mirror_dim
    mirrored_obs = np.full((new_grid_size, new_grid_size), np.nan)
    
    # Fill in the original data
    mirrored_obs[num_mirror_dim:(-num_mirror_dim),
                 num_mirror_dim:(-num_mirror_dim)] = observation
    
    # Add top and bottom mirrored data
    mirrored_obs[:num_mirror_dim, num_mirror_dim:(
      -num_mirror_dim)] = observation[-num_mirror_dim:, :]
    mirrored_obs[-num_mirror_dim:, num_mirror_dim:(
      -num_mirror_dim)] = observation[:num_mirror_dim, :]
    
    # Add left and right mirrored data
    mirrored_obs[:, :num_mirror_dim] = mirrored_obs[
      :, -(2*num_mirror_dim):(-num_mirror_dim)]
    mirrored_obs[:, -num_mirror_dim:] = mirrored_obs[
      :, num_mirror_dim:(2*num_mirror_dim)]
    
    observation = mirrored_obs
  
  return observation

def smooth2d(grid, smooth_kernel_dim=7, return_kernel=False):
  edge_augmented = mirror_edges(grid, smooth_kernel_dim-1)
  kernel = GAUSSIAN_2D_KERNELS[int(2*smooth_kernel_dim-1)]
  convolved = signal.convolve2d(edge_augmented, kernel, mode="valid")
  
  if return_kernel:
    return convolved, kernel
  else:
    return convolved

def move_ship_row_col(row, col, direction, size):
  if direction == "NORTH":
    return (size-1 if row == 0 else row-1, col)
  elif direction == "SOUTH":
    return (row+1 if row < (size-1) else 0, col)
  elif direction == "EAST":
    return (row, col+1 if col < (size-1) else 0)
  elif direction == "WEST":
    return (row, size-1 if col == 0 else col-1)
  elif direction is None:
    return (row, col)
  
def add_warped_kernel(grid, kernel, row, col):
  kernel_dim = kernel.shape[0]
  assert kernel_dim % 2 == 1
  half_kernel_dim = int((kernel_dim-1)/2)
  
  addition = np.zeros_like(grid)
  addition[:kernel_dim, :kernel_dim] = kernel
  row_roll = row-half_kernel_dim
  col_roll = col-half_kernel_dim
  addition = np.roll(addition, row_roll, axis=0)
  addition = np.roll(addition, col_roll, axis=1)
  
  return grid + addition

def get_config_actions(config, observation, player_obs, env_config,
                       verbose=False, version_1=False):
  call_module = rule_actions_v1 if version_1 else rule_actions_v2
  return call_module.get_config_actions(
    config, observation, player_obs, env_config, verbose)

def get_config_or_callable_actions(config_or_callable, observation, player_obs,
                                   env_observation, env_config, verbose=False):
  if isinstance(config_or_callable, dict):
    return get_config_actions(config_or_callable, observation, player_obs,
                              env_config, verbose)
  else:
    mapped_actions = config_or_callable(env_observation, env_config)
    
    # Infer the amount of halite_spent from the actions
    halite_spent = 0
    for k in mapped_actions:
      if mapped_actions[k] == SPAWN:
        halite_spent += env_config.spawnCost
      elif mapped_actions[k] == CONVERT:
        halite_spent += env_config.convertCost
    
    return mapped_actions, halite_spent, None

def get_next_config_settings(
    opt, config_keys, num_games, num_repeat_first_configs, config_ranges):
  num_suggested = int(np.ceil(num_games/num_repeat_first_configs))
  if opt is None:
    suggested = [[config_ranges[k] for k in config_keys] for _ in range(
      num_suggested)]
    suggested_configs = [config_ranges for _ in range(num_suggested)]
  else:
    suggested = opt.ask(num_suggested)
    suggested_configs = [suggested_to_config(s, config_keys) for s in suggested]
  
  return suggested, suggested_configs

def suggested_to_config(suggested, config_keys):
  config = {}
  for i, k in enumerate(config_keys):
    config[k] = suggested[i]
  for k in config_keys:
    if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
      config[k] /= config['halite_config_setting_divisor']
      
  return config

def sample_from_config_or_path(config_or_path, return_callable):
  if isinstance(config_or_path, str):
    agent_file = environment_utils.read_file(config_or_path)
    if return_callable:
      return environment_utils.get_last_callable(agent_file)
    else:
      return agent_file
  else:
    return sample_from_config(config_or_path)

def sample_from_config(config):
  if not isinstance(config[list(config.keys())[0]], list):
    # The config is already sampled from - just return the original config
    return config
  
  sampled_config = {}
  for k in config:
    if config[k][0][0] == config[k][0][1]:
      sampled_config[k] = config[k][0][0]
    else:
      if config[k][1] == "float":
        sampled_config[k] = np.random.uniform(config[k][0][0], config[k][0][1])
      elif config[k][1] == "int":
        sampled_config[k] = np.random.randint(
          np.floor(config[k][0][0]), np.ceil(config[k][0][1]))
      else:
        raise ValueError("Unsupported sampling type '{}'".format(config[k][1]))
  
  for k in config:
    if k in HALITE_MULTIPLIER_CONFIG_ENTRIES:
      sampled_config[k] /= sampled_config['halite_config_setting_divisor']
  
  return sampled_config

def fixed_pool_sample_probs(opponent_names):
  sample_probs = np.zeros(len(opponent_names))
  for i, n in enumerate(opponent_names):
    sample_probs[i] = FIXED_POOL_AGENT_WEIGHTS[n]
    
  return sample_probs/sample_probs.sum()

def record_videos(agent_path, num_agents_per_game, extension_override=None,
                  config_override_agents=None):
  print("Generating videos of iteration {}".format(agent_path))
  env = make_environment(
    "halite", configuration={"agentExec": "LOCAL"})#, configuration={"agentTimeout": 10000, "actTimeout": 10000})
  config = load_configs([agent_path])[0]
  env_configuration = env.configuration
  
  def my_agent(observation, config_id):
    config = AGENT_CONFIGS[config_id]
    active_id = observation.player
    current_observation = utils.structured_env_obs(
      env_configuration, observation, active_id)
    player_obs = observation.players[active_id]
    
    mapped_actions, _, _ = get_config_or_callable_actions(
      config, current_observation, player_obs, observation, env_configuration)
    
    return mapped_actions
  
  if config_override_agents is None:
    AGENT_CONFIGS = []
    for i in range(num_agents_per_game):
      AGENT_CONFIGS.append(sample_from_config(config))
  else:
    AGENT_CONFIGS = [sample_from_config_or_path(
      p, return_callable=True) for p in config_override_agents]
  
  # For some reason this needs to be verbose - list comprehension breaks the
  # stochasticity of the agents.
  config_id_agents = [
    lambda observation: my_agent(observation, 0),
    lambda observation: my_agent(observation, 1),
    lambda observation: my_agent(observation, 2),
    lambda observation: my_agent(observation, 3),
    ][:num_agents_per_game]
  
  for video_type in ["self play", "random opponent"]:
    env.reset(num_agents=num_agents_per_game)
    agents = config_id_agents if video_type == "self play" else [
      config_id_agents[0]] + ["random"]*(num_agents_per_game-1)
    
    env.run(agents)
    
    if config_override_agents is not None and video_type == "self play":
      video_type = "; ".join([
        a.rsplit('/', 1)[-1][:-3] for a in config_override_agents[1:]])
    
    # Save the HTML recording in the videos folder
    game_recording = env.render(mode="html", width=800, height=600)
    folder, extension = tuple(agent_path.rsplit('/', 1))
    videos_folder = os.path.join(folder, 'Videos')
    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    ext = extension[:-5] if extension_override is None else extension_override
    video_path = os.path.join(videos_folder, ext+' - '+video_type+'.html')
    with open(video_path,"w") as f:
      f.write(game_recording)