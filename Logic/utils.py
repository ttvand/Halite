import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import base64 as b64
from kaggle_environments import make as make_environment
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import tensorflow as tf
# import time


###############################################################################
# Acting utilities                                                            #
###############################################################################

ACTION_MAPPING = {
  0: "NORTH",
  1: "SOUTH",
  2: "EAST",
  3: "WEST",
  4: "CONVERT",
  5: "SPAWN",
  6: None,
  }

def get_input_output_shapes(config):
  # Create a new environment, perform the preprocessing and record the shape
  env = make_environment('halite')
  env.reset(num_agents=config['num_agents_per_game'])
  env_configuration = env.configuration
  env_observation = env.state[0].observation
  obs_input = state_to_input(structured_env_obs(
    env_configuration, env_observation, active_id=0))
  num_actions = len(ACTION_MAPPING)
  
  return obs_input.shape, num_actions

def get_action_costs():
  # Create a new environment, read the config and record the action costs
  env = make_environment('halite')
  action_costs = np.zeros((len(ACTION_MAPPING)))
  for k in ACTION_MAPPING:
    if ACTION_MAPPING[k] == "CONVERT":
      action_costs[k] = env.configuration.convertCost
    elif ACTION_MAPPING[k] == "SPAWN":
      action_costs[k] = env.configuration.spawnCost
    else:
      action_costs[k] = 0
      
  return action_costs

def add_action_costs_to_config(config):
  config['action_costs'] = get_action_costs()

def row_col_from_square_grid_pos(pos, size):
  col = pos % size
  row = pos // size
  
  return row, col

def get_base_pos(base_data, grid_size):
  base_pos = np.zeros((grid_size, grid_size))
  for _, v in base_data.items():
    row, col = row_col_from_square_grid_pos(v, grid_size)
    base_pos[row, col] = 1
  
  return base_pos

def get_ship_halite_pos(ship_data, grid_size):
  ship_pos = np.zeros((grid_size, grid_size))
  ship_halite = np.zeros((grid_size, grid_size))
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
  relative_step = step/num_episode_steps
  
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
    }
  

def select_action_from_q(valid_actions, q_values, epsilon_greedy,
                         exploration_parameter, pick_first_on_tie):
  if valid_actions.dtype == np.bool:
    valid_actions = np.where(valid_actions)[0]
  
  if epsilon_greedy:
    # Select the best valid or a valid exploratory action
    best_q = q_values[valid_actions].max()
    best_a_ids = np.where(q_values[valid_actions] == best_q)[0]
    if pick_first_on_tie:
      best_a_ids = best_a_ids[:1]
    best_a = valid_actions[np.random.choice(best_a_ids)]
    exploratory_a = np.random.choice(valid_actions)
    explore = np.random.uniform() < exploration_parameter
    action = exploratory_a if explore else best_a
  else:
    # -500 to avoid overflow when exploration parameter is in [1e-3, 1]
    if exploration_parameter == 0:
      # Select the best valid action
      best_q = q_values[valid_actions].max()
      best_a_ids = np.where(q_values[valid_actions] == best_q)[0]
      if pick_first_on_tie:
        best_a_ids = best_a_ids[:1]
      action = valid_actions[np.random.choice(best_a_ids)]
    else:
      temp_valid_qs = np.array(
        [np.exp(q/exploration_parameter-500) for q in q_values[valid_actions]])
      sampled_q_value = np.random.choice(temp_valid_qs,
                                         p=temp_valid_qs/temp_valid_qs.sum())
      sampled_a_ids = np.where(temp_valid_qs == sampled_q_value)[0]
      if pick_first_on_tie:
        sampled_a_ids = sampled_a_ids[:1]
      action = valid_actions[np.random.choice(sampled_a_ids)]
    
  return action

# Predict network outputs where the number of inputs can be large. Use batching
# when there are more inputs than the max_batch_size
def my_keras_predict(model, inputs, max_batch_size=200):
  num_inputs = inputs[0].shape[0]
  num_batches = int(np.ceil(num_inputs/max_batch_size))
  
  outputs = []
  for i in range(num_batches):
    end_id = num_inputs if i == (num_batches-1) else (i+1)*max_batch_size
    batch_inputs = [input_mod[i*max_batch_size:end_id] for input_mod in inputs]
    batch_inputs = [b.astype(np.float32) for b in batch_inputs]
    if tf.__version__[0] == '2':
      # import pdb; pdb.set_trace()
      # t=time.time(); model(batch_inputs); print(time.time()-t)
      batch_outputs = model(batch_inputs)
      if isinstance(batch_outputs, list):
        batch_outputs = [b.numpy() for b in batch_outputs]
      else:
        batch_outputs = [batch_outputs.numpy()]
    else:
      batch_outputs = [model.predict(batch_inputs)]
    outputs.append(batch_outputs)
    
  # Transpose the outputs
  outputs_transposed = list(map(list, zip(*outputs)))
  return [np.concatenate(o) for o in outputs_transposed]

def get_key_q_valid(q_values, player_obs, configuration, rewards_bases_ships):
  shipyard_keys = list(player_obs[1].keys())
  ship_keys = list(player_obs[2].keys())
  grid_size = configuration.size
  
  key_q_valid = []
  
  base_valid = np.ones((7), dtype=np.bool)
  base_valid[:5] = 0
  for k in shipyard_keys:
    row, col = row_col_from_square_grid_pos(player_obs[1][k], grid_size)
    
    # No spawning when my agent currently has a ship at the base (clear waste)
    base_valid[5] = not rewards_bases_ships[0][2][row][col]
    key_q_valid.append((k, q_values[row, col], base_valid, row, col))
  
  ship_valid = np.ones((7), dtype=np.bool)
  ship_valid[5] = 0
  for k in ship_keys:
    row, col = row_col_from_square_grid_pos(player_obs[2][k][0], grid_size)
    
    # No converting at the location of my or opponent bases
    ship_valid[4] = True
    for i in range(len(rewards_bases_ships)):
      ship_valid[4] = ship_valid[4] and not rewards_bases_ships[i][1][row][col]
    key_q_valid.append((k, q_values[row, col], ship_valid, row, col))
    
  return key_q_valid

# Q value based main function to act a single step
def get_agent_q_and_a(network, observation, player_obs, configuration,
                      epsilon_greedy, exploration_parameter,
                      action_costs, pick_first_on_tie):
  # Preprocess the state so it can be fed in to the network
  obs_input = [np.expand_dims(state_to_input(observation), 0)]
  
  # Obtain the q values
  q_values = my_keras_predict(network, obs_input)[0][0]
  
  # Determine valid actions for each of the ships/shipyards
  all_key_q_valid = get_key_q_valid(
    q_values, player_obs, configuration, observation['rewards_bases_ships'])
  
  all_mapped_actions = {}
  grid_size = configuration.size
  num_actions = len(ACTION_MAPPING)
  actions = -1*np.ones((grid_size, grid_size)).astype(np.int32)
  valid_actions = np.zeros((grid_size, grid_size, num_actions), dtype=np.bool)
  action_budget = player_obs[0]
  for i, (k, q_sub_values, valid_sub_actions, r, c) in enumerate(
      all_key_q_valid):
    # Set actions we can't afford to invalid
    valid_sub_actions &= action_costs <= action_budget
    action_id = select_action_from_q(valid_sub_actions, q_sub_values,
                                     epsilon_greedy, exploration_parameter,
                                     pick_first_on_tie)
    valid_actions[r, c] = valid_sub_actions
    actions[r, c] = action_id
    action_budget -= action_costs[action_id]
    mapped_action = ACTION_MAPPING[action_id]
    if mapped_action is not None:
      all_mapped_actions[k] = mapped_action
      
  valid_actions = np.stack(valid_actions, 0)
  
  return (obs_input, q_values, actions, all_mapped_actions, valid_actions)


###############################################################################
# Learning utilities                                                          #
###############################################################################

# One step mixed q learning target computation
def one_step_mixed_q_targets(next_mixed_q_vals, experience):
  terminal_rewards = np.array([e.episode_reward for e in experience])
  last_episode_actions = np.array([e.last_episode_action for e in experience])
  target_qs = np.where(last_episode_actions, terminal_rewards,
                       next_mixed_q_vals).astype(np.float32)
  
  return target_qs

def aggregate_next_mixed_q_vals(
    experience, episode_ids, this_q_vals, num_agents):
  valid_actions = np.stack([e.valid_actions for e in experience])
  num_actions = (valid_actions.sum(-1) > 0).sum((1, 2))
  best_q_sums = (valid_actions*this_q_vals).max(-1).sum((1, 2))
  mean_best_valid_qs = (best_q_sums/num_actions)
  
  # Compute the episode id offsets (agent and episode step offsets are easy).
  episode_steps = np.array([e.episode_step for e in experience])
  agent_ids = np.array([e.active_id for e in experience])
  episode_step_counts = np.bincount(episode_ids)
  last_step_ids = np.cumsum(episode_step_counts)-1
  episode_durations = episode_steps[last_step_ids]+1
  episode_offsets = np.zeros_like(episode_steps)
  start_id = 0
  for i in range(episode_durations.size):
    offset = np.sum(episode_durations[:i])*num_agents
    episode_offsets[start_id:(start_id + episode_step_counts[i])] = offset
    start_id += episode_step_counts[i]
  
  # Set Q values of inactive agents to zero and fill the other data
  # appropriately
  offsets = episode_offsets+episode_steps*num_agents+agent_ids
  mean_best_valid_qs_filled = np.zeros((episode_durations.sum()*num_agents))
  try:
    mean_best_valid_qs_filled[offsets] = mean_best_valid_qs
  except:
    import pdb; pdb.set_trace()
    x=1
  
  # Obtain the next q-values for all agents
  mean_best_valid_qs = mean_best_valid_qs_filled.reshape(
    (-1, num_agents))
  
  next_q_vals = np.zeros_like(mean_best_valid_qs_filled, dtype=np.float)
  
  # Assume that the terminal reward is 0.5-mean for now
  active_agents_count = (mean_best_valid_qs > 0).sum(1)
  q_sum_target_options = np.cumsum(np.flip(np.arange(num_agents)/(num_agents-1)))
  q_sum_targets = q_sum_target_options[active_agents_count-1]
  for i in range(num_agents):
    other_cols = np.arange(num_agents)
    other_cols = np.delete(other_cols, i)
    this_q_vals = mean_best_valid_qs[:, i]
    other_q_vals = mean_best_valid_qs[:, other_cols].sum(-1)
    
    next_q_vals[i::num_agents] = this_q_vals/(
      this_q_vals+other_q_vals)*q_sum_targets
    
  next_q_vals = np.concatenate([next_q_vals[num_agents:],
                                np.full((num_agents), np.inf)])[offsets]
  next_q_vals = next_q_vals.astype(np.float32)
  
  return next_q_vals

# Get the q learning observations, targets and observation weights
def mixed_nega_q_learning(
    model, experience, episode_ids, nan_coding_value, symmetric_experience,
    num_agents_per_game):
  # Evaluate the q values of the current and next state for all observations
  this_states = np.stack([e.current_obs for e in experience])
  this_q_vals = my_keras_predict(model, [this_states])[0]
  
  # Filter out next Q-values where the action is not valid. Filtered out since
  # every target computation performs a max operation.
  next_mixed_q_vals = aggregate_next_mixed_q_vals(
    experience, episode_ids, this_q_vals, num_agents_per_game)
    
  # Compute the target action values
  target_qs = one_step_mixed_q_targets(next_mixed_q_vals, experience)
  
  # Set the target for the action that was selected
  num_actions = len(ACTION_MAPPING)
  grid_size = this_states.shape[1]
  actions = np.stack([e.actions for e in experience])
  one_hot_actions = (np.arange(num_actions) == actions[..., None]).astype(bool)
  all_target_qs = np.where(
    one_hot_actions,
    np.tile(target_qs.reshape((-1, 1, 1, 1)),
            [1, grid_size, grid_size, num_actions]),
    nan_coding_value*np.ones_like(this_q_vals))
  
  if symmetric_experience:
    # TODO: Add symmetric observations and symmetric targets.
    # IMPORTANT: mind the violation of symmetry in the actions
    pass
    
  return this_states, all_target_qs

# Masked mse loss - values equal to mask_val are ignored in the loss
def masked_mse(y, p, mask_val):
  mask = K.cast(K.not_equal(y, mask_val), K.floatx())
  if tf.__version__[0] == '2':
    masked_loss = tf.losses.mse(y*mask, p*mask)
  else:
    mask = K.cast(mask, 'float32')
    masked_loss = K.mean(tf.math.square(p*mask - y*mask), axis=-1)
    # masked_loss = tf.compat.v1.losses.mean_squared_error(y*mask, p*mask)
    
  return masked_loss
    
# Make the masked mse loss
def make_masked_mse(nan_coding_value):
  def loss(y, p):
    return masked_mse(y, p, mask_val=nan_coding_value)
  
  return loss


###############################################################################
# Data utilities                                                              #
###############################################################################

# Increment the iteration id of some .h5 model path
def increment_iteration_id(path):
  underscore_parts = path.split('_')
  id_ = int(underscore_parts[-1].split('.')[0])
  incremented_id = id_+1
  new_path = '_'.join(underscore_parts[:-1]) + '_' + str(
    incremented_id ) + '.h5'
  
  return new_path

# Load all keras models for the given paths
def load_models(paths):
  models = []
  for p in paths:
    if tf.__version__[0] == '2':
      K.clear_session()
    models.append(load_model(p, custom_objects={'loss': make_masked_mse(0)}))
    
  return models

# Get the path in a folder with the highest iteration id
def get_most_recent_agent_extension(folder):
  agent_paths = [f for f in os.listdir(folder) if f[-3:]=='.h5']
  
  # Order the agent models on their iteration ids
  agent_paths = decreasing_sort_on_iteration_id(agent_paths)
  most_recent_agent_extension = agent_paths[0]
  
  return most_recent_agent_extension

# Order paths in a decreasing manner (sort on iteration id)
def decreasing_sort_on_iteration_id(paths):
  # Drop everything after the underscore and before the dot to get the ids
  ids = np.array([p.split('_')[1].split('.')[0] for p in paths]).astype(np.int)
  paths_np = np.array(paths)
  
  sort_ids = np.argsort(-ids)
  return paths_np[sort_ids].tolist()

def player_state_to_array(player_state, reward_divisor, ship_divisor):
  base_pos = np.expand_dims(player_state[1], -1)
  ship_pos = np.expand_dims(player_state[2], -1)
  ship_halite = np.expand_dims(player_state[3], -1)/ship_divisor
  reward = np.ones_like(base_pos)*player_state[0]/reward_divisor
  
  out = np.concatenate(
    [reward, base_pos, ship_pos, ship_halite], -1)
  
  return out

# Combine the state to a single numpy array so it can be fed in to the
# network
def state_to_input(observation, reward_halite_divisor=100000,
                   cell_halite_divisor=100000, ship_halite_divisor=100000):
  # Combine the Halite count, relative step fraction, my and opponent
  # ship and base information to a single numpy array
  halite_count = np.expand_dims(observation['halite'], -1)/cell_halite_divisor
  log_halite_count = np.log10(1+halite_count)
  halite_count = np.minimum(halite_count, np.ones_like(halite_count))
  relative_step = np.ones_like(halite_count)*observation['relative_step']
  my_data = player_state_to_array(
    observation['rewards_bases_ships'][0], reward_halite_divisor,
    ship_halite_divisor)
  
  opponent_data = []
  for i in range(1, len(observation['rewards_bases_ships'])):
    opponent_data.append(player_state_to_array(
      observation['rewards_bases_ships'][i], reward_halite_divisor,
      ship_halite_divisor))
  
  state_inputs = np.concatenate(
    [halite_count, log_halite_count, relative_step, my_data] + opponent_data,
    -1).astype(np.float32)
  
  return state_inputs

# Custom class to reuse data of subsequent interations with the environment
# FIFO buffer. 
class ExperienceBuffer:
  def __init__(self, buffer_size):
    self.buffer_size = buffer_size
    self.episode_offset = 0
    self.data = []
    self.episode_ids = np.array([], dtype=np.int)
    
  def add(self, data):
    episode_ids = np.array([d.game_id for d in data])
    num_episodes = episode_ids[-1] + 1
    if num_episodes > self.buffer_size:
      # Keep most recent experience of the experience batch
      data = data[
        np.where(episode_ids == (num_episodes-self.buffer_size))[0][0]:]
      self.data = data
      self.episode_ids = episode_ids
      self.episode_offset = 0
      return
      
    episode_ids = episode_ids + self.episode_offset
    self.data += data
    self.episode_ids = np.concatenate([self.episode_ids, episode_ids])
    
    unique_episode_ids = pd.unique(self.episode_ids)
    if unique_episode_ids.size > self.buffer_size:
      cutoff_index = np.where(self.episode_ids == unique_episode_ids[
        unique_episode_ids.size-self.buffer_size])[0][0]
      self.data = self.data[cutoff_index:]
      self.episode_ids = self.episode_ids[cutoff_index:]
    self.episode_offset += num_episodes
    
  def get_data(self):
    return self.data
  
  def size(self):
    return len(self.data)
  

###############################################################################
# Vizualization utilities                                                     #
###############################################################################

def update_learning_progress(experiment_name, data_vals): 
  # Append a line to the learning progress line if the file exists. Otherwise:
  # create it
  this_folder = os.path.dirname(__file__)
  agents_folder = os.path.join(this_folder, '../Agents/' + experiment_name)
  progress_path = os.path.join(agents_folder, 'learning_progress.csv')
  
  if os.path.exists(progress_path):
    progress = pd.read_csv(progress_path)
    progress.loc[progress.shape[0]] = data_vals
  else:
    progress = pd.DataFrame(data_vals, index=[0])
  
  progress.to_csv(progress_path, index=False)
  
  
def record_videos(agent_path, num_agents_per_game):
  print("Generating videos of iteration {}".format(agent_path))
  env = make_environment(
    "halite", configuration={"agentExec": "LOCAL"})#, configuration={"agentTimeout": 10000, "actTimeout": 10000})
  model = load_models([agent_path])[0]
  action_costs = get_action_costs()
  env_configuration = env.configuration
  def my_agent(observation):
    active_id = observation.player
    current_observation = structured_env_obs(
      env_configuration, observation, active_id)
    player_obs = observation.players[active_id]
    
    # Preprocess the state so it can be fed in to the network
    obs_input = np.expand_dims(state_to_input(current_observation), 0)
    
    # Obtain the q values
    q_values = model(obs_input).numpy()[0]
    
    # Determine valid actions for each of the ships/shipyards
    all_key_q_valid = get_key_q_valid(
      q_values, player_obs, env_configuration,
      current_observation['rewards_bases_ships'])
    
    mapped_actions = {}
    action_budget = player_obs[0]
    for i, (k, q_sub_values, valid_sub_actions, r, c) in enumerate(
        all_key_q_valid):
      # Set actions we can't afford to invalid
      valid_sub_actions &= action_costs <= action_budget
      valid_sub_actions
      valid_sub_actions = np.where(valid_sub_actions)[0]
      best_q = q_sub_values[valid_sub_actions].max()
      best_a_id = np.where(q_sub_values[valid_sub_actions] == best_q)[0][0]
      action_id = valid_sub_actions[best_a_id]
      
      # Hard coded epsilon greedy exploration
      if np.random.uniform() < 0.05:
        action_id = np.random.choice(valid_sub_actions)
      
      action_budget -= action_costs[action_id]
      mapped_action = ACTION_MAPPING[action_id]
      if mapped_action is not None:
        mapped_actions[k] = mapped_action
       
    return mapped_actions
  
  for video_type in ["random opponent", "self play"]:
    env.reset(num_agents=num_agents_per_game)
    agents = [my_agent]*num_agents_per_game if video_type == "self play" else [
      my_agent] + ["random"]*(num_agents_per_game-1)
    env.run(agents)
    
    # Save the HTML recording in the videos folder
    game_recording = env.render(mode="html", width=800, height=600)
    folder, extension = tuple(agent_path.rsplit('/', 1))
    videos_folder = os.path.join(folder, 'Videos')
    Path(videos_folder).mkdir(parents=True, exist_ok=True)
    video_path = os.path.join(
      videos_folder, extension[:-3] + ' - ' + video_type + '.html')
    with open(video_path,"w") as f:
      f.write(game_recording)
  
    
###############################################################################
# Submission utilities                                                        #
###############################################################################

# Bit 64 encoding of a pickled object
def serialize(value, verbose=True):
  serialized = pickle.dumps(value)
  if verbose:
    print('Length of serialized object:', len(serialized))
  return b64.b64encode(serialized)

# Bit 64 decoding followed by a pickle load
def deserialize(serialized):
  data_byte = b64.b64decode(serialized)
  value = pickle.loads(data_byte)
  return value
    
    
    
    