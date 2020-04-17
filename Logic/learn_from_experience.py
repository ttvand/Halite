# Update the keras model
import gc
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import utils



def update_agent(experience, episode_ids, agent_path, config, agent_config):
  model = load_model(agent_path, custom_objects={
    'loss': utils.make_masked_mse(0)})
  mean_losses = []
  nan_coding_value = config['nan_coding_value']
  adam = Adam(lr=config['learning_rate'])
  model.compile(optimizer=adam, loss=utils.make_masked_mse(nan_coding_value))
  
  scrambled_episode_ids = np.random.permutation(np.unique(episode_ids))
  num_episodes = scrambled_episode_ids.size
  subset_max_episodes = config["max_episodes_per_learning_update_q_learning"]
  num_learn_updates = int(np.ceil(num_episodes / subset_max_episodes))
  for i in range(num_learn_updates):
    last_id = min(num_episodes, (i+1)*subset_max_episodes)
    subset_episode_ids = scrambled_episode_ids[
      (i*subset_max_episodes):last_id]
    subset_mask = np.isin(episode_ids, subset_episode_ids).tolist()
    subset_experience = [e for (e, m) in zip(experience, subset_mask) if m]
    subset_experience_episode_ids = episode_ids[subset_mask]
    
    x_train, y_train = utils.q_learning(
      model, subset_experience, subset_experience_episode_ids,
      nan_coding_value, config['symmetric_experience'],
      agent_config['num_agents_per_game'], config['reward_type'],
      config['halite_change_discount'])
    
    history = model.fit(
      x_train,
      y_train,
      batch_size=config['batch_size'], 
      epochs=config['num_epochs'],
      )
    
    del subset_experience
    del x_train
    del y_train
    gc.collect()
    
    mean_losses.append(history.history['loss'])
  
  # Save the updated agent
  model.save(agent_path)
    
  return np.array(mean_losses).mean()
    
    
    
    
    
    
    
    
    
    
  

  
