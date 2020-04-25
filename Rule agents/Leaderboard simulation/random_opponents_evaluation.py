from kaggle_environments import evaluate as evaluate_game
from kaggle_environments import make as make_environment
from kaggle_environments import utils as environment_utils
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path


# Evaluation parameters - agents from the agents pool are selected
# independently with replacement to compete in a submission format that
# resembles the Kaggle leaderboard.
# Execution time scales linearly with the number of evaluation games and is
# independent of the number of agents in the pool.
num_agents_per_game = 4
num_games = 100
use_multiprocessing = True
generate_pair_videos = True


# List the path of all evaluated agents
this_folder = os.path.dirname(__file__)
agents_folder = os.path.join(this_folder, 'Agents')
agent_extensions = np.array(
  [f for f in os.listdir(agents_folder) if f[-3:]=='.py'])
num_agents = len(agent_extensions)
agent_full_paths = [os.path.join(agents_folder, e) for e in agent_extensions]

# Simulate games, reloading the file on each interaction since some agents
# are stochastic
scrambled_agent_ids = np.random.permutation(num_games*num_agents_per_game)
agent_ids = np.mod(scrambled_agent_ids, num_agents).reshape([num_games, -1])

def evaluate_single_episode(agent_ids, agent_paths):
  num_agents = agent_ids.size
  agents = []
  for id_ in agent_ids:
    my_agent_file = environment_utils.read_file(agent_full_paths[id_])
    agents.append(environment_utils.get_last_callable(my_agent_file))
  
  halite_scores = evaluate_game(
    "halite", agents, num_episodes=1,
    configuration={"agentExec": "LOCAL",
                   # "agentTimeout": 10000, # Uncomment while debugging 
                   # "actTimeout": 10000, # Uncomment while debugging 
                   })[0]
  
  halite_scores = [-1 if h is None else h for h in halite_scores]
  episode_rewards = np.zeros((num_agents))
  for i in range(0, num_agents-1):
    for j in range(i+1, num_agents):
      if halite_scores[i] == halite_scores[j]:
        first_score = 0.5
      else:
        first_score = int(halite_scores[i] > halite_scores[j])
      
      episode_rewards[i] += first_score
      episode_rewards[j] += 1-first_score
  episode_rewards /= (num_agents-1)
  
  return episode_rewards, halite_scores
  

# Collect episode rewards for all episodes
if use_multiprocessing:
  with mp.Pool(processes=mp.cpu_count()-1) as pool:
    results = [pool.apply_async(
                evaluate_single_episode, args=(
                  agent_ids[i], agent_full_paths,)) for i in range(num_games)]
    episode_rewards, halite_scores = list(zip(*[p.get() for p in results]))
else:
  episode_rewards = []
  halite_scores = []
  for game_id in range(num_games):
    single_episode_rewards, single_episode_halite = evaluate_single_episode(
        agent_ids[game_id], agent_full_paths)
    episode_rewards.append(single_episode_rewards)
    halite_scores.append(single_episode_halite)
    
# Compute the average rewards for all agents and present the ranking
agent_ids_flat = agent_ids.flatten()
episode_rewards_flat = np.array(episode_rewards).flatten()
halite_scores_flat = np.array(halite_scores).flatten()
avg_rewards = np.zeros(num_agents)
avg_terminal_halite = np.zeros(num_agents)
num_agent_games = np.zeros(num_agents)
for i in range(num_agents):
  agent_reward_ids = np.where(agent_ids_flat == i)[0]
  avg_rewards[i] = episode_rewards_flat[agent_reward_ids].mean()
  avg_terminal_halite[i] = halite_scores_flat[agent_reward_ids].mean()
  num_agent_games[i] = agent_reward_ids.size
  
agent_rank_ids = np.argsort(-avg_rewards)
results = pd.DataFrame.from_dict({
  'Rank': np.arange(num_agents)+1,
  'Agent': agent_extensions[agent_rank_ids],
  'Average episode reward': avg_rewards[agent_rank_ids],
  'Average terminal halite': avg_terminal_halite[agent_rank_ids],
  'Num evaluation games': num_agent_games[agent_rank_ids],
  }
  )
print(results)

if generate_pair_videos:
  env = make_environment("halite", configuration={"agentExec": "LOCAL"})
  for i in range(0, num_agents-1):
    first_agent_file = environment_utils.read_file(agent_full_paths[i])
    first_agent = environment_utils.get_last_callable(first_agent_file)
    for j in range(i+1, num_agents):
      ext = agent_extensions[i][:-3] + " ***versus*** " + agent_extensions[j][:-3]
      
      second_agent_file = environment_utils.read_file(agent_full_paths[j])
      second_agent = environment_utils.get_last_callable(second_agent_file)
      
      agents = [first_agent, second_agent, second_agent, first_agent]
      env.reset(num_agents=num_agents_per_game)
      env.run(agents)
      
      # Save the HTML recording in the videos folder
      game_recording = env.render(mode="html", width=800, height=600)
      videos_folder = os.path.join(agents_folder, '../Videos')
      Path(videos_folder).mkdir(parents=True, exist_ok=True)
      video_path = os.path.join(videos_folder, ext+'.html')
      with open(video_path,"w") as f:
        f.write(game_recording)
