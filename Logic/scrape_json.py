import json
from kaggle_environments import get_episode_replay
from kaggle_environments import list_episodes_for_submission
import numpy as np
import os
from pathlib import Path
import time

my_submission = [17114281, 17114329][0]

# Returns metadata for all episodes for a particular submission
# You can find your submission id at the top of the list of episodes for your agent on the leaderboard
submission_episodes = list_episodes_for_submission(my_submission)['result']['episodes'][1:]
num_episodes = len(submission_episodes)
episode_ids = [e['id'] for e in submission_episodes]
agent_ids = np.array([a['submissionId'] for e in submission_episodes for a in (
  e['agents'])]).reshape([num_episodes, -1])
my_agent_ids = np.where(agent_ids == my_submission)[1]

# Path handling
this_folder = os.path.dirname(__file__)
data_folder = os.path.join(this_folder, '../Rule agents/Leaderboard replays/'+(
  str(my_submission)))
Path(data_folder).mkdir(parents=True, exist_ok=True)

# Iterate over all episodes to get the json replay in reverse order (more
# recent games are probably more interesting)
for i in range(num_episodes-1, -1, -1):
  episode_id = episode_ids[i]
  f = os.path.join(data_folder,
                   str(episode_id) + '-' + str(my_agent_ids[i]) + '.json')
  if not os.path.exists(f):
    replay = get_episode_replay(episode_id)['result']['replay']
    with open(f, 'w') as outfile:
      outfile.write(json.dumps(replay, indent=4))
      
    time.sleep(1.02)