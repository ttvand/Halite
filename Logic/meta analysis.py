import json
import numpy as np
import os
import pandas as pd

my_submissions = [17114281, 17114329, 17168045, 17170621, 17170654,
                  17170690, 17171012, 17183645, 17187266,
                  17190682, 17190822, 17190934, 17195646]
this_folder = os.path.dirname(__file__)
replay_folder = os.path.join(
  this_folder, '../Rule agents/Leaderboard replays/')
all_replay_game_results = []
for my_submission in my_submissions:
  data_folder = os.path.join(replay_folder, str(my_submission))
  json_files = np.sort(
    [f for f in os.listdir(data_folder) if f[-4:] == "json"])
  
  for i, f in enumerate(json_files):
    json_path = os.path.join(data_folder, f)
    with open(json_path) as f:
      raw_data = json.load(f)
      json_data = json.loads(raw_data)
      rewards = np.array([d if d is not None else -1 for d in json_data[
        'rewards']])
      my_id = np.where(np.array([d in [
        "Tom Van de Wiele", '"Tom Van de Wiele"'] for d in json_data[
          'info']['TeamNames']]))[0][0]
      my_lost_encounters = (rewards > rewards[my_id]).sum()
      all_replay_game_results.append((my_submission, i, my_lost_encounters))
      
df = pd.DataFrame(all_replay_game_results, columns=[
  'Submission', 'Game id', 'Num Losses'])

wide_df = df.pivot(index='Game id', columns='Submission')[['Num Losses']]
save_path = replay_folder + "Meta analysis.csv"
wide_df.to_csv(save_path)