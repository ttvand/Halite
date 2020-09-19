import numpy as np
import os
import pandas as pd
import requests


team_name_ids = {
  "Raine Force": 5423724,
  "Tom Van de Wiele": 4714287,
  }

def getTeamEpisodes(team_id):
    # request
    base_url = "https://www.kaggle.com/requests/EpisodeService/"
    list_url = base_url + "ListEpisodes"
    r = requests.post(list_url, json = {"teamId":  int(team_id)})
    rj = r.json()

    # update teams list
    r = requests.post(list_url, json = {"teamId": team_id})
    rj = r.json()
    teams_df = pd.DataFrame(rj['result']['teams'])
    teams_df_new = pd.DataFrame(rj['result']['teams'])
    
    if len(teams_df.columns) == len(teams_df_new.columns) and (teams_df.columns == teams_df_new.columns).all():
        teams_df = pd.concat( (teams_df, teams_df_new.loc[[c for c in teams_df_new.index if c not in teams_df.index]] ) )
        teams_df.sort_values('publicLeaderboardRank', inplace = True)
#         print('{} teams on file'.format(len(teams_df)))
    else:
        print('teams dataframe did not match')
    
    # make df
    team_episodes = pd.DataFrame(rj['result']['episodes'])
    team_episodes['avg_score'] = -1;
    
    for i in range(len(team_episodes)):
        agents = team_episodes['agents'].loc[i]
        agent_scores = [a['updatedScore'] for a in agents if a['updatedScore'] is not None]
        team_episodes.loc[i, 'submissionId'] = [a['submissionId'] for a in agents if a['submission']['teamId'] == team_id][0]
        team_episodes.loc[i, 'updatedScore'] = [a['updatedScore'] for a in agents if a['submission']['teamId'] == team_id][0]
        
        if len(agent_scores) > 0:
            team_episodes.loc[i, 'avg_score'] = np.mean(agent_scores)

    for sub_id in team_episodes['submissionId'].unique():
        sub_rows = team_episodes[ team_episodes['submissionId'] == sub_id ]
        max_time = max( [r['seconds'] for r in sub_rows['endTime']] )
        final_score = max( [r['updatedScore'] for r_idx, (r_index, r) in enumerate(sub_rows.iterrows())
                                if r['endTime']['seconds'] == max_time] )

        team_episodes.loc[sub_rows.index, 'final_score'] = final_score
        
    team_episodes.sort_values('avg_score', ascending = False, inplace=True)
    return rj, team_episodes, teams_df
  
def get_submisison_progress(episodes, team_name, team_id):
  submission_progress = {}
  num_epsiodes = episodes.shape[0]
  episodes = episodes.sort_values("id")
  for i in range(num_epsiodes):
    agents = episodes.iloc[i].agents
    this_agent_id = np.where(np.array([
      a['submission']['teamId'] for a in agents]) == team_id)[0][0]
    submission_id = agents[this_agent_id]['submissionId']
    new_score = agents[this_agent_id]['updatedScore']
    if submission_id in submission_progress:
      submission_progress[submission_id].append(new_score)
    else:
      submission_progress[submission_id] = [new_score]
      
  # Convert to a single combined csv
  all_data = []
  for submission_id in submission_progress:
    if len(submission_progress[submission_id]) < 2:
      continue
    scores = np.array(submission_progress[submission_id])
    num_scores = scores.size
    game_ids = np.arange(num_scores)-(num_scores-1)
    df = pd.DataFrame({'team_name': team_name, 'submission_id': submission_id,
                       'game_id': game_ids, 'score': scores})
    all_data.append(df)
    
  return pd.concat(all_data)
  
  
all_team_episodes = {}
all_team_submission_progress = {}
for team_name in team_name_ids:
  team_id = team_name_ids[team_name]
  all_team_episodes[team_name] = getTeamEpisodes(team_id)
  all_team_submission_progress[team_name] = get_submisison_progress(
    all_team_episodes[team_name][1], team_name, team_id)
  
all_submission_progress = pd.concat(
  list(all_team_submission_progress.values()))
this_folder = os.path.dirname(__file__)
replay_folder = os.path.join(
  this_folder, '../Rule agents/Leaderboard replays/')
save_path = replay_folder + "Raine Force comparison.csv"
all_submission_progress.to_csv(save_path)