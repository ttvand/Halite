# Source: https://www.kaggle.com/hubcity/simple-bad-agents-halite

def yard_only_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    num_yards = len(me[1])
    if num_yards == 0:
        first_ship = next(iter(me[2].keys()))
        actions = { first_ship: "CONVERT" }
    return actions
