# Source: https://www.kaggle.com/hubcity/simple-bad-agents-halite

def single_base_no_spawns_agent(obs, config):
    me = obs.players[obs.player]
    num_ships = len(me[2])
    if num_ships == 1:
        first_ship = next(iter(me[2].keys()))
        actions = {first_ship: "CONVERT"}
    else:
        actions = {}
    return actions
