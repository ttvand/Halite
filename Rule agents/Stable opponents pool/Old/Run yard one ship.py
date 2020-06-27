# Source: https://www.kaggle.com/hubcity/simple-bad-agents-halite
import random 

from kaggle_environments.envs.halite.halite import get_to_pos

def create_enemy_ship_possible_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    for idx, opp in enumerate(obs.players):
        if idx == player:
            continue
        for ship in opp[2].values():
            map[ship[0]] = 1
            for dir in ["NORTH", "SOUTH", "EAST", "WEST"]:
                map[get_to_pos(config.size, ship[0], dir)] = 1
    return map

def create_enemy_yard_map(obs, config):
    map = [0] * config.size * config.size
    player = obs.player
    for idx, opp in enumerate(obs.players):
        if idx == player:
            continue
        for yard_pos in opp[1].values():
            map[yard_pos] = 1
    return map

def runaway_agent(obs, config):
    me = obs.players[obs.player]
    first_ship = next(iter(me[2].keys()))
    pos = me[2][first_ship][0]
    cargo = me[2][first_ship][1]
    esp = create_enemy_ship_possible_map(obs, config)
    ey = create_enemy_yard_map(obs, config)
    bad_square = [a+b for a,b in zip(esp, ey)]
    # ignore negative halite
    good_square = [x if x >= 0 else 0 for x in obs.halite]
    square_score = [-b if b > 0 else g for b,g in zip(bad_square, good_square)]
    moves = ["NORTH", "SOUTH", "EAST", "WEST"]
    random.shuffle(moves)
    best_score = square_score[pos]
    best_move = ""
    actions = {}
    for move in moves:
        new_pos = get_to_pos(config.size, pos, move)
        pos_score = square_score[new_pos]
        if pos_score > best_score or (pos_score == 0 and best_score == 0):
            best_score = pos_score
            best_move = move
            actions = { first_ship: best_move }
    return actions

def run_yard_agent(obs, config):
    actions = {}
    me = obs.players[obs.player]
    if len(me[2]) > 0:  # if I have a ship
        first_ship = next(iter(me[2].keys()))
        if me[2][first_ship][1] > config.convertCost:
            actions = { first_ship: "CONVERT" }
        else:
            actions = runaway_agent(obs, config)
    return actions

def run_yard_one_agent(obs, config):
    me = obs.players[obs.player]
    num_ships = len(me[2])
    if num_ships == 0:
        first_yard = next(iter(me[1].keys()))
        actions = { first_yard: "SPAWN" }
    else:
        actions = run_yard_agent(obs, config)
    return actions
