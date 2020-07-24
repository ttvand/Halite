import numpy as np
import random

#FUNCTIONS###################################################
def get_map(obs):
    """ get map as two dimensional array of objects and set amounts of halite in each cell """
    game_map = []
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                # value will be ID of owner
                "shipyard": None,
                # value will be ID of owner
                "ship": None,
                # value will be amount of halite
                "ship_cargo": None,
                # amount of halite
                "halite": obs.halite[conf.size * y + x]
            })
    return game_map

def get_swarm_units_coords_and_update_map(s_env):
    """ get lists of coords of Swarm's units and update locations of ships and shipyards on the map """
    # arrays of (x, y) coords
    swarm_shipyards_coords = []
    swarm_ships_coords = []
    # place on the map locations of units of every player
    for player in range(len(s_env["obs"].players)):
        # place on the map locations of every shipyard of the player
        shipyards = list(s_env["obs"].players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map
            s_env["map"][x][y]["shipyard"] = player
            if player == s_env["obs"].player:
                swarm_shipyards_coords.append((x, y))
        # place on the map locations of every ship of the player
        ships = list(s_env["obs"].players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            s_env["map"][x][y]["ship"] = player
            s_env["map"][x][y]["ship_cargo"] = ship[1]
            if player == s_env["obs"].player:
                swarm_ships_coords.append((x, y))
    return swarm_shipyards_coords, swarm_ships_coords

def get_c(c):
    """ get coordinate, considering donut type of the map """
    return c % conf.size

def clear(x, y, player, game_map):
    """ check if cell is safe to move in """
    # if there is no shipyard, or there is player's shipyard
    # and there is no ship
    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and
            game_map[x][y]["ship"] == None):
        return True
    return False

def move_ship(x_initial, y_initial, actions, s_env, ship_index):
    """ move the ship according to first acceptable tactic """
    ok, actions = boarding(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    ok, actions = go_for_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    ok, actions = unload_halite(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)
    if ok:
        return actions
    return standard_patrol(x_initial, y_initial, s_env["ships_keys"][ship_index], actions, s_env, ship_index)

def boarding(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ Yo Ho Ho and a Bottle of Rum!!! """
    # direction of hostile ship with biggest prize
    biggest_prize = None
    for d in range(len(directions_list)):
        x = directions_list[d]["x"](x_initial)
        y = directions_list[d]["y"](y_initial)
        # if hostile ship is there, has enough halite and safe for boarding
        if (s_env["map"][x][y]["ship"] != s_env["obs"].player and
                s_env["map"][x][y]["ship"] != None and
                s_env["map"][x][y]["ship_cargo"] > s_env["ships_values"][ship_index][1] and
                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"],
                                          s_env["ships_values"][ship_index][1] + s_env["map"][x][y]["ship_cargo"])):
            # if current hostile ship has more than ship with biggest prize
            if biggest_prize == None or s_env["map"][x][y]["ship_cargo"] > biggest_prize:
                biggest_prize = s_env["map"][x][y]["ship_cargo"]
                direction = directions_list[d]["direction"]
                direction_x = x
                direction_y = y
    # if hostile ship is there, has enough halite and safe for boarding
    if biggest_prize != None and np.random.uniform() < chase_probability:
        actions[ship_id] = direction
        s_env["map"][x_initial][y_initial]["ship"] = None
        s_env["map"][direction_x][direction_y]["ship_cargo"] = -1
        return True, actions
    return False, actions
    
def go_for_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ ship will go to safe cell with enough halite, if it is found """
    # biggest amount of halite among scanned cells
    most_halite = low_amount_of_halite
    for d in range(len(directions_list)):
        x = directions_list[d]["x"](x_initial)
        y = directions_list[d]["y"](y_initial)
        # if cell is safe to move in
        if (clear(x, y, s_env["obs"].player, s_env["map"]) and
                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1])):
            # if current cell has more than biggest amount of halite
            if s_env["map"][x][y]["halite"] > most_halite:
                most_halite = s_env["map"][x][y]["halite"]
                direction = directions_list[d]["direction"]
                direction_x = x
                direction_y = y
    # if cell is safe to move in and has substantial amount of halite
    if most_halite > low_amount_of_halite:
        actions[ship_id] = direction
        s_env["map"][x_initial][y_initial]["ship"] = None
        s_env["map"][direction_x][direction_y]["ship"] = s_env["obs"].player
        return True, actions
    return False, actions

def unload_halite(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ unload ship's halite if there is any and Swarm's shipyard is near """
    if s_env["ships_values"][ship_index][1] > 0:
        for d in range(len(directions_list)):
            x = directions_list[d]["x"](x_initial)
            y = directions_list[d]["y"](y_initial)
            # if shipyard is there and unoccupied
            if (clear(x, y, s_env["obs"].player, s_env["map"]) and
                    s_env["map"][x][y]["shipyard"] == s_env["obs"].player):
                actions[ship_id] = directions_list[d]["direction"]
                s_env["map"][x_initial][y_initial]["ship"] = None
                s_env["map"][x][y]["ship"] = s_env["obs"].player
                return True, actions
    return False, actions

def standard_patrol(x_initial, y_initial, ship_id, actions, s_env, ship_index):
    """ 
        ship will move in expanding circles clockwise or counterclockwise
        until reaching maximum radius, then radius will be minimal again
    """
    directions = ships_data[ship_id]["directions"]
    # set index of direction
    i = ships_data[ship_id]["directions_index"]
    direction_found = False
    for j in range(len(directions)):
        x = directions[i]["x"](x_initial)
        y = directions[i]["y"](y_initial)
        # if cell is ok to move in
        if (clear(x, y, s_env["obs"].player, s_env["map"]) and
                (s_env["map"][x][y]["shipyard"] == s_env["obs"].player or
                not hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][ship_index][1]))):
            ships_data[ship_id]["moves_done"] += 1
            # apply changes to game_map, to avoid collisions of player's ships next turn
            s_env["map"][x_initial][y_initial]["ship"] = None
            s_env["map"][x][y]["ship"] = s_env["obs"].player
            # if it was last move in this direction
            if ships_data[ship_id]["moves_done"] >= ships_data[ship_id]["ship_max_moves"]:
                ships_data[ship_id]["moves_done"] = 0
                ships_data[ship_id]["directions_index"] += 1
                # if it is last direction in a list
                if ships_data[ship_id]["directions_index"] >= len(directions):
                    ships_data[ship_id]["directions_index"] = 0
                    ships_data[ship_id]["ship_max_moves"] += 1
                    # if ship_max_moves reached maximum radius expansion
                    if ships_data[ship_id]["ship_max_moves"] > max_moves_amount:
                        ships_data[ship_id]["ship_max_moves"] = 1
            actions[ship_id] = directions[i]["direction"]
            direction_found = True
            break
        else:
            # loop through directions
            i += 1
            if i >= len(directions):
                i = 0
    # if ship is not on shipyard and surrounded by opponent's units
    if not direction_found and s_env["map"][x_initial][y_initial]["shipyard"] == None:
        # if there is enough halite to convert
        if s_env["ships_values"][ship_index][1] >= conf.convertCost:
            actions[ship_id] = "CONVERT"
            s_env["map"][x_initial][y_initial]["ship"] = None
        # move to any empty cell
        else:
            for i in range(len(directions)):
                x = directions[i]["x"](x_initial)
                y = directions[i]["y"](y_initial)
                # if cell is empty or it is opponent's shipyard
                if clear(x, y, s_env["obs"].player, s_env["map"]) or s_env["map"][x][y]["shipyard"] != None:
                    # apply changes to game_map, to avoid collisions of player's ships next turn
                    s_env["map"][x_initial][y_initial]["ship"] = None
                    s_env["map"][x][y]["ship"] = s_env["obs"].player
                    actions[ship_id] = directions[i]["direction"]
                    break
    return actions

def get_directions(i0, i1, i2, i3):
    """ get list of directions in a certain sequence """
    return [directions_list[i0], directions_list[i1], directions_list[i2], directions_list[i3]]

def hostile_ship_near(x, y, player, m, cargo):
    """ check if hostile ship is in one move away from game_map[x][y] and has less or equal halite """
    # m = game map
    n = get_c(y - 1)
    e = get_c(x + 1)
    s = get_c(y + 1)
    w = get_c(x - 1)
    if (
            (m[x][n]["ship"] != player and m[x][n]["ship"] != None and m[x][n]["ship_cargo"] <= cargo) or
            (m[x][s]["ship"] != player and m[x][s]["ship"] != None and m[x][s]["ship_cargo"] <= cargo) or
            (m[e][y]["ship"] != player and m[e][y]["ship"] != None and m[e][y]["ship_cargo"] <= cargo) or
            (m[w][y]["ship"] != player and m[w][y]["ship"] != None and m[w][y]["ship_cargo"] <= cargo)
        ):
        return True
    return False

def define_some_globals(configuration):
    """ define some of the global variables """
    global conf
    global convert_threshold
    global max_moves_amount
    global globals_not_defined
    conf = configuration
    convert_threshold = conf.convertCost + conf.spawnCost * 2
    max_moves_amount = conf.size
    globals_not_defined = False

def adapt_environment(observation, configuration):
    """ adapt environment for the Swarm """
    s_env = {}
    s_env["obs"] = observation
    if globals_not_defined:
        define_some_globals(configuration)
    s_env["map"] = get_map(s_env["obs"])
    s_env["swarm_halite"] = s_env["obs"].players[s_env["obs"].player][0]
    s_env["swarm_shipyards_coords"], s_env["swarm_ships_coords"] = get_swarm_units_coords_and_update_map(s_env)
    s_env["ships_keys"] = list(s_env["obs"].players[s_env["obs"].player][2].keys())
    s_env["ships_values"] = list(s_env["obs"].players[s_env["obs"].player][2].values())
    s_env["shipyards_keys"] = list(s_env["obs"].players[s_env["obs"].player][1].keys())
    return s_env
    
def actions_of_ships(s_env):
    """ actions of every ship of the Swarm """
    global movement_tactics_index
    actions = {}
    shipyards_amount = len(s_env["shipyards_keys"])
    for i in range(len(s_env["swarm_ships_coords"])):
        x = s_env["swarm_ships_coords"][i][0]
        y = s_env["swarm_ships_coords"][i][1]
        # if this is a new ship
        if s_env["ships_keys"][i] not in ships_data:
            ships_data[s_env["ships_keys"][i]] = {
                "moves_done": 0,
                "ship_max_moves": random.randint(1, max_moves_amount),
                "directions": movement_tactics[movement_tactics_index]["directions"],
                "directions_index": 0
            }
            movement_tactics_index += 1
            if movement_tactics_index >= movement_tactics_amount:
                movement_tactics_index = 0
        elif s_env["obs"].step == (conf.episodeSteps - 2) and s_env["ships_values"][i][1] >= conf.convertCost:
            actions[s_env["ships_keys"][i]] = "CONVERT"
            s_env["map"][x][y]["ship"] = None
        # if there is no shipyards and enough halite to spawn few ships
        elif (shipyards_amount == 0 and 
                (s_env["swarm_halite"] + s_env["ships_values"][i][1]) >= convert_threshold):
            s_env["swarm_halite"] = s_env["swarm_halite"] + s_env["ships_values"][i][1] - conf.convertCost
            actions[s_env["ships_keys"][i]] = "CONVERT"
            s_env["map"][x][y]["ship"] = None
            shipyards_amount += 1
        else:
            # if this cell has low amount of halite or hostile ship is near
            if (s_env["map"][x][y]["halite"] < low_amount_of_halite or
                    hostile_ship_near(x, y, s_env["obs"].player, s_env["map"], s_env["ships_values"][i][1])):
                actions = move_ship(x, y, actions, s_env, i)
    return actions
     
def actions_of_shipyards(actions, s_env):
    """ actions of every shipyard of the Swarm """
    ships_amount = len(s_env["ships_keys"])
    # spawn ships from every shipyard, if possible
    # iterate through shipyards starting from last created
    for i in range(len(s_env["swarm_shipyards_coords"]))[::-1]:
        if s_env["swarm_halite"] >= conf.spawnCost and ships_amount < spawn_limit:
            x = s_env["swarm_shipyards_coords"][i][0]
            y = s_env["swarm_shipyards_coords"][i][1]
            # if there is currently no ship on shipyard
            if clear(x, y, s_env["obs"].player, s_env["map"]):
                s_env["swarm_halite"] -= conf.spawnCost
                actions[s_env["shipyards_keys"][i]] = "SPAWN"
                s_env["map"][x][y]["ship"] = s_env["obs"].player
                ships_amount += 1
        else:
            break
    return actions


#GLOBAL_VARIABLES#############################################
conf = None
# max amount of moves in one direction before turning
max_moves_amount = None
# threshold of harvested by a ship halite to convert
convert_threshold = None
# object with ship ids and their data
ships_data = {}
# initial movement_tactics index
movement_tactics_index = 0
# amount of halite, that is considered to be low
low_amount_of_halite = 50
# limit of ships to spawn - MADE STOCHASTIC BY TOM
spawn_limit = random.choice([10 + i for i in range(15)])
# TOM ADDED: probability of chasing a nearby ship with more halite
chase_probability = np.random.uniform(0.5, 1.0)
# not all global variables are defined
globals_not_defined = True

# list of directions
directions_list = [
    {
        "direction": "NORTH",
        "x": lambda z: z,
        "y": lambda z: get_c(z - 1)
    },
    {
        "direction": "EAST",
        "x": lambda z: get_c(z + 1),
        "y": lambda z: z
    },
    {
        "direction": "SOUTH",
        "x": lambda z: z,
        "y": lambda z: get_c(z + 1)
    },
    {
        "direction": "WEST",
        "x": lambda z: get_c(z - 1),
        "y": lambda z: z
    }
]

# list of movement tactics
movement_tactics = [
    # N -> E -> S -> W
    {"directions": get_directions(0, 1, 2, 3)},
    # S -> E -> N -> W
    {"directions": get_directions(2, 1, 0, 3)},
    # N -> W -> S -> E
    {"directions": get_directions(0, 3, 2, 1)},
    # S -> W -> N -> E
    {"directions": get_directions(2, 3, 0, 1)},
    # E -> N -> W -> S
    {"directions": get_directions(1, 0, 3, 2)},
    # W -> S -> E -> N
    {"directions": get_directions(3, 2, 1, 0)},
    # E -> S -> W -> N
    {"directions": get_directions(1, 2, 3, 0)},
    # W -> N -> E -> S
    {"directions": get_directions(3, 0, 1, 2)},
]
movement_tactics_amount = len(movement_tactics)


#THE_SWARM####################################################
def swarm_agent(observation, configuration, **kwargs):
    """ RELEASE THE SWARM!!! """
    s_env = adapt_environment(observation, configuration)
    actions = actions_of_ships(s_env)
    actions = actions_of_shipyards(actions, s_env)

    if observation.step == 0:
        actions = {list(observation.players[observation.player][2].keys())[0]: "CONVERT"}

    return actions
