# for Debug/Train previous line should be commented out, uncomment to write submission.py 

import numpy as np
import datetime as datetime

print_log = False;
log = []

def logit(text):
    log.append(text)
def reset_game_map(obs):
    """ redefine game_map as two dimensional array of objects and set amounts of halite in each cell """
    global game_map
    game_map = []
    for x in range(conf.size):
        game_map.append([])
        for y in range(conf.size):
            game_map[x].append({
                "shipyard": None,
                "ship": None,
                "halite": obs.halite[conf.size * y + x],
                "targeted": 0,
                'enemy_halite': 0,
                
                "lightest_enemy_touching": 1e6,
                'enemies_touching': 0,
                'enemy_halite_touching': 0,
                
                'lightest_enemy_nearby': 1e6,
                'enemies_nearby': 0,
                'enemy_halite_nearby': 0, 
                
                'lightest_enemy_within_3': 1e6,
                'enemies_within_3': 0,
                'enemy_halite_within_3': 0, 
                
                "please_move": 0,
                "touching_base": False,
            })

def get_my_units_coords_and_update_game_map(obs):
    """ get lists of coords of my units and update locations of ships and shipyards on the map """
    # arrays of (x, y) coords
    global game_map
    my_shipyards_coords = []
    my_ships_coords = []
    
    for player in range(len(obs.players)):
        shipyards = list(obs.players[player][1].values())
        for shipyard in shipyards:
            x = shipyard % conf.size
            y = shipyard // conf.size
            # place shipyard on the map
            game_map[x][y]["shipyard"] = player
            if player == obs.player:
                my_shipyards_coords.append((x, y))
                for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:
                    game_map[spot[0]][spot[1]]["touching_base"] = True
        
        ships = list(obs.players[player][2].values())
        for ship in ships:
            x = ship[0] % conf.size
            y = ship[0] // conf.size
            # place ship on the map
            game_map[x][y]["ship"] = player
            if player == obs.player: # mine
                my_ships_coords.append((x, y))
            else: # enemy ship
                game_map[x][y]["enemy_halite"] = ship[1]
                for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:
                    
                    game_map[spot[0]][spot[1]]["enemies_touching"] += 1
                    game_map[spot[0]][spot[1]]["enemy_halite_touching"] += ship[1]
                    
                    if ship[1] < game_map[spot[0]][spot[1]]["lightest_enemy_touching"]:
                        game_map[spot[0]][spot[1]]["lightest_enemy_touching"] = ship[1]
                
                for spot in [(x, y), 
                              (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1)),
                               (c(x-2), y), (c(x+2), y), (x, c(y-2)), (x, c(y+2)),
                                    (c(x-1), c(y-1)), (c(x-1), c(y+1)), (c(x+1), c(y-1)),  (c(x+1), c(y+1)),]:
                    
                    game_map[spot[0]][spot[1]]["enemies_nearby"] += 1
                    game_map[spot[0]][spot[1]]["enemy_halite_nearby"] += ship[1]
                    
                    if ship[1] < game_map[spot[0]][spot[1]]["lightest_enemy_nearby"]:
                        game_map[spot[0]][spot[1]]["lightest_enemy_nearby"] = ship[1]
                    
                for spot in [(x, y), 
                              (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1)),
                               (c(x-2), y), (c(x+2), y), (x, c(y-2)), (x, c(y+2)),
                                    (c(x-1), c(y-1)), (c(x-1), c(y+1)), (c(x+1), c(y-1)),  (c(x+1), c(y+1)),
                               (c(x-3), y), (c(x+3), y), (x, c(y-3)), (x, c(y+3)),
                                  (c(x-2), c(y-1)), (c(x-2), c(y-1)), (c(x+2), c(y-1)),  (c(x+2), c(y+1)),
                                  (c(x-1), c(y-2)), (c(x-1), c(y+2)), (c(x+1), c(y-2)),  (c(x+1), c(y+2)),
                            ]:
                    
                    if ship[1] < game_map[spot[0]][spot[1]]["lightest_enemy_within_3"]:
                        game_map[spot[0]][spot[1]]["lightest_enemy_within_3"] = ship[1]
                        
                    game_map[spot[0]][spot[1]]["enemies_within_3"] += 1
                    game_map[spot[0]][spot[1]]["enemy_halite_within_3"] += ship[1]    
    
    return my_shipyards_coords, my_ships_coords

def get_x(x):
    return (x % conf.size)

def get_y(y):
    return (y % conf.size)

def c(s):
    return (s % conf.size)

def enemy_ship_near(x, y, player, ship_halite):
    """ check if enemy ship can attack a square """
#     for spot in [game_map[x][y], game_map[c(x-1)][y], game_map[c(x+1)][y], game_map[x][c(y-1)], game_map[x][c(y+1)] ]:
#         if (spot["ship"] != None and spot["ship"] != player 
#                      and spot["enemy_halite"] <= ship_halite): 
#             return True
#     return False
    if game_map[x][y]["lightest_enemy_touching"] <= ship_halite:
        return True
    else:
        return False

def threat(x, y, player, ship_halite):
    spot = game_map[get_x(x)][get_y(y)]
    return ( (spot["ship"] != None and spot["ship"] != player 
                     and spot["enemy_halite"] <= ship_halite) 
             or (spot["shipyard"] != None and spot["shipyard"] != player ) )
    

def nearestThreateningShip(x, y, player, ship_halite):
    for dist in range(conf.size):
        for xs in range(dist + 1):
            ys = dist-xs;
            if ( threat(x + xs, y + ys, player, ship_halite) or
                 threat(x - xs, y + ys, player, ship_halite) or
                 threat(x + xs, y - ys, player, ship_halite) or
                 threat(x - xs, y - ys, player, ship_halite) ):
                logit('nearest threat is {} units away'.format(dist))
                return dist;
            
    else:
        logit('no threats?')
        return 100;

# def isBase(x, y, player):
#     if game_map[x][y]["shipyard"] != None and game_map[x][y]["shipyard"] == player:
#         return True
#     else:
#         return False
    
# def touchingBase(x, y, player):
#     for spot in [game_map[x][y], game_map[c(x-1)][y], game_map[c(x+1)][y], game_map[x][c(y-1)], game_map[x][c(y+1)] ]:
#         if spot["shipyard"] != None and spot["shipyard"] == player:
#             return True
#     return False

'''check whether current spot and the four spots can move to are all threatened'''
def surrounded(x, y, player, halite_onboard):
    for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:
        if not enemy_ship_near(*spot, player, halite_onboard):
            return False
    return True

'''check if have only one possible move'''
def onlyOneOption(x, y, player, halite_onboard, moveable = False):
    viable_spot = []
    for spot in [(x, y), (c(x-1), y), (c(x+1), y), (x, c(y-1)), (x, c(y+1))]:
        if clearArea(*spot, player, halite_onboard, moveable):
            if len(viable_spot) > 0: # if have multiple spots
                return None
            else:
                viable_spot = spot
    if len(viable_spot) == 0:
        logit("should have been flagged as surrounded")
        return None
        
    return viable_spot

def clear(x, y, player, halite_onboard, moveable = False):
    """ check if cell is clear to move into """
    if ((game_map[x][y]["shipyard"] == player or game_map[x][y]["shipyard"] == None) and
            ( (game_map[x][y]["ship"] == player and moveable)   or   (game_map[x][y]["ship"] == None)  or 
              ( game_map[x][y]["ship"] !=  player  and game_map[x][y]["enemy_halite"] > halite_onboard)) ):
        return True
    return False

def clearArea(x, y, player, halite_onboard, moveable = False):
    return (clear(x, y, player, halite_onboard, moveable) and not enemy_ship_near(x, y, player, halite_onboard))
# def distanceTo(xf, yf, x_pos, y_pos):
#     return np.min( (   (xf - x_pos) % conf.size, conf.size - ((xf - x_pos) % conf.size) ) ) \
#             + np.min( (   (yf - y_pos) % conf.size, conf.size - ((yf - y_pos) % conf.size) ) ) 

def distanceTo(xf, yf, x_pos, y_pos):
    xd = abs(xf - x_pos)
    yd = abs(yf - y_pos)
    return min(xd, 21-xd) + min(yd, 21-yd)

def directionTo(xf, yf, x_pos, y_pos):
    return (0 if xf==x_pos else (1 if (xf - x_pos) % conf.size <= conf.size // 2 else -1)), \
              (0 if yf==y_pos else (1 if (yf - y_pos) % conf.size <= conf.size // 2 else -1)),

# ship values 
SHIP_MINE_PCT = 0.2  
SHIP_HUNT_PCT = 0.10     
AVG_HALITE_PWR = 0.8
ENEMY_HALITE_PWR = 0.8

# final drop-offs
MIN_FINAL_DROPOFF = 100   
PCTILE_DROPOFF = 95   
RETURN_HOME = 360  
MUST_DROPOFF = 2000

# mining variables
EXPECTED_MINE_TURNS = 3     # possibly lower//
EXPECTED_MINE_FRAC = np.sum([ 0.25 * (1 - 0.25 + 0.02) ** i for i in range(int(EXPECTED_MINE_TURNS))])

INERTIA = 1.1   
DISTANCE_DECAY = 0.04   

# return and conversion variables
RETURN_CHANCE = 0.5                        
YARD_CONVERSION_CHANCE = 0.25    
BASE_YARDS = 0.5
SHIP_TO_BASE_MULT = 2.5    

# raid variables
RAID_BASE_TURNS = 1 / 2      # WILDCARD -- may also generate a lot of variety
RAID_DISTANCE_DECAY = 0.15 
RAID_RETARGET_DECAY = 0.35  

# deprecated
RETURN_RESIDUAL = 0 
MOVE_MULTIPLIER = 1  
RAID_ONBOARD_PENALTY = 0
RAID_MINE_RESIDUAL = 0    


# raid parameters - enemies and halite nearby
RAID_MULTIPLIER = 0.10    # likely increase this//
REHT_MULTIPLIER = 0.05  
REHN_MULTIPLIER = 0.03   
REH3_MULTIPLIER = 0.015 

RLEN_PENALTY = -60     # been moving higher
RLE3_PENALTY = -30  

# mining parameters - nearness to base
MINE_BASE_DIST_MULT = 0.0

# mining parameters - enemies nearby
MET_ADJUSTMENT = 0
MEN_ADJUSTMENT = 0
ME3_ADJUSTMENT = 0

MLET_FWD = 0.25
MLEN_FWD = 0.25
MLE3_FWD = 0.25

MLET_PENALTY = -150   # probably slash this //
MLEN_PENALTY = -10
MLE3_PENALTY = 0

MEHT_BONUS = 0
MEHN_BONUS = 0
MEH3_BONUS = 0


def findBestSpot(x_pos, y_pos, player, my_halite, halite_onboard, 
                 my_shipyards_coords, num_shipyards, avg_halite, step, num_ships,
                   live_run = False):   
    moveable = not live_run
    
    must_move = enemy_ship_near(x_pos, y_pos, player, halite_onboard)
    ship_inertia = (INERTIA if not game_map[x_pos][y_pos]["touching_base"] else 1)
    
    current_halite = game_map[x_pos][y_pos]["halite"]
    best_spot = (x_pos, y_pos, np.floor(current_halite \
                                            * EXPECTED_MINE_FRAC / EXPECTED_MINE_TURNS \
                                             * (ship_inertia) 
                                              * (0 if must_move else 1) ), 'remain')
    
    if surrounded(x_pos, y_pos, player, halite_onboard):
        if halite_onboard > conf.convertCost or my_halite > conf.convertCost:
            logit('emergency conversion at ({}, {}) to preserve {} halite'.format(x_pos, y_pos, halite_onboard))
            return (x_pos, y_pos, np.floor(halite_onboard) + conf.convertCost, 'conversion') 
        else:
            logit('surrounded at ({}, {}) but not enough cash for emergency conversion'.format(x_pos, y_pos))
    
    ooo = onlyOneOption(x_pos, y_pos, player, halite_onboard, moveable)
    if ooo is not None:
        logit('only one option from ({}, {}), heading {}, {}'.format(x_pos, y_pos, ooo[0], ooo[1]))
        return ( c(x_pos + ooo[0]), c(y_pos + ooo[1]), np.floor(halite_onboard) + conf.convertCost, 'only one option' )
    
    if halite_onboard > conf.convertCost:
        closest_yard, min_dist = findNearestYard(my_shipyards_coords, x_pos, y_pos)
        spot_value = halite_onboard * YARD_CONVERSION_CHANCE * min_dist / 20 \
                       * (1 / ( BASE_YARDS + num_shipyards ) )  #* (50 / (avg_halite + BASE_AVG_HALITE))
        if spot_value > best_spot[2]:
            best_spot = (x_pos, y_pos, np.floor(spot_value), 'conversion')
    
    ship_value = getShipValue(num_ships, step)
    return_penalty = ( (ship_value - conf.spawnCost) 
                         if ( my_halite > conf.spawnCost and ship_value > conf.spawnCost ) else 0)
    
    # consider returning
    return_value = 0; nearest_base = 20;
    for shipyard_coords in my_shipyards_coords:
        x, y = shipyard_coords
        x_dir, y_dir = directionTo(x, y, x_pos, y_pos)
         
        # if dangerous spot or both paths are dangerous, ignore;
        if ( not clearArea(x, y, player, halite_onboard, moveable) or
            ( not clearArea( (x_pos + x_dir) % conf.size, y_pos, player, halite_onboard, moveable)  and  
               not clearArea( x_pos, (y_pos + y_dir) % conf.size, player, halite_onboard, moveable)) ):
            continue;

        
        dist = distanceTo(x, y, x_pos, y_pos) #  * (step + base_step) ** 2 / (base_step + 400)
        
        if dist < nearest_base:
            nearest_base = dist
        
        spot_value = ( halite_onboard * RETURN_CHANCE  
                        / ( dist * MOVE_MULTIPLIER + 1e-6) 
                             * np.min( ( np.sqrt(step/ 30), 1 ) )
                             * np.clip( ( ship_value / conf.spawnCost), 1, 1.5)         
                      )
        if spot_value > return_value:
            return_value = spot_value
        if spot_value > best_spot[2] and not enemy_ship_near(x, y, player, halite_onboard):
            best_spot = (x, y, np.floor(spot_value), 'dropoff')

    for x in range(conf.size):
        for y in range(conf.size):
            x_dir, y_dir = directionTo(x, y, x_pos, y_pos)
             
            # if dangerous spot or both paths are dangerous, ignore;
            if ( not clearArea(x, y, player, halite_onboard, moveable) or
                ( not clearArea( (x_pos + x_dir) % conf.size, y_pos, player, halite_onboard, moveable)  and  
                   not clearArea( x_pos, (y_pos + y_dir) % conf.size, player, halite_onboard, moveable)) ):
                continue;
            

                  
                    
                    
                    
                    
            # Consider Raiding: 
            spot_raid_value = 0
            if game_map[x][y]["enemy_halite"] > halite_onboard:
                dist = distanceTo(x, y, x_pos, y_pos)
                spot_raid_value = ( (game_map[x][y]["enemy_halite"] * RAID_MULTIPLIER  
                                - RAID_ONBOARD_PENALTY * halite_onboard  
                                    
                                     + game_map[x][y]['enemy_halite_touching'] * REHT_MULTIPLIER
                                     + game_map[x][y]['enemy_halite_nearby'] * REHN_MULTIPLIER
                                     + game_map[x][y]['enemy_halite_within_3'] * REH3_MULTIPLIER
                                     
                                     + (RLEN_PENALTY if game_map[x][y]['lightest_enemy_nearby'] < halite_onboard 
                                                             else 0) 
 
                                     + (RLE3_PENALTY if game_map[x][y]['lightest_enemy_within_3'] < halite_onboard 
                                                             else 0) 
 
                                     
                                    )
                             
                            / ( RAID_BASE_TURNS + dist * MOVE_MULTIPLIER )   
                                    * np.exp( - RAID_DISTANCE_DECAY * dist )  
                               * np.exp( - RAID_RETARGET_DECAY * game_map[x][y]["targeted"] ) 
                                  )
                    
                if spot_raid_value > best_spot[2]:
                    best_spot = (x, y, np.floor(spot_raid_value), 'raiding')
            
            
            
            # Mining: consider if must move or beats current spot, and not yet targeted
            if  ( ( must_move or game_map[x][y]["halite"] > current_halite * ship_inertia )
                    and game_map[x][y]["targeted"] == 0):
                dist = distanceTo(x, y, x_pos, y_pos)
                spot_value = ( game_map[x][y]["halite"] * EXPECTED_MINE_FRAC 
                            / ( EXPECTED_MINE_TURNS + dist * MOVE_MULTIPLIER 
                                                     + nearest_base * MINE_BASE_DIST_MULT )  
                                    * np.exp( - DISTANCE_DECAY * dist)
                                       * (0.7 if game_map[x][y]["touching_base"] else 1)
                              +  ( (return_value * RETURN_RESIDUAL - return_penalty) 
                                           if game_map[x][y]["touching_base"] else 0)
                              
                              + game_map[x][y]["enemies_touching"] * MET_ADJUSTMENT
                              + game_map[x][y]["enemies_nearby"] * MEN_ADJUSTMENT 
                              + game_map[x][y]["enemies_within_3"] * ME3_ADJUSTMENT
                                     
                              + game_map[x][y]["enemy_halite_touching"] * MEHT_BONUS
                              + game_map[x][y]["enemy_halite_nearby"] * MEHN_BONUS 
                              + game_map[x][y]["enemy_halite_within_3"] * MEH3_BONUS
                      
                              + (MLET_PENALTY if (game_map[x][y]['lightest_enemy_touching'] 
                                                    < halite_onboard + game_map[x][y]["halite"] * MLET_FWD)
                                                     else 0) 
                              + (MLEN_PENALTY if (game_map[x][y]['lightest_enemy_nearby'] 
                                                    < halite_onboard + game_map[x][y]["halite"] * MLEN_FWD)
                                                     else 0) 

                              + (MLE3_PENALTY if (game_map[x][y]['lightest_enemy_within_3'] 
                                                    < halite_onboard + game_map[x][y]["halite"] * MLE3_FWD)
                                                     else 0) 
 
                              + spot_raid_value * RAID_MINE_RESIDUAL)
                
                
                if spot_value > best_spot[2]:
                    best_spot = (x, y, np.floor(spot_value), 'mining')

   
    if best_spot[0] == x_pos and best_spot[1] == y_pos: # staying put
        sp = True
    else:  # if moving;
        if live_run:
            game_map[best_spot[0]][best_spot[1]]["targeted"] += 1

    if best_spot[3] == 'dropoff':
        logit('   moving from ({}, {}) to ({}, {}) to dropoff {} halite'.format(
            x_pos, y_pos, best_spot[0], best_spot[1], halite_onboard))
    if best_spot[3] == 'raiding':
        logit('attempting raid of ({}, {}) from ({}, {}) to gain {} halite'.format(
            best_spot[0], best_spot[1], x_pos, y_pos, game_map[best_spot[0]][best_spot[1]]["enemy_halite"] ))
        
    
    
    return best_spot

def findNearestYard(my_shipyards_coords, x, y):
    """ find nearest shipyard to deposit there"""
    min_dist = conf.size * 2
    closest_yard = 0
    for yard_idx, yard in enumerate(my_shipyards_coords):
        dist = np.min( (  ((x - my_shipyards_coords[yard_idx][0]) % conf.size), 
                                (21 - ((x - my_shipyards_coords[yard_idx][0]) % conf.size))  ) ) \
          + np.min( (  ((y - my_shipyards_coords[yard_idx][1]) % conf.size), 
                     (21 - ((y - my_shipyards_coords[yard_idx][1]) % conf.size))  ) )
        if dist < min_dist:
            min_dist = dist;
            closest_yard = yard_idx
    return closest_yard, min_dist
            
def moveTo(x_initial, y_initial, x_target, y_target, ship_id, halite_onboard, player, actions):
    """ move toward target as quickly as possible without collision (or later, bad collision)"""
    if (x_target - x_initial) % conf.size <=  ( 1 + conf.size) // 2 :
        # move down
        x_dir = 1;
        x_dist = (x_target - x_initial) % conf.size
    else:
        # move up
        x_dir = -1;
        x_dist = (x_initial - x_target) % conf.size
    
    if (y_target - y_initial) % conf.size <=  ( 1 + conf.size) // 2 :
        # move down
        y_dir = 1;
        y_dist = (y_target - y_initial) % conf.size
    else:
        # move up
        y_dir = -1;
        y_dist = (y_initial - y_target) % conf.size
    
    action = None
    if x_dist > y_dist:
        # move X first if can;
        if clearArea( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):
            action = ('WEST' if x_dir <0 else 'EAST')
        elif clearArea( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :
            action = ('NORTH' if y_dir < 0 else 'SOUTH')
        
    else:
        # move Y first if can
        if clearArea( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :
            action = ('NORTH' if y_dir < 0 else 'SOUTH')
        elif clearArea( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):
            action = ('WEST' if x_dir <0 else 'EAST')
    
    # if area was not clear, then just move whoever is currently empty
    if enemy_ship_near(x_initial, y_initial, player, halite_onboard):
        logit('moving into traffic from ({}, {})'.format(x_initial, y_initial))
        if x_dist > y_dist:
            # move X first if can;
            if clear( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):
                action = ('WEST' if x_dir <0 else 'EAST')
            elif clear( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :
                action = ('NORTH' if y_dir < 0 else 'SOUTH')

        else:
            # move Y first if can
            if clear( x_initial, ( y_initial + y_dir) % conf.size, player, halite_onboard) :
                action = ('NORTH' if y_dir < 0 else 'SOUTH')
            elif clear( ( x_initial + x_dir) % conf.size, y_initial, player, halite_onboard):
                action = ('WEST' if x_dir <0 else 'EAST')
            
    if action is not None:
        game_map[x_initial][y_initial]["ship"] = None
        actions[ship_id] = action
        
    if action == 'NORTH':
        game_map[x_initial][(y_initial - 1) % conf.size]["ship"] = player
    elif action == 'SOUTH':
        game_map[x_initial][(y_initial + 1) % conf.size]["ship"] = player
    elif action == 'WEST':
        game_map[(x_initial - 1) % conf.size][y_initial]["ship"] = player
    elif action == 'EAST':
        game_map[(x_initial + 1) % conf.size][y_initial]["ship"] = player
    
    
    return actions

# def get_directions(i0, i1, i2, i3):
#     """ get list of directions in a certain sequence """
#     return [directions[i0], directions[i1], directions[i2], directions[i3]]

def orderBasesForSpawning(shipyards_keys, my_shipyards_coords, player, my_ships_coords):
    base_values = np.zeros(len(my_shipyards_coords))
    for yard_idx, yard_coords in enumerate(my_shipyards_coords):
        x = yard_coords[0]
        y = yard_coords[1]
        
        # ensure base is clear
        if not clear(x, y, player, 0):
            continue;
        
        nearby_halite = 0
        nearby_enemy_halite = 0
        nearby_ships = 0.1
        
        BASE_MINING_TURNS = 3
        BASE_SHIP_TURNS = 0.5
        RADIUS = 10
        ENEMY_DISTANCE_DECAY = 0.10
        
        BASE_SHIPS_PWR = 0
        
        RAIDING_EFFICIENCY = 0.03  
        
        # look at halite nearby (favor lots of halite nearby)
        for xs in range(-RADIUS, RADIUS + 1):
            for ys in range(-RADIUS, RADIUS + 1):
                nearby_halite += ( game_map[c(x + xs)][c(y + ys)]["halite"] 
                                            / (BASE_MINING_TURNS + 2 * ( abs(xs) + abs(ys) ) ) )
                nearby_enemy_halite += ( game_map[c(x + xs)][c(y + ys)]["enemy_halite"] 
                                                  * np.exp( - ENEMY_DISTANCE_DECAY * ( abs(xs) + abs(ys) ) ) )
        
        for s in my_ships_coords:
            nearby_ships +=  ( 1 / (BASE_SHIP_TURNS + distanceTo( s[0], s[1], x, y) ) )
            
        
        base_values[yard_idx] = (  ( nearby_halite + RAIDING_EFFICIENCY * nearby_enemy_halite ) 
                                       / ( nearby_ships ** BASE_SHIPS_PWR ) ) 
        
    priorities = np.argsort(base_values)[::-1]
    logit( [int(base_values[i]) for i in priorities])
    
    my_shipyards_coords = [ my_shipyards_coords[i] for i in priorities]
    shipyards_keys = [shipyards_keys[i] for i in priorities] 
    
    return shipyards_keys, my_shipyards_coords

def getAverageHalite():
    total_halite = 0
    for x in range(conf.size):
        for y in range(conf.size):
            total_halite += game_map[x][y]["halite"]
    return total_halite / (conf.size ** 2)
    

def getAverageEnemyHalite():
    total_enemy_halite = 0
    for x in range(conf.size):
        for y in range(conf.size):
            total_enemy_halite += game_map[x][y]["enemy_halite"]
    return total_enemy_halite / (conf.size ** 2)

def define_some_globals(config):
    """ define some of the global variables """
    global conf
    global convert_plus_spawn_cost
    global globals_not_defined
    conf = config
    convert_plus_spawn_cost = conf.convertCost + conf.spawnCost
    globals_not_defined = False


############################################################################
conf = None
game_map = [] 
convert_plus_spawn_cost = None  
 
globals_not_defined = True 

def getShipValue(num_ships, step):
    return (  (SHIP_MINE_PCT * getAverageHalite() ** AVG_HALITE_PWR  +
                            SHIP_HUNT_PCT * getAverageEnemyHalite() ** ENEMY_HALITE_PWR ) 
                            * (10 / num_ships)
                              * (400 - step) 
            
            + (1000 if num_ships < 3 else 0)
            + (3000 if num_ships < 2 else 0)
           
           ) 


    
def strategicShipSpawning(my_halite, actions, num_ships, shipyards_keys, my_shipyards_coords, step, player):
    for i in range(len(my_shipyards_coords)):
        ship_value  =  getShipValue(num_ships, step)
        logit('Ship Value: {:.0f}'.format(ship_value))
        
        if (my_halite < conf.spawnCost):
            logit('  not enough halite to spawn ship')
            break;
            
        if ship_value < conf.spawnCost:
            break;

        x = my_shipyards_coords[i][0]
        y = my_shipyards_coords[i][1]
        if clear(x, y, player, 0):
            my_halite -= conf.spawnCost
            actions[shipyards_keys[i]] = "SPAWN"
            num_ships += 1
            game_map[x][y]["ship"] = player
            logit('spawning new ship at ({}, {})'.format(x, y))

                
    return my_halite, actions


def timeout(start_time, player, step, rd, ship, config):

    TIMEOUT = 0.7 * config.actTimeout *  1e6   # * 100e3  
    
    if (datetime.datetime.now() - start_time).microseconds > TIMEOUT:
        print('AGENT {} TIMED OUT ON STEP {} - for Round {} and ship number {}'.format(player, step, rd, ship))
        return True;
    else:
        return False;

def swarm_agent(obs, config, **kwargs):
    start_time = datetime.datetime.now()
    logit('\nStep {}'.format(obs.step))
    if globals_not_defined:
        define_some_globals(config)
    actions = {}
    my_halite = obs.players[obs.player][0]
    
    reset_game_map(obs)
    my_shipyards_coords, my_ships_coords = get_my_units_coords_and_update_game_map(obs)
    num_shipyards = len(my_shipyards_coords)
    
    ships_keys = list(obs.players[obs.player][2].keys()); ships_values = list(obs.players[obs.player][2].values())
    shipyards_keys = list(obs.players[obs.player][1].keys())
    avg_halite = getAverageHalite()

    # order the actions of ships
    move_values = []; # moves = []
    for i in range(len(my_ships_coords)):
        if timeout(start_time, obs.player, obs.step, 1, i, config):
            return actions;
        halite_onboard = ships_values[i][1]
        x = my_ships_coords[i][0]; y = my_ships_coords[i][1]
        
        x_target, y_target, spot_value, purpose = findBestSpot(x, y, obs.player, my_halite, halite_onboard,
                                                                          my_shipyards_coords, num_shipyards,
                                                                         avg_halite, obs.step, len(ships_keys), 
                                                                           False)
        x_dir, y_dir = directionTo(x_target, y_target, x, y)
#         moves.append( c(x + x_dir), c(y + y_dir) )
        if spot_value > game_map[c(x + x_dir)][c(y + y_dir)]["please_move"]:
            game_map[c(x + x_dir)][c(y + y_dir)]["please_move"] = spot_value + 1
            
        if purpose == 'only one option':
            logit('only one option for ship at ({}, {}) with {} halite onboard'.format(x, y, halite_onboard))
        move_values.append(int(spot_value))
    
    # check for any 'please move' requests
    for i in range(len(my_ships_coords)):
        x = my_ships_coords[i][0]; y = my_ships_coords[i][1]
        if game_map[x][y]["please_move"] > move_values[i]:
            move_values[i] = game_map[x][y]["please_move"]
            logit("please move off ({}, {}) with weight {}".format(x, y, move_values[i]))
      
    priorities = np.argsort(move_values)[::-1]
    logit( [move_values[i] for i in priorities])
    
    my_ships_coords = [my_ships_coords[i] for i in priorities]
    ships_keys = [ships_keys[i] for i in priorities]
    ships_values = [ships_values[i] for i in priorities]
    
    logit('\n - actions - \n')
        
    # execute ship actions
    for i in range(len(my_ships_coords)):
        if timeout(start_time, obs.player, obs.step, 2, i, config):
            return actions;
        
        halite_onboard = ships_values[i][1]
        x = my_ships_coords[i][0]; y = my_ships_coords[i][1]

        # no yards or enormous halite surplus, must convert;
        if ( (num_shipyards==0 and (my_halite >= conf.convertCost or halite_onboard >= convert_plus_spawn_cost))
               or (halite_onboard >= convert_plus_spawn_cost * SHIP_TO_BASE_MULT) ):
            actions[ships_keys[i]] = "CONVERT"
            num_shipyards += 1
            logit('forced conversion')
            continue;
           
        # '''final dropoff or must dropoff'''
        elif ( (obs.step > RETURN_HOME) and 
              (  (halite_onboard > MIN_FINAL_DROPOFF) or 
                       (halite_onboard >  np.percentile( [s[1] for s in ships_values], PCTILE_DROPOFF) ) )
           or    (halite_onboard > MUST_DROPOFF  ) ):
            
            if len(my_shipyards_coords) > 0:
                closest_yard, min_dist = findNearestYard(my_shipyards_coords, x, y)
                actions = moveTo(x, y, *my_shipyards_coords[closest_yard], ships_keys[i], halite_onboard,
                                         obs.player, actions)
                
        # '''figure out best move''' 
        else:
            x_target, y_target, new_spot_halite, purpose = findBestSpot(x, y, obs.player, my_halite, halite_onboard,
                                                                          my_shipyards_coords, num_shipyards,
                                                                avg_halite, obs.step, len(ships_keys), True)
            if not (x_target == x and y_target == y):
#                 logit('aiming to get to ({}, {}) from ({}, {})'.format(x_target,y_target, x ,y))
                actions = moveTo(x, y, x_target, y_target, ships_keys[i], halite_onboard, obs.player, actions)
            elif purpose == 'conversion':
                actions[ships_keys[i]] = "CONVERT"
                logit('converting to base')
                num_shipyards += 1

    if my_halite >= conf.spawnCost:
        shipyards_keys, my_shipyards_coords = orderBasesForSpawning(shipyards_keys, my_shipyards_coords,
                                                                       obs.player, my_ships_coords )

    # YARDS AND SPAWNING:
    # auto-spawn if no ships, or first 30 turns
    if ( len(ships_keys) == 0 or obs.step <= 30) and my_halite >= conf.spawnCost:
        for i in range(len(my_shipyards_coords)):
            if my_halite >= conf.spawnCost:
                x = my_shipyards_coords[i][0]
                y = my_shipyards_coords[i][1]
                if clear(x, y, obs.player, 0):
                    my_halite -= conf.spawnCost
                    actions[shipyards_keys[i]] = "SPAWN"
                    game_map[x][y]["ship"] = obs.player
                    logit('auto-spawn')

    # strategic spawning:
    else:
        my_halite, actions = strategicShipSpawning(my_halite, actions, len(ships_keys),
                                                   shipyards_keys, my_shipyards_coords, 
                                                   obs.step, obs.player)
        
    return actions
