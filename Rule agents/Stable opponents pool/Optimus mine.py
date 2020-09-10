import time
import copy
import sys
import math
import collections
import pprint
import numpy as np
import scipy.optimize
import scipy.ndimage
from kaggle_environments.envs.halite.helpers import *
import kaggle_environments
'''
Initialization code can run when the file is loaded.  The first call to agent() is allowed 30 sec
The agent function is the last function in the file (does not matter its name)
agent must run in less than 6 sec
Syntax errors when running the file do not cause any message, but cause your agent to do nothing

'''
CONFIG_MAX_SHIPS=20

#print('kaggle version',kaggle_environments.__version__)
#### Global variables
all_actions=[ShipAction.NORTH, ShipAction.EAST,ShipAction.SOUTH,ShipAction.WEST]
all_dirs=[Point(0,1), Point(1,0), Point(0,-1), Point(-1,0)]
start=None
num_shipyard_targets=4
size=None
# Will keep track of whether a ship is collecting halite or carrying cargo to a shipyard
ship_target={}   # where to go to collect
me=None
did_init=False
quiet=False
C=None
class Obj:
  pass
# will hold global data for this turn, updating as we set actions.
# E.g. number of ships, amount of halite
# taking into account the actions set so far.  Also max_ships, etc.
turn=Obj()
### Data
#turns_optimal[CHratio, round_trip_travel] for mining
#See notebook on optimal mining kaggle.com/solverworld/optimal-mining
turns_optimal=np.array(
  [[0, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8],
   [0, 1, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7],
   [0, 0, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 7],
   [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6],
   [0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6],
   [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5],
   [0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#### Functions
def print_enemy_ships(board):
  print('\nEnemy Ships')
  for ship in board.ships.values():
    if ship.player_id != me.id:
      print('{:6}  {} halite {}'.format(ship.id,ship.position,ship.halite))
      
def print_actions(board):
  print('\nShip Actions')
  for ship in me.ships:
    print('{:6}  {}  {} halite {}'.format(ship.id,ship.position,ship.next_action,ship.halite))
  print('Shipyard Actions')
  for sy in me.shipyards:
    print('{:6}  {}  {}'.format(sy.id,sy.position,sy.next_action))

def print_none(*args):
  pass

def compute_max_ships(step):
  #This needs to be tuned, perhaps based on amount of halite left
  if step < 200:
    return CONFIG_MAX_SHIPS
  elif step < 300:
    return CONFIG_MAX_SHIPS-2
  elif step < 350:
    return CONFIG_MAX_SHIPS-4
  else:
    return CONFIG_MAX_SHIPS-5

def set_turn_data(board):
  #initialize the global turn data for this turn
  turn.num_ships=len(me.ships)
  turn.max_ships=compute_max_ships(board.step)
  turn.total_halite=me.halite
  #this is matrix of halite in cells
  turn.halite_matrix=np.reshape(board.observation['halite'], (board.configuration.size,board.configuration.size))
  turn.num_shipyards=len(me.shipyards)
  #compute enemy presence and enemy halite matrices
  turn.EP,turn.EH,turn.ES=gen_enemy_halite_matrix(board)
  #filled in by shipid as a ship takes up a square
  turn.taken={}
  turn.last_episode = (board.step == (board.configuration.episode_steps-2))
  
def init(obs,config):
  #This is only called on first call to agent()
  #Do initalization things
  global size
  global print
  if hasattr(config,'myval') and config.myval==9 and not quiet:
    #we are called locally, so leave prints OK
    pass
  else:
    #we are called in competition, quiet output
    print=print_none
    pprint.pprint=print_none
  size = config.size

def limit(x,a,b):
  if x<a:
    return a
  if x>b:
    return b
  return x
  
def num_turns_to_mine(C,H,rt_travel):
  #How many turns should we plan on mining?
  #C=carried halite, H=halite in square, rt_travel=steps to square and back to shipyard
  if C==0:
    ch=0
  elif H==0:
    ch=turns_optimal.shape[0]
  else:
    ch=int(math.log(C/H)*2.5+5.5)
    ch=limit(ch,0,turns_optimal.shape[0]-1)
  rt_travel=int(limit(rt_travel,0,turns_optimal.shape[1]-1))
  return turns_optimal[ch,rt_travel]

def halite_per_turn(carrying, halite,travel,min_mine=1):
  #compute return from going to a cell containing halite, using optimal number of mining steps
  #returns halite mined per turn, optimal number of mining steps
  #Turns could be zero, meaning it should just return to a shipyard (subject to min_mine)
  turns=num_turns_to_mine(carrying,halite,travel)
  if turns<min_mine:
    turns=min_mine
  mined=carrying+(1-.75**turns)*halite
  return mined/(travel+turns), turns
  
def move(pos, action):
  ret=None
  #return new Position from pos when action is applied
  if action==ShipAction.NORTH:
    ret=pos+Point(0,1)
  if action==ShipAction.SOUTH:
    ret=pos+Point(0,-1)
  if action==ShipAction.EAST:
    ret=pos+Point(1,0)
  if action==ShipAction.WEST:
    ret=pos+Point(-1,0)
  if ret is None:
    ret=pos
  #print('move pos {} {} => {}'.format(pos,action,ret))
  return ret % size

def dirs_to(p1, p2, size=21):
  #Get the actions you should take to go from Point p1 to Point p2
  #using shortest direction by wraparound
  #Args: p1: from Point
  #      p2: to Point
  #      size:  size of board
  #returns: list of directions, tuple (deltaX,deltaY)
  #The list is of length 1 or 2 giving possible directions to go, e.g.
  #to go North-East, it would return [ShipAction.NORTH, ShipAction.EAST], because
  #you could use either of those first to go North-East.
  #[None] is returned if p1==p2 and there is no need to move at all
  deltaX, deltaY=p2 - p1
  if abs(deltaX)>size/2:
    #we wrap around
    if deltaX<0:
      deltaX+=size
    elif deltaX>0:
      deltaX-=size
  if abs(deltaY)>size/2:
    #we wrap around
    if deltaY<0:
      deltaY+=size
    elif deltaY>0:
      deltaY-=size
  #the delta is (deltaX,deltaY)
  ret=[]
  if deltaX>0:
    ret.append(ShipAction.EAST)
  if deltaX<0:
    ret.append(ShipAction.WEST)
  if deltaY>0:
    ret.append(ShipAction.NORTH)
  if deltaY<0:
    ret.append(ShipAction.SOUTH)
  if len(ret)==0:
    ret=[None]  # do not need to move at all
  return ret, (deltaX,deltaY)

def shipyard_actions():
  #spawn a ship as long as there is no ship already moved to this shipyard
  for sy in me.shipyards:
    if turn.num_ships < turn.max_ships:
      if turn.total_halite >= 500 and sy.position not in turn.taken:
        #spawn one
        sy.next_action = ShipyardAction.SPAWN
        turn.taken[sy.position]=1
        turn.num_ships+=1
        turn.total_halite-=500

def gen_enemy_halite_matrix(board):
  #generate matrix of enemy positions:
  #EP=presence of enemy ship
  #EH=amount of halite in enemy ships
  #ES=presence of enemy shipyards
  EP=np.zeros((size,size))
  EH=np.zeros((size,size))
  ES=np.zeros((size,size))
  for id,ship in board.ships.items():
    if ship.player_id != me.id:
      EH[ship.position.y,ship.position.x]=ship.halite
      EP[ship.position.y,ship.position.x]=1
  for id, sy in board.shipyards.items():
    if sy.player_id != me.id:
      ES[sy.position.y,sy.position.x]=1
  return EP,EH,ES

def dist(a,b):
  #Manhattan distance of the Point difference a to b, considering wrap around
  action,step=dirs_to(a, b, size=21) 
  return abs(step[0]) + abs(step[1])

def nearest_shipyard(pos):
  #return distance, position of nearest shipyard to pos.  100,None if no shipyards
  mn=100
  best_pos=None
  for sy in me.shipyards:
    d=dist(pos, sy.position)
    if d<mn:
      mn=d
      best_pos=sy.position
  return mn,best_pos
  
def assign_targets(board,ships):
  #Assign the ships to a cell containing halite optimally
  #set ship_target[ship_id] to a Position
  #We assume that we mine halite containing cells optimally or return to deposit
  #directly if that is optimal, based on maximizing halite per step.
  #Make a list of pts containing cells we care about, this will be our columns of matrix C
  #the rows are for each ship in collect
  #computes global dict ship_tagert with shipid->Position for its target
  #global ship targets should already exist
  old_target=copy.copy(ship_target)
  ship_target.clear()
  if len(ships)==0:
    return
  halite_min=50
  pts1=[]
  pts2=[]
  for pt,c in board.cells.items():
    assert isinstance(pt,Point)
    if c.halite > halite_min:
      pts1.append(pt)
  #Now add duplicates for each shipyard - this is directly going to deposit
  for sy in me.shipyards:
    for i in range(num_shipyard_targets):
      pts2.append(sy.position)
  #this will be the value of assigning C[ship,pt]
  C=np.zeros((len(ships),len(pts1)+len(pts2)))
  #this will be the optimal mining steps we calculated
  for i,ship in enumerate(ships):
    for j,pt in enumerate(pts1+pts2):
      #two distances: from ship to halite, from halite to nearest shipyard
      d1=dist(ship.position,pt)
      d2,shipyard_position=nearest_shipyard(pt)
      if shipyard_position is None:
        #don't know where to go if no shipyard
        d2=1
      #value of target is based on the amount of halite per turn we can do
      my_halite=ship.halite
      if j < len(pts1):
        #in the mining section
        v, mining=halite_per_turn(my_halite,board.cells[pt].halite, d1+d2)
        #mining is no longer 0, due to min_mine (default)
      else:
        #in the direct to shipyard section
        if d1>0:
          v=my_halite/d1
        else:
          #we are at a shipyard
          v=0
      if board.cells[pt].ship and board.cells[pt].ship.player_id != me.id:
        #if someone else on the cell, see how much halite they have
        #enemy ship
        enemy_halite=board.cells[pt].ship.halite
        if enemy_halite <= my_halite:
          v = -1000   # don't want to go there
        else:
          if d1<5:
            #attack or scare off if reasonably quick to get there
            v+= enemy_halite/(d1+1)  # want to attack them or scare them off
      #print('shipid {} col {} is {} with {:8.1f} score {:8.2f}'.format(ship.id,j, pt,board.cells[pt].halite,v))
      C[i,j]=v
  print('C is {}'.format(C.shape))
  #Compute the optimal assignment
  row,col=scipy.optimize.linear_sum_assignment(C, maximize=True)
  #so ship row[i] is assigned to target col[j]
  #print('got row {} col {}'.format(row,col))
  #print(C[row[0],col[0]])
  pts=pts1+pts2
  for r,c in zip(row,col):
    ship_target[ships[r].id]=pts[c]
  #print out results
  print('\nShip Targets')
  print('Ship      position          target')
  for id,t in ship_target.items():
    st=''
    ta=''
    if board.ships[id].position==t:
      st='MINE'
    elif len(me.shipyards)>0 and t==me.shipyards[0].position:
      st='SHIPYARD'
    if id not in old_target or old_target[id] != ship_target[id]:
      ta=' NEWTARGET'
    print('{0:6}  at ({1[0]:2},{1[1]:2})  assigned ({2[0]:2},{2[1]:2}) h {3:3} {4:10} {5:10}'.format(
      id, board.ships[id].position, t, board.cells[t].halite,st, ta))

  return

def make_avoidance_matrix(myship_halite):
  #make a matrix of True where we want to avoid, uses
  #turn.EP=enemy position matrix
  #turn.EH=enemy halite matrix
  #turn.ES=enemy shipyard matrix
  filter=np.array([[0,1,0],[1,1,1],[0,1,0]])
  bad_ship=np.logical_and(turn.EH <= myship_halite,turn.EP)
  avoid=scipy.ndimage.convolve(bad_ship, filter, mode='wrap',cval=0.0)
  avoid=np.logical_or(avoid,turn.ES)
  return avoid

def make_attack_matrix(myship_halite):
  #make a matrix of True where we would want to move to attack an enemy ship
  #for now, we just move to where the ship is.
  #turn.EP=enemy position matrix
  #turn.EH=enemy halite matrix
  attack=np.logical_and(turn.EH > myship_halite,turn.EP)
  #print('attack',attack)
  return attack

def get_max_halite_ship(board, avoid_danger=True):
  #Return my Ship carrying max halite, or None if no ships
  #NOTE: creating avoid matrix again!
  mx=-1
  the_ship=None
  for ship in me.ships:
    x=ship.position.x
    y=ship.position.y
    avoid=make_avoidance_matrix(ship.halite)
    if ship.halite>mx and (not avoid_danger or not avoid[y,x]):
      mx=ship.halite
      the_ship=ship
  return the_ship

def remove_dups(p):
  #remove duplicates from a list without changing order
  #Not efficient for long lists
  ret=[]
  for x in p:
    if x not in ret:
      ret.append(x)
  return ret

def matrix_lookup(matrix,pos):
  return matrix[pos.y,pos.x]

def ship_converts(board):
  #if no shipyard, convert the ship carrying max halite unless it is in danger
  if turn.num_shipyards==0 and not turn.last_episode:
    mx=get_max_halite_ship(board)
    if mx is not None:
      if mx.halite + turn.total_halite > 500:
        mx.next_action=ShipAction.CONVERT
        turn.taken[mx.position]=1
        turn.num_shipyards+=1
        turn.total_halite-=500
  #Now check the rest to see if they should convert
  for ship in me.ships:
    if ship.next_action:
      continue
    #CHECK if in danger without escape, convert if h>500
    avoid=make_avoidance_matrix(ship.halite)
    z=[matrix_lookup(avoid,move(ship.position,a)) for a in all_actions]
    if np.all(z) and ship.halite > 500:
      ship.next_action=ShipAction.CONVERT
      turn.taken[ship.position]=1
      turn.num_shipyards+=1
      turn.total_halite-=500
      print('ship id {} no escape converting'.format(ship.id))
    #CHECK if last step and > 500 halite, convert
    if turn.last_episode and ship.halite > 500:
      ship.next_action=ShipAction.CONVERT
      turn.taken[ship.position]=1
      turn.num_shipyards+=1
      turn.total_halite-=500
      
def ship_moves(board, np_rng):
  ships=[ship for ship in me.ships if ship.next_action is None]
  #update ship_target
  assign_targets(board,ships)
  #For all ships without a target, we give them a random movement (we will check below if this
  actions={}   # record actions for each ship
  for ship in ships:
    if ship.id in ship_target:
      a,delta = dirs_to(ship.position, ship_target[ship.id],size=size)
      actions[ship.id]=a
    else:
      actions[ship.id]=[np_rng.choice(all_actions)]
      
  for ship in ships:
    action=None
    x=ship.position
    #generate matrix of places to attack and places to avoid
    avoid=make_avoidance_matrix(ship.halite)
    attack=make_attack_matrix(ship.halite)
    #see if there is a attack options
    action_list=actions[ship.id]+[None]+all_actions
    #see if we should add an attack diversion to our options
    #NOTE: we will avoid attacking a ship that is on the avoid spot - is this a good idea?
    for a in all_actions:
      m=move(x,a)
      if attack[m.y,m.x]:
        print('ship id {} attacking {}'.format(ship.id,a))
        action_list.insert(0,a)
        break
    #now try the options, but don't bother repeating any
    action_list=remove_dups(action_list)
    for a in action_list:
      m=move(x,a)
      if avoid[m.y,m.x]:
        print('ship id {} avoiding {}'.format(ship.id,a))
      if m not in turn.taken and not avoid[m.y,m.x]:
        action=a
        break
    ship.next_action=action
    turn.taken[m]=1
    
# Returns the commands we send to our ships and shipyards, must be last function in file
def agent(obs, config, **kwargs):
#  print("Optimus mine!")
  global size
  global start
  global prev_board
  global me
  global did_init
  #Do initialization 1 time
  start_step=time.time()
  if start is None:
    start=time.time()
  if not did_init:
    init(obs,config)
    did_init=True
  board = Board(obs, config)
  me=board.current_player
  set_turn_data(board)
  print('==== step {} sim {}'.format(board.step,board.step+1))
  print('ships {} shipyards {}'.format(turn.num_ships,turn.num_shipyards))
  print_enemy_ships(board)
  ship_converts(board)
  rng_action_seed = kwargs.get('rng_action_seed', 0)
  ship_moves(board, np.random.RandomState(int(rng_action_seed+obs.step)))
  shipyard_actions()
  print_actions(board)
  print('time this turn: {:8.3f} total elapsed {:8.3f}'.format(time.time()-start_step,time.time()-start))
  return me.next_actions
