import copy
import datetime
import heapq
import logging
import numpy as np
from operator import itemgetter, attrgetter
from queue import Queue
import sys
from kaggle_environments.envs.halite.helpers import *

MY_DATEFMT = '%M:%S'
def easy_log(s, loglevel='D'):
    pass  # print(f'{datetime.datetime.now().strftime(MY_DATEFMT)}{loglevel.upper()[0]} {s}')
easy_log('ini begin')

MAX_HALITE = 500
MAP_SIZE = 21
HALF_MAP_SIZE = MAP_SIZE // 2
ROWS = MAP_SIZE
COLS = MAP_SIZE

PLAYERS = 4

MOVE = [
        None,
        ShipAction.NORTH,
        ShipAction.EAST,
        ShipAction.SOUTH,
        ShipAction.WEST,
        ]
LEN_MOVE = len(MOVE)
I_MINE = 0
I_NORTH = 1
I_EAST = 2
I_SOUTH = 3
I_WEST = 4
I_NORTH_EAST = 5
I_SOUTH_EAST = 6
I_SOUTH_WEST = 7
I_NORTH_WEST = 8
I_CONVERT = 5
def ship_action_to_int(action, convert_aware=False):
    if action is None:
        return I_MINE
    elif isinstance(action, int):
        return action
    elif action == ShipAction.NORTH:
        return I_NORTH
    elif action == ShipAction.EAST:
        return I_EAST
    elif action == ShipAction.SOUTH:
        return I_SOUTH
    elif action == ShipAction.WEST:
        return I_WEST
    elif action == ShipAction.CONVERT:
        if convert_aware:
            return I_CONVERT
    return I_MINE

DY = [0, 1, 0, -1, 0]
DX = [0, 0, 1, 0, -1]
def position_to_ij(p):
    return ROWS - p[1] - 1, p[0]
def ij_to_position(i, j):
    return j, ROWS - i - 1
def mod_map_size_x(x):
    return (x + MAP_SIZE) % MAP_SIZE


def rotated_diff_position_impl(x0, x1):
    """x1 - x0 の値域 [-20, 20] を [-10, 10] へおさめたい"""
    d = x1 - x0
    if d < -HALF_MAP_SIZE:  # [-20, -11]
        return d + MAP_SIZE  # [1, 10]
    elif HALF_MAP_SIZE < d:  # [11, 20]
        return d - MAP_SIZE  # [-10, -1]
    return d
# memorize
def initialize_rotated_diff_position():
    t = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
    for x0 in range(MAP_SIZE):
        for x1 in range(MAP_SIZE):
            t[x0, x1] = rotated_diff_position_impl(x0, x1)
    t.flags.writeable = False
    return t
ROTATED_DIFF_POSITION = initialize_rotated_diff_position()
def rotated_diff_position(x0, x1):
    return ROTATED_DIFF_POSITION[x0, x1]
def distance_impl(x0, x1, y0, y1):
    dx = abs(rotated_diff_position(x0, x1))
    dy = abs(rotated_diff_position(y0, y1))
    return dx + dy
def initialize_distance():
    t = np.zeros((COLS, ROWS, COLS, ROWS), dtype=np.int32)
    for x0 in range(COLS):
        for y0 in range(ROWS):
            for x1 in range(COLS):
                for y1 in range(ROWS):
                    t[x0, y0, x1, y1] = distance_impl(x0=x0, y0=y0, x1=x1, y1=y1)
    t.flags.writeable = False
    return t
DISTANCE = initialize_distance()
def calculate_distance(p0, p1):
    return DISTANCE[p0[0], p0[1], p1[0], p1[1]]

def initialize_neighbor_positions(sup_d):
    """マンハッタン距離1ずつ増加していくときの全範囲x, yと距離d"""
    ts =[]
    us = []
    for d in range(sup_d):
        n_neighbors = 1 + (d * (d + 1) // 2) * 4
        t = np.zeros((n_neighbors, 3), dtype=np.int32)
        k = 0
        for dx in range(-d, d + 1):
            abs_dx = abs(dx)
            for dy in range(-d, d + 1):
                abs_dy = abs(dy)
                if d < abs_dx + abs_dy:
                    continue
                t[k, :] = dx, dy, abs_dx + abs_dy
                k += 1
        assert k == n_neighbors
        u = np.zeros((COLS, ROWS, n_neighbors, 3), dtype=np.int32)
        for x in range(COLS):
            for y in range(ROWS):
                for k, (dx, dy, d) in enumerate(t):
                    x1 = mod_map_size_x(x + dx)
                    y1 = mod_map_size_x(y + dy)
                    u[x, y, k, :] = x1, y1, d
        t.flags.writeable = False
        u.flags.writeable = False
        ts.append(t)
        us.append(u)
    return ts, us
NEIGHBOR_D_POSITIONS, NEIGHBOR_POSITIONS = initialize_neighbor_positions(sup_d=7)
def neighbor_d_positions(d):
    return NEIGHBOR_D_POSITIONS[d]
def neighbor_positions(d, p):
    return NEIGHBOR_POSITIONS[d][p[0], p[1]]



DISTANCE_TO_PREFERENCE = [0.62 + 0.02 * i for i in range(HALF_MAP_SIZE)] + [1.0] + [1.2 + 0.02 * i for i in range(HALF_MAP_SIZE)]
def distance_to_preference(d):
    return DISTANCE_TO_PREFERENCE[d + HALF_MAP_SIZE]

def preference_move_to_impl_2(x0, x1, dx_action):
    abs_dx0 = abs(rotated_diff_position(x0=x0, x1=x1))
    x_ = mod_map_size_x(x0 + dx_action)
    abs_dx_ = abs(rotated_diff_position(x0=x_, x1=x1))
    preference = 1.0
    dx2 = abs_dx_ - abs_dx0
    if dx2 < 0: # 距離縮んだ 遠いほうは abs_dx0
        preference *= distance_to_preference(abs_dx0)
    elif 0 < dx2:  # 遠ざかった 遠いほうは abs_dx_
        preference *= distance_to_preference(-abs_dx_)
    return preference


def preference_move_to_impl(x0, y0, x1, y1):
    """x0, y0: 現在位置; x1, y1: 目標位置"""
    preference = np.ones(LEN_MOVE, dtype=np.float32)
    for i_action in range(LEN_MOVE):
        preference[i_action] *= preference_move_to_impl_2(
                x0=x0, x1=x1, dx_action=DX[i_action])
        preference[i_action] *= preference_move_to_impl_2(
                x0=y0, x1=y1, dx_action=DY[i_action])
    if x0 == x1 and y0 == y1:
        preference[0] *= 1.5
    return preference

def initialize_preference_move_to():
    t = np.zeros((COLS, ROWS, LEN_MOVE), dtype=np.float32)
    for x1 in range(COLS):
        for y1 in range(ROWS):
            t[x1, y1, :] = preference_move_to_impl(x0=0, y0=0, x1=x1, y1=y1)
    t.flags.writeable = False
    return t
PREFERENCE_MOVE_TO = initialize_preference_move_to()
def preference_move_to(p0, p1):
    x1 = mod_map_size_x(p1[0] - p0[0])
    y1 = mod_map_size_x(p1[1] - p0[1])
    return PREFERENCE_MOVE_TO[x1, y1]



def calculate_next_position(position, next_action):
    """next_action で移動した後の座標を求める"""
    p = position
    if isinstance(next_action, int):
        next_action = MOVE[next_action]
    assert (next_action is None) or isinstance(next_action, ShipAction)
    if next_action == ShipAction.NORTH:
        p = Point(x=p[0], y=(p[1] + 1) % ROWS)
    elif next_action == ShipAction.EAST:
        p = Point(x=(p[0] + 1) % COLS, y=p[1])
    elif next_action == ShipAction.SOUTH:
        p = Point(x=p[0], y=(p[1] + ROWS - 1) % ROWS)
    elif next_action == ShipAction.WEST:
        p = Point(x=(p[0] + COLS - 1) % COLS, y=p[1])
    return p


def direction_to_str(next_action):
    if next_action == ShipAction.NORTH:
        return '^'
    elif next_action == ShipAction.EAST:
        return '>'
    elif next_action == ShipAction.SOUTH:
        return 'v'
    elif next_action == ShipAction.WEST:
        return '<'
    elif next_action == ShipAction.CONVERT:
        return 'c'
    return '.'


I_FLAG_NEXT_SHIP_POSITION = 0
I_FLAG_MINE_D2 = 1
I_FLAG_MINE_D3 = 2
I_FLAG_MINE_D4 = 3
I_FLAG_GO_HOME_STRAIGHT = 4
N_FLAG_TYPES = 5
I_SCORE_SURROUNDED_HALITE_GROUND = 0  # 隣接4マスのうち何マス halite 増産するか 途中更新なし shipyard 生成破壊で変わりうることに注意
I_SCORE_EMPTY_OPPONENT_D2 = 1
I_SCORE_NON_EMPTY_OPPONENT_D2 = 2
I_SCORE_OPPONENT_D2 = 3
I_SCORE_OPPONENT_D3 = 4
I_SCORE_OPPONENT_SHIPYARD_D6 = 5
I_SCORE_ALLY_SHIPYARD_D1 = 6
I_SCORE_ALLY_SHIPYARD_D4 = 7
I_SCORE_ALLY_SHIPYARD_D7 = 8
I_SCORE_OPPONENT_SHIPYARD_D2 = 9
I_SCORE_OPPONENT_SHIPYARD_D3 = 10
I_SCORE_OPPONENT_SHIPYARD_D4 = 11
I_SCORE_ALLY_D2 = 12
I_SCORE_EMPTY_ALLY_D4 = 13
I_SCORE_NON_EMPTY_ALLY_D4 = 14
I_SCORE_ALLY_D4 = 15
I_SCORE_HALITE = 16  # ただの地面 halite
I_SCORE_HALITE_D4 = 17  # 周囲Dマスの halite 和
I_SCORE_SHIPYARD_CANDIDATES_SUB = 18  # halite, 他のshipyard領域なら0
I_SCORE_SHIPYARD_CANDIDATES = 19  # 自分自身の位置と、他のshipyard領域を除く周囲のhalite
I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE = 20  # その場, 隣接 計5マスの敵haliteの最小
I_SCORE_ALLY_REACH = 21  # ally が到達できる最短ターン
I_SCORE_EMPTY_ALLY_REACH = 22  # empty ally が到達できる最短ターン
I_SCORE_NON_EMPTY_ALLY_REACH = 23
I_SCORE_OPPONENT_REACH = 24  # opponent が到達できる最短ターン
I_SCORE_EMPTY_OPPONENT_REACH = 25  # empty opponent が到達できる最短ターン
I_SCORE_NON_EMPTY_OPPONENT_REACH = 26
I_SCORE_REACH_ADVANTAGE = 27  # I_SCORE_OPPONENT_REACH - I_SCORE_ALLY_REACH
I_SCORE_DETOUR_REACH = 28  # shipyards has neighbor ZOC
I_SCORE_OPPONENT_DETOUR_REACH = 29
I_SCORE_DETOUR_ADVANTAGE = 30
I_SCORE_DANGER_ZONE = 31  # rectangle of 2 diagonal shipyards, with edge
I_SCORE_DANGER_ZONE_IN = 32  # without edge
I_SCORE_HUNT_ZONE = 33
I_SCORE_HUNT_ZONE_IN = 34
I_SCORE_FUTURE_HUNT_ZONE = 35
N_SCORE_TYPES = 36

class FlagsManager(object):
    def __init__(self):
        self.flags = np.zeros((N_FLAG_TYPES, ROWS), dtype=np.uint32)

    def parse(self, i=None, j=None, x=None, y=None):
        i_ = (ROWS - y - 1) if i is None else i
        j_ = x if j is None else j
        assert i_ is not None, f'i={i}, y={y}'
        assert j_ is not None, f'j={j}, x={x}'
        return i_, j_

    def set_all(self, i_flag_type):
        self.flags[i_flag_type, ...] = 0xFFFFFFFF

    def reset_all(self, i_flag_type):
        self.flags[i_flag_type, ...] = 0

    def set(self, i_flag_type, **kwargs):
        i, j = self.parse(**kwargs)
        self.flags[i_flag_type, i] |= (1 << j)

    def reset(self, i_flag_type, **kwargs):
        i, j = self.parse(**kwargs)
        self.flags[i_flag_type, i] &= ~(1 << j)

    def xor(self, i_flag_type, **kwargs):
        i, j = self.parse(**kwargs)
        self.flags[i_flag_type, i] ^= (1 << j)

    def get(self, i_flag_type, **kwargs):
        i, j = self.parse(**kwargs)
        return (self.flags[i_flag_type, i] >> j) & 1



def initialize_compound_interest(t_max=400, max_cell_halite=500, regen_rate=0.02):
    m = np.zeros((t_max + 1, max_cell_halite + 1), dtype=np.float32)
    m[0, :] = np.arange(max_cell_halite + 1)
    rate = 1.0 + regen_rate
    for t in range(t_max):
        m[t + 1, :] = np.minimum(m[t, :] * rate, max_cell_halite)
    # for h in range(4, max_cell_halite, 25):
        # logger.debug(f'h={h}, m[:, h]={m[:, h]}')
    return m
COMPOUND_INTEREST = initialize_compound_interest()


HUNT_IMPOSSIBLE = 9
DIAG_DIRECTIONS = ((I_NORTH, I_EAST), (I_EAST, I_SOUTH), (I_SOUTH, I_WEST), (I_WEST, I_NORTH))
def solve_hunt_impl2(m, c, result):
    if not c:
        if m == 0x1E:  # 全方向にいる
            return True, result
        return False, [HUNT_IMPOSSIBLE] * 4
    t = []
    diag, ij = c[0]
    for i in ij:
        # print(f'solve_hunt_impl2: m={m} c={c} result={result} c0={c[0]} i={i}')
        if i == 0:
            m_i = m
        else:
            m_i = m | (1 << i)
        r_i = result + [i]
        success_i, result_i = solve_hunt_impl2(m_i, c[1:], r_i)
        if success_i:
            return success_i, result_i
    return False, [HUNT_IMPOSSIBLE] * 4


def solve_hunt_impl(a):
    m0 = (a[0] << I_NORTH) | (a[1] << I_EAST) | (a[2] << I_SOUTH) | (a[3] << I_WEST)
    c = []
    for diag, b in enumerate(a[4:]):
        if b == 0:
            c.append([diag, (0,)])  # 単にない方角は0とする
        else:
            c.append([diag, DIAG_DIRECTIONS[diag]])
    return solve_hunt_impl2(m0, c, [])

def initialize_hunt_dp():
    # ラストの4が解で ne se sw nw に対して行くべき方向
    # (I_NORTH or I_EAST), (I_SOUTH or I_EAST), ...
    # 0だったら2shipsを両側に派遣しろという意味
    t = np.zeros((2, 2, 2, 2, 3, 3, 3, 3, 4), dtype=np.int32)
    for north in range(2):
        for east in range(2):
            for south in range(2):
                for west in range(2):
                    for ne in range(3):
                        for se in range(3):
                            for sw in range(3):
                                for nw in range(3):
                                    n = north
                                    e = east
                                    s = south
                                    w = west
                                    ne_ = ne
                                    se_ = se
                                    sw_ = sw
                                    nw_ = nw
                                    if ne == 2:
                                        n = 1
                                        e = 1
                                        ne_ = 0
                                    if se == 2:
                                        s = 1
                                        e = 1
                                        se_ = 0
                                    if sw == 2:
                                        s = 1
                                        w = 1
                                        sw_ = 0
                                    if nw == 2:
                                        n = 1
                                        w = 1
                                        nw_ = 0
                                    a = [n, e, s, w, ne_, se_, sw_, nw_]
                                    success, result = solve_hunt_impl(a)
                                    if success:
                                        if ne == 2:
                                            result[0] = 5
                                        if se == 2:
                                            result[1] = 5
                                        if sw == 2:
                                            result[2] = 5
                                        if nw == 2:
                                            result[3] = 5
                                    t[north, east, south, west, ne, se, sw, nw, :] = result
                                    # print(f'HUNT_DP[n{north} e{east} s{south} w{west} ne{ne} se{se} sw{sw} nw{nw}]={result}, success={success}')
    t.flags.writeable = False
    return t
HUNT_DP = initialize_hunt_dp()
DIRECTION_MAPPING = np.array([I_MINE, I_MINE, I_NORTH, -1, I_EAST, -1, I_NORTH_EAST, -1, I_SOUTH, -1, -1, -1, I_SOUTH_EAST, -1, -1, -1, I_WEST, -1, I_NORTH_WEST, -1, -1, -1, -1, -1, I_SOUTH_WEST, -1, -1, -1, -1, -1, -1, -1], dtype=np.int32)
DIRECTION_MAPPING.flags.writeable = False
assert DIRECTION_MAPPING[1 << I_MINE] == I_MINE
assert DIRECTION_MAPPING[1 << I_NORTH] == I_NORTH
assert DIRECTION_MAPPING[1 << I_EAST] == I_EAST
assert DIRECTION_MAPPING[1 << I_SOUTH] == I_SOUTH
assert DIRECTION_MAPPING[1 << I_WEST] == I_WEST
assert DIRECTION_MAPPING[(1 << I_NORTH) | (1 << I_EAST)] == I_NORTH_EAST
assert DIRECTION_MAPPING[(1 << I_EAST) | (1 << I_SOUTH)] == I_SOUTH_EAST
assert DIRECTION_MAPPING[(1 << I_SOUTH) | (1 << I_WEST)] == I_SOUTH_WEST
assert DIRECTION_MAPPING[(1 << I_WEST) | (1 << I_NORTH)] == I_NORTH_WEST

def initialize_both_move_to_halite_nash_equilibrium_impl2(halite, r_wait):
    h1 = int((halite) * 0.25)
    h2 = int((halite - h1) * 0.25)
    h3 = int((halite - h1 - h2) * 0.25)
    h23 = h2 + h3
    # -500, -500 | h1, h23
    # h23, h1    | r_wait, r_wait
    # 後続の掘りの差を出さないためにすぐSPAWNすると仮定
    # 本当は1ターン待つと 1.02倍されるので impl2 の r_wait の価値が上がるはずだが、参入できていない
    # h_next = min(MAX_HALITE, int(halite * 1.02))

    # r_p = -MAX_HALITE * p * q + h1 * p * (1 - q) + h23 * (1 - p) * q + r_wait * (1 - p) * (1 - q)
    # r_q = -MAX_HALITE * p * q + h23 * p * (1 - q) + h1 * (1 - p) * q + r_wait * (1 - p) * (1 - q)
    # dr_q / dq = -MAX_HALITE * p - h23 * p + h1 * (1 - p) - r_wait * (1 - p)
    #           = p * (-MAX_HALITE - h23 - h1 + r_wait) + (h1 - r_wait)
    #           = 0
    # となる p を求める
    p = max(0.0, (h1 - r_wait) / (MAX_HALITE + h23 + h1 - r_wait))
    r_p = -MAX_HALITE * p * p + h1 * p * (1 - p) + h23 * (1 - p) * p + r_wait * (1 - p) * (1 - p)
    return p, r_p

def initialize_both_move_to_halite_nash_equilibrium_impl(halite):
    r_ng = 0.0
    r_ok = MAX_HALITE
    p_ok = 0.0
    gamma = 0.9
    while 0.99 < abs(r_ok - r_ng):
        r_mid = 0.5 * (r_ng + r_ok)
        p, r_p = initialize_both_move_to_halite_nash_equilibrium_impl2(halite, r_mid * gamma)
        # print(f'h{halite} r_mid{r_mid:.1f} r_p{r_p:.1f} p{p:.6f}')
        if r_p < r_mid:  # ok
            r_ok = r_mid
            p_ok = p
        else:
            r_ng = r_mid
    return p_ok, r_ok
    
def initialize_both_move_to_halite_nash_equilibrium():
    t = np.zeros(MAX_HALITE + 1, dtype=np.float32)
    for halite in range(MAX_HALITE + 1):
        p, r = initialize_both_move_to_halite_nash_equilibrium_impl(halite)
        t[halite] = p
    t.flags.writeable = False
    return t
BOTH_MOVE_TO_HALITE_NASH_EQUILIBRIUM = initialize_both_move_to_halite_nash_equilibrium()
def both_move_to_halite_nash_equilibrium(halite):
    return BOTH_MOVE_TO_HALITE_NASH_EQUILIBRIUM[int(halite)]


class Project(object):
    def __init__(self, project_id, agent, ):
        self.project_id = project_id
        self.agent = agent
        self.priority = 1.0
        self.ships = {}  # key is ship_id, value is role
        self.shipyards = {}  # key is shipyard_id, value is role
        self.budget = 0  # Projectで複数stepにまたがって自由に使えるhalite

    def schedule(self):
        """Project 継続判断 (True で継続 False なら解体) と priority 設定"""
        return False

    def reserve_budget(self, halite):
        """
        free_haliteから予算をhaliteだけ追加確保する
        指定したhaliteが負ならfree_haliteへ戻るので
        Projectが使う際には, 先にfree_haliteを増やしてから
        reserve_ship なりreserve_shipyardするとよい

        次stepへ持ち越す事も可能
        すなわち確保したうえで自分が使わなければ次stepでも残っている
        次step高優先度projectにもとられない

        free_halite < haliteなら失敗し, 
        成功したらTrue 失敗したら現在budgetのままとなる
        """
        d_halite = max(-self.budget, halite)
        if self.agent.free_halite < d_halite:
            return False  # 足りない
        self.agent.free_halite -= d_halite
        self.budget += d_halite
        assert 0 <= self.budget, f'{self.budget}'
        return True

    def maintain_dead_staffs(self):
        staff_ids = []
        b = self.agent.board
        for ship_id in self.ships:
            if ship_id not in b.ships:
                staff_ids.append(ship_id)
        for shipyard_id in self.shipyards:
            if shipyard_id not in b.shipyards:
                staff_ids.append(shipyard_id)
        self.dismiss_project(staff_ids=staff_ids)

    def ships_generator(self, with_free=False, with_my_project=False):
        a = self.agent
        for ship in a.sorted_ships:
            if a.determined_ships.get(ship.id, None) is not None:
                continue
            project_id = a.belonging_project.get(ship.id, None)
            if (with_free and (project_id is None)) or (with_my_project and (project_id == self.project_id)):
                yield ship

    def run(self):
        """
        人員と予算と確保し, next_actionを決める
        必須人員他に確保されていたらまだ解体ありうる (return True で継続)
        前step確保していたstaff_idsはscheduleでリリースしていない限り残る
        """
        # if -1e-6 < self.priority:
            # self.agent.log(s=f'prj={self.project_id} run prio={self.priority}')
        return False

    def discard(self):
        """
        project終了のため, 確保している人員を開放する
        予算を戻してもらう
        """
        a = self.agent
        a.log(step=a.board.step, id_=None, s=f'p{a.player_id} discard project_id={self.project_id}')
        assert 0 <= self.budget, self.budget
        self.reserve_budget(-self.budget)
        self.dismiss_project(
                staff_ids=list(self.ships.keys()) + list(self.shipyards.keys()))
        assert self.project_id not in a.belonging_project.values(), f'project_id={self.project_id} belonging_project={a.belonging_project}'
        if self.project_id in a.projects:
            del a.projects[self.project_id]

    def join_project(self, *args, staff_ids, **kwargs):
        a = self.agent
        for staff_id in staff_ids:
            project_id = a.belonging_project.get(staff_id, None)
            if project_id and project_id != self.project_id:
                project = a.projects.get(project_id, None)
                if project:
                    project.dismiss_project(staff_ids=[staff_id])
        return self.agent.join_project(*args, staff_ids=staff_ids, project_id=self.project_id, **kwargs)

    def dismiss_project(self, *args, **kwargs):
        return self.agent.dismiss_project(*args, project_id=self.project_id, **kwargs)


class DefenseShipyardProject(Project):
    """
    priorityはhuntに次ぐ
    Shipyardが壊されないように守る
    敵shipより手前に1shipはいるようにする
    張りこまれていたら, 
        相殺する
        deposit希望のshipを突っ込ませる運ゲーをマネジメント
    """

    def __init__(self, shipyard_id, *args, **kwargs):
        project_id = f'defense_yd{shipyard_id}'
        super().__init__(*args, project_id=project_id, **kwargs)
        self.shipyard_id = shipyard_id
        self.cancel_threshold = 4  # 隣で停止されたときに相殺決断するまで待つ猶予
        self.spawn_step_threshold = 250
        self.spawn_step_threshold_final = 390
        self.o = None
        self.empty_o_min_d = []
        self.last_confirmed_step = -1
        self.last_swapped_step = -1
        self.info = {}

    def shipyard_defender_strategy(self, ship):
        a = self.agent
        shipyard = a.board.shipyards[self.shipyard_id]
        o_min_d = self.info['opponent_distance']
        e_min_d = self.info.get('empty_ally_ship_distance', 99999)
        n_min_d = self.info.get('non_empty_ally_ship_distance', 99999)
        d = calculate_distance(ship.position, shipyard.position)
        free_steps = o_min_d - d
        original_free_steps = free_steps
        if 0 < ship.halite:
            free_steps -= 1  # 先回りして deposit する必要がある
        remaining_steps = max(0, 398 - a.board.step - d)
        free_steps = min(free_steps, remaining_steps)
        original_free_steps = min(original_free_steps, remaining_steps)
        a.log(step=a.board.step, id_=ship.id, s=f'yd{shipyard.id} shipyard_defender h{ship.halite} o_min_d={o_min_d} d={d} free_steps={free_steps}')
        if free_steps < 0:  # こいつ任命できないはずなんだが
            a.log(loglevel='WARNING', id_=ship.id, s=f'defender too far. project_id={self.project_id} h={ship.halite} o_min_d={o_min_d} info={self.info}')
            return a.ship_strategy(ship)
        # 基本は帰還
        priority = 10000 + a.calculate_collision_priority(ship)
        q_empty, forced = a.calculate_moving_ship_preference(
                ship=ship, position=shipyard.position, mode='escape', mine_threshold=None)
        if o_min_d == 1:  # 身を盾にして守る
            for k_action, cell_k in enumerate(a.neighbor_cells(a.board.cells[ship.position])):
                if cell_k.shipyard and cell_k.shipyard.player_id == a.player_id:
                    q_empty[k_action] = max(10.0, q_empty[k_action])
            return a.reserve_ship_by_q(ship, q=q_empty, priority=priority)
        # キリの良い数字になれるならさっさと帰る
        player_halite = a.board.current_player.halite
        short_halite = MAX_HALITE - (player_halite % MAX_HALITE)
        if free_steps == 0 or (player_halite < 1000 and short_halite <= ship.halite):  # 最短距離で帰る
            a.log(id_=ship.id, s=f'{ship.position} prj={self.project_id} free_steps{free_steps} q_empty{q_empty} back short_halite{short_halite}')
            return a.reserve_ship_by_q(ship, q=q_empty, priority=priority)
        # 今いる場所を掘れるかどうか検討しよう
        cell = a.board.cells[ship.position]
        mine_project = a.projects.get(f'mine_{ship.position[0]}_{ship.position[1]}', None)
        if mine_project:
            mine_threshold = mine_project.mining_halite_threshold
        else:
            mine_threshold = 160.0
        if mine_threshold:
            mode = 'mine'
        else:
            mode = 'escape'
        if mine_threshold < cell.halite:
            q_mine, forced = a.calculate_moving_ship_preference(
                    ship=ship, position=ship.position, mode=mode, mine_threshold=mine_threshold)
        else:
            q_mine = np.copy(q_empty)
        if mine_threshold <= cell.halite:  # 危険な時はq_mine使わないので停止危険チェックはしなくてOK
            q_mine[0] = max(4.0, q_mine[0])
        if free_steps == 1:  # 最短 / 1回停止可能
            if mine_threshold <= cell.halite:
                if 0 < ship.halite:  # 掘っても free_steps は減らない
                    a.log(id_=ship.id, s=f'{ship.position} prj={self.project_id} free_steps{free_steps} q_mine{q_mine} mine_thre{mine_threshold}')
                    return a.reserve_ship_by_q(ship, q=q_mine, priority=priority)
                elif d == 1 and mine_threshold <= cell.halite:
                    # もし唯一の敵をブロックする位置にいるのなら, halite回収できる
                    for cell_i in a.neighbor_cells(a.board.cells[shipyard.position]):
                        if cell_i.position == ship.position:
                            continue
                        i_, j_ = position_to_ij(cell_i.position)
                        if a.scores[I_SCORE_OPPONENT_REACH, i_, j_] < 1.5:
                            # blockできていないので大人しく帰る
                            return a.reserve_ship_by_q(ship, q=q_empty, priority=priority)
                    a.log(id_=ship.id, s=f'{ship.position} prj={self.project_id} blocking. q_mine={q_mine}')
                    return a.reserve_ship_by_q(ship, q=q_mine, priority=priority)
            a.log(id_=ship.id, s=f'{ship.position} prj={self.project_id} free_steps{free_steps} q_empty{q_empty} back')
            return a.reserve_ship_by_q(ship, q=q_empty, priority=priority)
        ship_cell = a.board.cells[ship.position]
        if ((free_steps < 2) or
                (0 < ship.halite) or 
                (free_steps == 3 and ship.halite == 0 and 0 < d and 80.0 < ship_cell.halite)):
            # 離れる方向へ移動しても掘れないので自重する
            has_non_empty_ship = False
            for k_action, cell_k in enumerate(a.neighbor_cells(a.board.cells[ship.position])):
                d_k = calculate_distance(cell_k.position, shipyard.position)
                if d < d_k:
                    q_mine[k_action] *= 0.3
                if d == 0:
                    if cell_k.ship and cell_k.ship.player_id == a.player_id and 0 < cell_k.ship.halite:
                        has_non_empty_ship = True
            if has_non_empty_ship:  # 帰り道を邪魔しない
                q_mine[I_NORTH:] *= 10.0
            a.log(id_=ship.id, s=f'{ship.position} prj={self.project_id} free_steps{free_steps} q_mine{q_mine} back with mine has_non_empty_ship={has_non_empty_ship} d{d} original_free_steps{original_free_steps} mine_thre{mine_threshold} mine_prj{mine_project}')
            return a.reserve_ship_by_q(ship, q=q_mine, priority=priority)
        # 離れて1回回収するための条件を満たしたので、周囲を探索
        positions = neighbor_positions(d=(2 if d == 0 else 1), p=ship.position)
        max_position = shipyard.position
        max_halite_diff = 0
        for x1, y1, d1 in positions:
            p1 = Point(x=x1, y=y1)
            cell1 = a.board.cells[p1]
            if cell1.halite < 1e-6:
                continue
            mine_project1 = a.projects.get(f'mine_{x1}_{y1}', None)
            if mine_project1:
                if mine_project1.ships:
                    continue  # 競合を避ける
                mine_threshold1 = mine_project1.halite_threshold
            else:
                mine_threshold1 = 40.0
            halite_diff = cell1.halite - mine_threshold1
            if max_halite_diff < halite_diff:
                max_position = p1
                max_halite_diff = halite_diff
        q_explore, forced = a.calculate_moving_ship_preference(
                ship=ship, position=max_position, mode='mine', mine_threshold=mine_threshold)
        if d == 0 and 2 < o_min_d:  # 脅威がない場合のshipyard上での停止は避ける
            q_explore[0] *= 0.4
        a.log(id_=ship.id, s=f'{ship.position} prj={self.project_id} free_steps{free_steps} q_explore{q_explore} max{max_position} h_diff{max_halite_diff}')
        return a.reserve_ship_by_q(ship, q=q_explore, priority=priority)
            
    def defense_shipyard_strategy_ships_generator(self):
        """
        MineProjectは制御を奪ってよい
        EscortProjectに自分自身が属しているshipは制御を奪ってよい
        """
        a = self.agent
        for ship in a.sorted_ships:
            determined = a.determined_ships.get(ship.id, None)
            if determined is not None:
                continue
            project_id = a.belonging_project.get(ship.id, None)
            if project_id is None:
                yield ship
            elif project_id == self.project_id:
                yield ship
            elif project_id[:6] == 'escort':
                escort_project = a.projects.get(project_id, None)
                if not escort_project:
                    yield ship
                elif escort_project.has_myself():
                    yield ship
            elif project_id[:4] == 'mine':
                yield ship

    def find_confirmed_shipyard(self, shipyard):
        """近くにally shipyardがあるなら、壊されてもいいや"""
        a = self.agent
        for x_k, y_k, d_k in neighbor_positions(d=2, p=shipyard.position):
            if d_k == 0:
                continue  # 自分自身
            cell_k = a.board.cells[x_k, y_k]
            shipyard_k = cell_k.shipyard
            if shipyard_k is None:
                continue
            if shipyard_k.player_id != a.player_id:
                continue
            project_id = f'defense_yd{shipyard_k.id}'
            project = a.projects.get(project_id, None)
            if project is None:
                continue
            last_confirmed_step = project.last_confirmed_step
            if a.board.step <= last_confirmed_step:
                return shipyard_k
        return None


    def schedule(self):
        super().schedule()
        self.info = {}
        self.maintain_dead_staffs()
        a = self.agent
        shipyard = a.board.shipyards.get(self.shipyard_id, None)
        if shipyard is None or 397 <= a.board.step:
            return False
        len_ships = len(a.board.current_player.ships)
        if len_ships == 1 and a.board.current_player.ships[0].halite == 0 and a.board.current_player.halite < MAX_HALITE:
            a.log(loglevel='WARNING', id_=shipyard.id, s=f'prj={self.project_id} last_escape_mode')
            return False  # 逃げ専
        self.position = shipyard.position
        self.i, self.j = position_to_ij(self.position)
        self.shipyard_cell = a.board.cells[self.position]

        confirmed_shipyard = self.find_confirmed_shipyard(shipyard)
        if confirmed_shipyard:
            a.log(id_=shipyard.id, s=f'confirmed_shipyard{confirmed_shipyard.id} found')
            return False

        o_min_d = 99999  # opponent
        empty_o_min_d = 99999
        self.o = None
        for shipyard_i in a.board.shipyards.values():
            if shipyard_i.player_id == a.player_id:
                continue
            # spawn に 1 step かかるので +1
            d = 1 + calculate_distance(self.position, shipyard_i.position)
            if d < o_min_d:
                self.o = shipyard_i
                o_min_d = d
                empty_o_min_d = d
        for ship in a.board.ships.values():
            if ship.player_id == a.player_id:
                continue
            d = calculate_distance(self.position, ship.position)
            if d < o_min_d:
                self.o = ship
                o_min_d = d
            if ship.halite == 0 and d < empty_o_min_d:
                empty_o_min_d = d
        self.info['shipyard_id'] = shipyard.id
        self.info['opponent_id'] = None if self.o is None else self.o.id
        self.info['opponent_distance'] = o_min_d
        self.empty_o_min_d.append(empty_o_min_d)
        if self.o is None:
            self.priority = -100.0
        elif a.board.step < 9:
            self.priority = -1.0
        else:
            self.priority = 200000. - o_min_d * 10000. + a.scores[I_SCORE_HALITE_D4, self.i, self.j]
        if self.priority < 0.0:
            self.dismiss_project(staff_ids=list(self.ships.keys()))
        self.last_confirmed_step = a.board.step
        return True

    def should_cancel(self, *args, **kwargs):
        result, condition = self.should_cancel_impl(*args, **kwargs)
        self.agent.log(s=f'prj={self.project_id} should_cancel={result} {condition}')

    def should_cancel_impl(self, e, o, cell_o, e_min_d, o_min_d):
        a = self.agent
        condition = ''
        if 2 < o_min_d:
            return False, '2<o_min_d'
        if e_min_d != 0:
            return False, 'e_min_d!=0'
        if len(self.empty_o_min_d) < 5:
            return False, f'len(empty_o_min_d)={len(self.empty_o_min_d)}'  # 最初期
        n_min_d = int(1e-6 + a.scores[I_SCORE_NON_EMPTY_ALLY_REACH, self.i, self.j])
        condition += f' n_min_d{n_min_d}'
        if 2 < n_min_d:
            condition += ' 2<n_min_d'
            return False, condition  # depositしたいshipがいないなら放置しよう
        condition0 = (3 == np.sum((np.array(self.empty_o_min_d[-3:]) <= 1).astype(np.int32)))
        condition1 = (4 <= np.sum((np.array(self.empty_o_min_d[-5:]) <= 1).astype(np.int32)))
        condition += f' cond0={condition0} cond1={condition1} empty_o_min_d{self.empty_o_min_d[-5:]}'
        if not (condition0 or condition1):
            return False, condition
        previous_o = a.previous_board.ships.get(o.id, None)
        # 往復にせよ停止にせよ、1ターン前の位置へ突っ込めばよい
        for k_action, cell_k in enumerate(a.neighbor_cells(a.previous_board.cells[self.position])):
            if k_action == 0:
                continue
            ship_k = cell_k.ship
            if ship_k is None or 0 < ship_k.halite or ship_k.player_id == a.player_id:
                continue
            condition += f' k{k_action}_found'
            return ship_k.position, condition
        return False, condition

    def search_empty_ally(self):
        """shipyardの隣にいるempty_shipを探す"""
        ship_candidates = list(self.defense_shipyard_strategy_ships_generator())
        for i_action, cell_i in enumerate(self.agent.neighbor_cells(self.shipyard_cell)):
            if i_action == 0:
                continue
            ship_i = cell_i.ship
            if ship_i is None:
                continue
            if 0 < ship_i.halite:
                continue
            if ship_i in ship_candidates:
                return ship_i
        return None

    def run_cancel(self, e, target_position):
        """return done, safe, spawned"""
        a = self.agent
        defender = None
        ally_ship = self.search_empty_ally()
        shipyard = a.board.shipyards[self.shipyard_id]
        if ally_ship:
            self.join_project(staff_ids=[ally_ship.id], role='defender_ally', forced=True)
            a.moving_ship_strategy(ship=ally_ship, position=self.position, mode='cancel_without_shipyard', mine_threshold=None)
            a.moving_ship_strategy(ship=e, position=target_position, mode='cancel_without_shipyard', mine_threshold=None)
            a.log(id_=e.id, s=f'prj={self.project_id} cancel to {target_position}, with ally s{ally_ship.id}')
            return True, True, False
        elif MAX_HALITE <= a.free_halite + self.budget:
            # SPAWN可能
            if a.flags.get(I_FLAG_NEXT_SHIP_POSITION, i=self.i, j=self.j):
                a.log(loglevel='warning', s=f'prj={self.project_id} cannot spawn because someone will return type A')
                a.reserve_shipyard(shipyard, None)  # 誰かが帰還するのでSPAWNしない
                a.moving_ship_strategy(ship=e, position=target_position, mode='cancel_without_shipyard', mine_threshold=None)
                return True, False, False
            elif self.spawn_step_threshold <= a.board.step and 1 < len(a.board.current_player.shipyards):
                # もう SPAWN するのは勿体ないので, 抑止力で祈る
                self.reserve_budget(MAX_HALITE - self.budget)
                a.log(id_=e.id, s=f'prj={self.project_id} cancel to {target_position}, with fake SPAWN')
                a.reserve_shipyard(shipyard, None)
                a.moving_ship_strategy(ship=e, position=target_position, mode='cancel_without_shipyard', mine_threshold=None)
                return True, False, False
            else:
                self.reserve_budget(-self.budget)
                a.log(id_=e.id, s=f'prj={self.project_id} cancel to {target_position}, with SPAWN')
                a.reserve_shipyard(shipyard, ShipyardAction.SPAWN)
                a.moving_ship_strategy(ship=e, position=target_position, mode='cancel_without_shipyard', mine_threshold=None)
                return True, True, True
        return False, True, False

    def run(self):
        """防衛担当者を決める"""
        super().run()
        a = self.agent
        shipyard = a.board.shipyards[self.shipyard_id]
        if self.shipyard_id not in self.shipyards:
            self.join_project(staff_ids=[self.shipyard_id], role='defended')

        if self.priority < 0.0:  # 何もしなくてよい. 単にプロジェクト継続
            self.info['safe'] = True
            self.info['empty_ally_ship_id'] = None
            self.info['empty_ally_ship_distance'] = None
            self.info['non_empty_ally_ship_id'] = None
            self.info['non_empty_ally_ship_distance'] = None
            return True

        if a.board.step < self.spawn_step_threshold and 0 < self.budget and MAX_HALITE <= a.free_halite + self.budget:
            # Expeditionから引き継いだbudgetをさっさと使う
            self.reserve_budget(-self.budget)
            a.reserve_shipyard(shipyard, ShipyardAction.SPAWN)
            self.dismiss_project(staff_ids=list(self.ships.keys()))
            return True

        cell = a.board.cells[self.position]
        o_min_d = self.info['opponent_distance']
        e_min_d = 99999  # empty
        e = None
        n_min_d = 99999  # non empty ally
        n = None
        for ship in self.defense_shipyard_strategy_ships_generator():
            d = calculate_distance(shipyard.position, ship.position)
            if ship.halite == 0:
                if (d < e_min_d):
                    e = ship
                    e_min_d = d
            elif d < o_min_d:
                if ((d < n_min_d)
                        or ((d == n_min_d) and (n.halite < ship.halite))):
                    n = ship
                    n_min_d = d
        if e:
            self.info['empty_ally_ship_id'] = e.id
            self.info['empty_ally_ship_distance'] = e_min_d
        if n:
            self.info['non_empty_ally_ship_id'] = n.id
            self.info['non_empty_ally_ship_distance'] = n_min_d
        safe = False
        spawned = False
        defender = None  # デフォルト挙動させたい場合に指定する
        min_d = 99999
        if n_min_d < o_min_d:  # deposit 間に合う
            self.dismiss_project(staff_ids=list(self.ships.keys()))
            self.join_project(staff_ids=[n.id], role='defender')
            defender = n
            safe = True
        elif e_min_d <= o_min_d:
            self.dismiss_project(staff_ids=list(self.ships.keys()))
            self.join_project(staff_ids=[e.id], role='defender')
            opponent_id = self.info['opponent_id']
            previous_o = a.previous_board.ships.get(opponent_id, None)
            o = a.board.ships.get(opponent_id, None)
            if o is None:
                o = a.board.shipyards.get(opponent_id, None)
            # a.log(f'prj={self.project_id} p{a.player_id} self.shipyard_id={self.shipyard_id} opponent_id={opponent_id} previous_o{previous_o} o{o} o_min_d={o_min_d}')
            cell_o = a.board.cells[o.position]
            target_position = self.should_cancel(e=e, o=o, cell_o=cell_o, e_min_d=e_min_d, o_min_d=o_min_d)
            if target_position:
                # RestrainShipyardProject 対策で相殺しに行く
                done, safe, spawned = self.run_cancel(e=e, target_position=target_position)
            else:
                defender = e
                safe = True
                spawned = False
        elif MAX_HALITE <= a.free_halite + self.budget:
            someone_arrived = a.flags.get(I_FLAG_NEXT_SHIP_POSITION, i=self.i, j=self.j)
            if someone_arrived:
                a.reserve_shipyard(shipyard, None)  # 誰かが帰還する
                safe = False
                a.log(loglevel='warning', s=f'id={self.project_id} cannot spawn because someone will return type B')
            elif ((self.spawn_step_threshold_final <= a.board.step)
                    or
                    (self.spawn_step_threshold <= a.board.step
                    and a.scores[I_SCORE_HALITE_D4, self.i, self.j] < 1000.
                    and 1 < len(a.board.current_player.shipyards))):
                # もう SPAWN するのは勿体ないので, 抑止力で祈る
                self.reserve_budget(MAX_HALITE - self.budget)
                a.reserve_shipyard(shipyard, None)
                safe = False
                a.log(f'prj={self.project_id} stop spawning because the shipyard is not tasty')
            else:
                self.reserve_budget(-self.budget)
                a.reserve_shipyard(shipyard, ShipyardAction.SPAWN)
                spawned = True
                safe = True
                a.log(f'prj={self.project_id} spawning to protect')
        someone_arrived = a.flags.get(I_FLAG_NEXT_SHIP_POSITION, i=self.i, j=self.j)
        if ((not spawned)
                and (not someone_arrived)
                and (MAX_HALITE <= a.free_halite + self.budget)):
            # 不要不急のspawn
            can_spawn = a.can_spawn(shipyard, budget=self.budget)
            if can_spawn:
                a.log(f'prj={self.project_id} unneeded spawn freeh{a.free_halite} budget{self.budget}')
                self.reserve_budget(-self.budget)
                a.reserve_shipyard(shipyard, ShipyardAction.SPAWN)
                spawned = True
        if defender:  # デフォルト挙動
            defender = self.swap_project(defender)
            self.shipyard_defender_strategy(defender)
        self.info['safe'] = safe
        a.log(s=f'id={self.project_id} run priority={self.priority} shipyard_info={self.info} ships={self.ships} spawned={spawned}')
        return True

    def swap_project(self, ship):
        """shipyard上待機が原因で味方をブロックしないように役割スワップを検討する"""
        a = self.agent
        shipyard = a.board.shipyards[self.shipyard_id]
        if shipyard.position != ship.position:
            return ship # 対象 ship が shipyard 上で守っているとき限定
        if a.board.step - 1 <= self.last_swapped_step:
            # 連続してswapするとデッドロックしうるので回避
            return ship
        to_sort = []
        for k_action, cell_k in enumerate(a.neighbor_cells(a.board.cells[ship.position])):
            if k_action == 0:
                continue
            ship_k = cell_k.ship
            if not ship_k:
                continue
            if ship_k.player_id != a.player_id:
                continue
            if 0 < ship_k.halite:
                continue
            project_id_k = a.belonging_project.get(ship_k.id, None)
            if project_id_k is None:
                continue
            project_k = a.projects.get(project_id_k, None)
            if project_k is None:
                continue
            target_position = None
            if project_id_k[:6] == 'escort':
                if project_k.target_ship_id in [ship.id, ship_k.id]:
                    continue  # 自分自身のescortは放置
                target_ship = a.board.ships.get(project_k.target_ship_id, None)
                if target_ship is None:
                    continue
                target_position = target_ship.position
                priority = 10.0
            elif project_id_k[:4] == 'mine':
                target_position = project_k.position
                priority = 100.0
            else:
                continue
            d_k = calculate_distance(ship_k.position, target_position)
            d = calculate_distance(ship.position, target_position)
            if d < d_k:
                priority += d_k
                to_sort.append((priority, ship_k.id, ship_k, project_k, target_position))
        if not to_sort:
            return ship
        # join, dismiss 処理をする
        priority, swapped_ship_id, swapped_ship, project, target_position = sorted(to_sort, reverse=True)[0]
        ship_role = self.ships[ship.id]
        swapped_ship_role = project.ships[swapped_ship_id]
        self.dismiss_project(staff_ids=[ship.id])
        project.dismiss_project(staff_ids=[swapped_ship_id])
        self.join_project(staff_ids=[swapped_ship_id], role=ship_role)
        project.join_project(staff_ids=[ship.id], role=swapped_ship_role)
        if project.project_id[:4] == 'mine':
            # 取り巻きも Project移動しなければならない
            old_escort_project_id = f'escort{swapped_ship.id}'
            old_escort_project = a.projects.get(old_escort_project_id, None)
            ship_ids = []
            if old_escort_project:
                ship_ids = list(old_escort_project.ships.keys())
                old_escort_project.dismiss_project(staff_ids=ship_ids)
            new_escort_project_id = f'escort{ship.id}'
            new_escort_project = a.projects.get(new_escort_project_id, None)
            if new_escort_project:
                new_escort_project.join_project(staff_ids=ship_ids, role='defender')
            a.log(id_=ship.id, s=f'swapped. old_escort_prj={old_escort_project_id}({len(old_escort_project.ships)} new_escort_prj={new_escort_project_id}({len(new_escort_project.ships)})')
            a.log(id_=swapped_ship_id, s=f'swapped. old_escort_prj={old_escort_project_id}({len(old_escort_project.ships)} new_escort_prj={new_escort_project_id}({len(new_escort_project.ships)})')
        a.log(id_=ship.id, s=f'swapped. s{ship.id}(prj={a.belonging_project.get(ship.id, None)}) s{swapped_ship_id}({a.belonging_project.get(swapped_ship_id, None)}) dk{d_k} target{target_position}')
        a.log(id_=swapped_ship_id, s=f'swapped. s{ship.id}(prj={a.belonging_project.get(ship.id, None)}) s{swapped_ship_id}({a.belonging_project.get(swapped_ship_id, None)}) dk{d_k} target{target_position}')
        self.last_swapped_step = a.board.step
        return swapped_ship

class RestrainShipyardProject(Project):
    """
    NOTE: best agent does not use this project
    敵のshipyardの隣に1ship常駐し、敵の動向を見る
    deposit防ぎつつ、相殺もされないのがベスト
    無視してshipyard空けたり
    depositしてくるようならタイミング見計らって特攻する
    相殺してくるようなら互いに損なので適度に逃げる
    メイン目的はHuntProjectの逃げ道防ぐ布石
    """
    def __init__(self, shipyard_id, *args, **kwargs):
        project_id = f'restrain_yd{shipyard_id}'
        super().__init__(*args, project_id=project_id, **kwargs)
        self.shipyard_id = shipyard_id
        self.shipyard = None
        self.shipyards[self.shipyard_id] = 'target_shipyard'
        self.zero_halite_positions = []
        self.stop_canceled_count = [0, 0]
        self.move_canceled_count = [0, 0]
        self.d_max_camp_positon_from_ally_shipyard = 7

    def feedback(self, shipyard):
        """味方が前いた位置に相殺しに来ているかをチェック"""
        a = self.agent
        if a.previous_board is None:
            return
        for ship_id, role in self.ships.items():
            ship0 = a.previous_board.ships.get(ship_id, None)
            if ship0 is None:
                continue
            ship1 = a.board.ships.get(ship_id, None)
            vibrate_canceled = 0
            stop_canceled = 0
            if ship1 is None:
                vibrate_canceled = 1
                stop_canceled = 1
            cell1 =  a.board.cells[ship0.position]
            if cell1.ship and cell1.ship.player_id == shipyard.player_id:
                stop_canceled = 1
            if role == 'stop':
                self.stop_canceled_count[0] += stop_canceled
                self.stop_canceled_count[1] += 1
            elif role == 'vibrate':
                self.vibrate_canceled_count[0] += vibrate_canceled
                self.vibrate_canceled_count[1] += 1
        # 全体を見る
        if 2 <= a.opponent_history[self.shipyard.player_id]['cancel_against_shipyard_attack'][0]:
            self.stop_canceled_count[0] = max(1, self.stop_canceled_count[0])
            self.stop_canceled_count[1] = max(1, self.stop_canceled_count[1])

    def can_stop(self):
        if 0 < self.stop_canceled_count[0]:
            return False
        if self.d_max_camp_positon_from_ally_shipyard < self.d_ally_shipyard:
            return False
        return True

    def can_vibrate(self):
        if 0 < self.vibrate_canceled_count[0]:
            return False
        if self.d_max_camp_positon_from_ally_shipyard < self.d_ally_shipyard:
            return False
        return True

    def schedule(self):
        super().schedule()
        a = self.agent
        self.shipyard = a.board.shipyards.get(self.shipyard_id, None)
        if self.shipyard is None:
            return False
        self.feedback(self.shipyard)

        # HuntProjectで連携したいShipyardの方向を探す
        self.d_ally_shipyard = a.ally_shipyard_distances[self.shipyard_id]['min']
        self.ally_shipyard = None
        for ally_shipyard_id_k, d_ally_shipyard_k in a.ally_shipyard_distances[self.shipyard_id].items():
            if ally_shipyard_id_k == 'min':
                continue
            if d_ally_shipyard_k == self.d_ally_shipyard:
                self.ally_shipyard = a.board.shipyards.get(ally_shipyard_id_k, None)
                if self.ally_shipyard:
                    break
        if self.ally_shipyard is None:
            self.priority = -1.0
            return True

        # ally_shipyardから掘った opponent_ship が逃げていく方向
        self.opponent_return_directions = np.zeros(LEN_MOVE, dtype=np.bool)
        self.dx = rotated_diff_position(self.ally_shipyard.position[0], self.shipyard.position[0])
        self.dy = rotated_diff_position(self.ally_shipyard.position[1], self.shipyard.position[1])
        if self.dx <= -2:
            self.opponent_return_directions[I_WEST] = True
        if 2 <= self.dx:
            self.opponent_return_directions[I_EAST] = True
        if self.dy <= -2:
            self.opponent_return_directions[I_SOUTH] = True
        if 2 <= self.dy:
            self.opponent_return_directions[I_NORTH] = True


        a.log(s=f'prj={self.project_id} ships={list(self.ships.keys())} d_ally_shipyard={self.d_ally_shipyard} op_return_dir{self.opponent_return_directions}')
        self.join_project(staff_ids=[self.shipyard_id], role='target_shipyard')
        # self.dismiss_project(staff_ids=list(self.ships.keys()))

        self.zero_halite_positions = []
        self.opponent_working = False
        self.already_in_position = False
        has_neighbor_opponent_shipyard = False
        for x, y, d in neighbor_positions(d=2, p=self.shipyard.position):
            if d == 0:
                continue
            cell_i = a.board.cells[x, y]
            shipyard_i = cell_i.shipyard
            if shipyard_i and shipyard_i.player_id != self.shipyard.player_id:
                # shipyardが密集しているのでスルー
                has_neighbor_opponent_shipyard = True
            ship_i = cell_i.ship
            if ship_i and ship_i.player_id not in [a.player_id, self.shipyard.player_id]:
                # 他の敵が仕事している
                self.opponent_working = True
            if d == 1 and cell_i.halite < 1e-6:
                self.zero_halite_positions.append(Point(x=x, y=y))
                if ship_i and ship_i.player_id == a.player_id:
                    self.already_in_position = True
        if 10 < self.d_ally_shipyard:
            self.priority = -1.0  # 初期位置より遠いと牽制する気起きん
        elif has_neighbor_opponent_shipyard:
            self.priority = -1.0  # 過密地帯なので勝手に牽制し合っている
        # elif (not self.already_in_position) and opponent_working:
            # self.priority = -1.0
        else:
            self.priority = len(self.zero_halite_positions) + 1.
        if self.priority < 0.0:
            self.dismiss_project(staff_ids=list(self.ships.keys()))
        a.log(id_=self.shipyard_id, s=f'prj={self.project_id} prio{self.priority} has_opyd={has_neighbor_opponent_shipyard} already_in_pos={self.already_in_position} op_working={self.opponent_working}') 
        return True

    def restrain_ships_generator(self):
        a = self.agent
        for ship in a.empty_ships:
            if a.determined_ships.get(ship.id, None) is not None:
                continue
            project_id = a.belonging_project.get(ship.id, None)
            if project_id is None:
                yield ship
            elif project_id == self.project_id:
                yield ship

    def calculate_target_position(self):
        a = self.agent
        p0 = self.ally_shipyard.position
        p1 = self.shipyard.position
        positions = [p0]
        p = p0
        for j in range(22):
            q = preference_move_to(p, p1)
            action = np.argmax(q)
            p = Point(x=mod_map_size_x(p[0] + DX[action]), y=mod_map_size_x(p[1] + DY[action]))
            positions.append(p)
            # a.log(s=f'prj={self.project_id} p{p} q{q} positions{positions}')
            if p == p1:
                break
        i = min(len(positions) - 2, self.d_max_camp_positon_from_ally_shipyard)
        a.log(s=f'prj={self.project_id} positions{positions} i{i}')
        return positions[i]

    def run(self):
        super().run()
        if self.priority < 0.0:
            return True
        a = self.agent

        # 自ship数が少ないときはやらない
        len_ships = len(a.board.current_player.ships)
        len_restrain_ships = 0
        len_staffs = len(self.ships)
        restrain_ships = []
        for staff_id, project_id in a.belonging_project.items():
            if project_id is None:
                continue
            if staff_id not in a.board.current_player.ships:
                continue
            if project_id[:8] == 'restrain':
                len_restrain_ships += 1
                restrain_ships.append((ship_id, project_id))
        len_restrain_ships_allowed = max(0, (len_ships - 15) // 3)
        len_diff = len_restrain_ships_allowed - len_restrain_ships
        if len_diff <= 0:
            max_len_ships = len_staffs
        else:
            max_len_ships = max(0, min(1, len_diff))
        if max_len_ships <= 0:
            return True

        target_position = self.calculate_target_position()
        a.log(s=f'prj={self.project_id} {target_position} len_ships{len_ships} len_restrain_ships{len_restrain_ships} len_staffs{len_staffs} len_restrain_ships_allowed{len_restrain_ships_allowed} len_diff{len_diff} max_len_ships{max_len_ships} restrain_ships{restrain_ships}')

        to_sort = []
        opponent_ships = []
        for opponent_ship in a.board.players[self.shipyard.player_id].ships:
            if opponent_ship.halite == 0:
                continue
            dx_opponent = rotated_diff_position(opponent_ship.position[0], self.shipyard.position[0])
            dy_opponent = rotated_diff_position(opponent_ship.position[1], self.shipyard.position[1])
            directions = np.zeros(LEN_MOVE, dtype=np.bool)
            if 0 < dy_opponent and self.opponent_return_directions[I_NORTH]:
                directions[I_NORTH] = True
            if 0 < dx_opponent and self.opponent_return_directions[I_EAST]:
                directions[I_EAST] = True
            if dy_opponent < 0 and self.opponent_return_directions[I_SOUTH]:
                directions[I_SOUTH] = True
            if dx_opponent < 0 and self.opponent_return_directions[I_WEST]:
                directions[I_WEST] = True
            if np.any(directions):
                opponent_ships.append((opponent_ship, dx_opponent, dy_opponent, directions))
            a.log(s=f'prj={self.project_id}{self.shipyard.position} test_op{opponent_ship.id}{opponent_ship.position} dxo{dx_opponent} dyo{dy_opponent} dirs{directions}')
        for ship in self.restrain_ships_generator():
            # ブロックできそうなshipはいる?
            d = calculate_distance(ship.position, target_position)
            # 正なら西にship, 東にshipyard
            dx = rotated_diff_position(ship.position[0], self.shipyard.position[0])
            # 正なら南にship, 北にshipyard
            dy = rotated_diff_position(ship.position[1], self.shipyard.position[1])
            score = 0
            target_opponent_ship = None
            next_direction = None
            can_stop = a.board.cells[ship.position].halite < 3.99
            condition = ''
            for opponent_ship, dx_opponent, dy_opponent, directions in opponent_ships:
                d_min_shipyard = a.opponent_shipyard_distances[opponent_ship.id]['min']
                d_shipyard = a.opponent_shipyard_distances[opponent_ship.id][self.shipyard_id]
                if d_min_shipyard < d_shipyard:
                    continue
                # 正なら西にally 東にop
                dx_ships = dx - dx_opponent
                # 正なら南にally 北にop
                dy_ships = dy - dy_opponent
                condition = f'dxs{dx_ships} dys{dy_ships}'
                abs_dx_ships = abs(dx_ships)
                abs_dy_ships = abs(dy_ships)
                if directions[I_NORTH] and 0 < dy < dy_opponent:
                    # 北 shipyard ally op 南
                    if abs_dx_ships <= abs_dy_ships:  # x軸合わせ間に合う
                        score = 1
                        target_opponent_ship = opponent_ship
                        condition += f' NORTH('
                        if abs_dx_ships == 0:
                            condition += f' abs_dx_ships0'
                            if 1 < abs_dy_ships:
                                condition += f' 1<abs_dy_ships'
                                next_direction = I_SOUTH
                            elif can_stop:
                                condition += f' can_stop'
                                next_direction = I_MINE
                            elif dx < 0:  # 西にshipyard
                                condition += f' dx<0'
                                next_direction = I_WEST
                            elif 0 < dx:  # 東にshipyard
                                condition += f' 0<dx'
                                next_direction = I_EAST
                            elif 1 < dy:
                                condition += f' 1<dy'
                                next_direction = I_NORTH
                            else:  # 運ゲー 
                                condition += f' good_luck'
                                next_direction = I_EAST
                        elif dx_ships < 0:  # 西にop
                            condition += f' dx_ships<0'
                            next_direction = I_WEST
                        else:
                            condition += f' 0<dx_ships'
                            next_direction = I_EAST
                        condition += f')'
                        break
                if directions[I_SOUTH] and dy_opponent < dy < 0:
                    # 北 op ally shipyard 南
                    if abs_dx_ships <= abs_dy_ships:  # x軸合わせ間に合う
                        score = 1
                        target_opponent_ship = opponent_ship
                        condition += f' SOUTH('
                        if abs_dx_ships == 0:
                            condition += f' abs_dx_ships0'
                            if 1 < abs_dy_ships:
                                condition += f' 1<abs_dy_ships'
                                next_direction = I_NORTH
                            elif can_stop:
                                condition += f' can_stop'
                                next_direction = I_MINE
                            elif dx < 0:  # 西にshipyard
                                condition += f' dx<0'
                                next_direction = I_WEST
                            elif 0 < dx:  # 東にshipyard
                                condition += f' 0<dx'
                                next_direction = I_EAST
                            elif dy < -1:
                                condition += f' dy<-1'
                                next_direction = I_SOUTH
                            else:  # 運ゲー 
                                condition += f' good_luck'
                                next_direction = I_EAST
                        elif dx_ships < 0:  # 西にop
                            condition += f' dx_ships<0'
                            next_direction = I_WEST
                        else:
                            condition += f' 0<dx_ships'
                            next_direction = I_EAST
                        condition += f')'
                        break
                if directions[I_WEST] and dx_opponent < dx < 0:
                    # 東 op ally shipyard 西
                    if abs_dy_ships <= abs_dx_ships:  # y軸合わせ間に合う
                        score = 1
                        target_opponent_ship = opponent_ship
                        condition += f' WEST('
                        if abs_dy_ships == 0:
                            condition += f' abs_dy_ships0'
                            if 1 < abs_dx_ships:
                                condition += f' 1<abs_dx_ships'
                                next_direction = I_EAST
                            elif can_stop:
                                condition += f' can_stop'
                                next_direction = I_MINE
                            elif dy < 0:  # 南にshipyard
                                condition += f' dy<0'
                                next_direction = I_SOUTH
                            elif 0 < dy:  # 北にshipyard
                                condition += f' 0<dy'
                                next_direction = I_NORTH
                            elif dx < -1:
                                condition += f' dx<-1'
                                next_direction = I_WEST
                            else:  # 運ゲー 
                                condition += f' good_luck'
                                next_direction = I_NORTH
                        elif dy_ships < 0:  # 南にop
                            condition += f' dy_ships<0'
                            next_direction = I_SOUTH
                        else:  # 北にop
                            condition += f' 0<dy_ships'
                            next_direction = I_NORTH
                        condition += f')'
                        break
                if directions[I_EAST] and 0 < dx < dx_opponent:
                    # 東 shipyard ally op 西
                    if abs_dy_ships <= abs_dx_ships:  # y軸合わせ間に合う
                        score = 1
                        target_opponent_ship = opponent_ship
                        condition += f' EAST('
                        if abs_dy_ships == 0:
                            condition += f' abs_dy_ships0'
                            if 1 < abs_dx_ships:
                                condition += f' 1<abs_dx_ships'
                                next_direction = I_WEST
                            elif can_stop:
                                condition += f' can_stop'
                                next_direction = I_MINE
                            elif dy < 0:  # 南にshipyard
                                condition += f' dy<0'
                                next_direction = I_SOUTH
                            elif 0 < dy:  # 北にshipyard
                                condition += f' 0<dy'
                                next_direction = I_NORTH
                            elif 1 < dx:
                                condition += f' 1<dx'
                                next_direction = I_EAST
                            else:  # 運ゲー 
                                condition += f' good_luck'
                                next_direction = I_NORTH
                        elif dy_ships < 0:  # 南にop
                            condition += f' dy_ships<0'
                            next_direction = I_SOUTH
                        else:  # 北にop
                            condition += f' 0<dy_ships'
                            next_direction = I_NORTH
                        condition += f')'
                        break
            to_sort.append((-score, d, dx, dy, ship.id, ship, target_opponent_ship, next_direction, condition))
        if len(to_sort) == 0:
            return True
        to_sort = sorted(to_sort)
        negative_score, d, dx, dy, ship_id, ship, target_opponent_ship, next_direction, condition = to_sort[0]
        score = -negative_score
        opponent_ship_id = None if target_opponent_ship is None else target_opponent_ship.id
        a.log(id_=ship_id, s=f'prj={self.project_id} {target_position} score{score} d{d} dx{dx} dy{dy}, op{opponent_ship_id} next_dir{next_direction} op_return_dir{self.opponent_return_directions} {condition}')
        self.dismiss_project(staff_ids=list(self.ships.keys()))
        role = 'escape'
        if next_direction is None:  # target_position へ移動する
            mode = 'escape'
            a.moving_ship_strategy(ship=ship, position=target_position, mode=mode, mine_threshold=None)
        else:
            mode = 'cancel'
            position = calculate_next_position(ship.position, next_direction)
            a.moving_ship_strategy(ship=ship, position=position, mode=mode, mine_threshold=None)
        self.join_project(staff_ids=[ship_id], role=role)
        return True


class EscortProject(Project):
    """
    対象のshipを守りながら移動
    対象自体は別Projectに属しているならそちら優先
    None ならまとめて管理 (defender に逃げ道潰されるのを防ぐため)
    """
    def __init__(self, target_ship_id, *args, defender_ship_id=None, **kwargs):
        self.target_ship_id = target_ship_id
        self.shipyard = None  # target_shipが帰還する予定のshipyard
        self.nearest_shipyard_id = None  # 同上
        self.defender_ship_id = defender_ship_id
        self.is_final = False
        super().__init__(*args, project_id=f'escort{self.target_ship_id}', **kwargs)

    def has_myself(self):
        return self.target_ship_id in self.ships.keys()

    def schedule(self):
        super().schedule()
        self.shipyard = None
        a = self.agent
        target_ship = a.board.ships.get(self.target_ship_id, None)
        if not target_ship:
            return False
        i_target, j_target = position_to_ij(target_ship.position)
        project_id = a.belonging_project.get(self.target_ship_id, None)
        is_project_mine = ((project_id is not None) and (project_id[:4] == 'mine'))
        self.can_safely_deposit = False
        if 0 == target_ship.halite:  # 普通護衛不要
            if is_project_mine:
                self.priority = 0.001
            else:
                self.priority = -1.0
                self.can_safely_deposit = True
        elif self.ships.get(self.target_ship_id, None):  # 自身が属している=帰還したい
            self.priority = (1.0 + target_ship.halite * 200)
        else:  # 護衛のみ
            self.priority = (1.0 + target_ship.halite * 100) / 1000

        self.shipyard, self.d_target_staff = a.find_nearest_ally_shipyard(target_ship.position)
        condition = ''
        if self.shipyard:
            self.target_staff = self.shipyard
            target_staff_id = self.target_staff.id if self.target_staff else None
            condition += 'shipyard'
        else:
            self.target_staff, self.d_target_staff = a.find_leader_ship(target_ship.position)
            condition += 'leader_ship'
        target_staff_id = self.target_staff.id if self.target_staff else None
        a.log(s=f'prj={self.project_id} {condition} target_staff_id={target_staff_id}')

        if self.shipyard and (not is_project_mine):
            # 間に合うか判定するのだが、
            # mine目的の時は今間に合っても暫く掘ると間に合わないかもしれない
            d_to_shipyard = calculate_distance(target_ship.position, self.shipyard.position)
            i_shipyard, j_shipyard = position_to_ij(self.shipyard.position)
            opponent_reach_shipyard = int(1e-6 + a.scores[I_SCORE_OPPONENT_REACH, i_shipyard, j_shipyard])
            i_target, j_target = position_to_ij(target_ship.position)
            opponent_reach_target = int(1e-6 + a.scores[I_SCORE_OPPONENT_REACH, i_target, j_target])
            if d_to_shipyard < opponent_reach_shipyard:
                # 間に合う
                self.can_safely_deposit = True
            elif d_to_shipyard <= 6 and d_to_shipyard < opponent_reach_target + opponent_reach_shipyard:
                # shipyard の反対側っぽいのでおおよそ安全
                a.log(id_=self.target_ship_id, s=f'{target_ship.position} EscortProject: target can ALMOST safely deposit to {self.shipyard.id}{self.shipyard.position}')
                self.can_safely_deposit = True

        original_defender_ship_id = self.defender_ship_id
        defender_ship = a.board.ships.get(self.defender_ship_id, None)
        if defender_ship:  # 現職の検証
            defender_ship_project_id = a.belonging_project.get(defender_ship.id, None)
            d = calculate_distance(defender_ship.position, target_ship.position)
            if (5 < d
                    or 0 < defender_ship.halite
                    or ((defender_ship_project_id is not None) and defender_ship_project_id != self.project_id)):
                # 不適当または雇えない
                self.dismiss_project(staff_ids=[self.defender_ship_id])
        elif self.defender_ship_id:
            self.dismiss_project(staff_ids=[self.defender_ship_id])
        if self.schedule_final(target_ship):
            self.is_final = True
            self.priority = 9999999.0 + target_ship.halite
        else:
            self.is_final = False
        if self.can_safely_deposit:
            # 一旦解散しつつpriority下げて様子見
            self.priority *= 0.01
            self.dismiss_project(staff_ids=list(self.ships.keys()))
        elif self.priority < 0.0:
            self.dismiss_project(staff_ids=list(self.ships.keys()))
        # a.log(id_=original_defender_ship_id, s=f'prj={self.project_id} prio{self.priority:.3f} def{self.defender_ship_id} belong{a.belonging_project.get(original_defender_ship_id, None)}')
        return True

    def update_defender_ship_id(self):
        self.defender_ship_id = None
        for ship_id, role in self.ships.items():
            if role == 'defender':
                self.defender_ship_id = ship_id

    def join_project(self, *args, **kwargs):
        super().join_project(*args, **kwargs)
        self.update_defender_ship_id()

    def dismiss_project(self, *args, **kwargs):
        super().dismiss_project(*args, **kwargs)
        self.update_defender_ship_id()

    def schedule_final(self, target_ship):
        a = self.agent
        if a.board.step <= 359:
            return False
        if a.determined_ships.get(self.target_ship_id, None) is not None:
            return False
        # 1歩進むのに倍ぐらいかかると予想
        # ラストescapeだと2手失うので1手余分に確保
        steps = 1 + self.d_target_staff + max(0, self.d_target_staff - 3)
        if a.board.step + steps < 399:
            return False
        return True

    def run(self):
        super().run()
        a = self.agent
        target_project_id = a.belonging_project.get(self.target_ship_id, None)
        is_defense_shipyard_project = ((target_project_id is not None) and (target_project_id[:10] == 'defense_yd'))
        if self.priority < 0.0 or is_defense_shipyard_project:
            self.dismiss_project(staff_ids=list(self.ships.keys()))
            return True
        if self.defender_ship_id and a.determined_ships.get(self.defender_ship_id, None):
            return True  # MineProject がrunしていることがある
        target_ship = a.board.ships[self.target_ship_id]
        determined = a.determined_ships.get(self.target_ship_id, None)
        if determined is None:
            # target_ship 未行動なら自身も加入させる
            self.join_project(staff_ids=[self.target_ship_id], role='target')

        defender = None
        if self.can_safely_deposit:  # defender 不要
            if self.defender_ship_id is not None:
                self.dismiss_project(staff_ids=[self.defender_ship_id])
        else:
            if self.defender_ship_id is not None:
                # HuntProjectに奪われているかもしれないので検証
                defender_ship_project_id = a.belonging_project.get(self.defender_ship_id, None)
                if ((defender_ship_project_id is not None)
                        and defender_ship_project_id != self.project_id):
                    self.defender_ship_id = None
            if self.defender_ship_id is not None:
                defender = a.board.ships[self.defender_ship_id]
            else:
                to_sort = []
                for ship in self.ships_generator(with_free=True):
                    if 0 < ship.halite:
                        continue
                    d = calculate_distance(ship.position, target_ship.position)
                    if 5 < d:
                        continue
                    to_sort.append((ship, d, ship.id))
                if not to_sort:  # 誰もいませんでした
                    if not self.is_final:
                        return True
                else:
                    to_sort = sorted(to_sort, key=itemgetter(1, 2))
                    defender = to_sort[0][0]
                    self.defender_ship_id = defender.id
            if defender:
                self.join_project(staff_ids=[defender.id], role='defender')

        # target_ship を先に移動
        if determined is None:
            mode = 'escape'
            mine_threshold = 3.99
            if self.is_final:
                if target_ship.halite == 0 and 5 <= len(a.board.current_player.ships):
                    # shipyard_attack
                    opponents = []
                    for player_id in range(PLAYERS):
                        if player_id != a.player_id:
                            opponents.append(player_id)
                    opponent_shipyard, d_opponent_shipyard = a.find_nearest_shipyard(
                            target_ship.position,
                            player_ids=opponents)
                    if opponent_shipyard:
                        target_position = opponent_shipyard.position
                        mode = 'cancel'
                        mine_threshold = None
                    else:
                        target_position = target_ship.position
                        mode = 'escape'
                        mine_threshold = None
                else:
                    target_position = self.target_staff.position
                    if 0 < target_ship.halite and self.d_target_staff <= 3:
                        mode = 'merge'
                        mine_threshold = None
            elif self.shipyard:
                target_position = self.shipyard.position
            elif defender:
                target_position = defender.position
            else:
                target_position = target_ship.position
            if 398 <= a.board.step and 500 < target_ship.halite and 1 < self.d_target_staff:
                a.reserve_ship(target_ship, ShipAction.CONVERT)
                a.log(id_=target_ship.id, s=f'h{target_ship.halite} convert at last turn')
            else:
                a.moving_ship_strategy(
                        target_ship, position=target_position,
                        mode=mode, mine_threshold=mine_threshold)
                a.log(id_=target_ship.id, s=f'{target_ship.position}->{target_position} by myself in EscortProject is_final={self.is_final} mode={mode}')
        else:
            target_position = None
        if defender:
            a.log(id_=defender.id, s=f'{defender.position} escort to {target_ship.id}{target_ship.position} is_final={self.is_final}')
            a.log(id_=target_ship.id, s=f'{target_ship.position} escorted by {defender.id}{defender.position} is_final={self.is_final}')
            qs = np.ones((LEN_MOVE, LEN_MOVE), dtype=np.float32)
            mode = 'escape'
            if determined is None and 0 < target_ship.halite:
                # 帰ろうとしているのでcancelも辞さない
                mode = 'cancel_without_shipyard'
            for i_action, cell_i in enumerate(a.neighbor_cells(a.board.cells[target_ship.position])):
                # cell_i: target_ship の次step移動先
                # target_position_i: target_shipの目標移動先
                if target_position:
                    target_position_i = target_position
                else:
                    target_position_i = cell_i.position
                # d_i = calculate_distance(defender.position, cell_i.position)
                target_d_i = calculate_distance(target_position_i, cell_i.position)
                # target_position よりも先回りする位置を目標地点とする
                to_sort_k = []
                for k_action, cell_k in enumerate(a.neighbor_cells(cell_i)):
                    target_d_k = calculate_distance(target_position_i, cell_k.position)
                    d_k = calculate_distance(defender.position, cell_k.position)
                    to_sort_k.append((target_d_k, d_k, k_action, cell_k))
                target_d_k, d_k, k_action, cell_k = sorted(to_sort_k)[0]
                if d_k <= 1: # 目標地点到達するときは先回りしているので相殺してよい
                    mode_i = mode
                else:  # 遠いときはcancelしても効果が薄い
                    mode_i = 'escape'
                q_i, forced = a.calculate_moving_ship_preference(
                        ship=defender, position=cell_k.position,
                        mode=mode_i, mine_threshold=None)
                a.log(id_=defender.id, s=f'[{i_action}{cell_i.position}] ka{k_action}{cell_k.position} tdk{target_d_k} dk{d_k} mode={mode_i}')
                qs[i_action, :] = q_i
            a.reserve_ship_by_q(
                    ship=defender, q=qs, depend_on=target_ship)
        return True

MINE_PRIORITY_BY_DISTANCE = [100.0 * (0.8**t) for t in range(99)]
def mine_priority_by_distance(d):
    if 0 <= d < len(MINE_PRIORITY_BY_DISTANCE):
        return MINE_PRIORITY_BY_DISTANCE[d]
    return 0.0

class MineProject(Project):
    """
    位置を指定して掘る
    """
    def __init__(self, position, *args, **kwargs):
        super().__init__(*args, project_id=f'mine_{position[0]}_{position[1]}', **kwargs)
        self.position = position
        self.i, self.j = position_to_ij(position)
        self.cell = None
        self.halite = 0.0
        self.halite_threshold = 80.0

        self.ally_reach = None
        self.empty_ally_reach = None
        self.ally_reach_v2 = None
        self.opponent_reach = None
        self.empty_opponent_reach = None
        self.opponent_reach_v2 = None

        self.elapsed_steps = 0
        self.neighbor_empty_opponent_counter = 0
        self.early_game_threshold = 50  # defender なしで project 成立するターン数
        self.last_swapped_step = -1
        self.last_mined_player_id = self.agent.player_id

    def run_miner(self, miner, forced=False):
        a = self.agent
        overwrite = False
        determined = a.determined_ships.get(miner.id, None)
        if determined is not None:
            overwrite = True
            if (not forced) or (determined != 'reserved'):
                a.log(loglevel='warning', id_=miner.id, s=f'prj={project_id} run_miner failed because determined{determined} or not forced')
                exit()
                return
            priority, ship_id, q, ship, forced_, depend_on = a.reserving_ships[miner.id]
            del a.reserving_ships[miner.id]
        else:
            priority = None
            depend_on = None

        previous_project_id = a.belonging_project.get(miner.id, None)
        if previous_project_id is None:
            self.join_project(staff_ids=[miner.id], role='miner')
        elif previous_project_id != self.project_id:
            self.join_project(staff_ids=[miner.id], role='miner', forced=True)

        miner_d = calculate_distance(miner.position, self.position)

        target_cell = a.board.cells[self.position]
        miner_cell = a.board.cells[miner.position]
        p_success = 1.0
        if miner_d == 0:
            mode = 'mine'
            q, forced_ = a.calculate_moving_ship_preference(
                    miner, position=self.position,
                    mode=mode, mine_threshold=self.mining_halite_threshold)
        elif miner_d == 1:  # 相殺で突っ込むべきか判断する
            opponent_exists = np.zeros(PLAYERS, dtype=np.int32)
            opponent_mining = False
            wait_opponent_mining = False
            for k_action, cell_k in enumerate(a.neighbor_cells(target_cell)):
                if cell_k.ship is None:
                    continue
                ship_k = cell_k.ship
                if ship_k.player_id == a.player_id:
                    continue
                if ship_k.halite < miner.halite:  # 無理
                    a.log(s=f'prj={self.project_id} k{k_action} ship_k{ship_k.id} h{ship_k.halite}')
                    self.d_halite = 0.0
                    return True
                if ship_k.halite == miner.halite:
                    if k_action == 0:  # 先着されているので待って追い出したい
                        opponent_mining = True
                        if miner_cell.halite <= 3.99 or (100. < miner_cell.halite < self.halite):  # 掘りあったら有利
                            wait_opponent_mining = True
                        a.log(id_=miner.id, loglevel='info', s=f'prj={self.project_id} k{k_action} ship_k{ship_k.id} th{self.halite} miner_cellh{miner_cell.halite} op_mining={opponent_mining} wait_op_mining={wait_opponent_mining}')
                    opponent_exists[ship_k.player_id] = 1
            p_stay = np.ones(PLAYERS, dtype=np.float32)
            # 敵が突っ込んでくる確率を見積り、nash均衡よりも高そうだったらnash均衡、そうでなければ突っ込む
            p_nash = both_move_to_halite_nash_equilibrium(target_cell.halite)
            for player_id in range(PLAYERS):
                if not opponent_exists[player_id]:
                    continue
                t, u = a.opponent_history[ship_k.player_id].get('cancel_both_move_to_mine', [0, 0])
                if u == 0:  # 初手は様子見
                    p_stay[player_id] = 0.0
                else:
                    p_stay[player_id] = (u - t) / u
            p_opponent_go = 1.0 - np.prod(p_stay)
            if opponent_mining:  # 掘らせておこう
                p_success = 0.0
            elif p_opponent_go < 1e-6 + p_nash:  # 敵は保守的なのでこっちはちょっと攻めよう
                p_success = 2.0 * p_nash
            else:  # nash均衡にしておく
                p_success = p_nash

            if np.random.rand() < p_success:
                mode = 'cancel'
            else:
                mode = 'escape'
            if np.any(opponent_exists):
                a.log(id_=miner.id, s=f'prj={self.project_id} h{miner.halite} p_nash{p_nash:.6f} p_stay{p_stay} p_opponent_go{p_opponent_go:.6f} p_success{p_success:.6f} mode={mode} op_mining={opponent_mining} wait_op_mining={wait_opponent_mining}')
            if wait_opponent_mining:  # 相殺覚悟で隣待ち
                mode = 'cancel'
                q, forced_ = a.calculate_moving_ship_preference(
                        miner, position=miner.position,
                        mode=mode, mine_threshold=3.99)
            else:
                q, forced_ = a.calculate_moving_ship_preference(
                        miner, position=self.position,
                        mode=mode, mine_threshold=self.mining_halite_threshold)
        elif miner_d == 2 and self.empty_opponent_reach == 0:
            # 相殺覚悟で突っ込みさっさと追い払う
            mode = 'cancel'
            q, forced_ = a.calculate_moving_ship_preference(
                    miner, position=self.position,
                    mode=mode, mine_threshold=None if miner.halite == 0 else 300.0)
        else:
            mode = 'escape'
            q, forced_ = a.calculate_moving_ship_preference(
                    miner, position=self.position,
                    mode=mode, mine_threshold=None if miner.halite == 0 else 300.0)

        a.reserve_ship_by_q(miner, q=q, forced=False, priority=priority, depend_on=depend_on)
        self.update_score()
        a.log(id_=miner.id, s=f'run_miner h{miner.halite} {miner.position}->{self.position} h{self.halite} prj={previous_project_id}->{self.project_id} prio{self.priority:.1f} noc{self.neighbor_empty_opponent_counter} h_thre{self.halite_threshold} mh_thre{self.mining_halite_threshold} mode={mode}')

    def swap_mine_project(self, miner, defender, escort_project):
        """
        役割スワップでターン節約
        miner, reserved, old_minerの就職先, 新たなminerに対応するescort_project
        """
        a = self.agent
        if a.board.step - 1 <= self.last_swapped_step:
            # 連続してswapするとデッドロックしうるので回避
            a.log(s=f'swap failed. last_swapped_step{self.last_swapped_step}')
            return miner, False, self, escort_project  # miner, defender, reserved, old_miner_project, old_defender_project
        defender_ship_id = defender.id if defender else None
        to_sort = []
        d0 = calculate_distance(miner.position, self.position)
        condition = f'ground{self.position} miner_id={miner.id}{miner.position} d0={d0}'
        for k_action, cell_k in enumerate(a.neighbor_cells(a.board.cells[miner.position])):
            if k_action == 0:
                continue
            ship_k = cell_k.ship
            if not ship_k:
                continue
            condition += f' [{k_action}] {ship_k.id}'
            determined = a.determined_ships.get(ship_k.id)
            reserved = None
            if determined is None:
                condition += f' not_reserved'
                reserved = False
            elif determined == 'reserved':
                condition += f' reserved'
                reserved = True
            else:
                condition += f' determined.'
                continue
            if ship_k.player_id != a.player_id:
                condition += f' p_diff.'
                continue
            if miner.halite != ship_k.halite:
                condition += f' h_diff.'
                continue  # can_offerが一致するかわからないので金額を同じとき限定にしておく
            project_id_k = a.belonging_project.get(ship_k.id, None)
            condition += f' prj={project_id_k}'
            if project_id_k is None:
                continue
            project_k = a.projects.get(project_id_k, None)
            if project_k is None:
                condition += f' prj_not_found.'
                continue
            target_position = None
            new_mine_project_k = None
            if project_id_k == f'escort{miner.id}':
                # 自分自身をescortしているdefenderとの交換だけにする
                if (defender is None):
                    condition += f' WARN_defender=None.'
                    continue  # 指定されているはずなんだが
                if (project_k.defender_ship_id is not None and project_k.defender_ship_id != defender_ship_id):
                    condition += f' WARN_defender_ship_id={project_k.defender_ship_id}!={defender_ship_id}.'
                    continue  # 指定されているはずなんだが
                target_ship = a.board.ships.get(project_k.target_ship_id, None)
                if target_ship is None or target_ship.id != miner.id:
                    condition += f' target_ship_not_found.'
                    continue
                target_position = target_ship.position
                condition += f' target_ship{target_ship.id}{target_position}'
                priority = 10.0
            elif project_id_k[:4] == 'mine':
                target_position = project_k.position
                condition += f' {target_position}'
                priority = 100.0
                new_mine_project_k = project_k
            else:
                condition += f' unsuitable_prj'
                continue
            d_k = calculate_distance(ship_k.position, target_position)
            d0_k = calculate_distance(ship_k.position, self.position)
            d = calculate_distance(miner.position, target_position)
            condition += f' d0{d0} dk{d_k} d{d} d0k{d0_k}'
            if d0 + d_k < d + d0_k:
                condition += f' far.'
                priority += 1000.
                continue  # 交換後合計が遠くなるならやらない
            elif d0 + d_k == d + d0_k and max(d0, d_k) <= max(d, d0_k):
                condition += f' imbal.'
                continue  # 遠いほうの距離を減らしたい
            priority += d_k
            to_sort.append((priority, ship_k.id, ship_k, new_mine_project_k, target_position, reserved))
        if not to_sort:
            a.log(id_=miner.id, s=f'prj={self.project_id} swap no_candidates. cond=({condition})')
            return miner, False, self, escort_project
        # join, dismiss 処理をする
        priority, swapped_ship_id, swapped_ship, new_mine_project, target_position, reserved = sorted(to_sort, reverse=True)[0]
        ship_role = self.ships[miner.id]
        self.dismiss_project(staff_ids=[miner.id])
        if new_mine_project is None:  # defenderとの交換
            if defender and swapped_ship_id == defender_ship_id:
                # swapped_shipをtarget_shipとするescort_projectへ変更する
                escort_project = a.projects.get(f'escort{swapped_ship_id}', None)
                if escort_project:
                    escort_project.join_project(staff_ids=[miner.id], role='defender')
                    condition += f' join_to_escort_prj={escort_project.project_id}'
                    return defender, False, escort_project, escort_project
            else:
                condition += ' WARN_new_mine_project=None swapped_ship_id{swapped_ship_id} defender_ship_id{defender_ship_id}.'
                a.log(id_=miner.id, s=f'prj={self.project_id} swap_failed. cond=({condition})')
            return miner, False, self, escort_project
        # miner 同士の交換
        condition += ' minerXminer'
        swapped_ship_escort_project = a.projects.get(f'escort{swapped_ship_id}', None)
        swapped_defender = None
        if swapped_ship_escort_project:
            swapped_defender_ship_id = swapped_ship_escort_project.defender_ship_id
            swapped_defender = a.board.ships.get(swapped_defender_ship_id, None)
        if swapped_defender:
            condition += f' has_swapped_defender{swapped_defender_ship_id}_{a.belonging_project.get(swapped_defender_ship_id, None)}'
        else:
            condition += 'no_swapped_defender'
        if defender_ship_id:
            # 自分に護衛がいるなら、相手にも護衛が確定していないと相手だけ解散して無駄足になりうる
            condition += f'has_defender{defender_ship_id}'
            if not swapped_defender:
                condition += f' no_swapped_defender'
                a.log(id_=miner.id, s=f'prj={self.project_id} swap_failed. cond=({condition})')
                return miner, False, self, escort_project
        elif swapped_defender_ship_id:
            # 自分に護衛がいないなら、相手にも護衛がいないときだけにしておく
            return miner, False, self, escort_project

        swapped_ship_role = new_mine_project.ships[swapped_ship_id]
        new_mine_project.dismiss_project(staff_ids=[swapped_ship_id])
        self.join_project(staff_ids=[swapped_ship_id], role=ship_role)
        new_mine_project.join_project(staff_ids=[miner.id], role=swapped_ship_role)
        condition += f' join_to_prj={new_mine_project.project_id}'
        a.log(id_=miner.id, s=f'prj={self.project_id} swapped. s{miner.id}(prj={a.belonging_project.get(miner.id, None)}) s{swapped_ship_id}({a.belonging_project.get(swapped_ship_id, None)}) d0{d0} dk{d_k} target{target_position} cond({condition})')
        a.log(id_=swapped_ship_id, s=f'prj={self.project_id} swapped. s{miner.id}(prj={a.belonging_project.get(miner.id, None)}) s{swapped_ship_id}({a.belonging_project.get(swapped_ship_id, None)}) d0{d0} dk{d_k} target{target_position} cond({condition})')
        self.last_swapped_step = a.board.step
        return swapped_ship, reserved, new_mine_project, swapped_ship_escort_project

    def mine_project_ships_generator(self, max_d):
        a = self.agent
        for ship in a.sorted_ships:
            if a.determined_ships.get(ship.id, None) is not None:
                continue
            d = calculate_distance(ship.position, self.position)
            if max_d < d:
                continue
            if d == 0 or 0 < ship.halite:
                neighbor_opponent_count = 0
                for k_action, cell_k in enumerate(a.neighbor_cells(a.board.cells[ship.position])):
                    i_k, j_k = position_to_ij(cell_k.position)
                    if 0.5 < a.scores[I_SCORE_EMPTY_OPPONENT_D2, i_k, j_k]:
                        neighbor_opponent_count += 1
                if 3 <= neighbor_opponent_count:  # 囲まれそう
                    continue
            if self.shipyard:
                nearest_shipyard, d_shipyard = a.find_nearest_shipyard(ship.position)
                if nearest_shipyard is not None and nearest_shipyard.id != self.shipyard.id:
                    if len(a.ships_by_shipyard[nearest_shipyard.id]) <= 2:
                        continue  # 過疎地帯からは引っ張ってこない
            project_id = a.belonging_project.get(ship.id, None)
            if project_id is None:
                yield ship, d
                continue
            project = a.projects.get(project_id, None)
            if project is None:
                yield ship, d
            elif project_id == self.project_id:
                yield ship, d
            elif project_id[:4] == 'mine':
                # 距離近めでhalite十分に多かったら上書き可能とする
                d_other = calculate_distance(ship.position, project.position)
                dd = d - d_other
                threshold = 99999.0
                if d <= d_other:
                    threshold = 1.0
                elif d <= d_other + 1:
                    threshold = 70.0
                elif d <= d_other + 2:
                    threshold = 140.0
                else:
                    threshold = 99999.0
                if project.d_halite + threshold < self.d_halite:
                    a.log(f'prj={self.project_id} other_prj={project_id} d{d} d_other{d_other} dh{self.d_halite} dh_other{project.d_halite}')
                    yield ship, d

    def update_score(self):
        a = self.agent
        if 0 == len(self.ships):
            return
        f = a.flags
        neighbor_positions = NEIGHBOR_POSITIONS[3][self.i, self.j]
        for i_k, j_k, d_k in neighbor_positions:
            if d_k < 2:
                f.set(I_FLAG_MINE_D2, i=i_k, j=j_k)
            if d_k < 3:
                f.set(I_FLAG_MINE_D3, i=i_k, j=j_k)
            if d_k < 4:
                f.set(I_FLAG_MINE_D4, i=i_k, j=j_k)
            
    def update_halite_threshold(self):
        a = self.agent
        step = a.board.step
        if a.previous_board:  # 掘られたかチェック
            previous_cell = a.previous_board.cells[self.position]
            cell = a.board.cells[self.position]
            if cell.halite + 1e-6 < previous_cell.halite:
                if previous_cell.ship:
                    self.last_mined_player_id = previous_cell.ship.player_id

        ally_d = 99999
        opponent_d = 99999
        condition = f'last_p{self.last_mined_player_id}'
        for shipyard in a.board.shipyards.values():
            d = calculate_distance(self.position, shipyard.position)
            if shipyard.player_id == a.player_id:
                if d < ally_d:
                    ally_d = d
            elif d < opponent_d:
                opponent_d = d
        danger_zone = a.scores[I_SCORE_DANGER_ZONE, self.i, self.j]
        hunt_zone = a.scores[I_SCORE_HUNT_ZONE_IN, self.i, self.j]
        condition += f' ally_d{ally_d} op_d{opponent_d} danger_zone{danger_zone:.0f} hunt_zone{hunt_zone:.0f}'
        nearest_ally_shipyard = a.nearest_ally_shipyard[self.i][self.j]
        if nearest_ally_shipyard:
            len_ships = len(a.ships_by_shipyard.get(nearest_ally_shipyard.id, []))
        else:
            len_ships = 0
        if ally_d <= opponent_d and danger_zone < 0.5:
            # 我々の土地
            if 3 <= len_ships and 0.5 < hunt_zone and self.last_mined_player_id == a.player_id:
                if step < 20:
                    self.halite_threshold = 200.
                elif step < 270:
                    self.halite_threshold = 200.0
                else:
                    self.halite_threshold = 20.0
            elif ally_d + 1 < opponent_d and ally_d <= 2 and self.last_mined_player_id == a.player_id:
                # shipyard至近距離 安全なので増やす
                if step < 20:
                    self.halite_threshold = 140.
                elif step < 270:
                    self.halite_threshold = 160.0 if ally_d <= 1 else 100.0
                else:
                    self.halite_threshold = 20.0
            else:
                if step < 100:
                    self.halite_threshold = 120.
                elif step < 200:
                    self.halite_threshold = 70.0
                elif step < 270:
                    self.halite_threshold = 50.0
                else:
                    self.halite_threshold = 20.0
        else:
            # 敵地なので吸いつくす
            if step < 100:
                self.halite_threshold = 100.
            elif step < 270:
                self.halite_threshold = 30.0
            else:
                self.halite_threshold = 20.0
        # 掘り始めたら1回余分に掘る
        self.mining_halite_threshold = self.halite_threshold * 0.7
        condition += f' h_thre{self.halite_threshold} mh_thre{self.mining_halite_threshold}'
        return condition

    def calculate_minable_halite(self, advantage):
        a = self.agent
        halite = self.halite
        disadvantage = -min(0, advantage)
        remaining_halite = self.halite * (0.75**disadvantage)

        if self.halite_threshold < halite:
            return halite - self.mining_halite_threshold, 'enough_h'
        elif self.mining_halite_threshold < halite:
            # 掘り始めたらしっかり掘る
            ship = a.board.cells[self.position].ship
            if ship and ship.player_id == a.player_id and 0 < ship.halite:
                return halite - self.mining_halite_threshold, 'thre_h_ok'
            else:
                return 0.0, 'thre_h_ng'
        else:
            return 0.0, 'short_h'

    def schedule(self):
        super().schedule()
        a = self.agent
        self.maintain_dead_staffs()
        self.cell = a.board.cells[self.position]
        self.halite = self.cell.halite
        if self.halite < 1e-6:
            return False
        condition = ''
        self.priority = 1.0

        self.shipyard, self.d_shipyard = a.find_nearest_shipyard(self.position)

        condition += self.update_halite_threshold()

        len_ships = len(a.board.current_player.ships)
        # 間に合わないなら掘らない
        if self.shipyard is None:
            condition += f' shipyard=None'
            if 389 < a.board.step:
                condition += f' 389<step.'
                self.priority = -1.0
            elif len_ships <= 1 and a.board.current_player.halite < 2 * MAX_HALITE:  # もう拠点を構えるのは不可能
                condition += f' escape_only len_ships{a.len_ships}.'
                # a.log(loglevel='WARNING', s=f'cond({condition})')
                self.priority = -1.0
        elif len_ships <= 1 and a.board.current_player.halite < 2 * MAX_HALITE:  # 逃げたほうが良い
            # condition += f' escape_only_with_yd len_ships{a.len_ships}.'
            # a.log(loglevel='WARNING', s=f'cond({condition})')
            self.priority = -1.0
        elif 399 < a.board.step + self.d_shipyard:
            condition += f' 399<step+{self.d_shipyard}.'
            self.priority = -1.0
        elif not a.flags.get(I_FLAG_GO_HOME_STRAIGHT, i=self.i, j=self.j):
            # 迂回が必要なら優先度下げる
            condition += f' must_detour'
            self.priority *= 0.1
            pass
        else:
            d_straight = calculate_distance(self.position, self.shipyard.position)
            detour_advantage = a.scores[I_SCORE_DETOUR_ADVANTAGE, self.i, self.j]
            condition += f' d_straight{d_straight} detour_adv{detour_advantage:.0f}'
            if 2 < d_straight and detour_advantage < -3.5:
                # 敵の強いところは避ける
                condition += ' detour_disadv'
                self.priority *= 0.1
                # self.priority = -1.0

        if self.priority < -1e-6:
            self.reset_project()
            a.log(f'prj={self.project_id} cond=({condition})')
            return True
        # shipyard無かったらどこかにできるかもしれないので継続
                
        if 20 < self.elapsed_steps:  # timeout 膠着状態かも
            condition += ' timeout'
            self.reset_project()

        # 敵とのレース
        self.ally_reach = int(1e-6 + a.scores[I_SCORE_ALLY_REACH, self.i, self.j])
        self.empty_ally_reach = int(1e-6 + a.scores[I_SCORE_EMPTY_ALLY_REACH, self.i, self.j])
        self.ally_reach_v2 = min(self.ally_reach + 1, self.empty_ally_reach)
        self.opponent_reach = int(1e-6 + a.scores[I_SCORE_OPPONENT_REACH, self.i, self.j])
        self.empty_opponent_reach = int(1e-6 + a.scores[I_SCORE_EMPTY_OPPONENT_REACH, self.i, self.j])
        self.opponent_reach_v2 = min(self.opponent_reach + 1, self.empty_opponent_reach)
        condition += f' a{self.ally_reach}_ea{self.empty_ally_reach}_o{self.opponent_reach}_eo{self.empty_opponent_reach}'
        if self.opponent_reach_v2 <= self.ally_reach_v2:
            # 敵が同時/先に着く場合、掘られた後 empty_ship で場所を取り返す
            advantage = -max(1, self.empty_ally_reach - 1 - self.opponent_reach)
            condition += f' op_territory'
        else:
            advantage = 0
        self.d_halite, condition_t = self.calculate_minable_halite(advantage)
        condition += f' adv{advantage} hth{self.halite_threshold} mth{self.mining_halite_threshold} dh{int(self.d_halite)} {condition_t}'
        if self.d_halite < 1e-6:
            self.priority = -1.0
        else:
            self.priority *= self.d_halite

        # 近いほど優先度高い
        ally_reach = int(1e-6 + a.scores[I_SCORE_ALLY_REACH, self.i, self.j])
        self.priority *= mine_priority_by_distance(self.ally_reach)

        if a.board.step < 2:
            self.priority = -1.0
            condition += ' too_early_game'
        has_neighbor_empty_opponent = False
        for i_action, cell_i in enumerate(a.neighbor_cells(self.cell)):
            shipyard = cell_i.shipyard
            if shipyard and shipyard.player_id != a.player_id:
                # 敵shipyardの隣は優先度下げる
                self.priority *= 0.1

            ship = cell_i.ship
            if not ship:
                continue
            if ship.player_id == a.player_id:
                if i_action != 0:
                    continue
                project_id = a.belonging_project.get(ship.id, None)
                # if project_id == self.project_id or project_id is None:
                    # 既に掘れるなら優先度上げる (他の移動に邪魔されないようにするため, ただしreserving_ships実装してからはこれだけだと不十分)
                    # pass  # ally_reach でもう上げている
            elif i_action == 0:  # 敵に先着された
                condition += ' op_reach'
                # self.priority = -1.0
            elif ship.halite == 0:
                condition += ' op_neighbor'
                has_neighbor_empty_opponent = True
        if has_neighbor_empty_opponent:
            self.neighbor_empty_opponent_counter += 1
            # if 3 <= self.neighbor_empty_opponent_counter:
                # self.priority = -1.0  # お見合い膠着回避
        else:
            self.neighbor_empty_opponent_counter = 0
        if self.priority < 0.0:
            condition += ' failed'
            self.reset_project()
        a.log(f'prj={self.project_id} schedule prio{self.priority:.1f} cond=({condition}) h_thre{self.halite_threshold}')
        return True

    def reset_project(self):
        self.elapsed_steps = 0
        if self.ships:
            self.dismiss_project(staff_ids=list(self.ships.keys()))

    def can_offer(self, ship_halite, d):
        """あんまりhalite量が多いならjoinさせない"""
        a = self.agent
        if ship_halite == 0:
            advantage = self.opponent_reach_v2 - d
            self.d_halite, condition = self.calculate_minable_halite(advantage=advantage)
            # あんまり遠いならjoinさせない
            return d <= 8 and 4.9 < self.d_halite
        if 1000. < ship_halite and 0 < d:
            return False
        advantage = self.opponent_reach - d - 1
        if 0 < advantage:
            self.d_halite = self.halite * (1.0 - (0.75 ** advantage))
            return True
        self.d_halite = self.halite * 0.25
        if d == 0:  # もうちょっと詳しくチェックする
            opponent_halite = int(1e-6 + a.scores[I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE, self.i, self.j])
            return ship_halite < opponent_halite
        elif d == 1:
            opponent_halite = 99999
            for x_k, y_k, d_k in neighbor_positions(d=2, p=self.position):
                cell_k = a.board.cells[x_k, y_k]
                ship_k = cell_k.ship
                if ship_k is None:
                    continue
                if ship_k.player_id == a.player_id:
                    continue
                opponent_halite = min(ship_k.halite, opponent_halite)
            return ship_halite < opponent_halite
        return False

    def is_current_ship_mining(self):
        a = self.agent
        cell = a.board.cells[self.position]
        if (cell.ship and cell.ship.player_id == a.player_id):
            determined = a.determined_ships.get(cell.ship.id, None)
            a.log(s=f'prj={self.project_id} is_current_ship_mining s{cell.ship.id} determined={determined}')
            if determined is None:
                pass
            elif determined == 'reserved':
                q = a.reserving_ships[cell.ship.id][2]
                if len(q.shape) == 2:
                    max_i_action = np.argmax(q[0])
                else:
                    max_i_action = np.argmax(q)
                if max_i_action == 0:
                    return True
            else:
                next_action = determined[0]
                if (next_action is None) or (next_action == I_MINE):
                    return True
        return False

    def should_have_defender(self, d_shipyard):
        a = self.agent
        # o0 = a.previous_len_opponent_ships // 5
        d_threshold = [99999, 6, 5, 4, 3]
        o1 = min(max(0, a.len_opponent_ships - 30) // 5, len(d_threshold) - 1)
        a.log(f'prj={self.project_id} should_have_defender d_yd{d_shipyard} o1{o1} d_thre{d_threshold[o1]}')
        return d_threshold[o1] < d_shipyard

    def run(self):
        super().run()
        a = self.agent
        if self.priority < 0.0:
            return True
        # 一旦解放
        self.dismiss_project(staff_ids=list(self.ships.keys()))
        # 他のProject (例えばDefenseShipyardProject)で掘っていることがある
        if self.is_current_ship_mining():
            a.log(s=f'prj={self.project_id} is_current_ship_mining is True')
            return True
        len_ships = len(a.board.current_player.ships)
        if len_ships < 4:
            i_flag = I_FLAG_MINE_D4
            max_d = [8, 1]
        elif len_ships < 8:
            i_flag = I_FLAG_MINE_D4
            max_d = [12, 2]
        elif len_ships < 10:
            i_flag = I_FLAG_MINE_D4
            max_d = [16, 2]
        elif len_ships < 20:
            i_flag = I_FLAG_MINE_D3
            max_d = [22, 1]
        else:
            i_flag = I_FLAG_MINE_D2
            max_d = [22, 1]
        if a.flags.get(i_flag, i=self.i, j=self.j):
            i_max_d = 1
        else:
            i_max_d = 0

        to_sort = []
        for ship, d in self.mine_project_ships_generator(max_d[i_max_d]):
            escorted_count = 0
            escort_project = a.projects.get(f'escort{ship.id}', None)
            if escort_project:
                for ship_id in escort_project.ships:
                    if ship_id == ship.id:
                        continue
                    escorted_count += 1
            to_sort.append((ship, d, -escorted_count, -ship.halite, ship.id))
        miner = None
        miner_d = 99999
        defender = None
        for i, (ship, d, negative_escorted_count, negative_halite, ship_id) in enumerate(sorted(to_sort, key=itemgetter(1, 2, 3, 4))):
            halite = abs(negative_halite)
            if not self.can_offer(ship_halite=halite, d=d):
                continue
            if miner is None:
                miner = ship
                miner_d = d
            elif halite == 0:
                if d <= miner_d + 3:  # 護衛候補が遠すぎるならいないのと同じ
                    defender = ship
                break
        if not miner:
            # a.log(s=f'prj={self.project_id} no miner max_d[{i_max_d}]{max_d[i_max_d]} len(to_sort){len(to_sort)}')
            self.d_halite = 0.0
            return True  # 人員確保できませんでした

        shipyard, d_shipyard = a.find_nearest_shipyard(self.position)
        if shipyard is None:
            defender = None
            escort_project = None
        elif not self.should_have_defender(d_shipyard):
            defender = None
            escort_project = None
        else:  # 中盤以降の遠征では defender 必須
            escort_project = a.projects[f'escort{miner.id}']
            if escort_project.defender_ship_id is None:
                if not defender:
                    a.log(s=f'prj={self.project_id} miner{miner.id} no defender')
                    self.d_halite = 0.0
                    return True
                escort_project.defender_ship_id = defender.id
                escort_project.join_project(staff_ids=[defender.id], role='defender')
            else:
                defender = a.board.ships[escort_project.defender_ship_id]
        defender_id = defender.id if defender else None
        a.log(id_=miner.id, s=f'd_yd={d_shipyard} defender{defender_id}')

        self.join_project(staff_ids=[miner.id], role='miner')
        old_miner = miner
        miner, reserved, old_miner_project, escort_project = self.swap_mine_project(miner, defender, escort_project)
        if old_miner_project.project_id == self.project_id:
            # かわりませんでした
            self.run_miner(miner)
        elif escort_project and old_miner_project.project_id == escort_project.project_id:
            # 護衛と交代しました
            defender = old_miner
            self.run_miner(miner)
        else:  # MineProject 同士で交換しました
            self.run_miner(miner, forced=True)
            if reserved:
                old_miner_project.run_miner(old_miner)
            else:
                pass  # old_miner は向こうのrunに任せる

        if escort_project:  # old_miner に対する defender も一緒に動かしてしまう
            escort_project.schedule()  # miner を join_project してから実行すること
            escort_project.run()
            defender_ship_id = escort_project.defender_ship_id
            defender = a.board.ships.get(defender_ship_id, None)
            a.log(f'miner={miner.id} old_miner={old_miner.id} defender={defender_id}')
            if defender:
                a.log(id_=defender_ship_id, s=f'{defender.position} hired by prj={self.project_id}, old_miner{old_miner.id}{old_miner.position} escort_project{escort_project.project_id} prio{escort_project.priority} defprj={a.belonging_project.get(defender_ship_id, None)}')
            a.log(id_=old_miner.id, s=f'{old_miner.position} hire defender{defender_ship_id} to escort prio{escort_project.priority} defprj={a.belonging_project.get(defender_ship_id, None)}')
        self.elapsed_steps += 1
        return True
# MineProject end


class ExpeditionProject(Project):
    """
    オイシイ土地へ遠征する
    現状CONVERTする前提
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_position = None
        self.early_game_threshold = 50
        self.halite_threshold = None
        self.mine_threshold_before_convert = 50.0
        self.last_converted_position = None
        self.ship_per_neighbor_positions = [None] * LEN_MOVE
        self.shipyard_halite_threshold = 1500.
    
    def schedule(self):
        super().schedule()
        a = self.agent
        self.maintain_dead_staffs()
        len_ships = len(a.board.current_player.ships)
        len_shipyards = len(a.board.current_player.shipyards)
        max_len_opponent_ships = np.max([len(a.board.players[player_id].ships) for player_id in range(PLAYERS)])
        if self.last_converted_position:
            shipyard = a.board.cells[self.last_converted_position].shipyard
            if shipyard and shipyard.player_id == a.player_id:
                # 前ターンにこの ExpeditionProject によって作った
                project_id = f'defense_yd{shipyard.id}'
                project = a.projects.get(project_id, None)
                if project:
                    budget = self.budget
                    self.reserve_budget(-self.budget)
                    project.reserve_budget(min(a.free_halite, MAX_HALITE))
                    a.log(id_=shipyard.id, s=f'hand over budget {self.project_id}={budget} -> {project_id}={project.budget}')
                else:
                    a.log(loglevel='warning', s=f'prj={self.project_id} hand over failed. {project_id} not found yd{shipyard.id}{shipyard.position}')
                    pass
            # 一旦解散
            return False
        if a.board.step < 2 or 300 <= a.board.step:
            return False
        if a.board.step < a.last_shipyard_attacked_step + 50:
            self.priority = -1.0  # 陥落したので控える
        elif (2 <= a.board.step < self.early_game_threshold) and (len_shipyards <= 1):
            self.priority = 1e8
        elif (2 <= a.board.step < 200) and (len_shipyards <= 2) and (len_shipyards * 7 <= len_ships):
            self.priority = 1e8
        elif (self.early_game_threshold <= a.board.step) and (len_shipyards * 6 <= len_ships) and (25000. < a.world_halite or 300.0 < a.halite_per_ship):
            # 敵が狩りタイプばっかりだと中盤起こる
            self.priority = 1e8
        elif len_shipyards <= 10 and len_shipyards * 4 < len_ships and max_len_opponent_ships <= len_ships:
            self.priority = 1e6
        elif self.ships:
            self.priority = 1e6
        else:
            self.priority = -1.0
        # if 0.0 < self.priority and 3 <= len_shipyards:
            # 4軒目以降は控えめ
            # self.reserve_budget(-self.budget)
            # self.priority = 100.0
        # self.halite_threshold = max(1400., 0.8 * a.best_scores[I_SCORE_HALITE_D4])
        return True

    def is_mining_convert_position(self):
        # convert予定地のmineを開始しているか
        mining_convert_position = False
        a = self.agent
        if self.target_position and a.previous_board:
            cell = a.board.cells[self.target_position]
            if cell.ship and cell.ship.player_id == a.player_id:
                previous_cell = a.previous_board.cells[self.target_position]
                mining_convert_position = (cell.halite < previous_cell.halite)
        return mining_convert_position

    def expedition_ships_generator(self):
        a = self.agent
        for ship in a.sorted_ships:
            if a.determined_ships.get(ship.id, None) is not None:
                continue
            project_id = a.belonging_project.get(ship.id, None)
            if project_id is None:
                yield ship
                continue
            elif project_id[:4] == 'hunt':
                continue
            elif project_id[:10] == 'defense_yd':
                continue
            yield ship

    def search_target_position(self):
        """
        self.target_positionを設定
        """
        a = self.agent
        queue = Queue()
        visited = np.zeros((2, COLS, ROWS), dtype=np.bool)
        best_score = -1.0
        best_position = None
        best_d = None
        best_ship = None
        for ship in self.expedition_ships_generator():
            is_empty_ship = 1 if ship.halite == 0 else 0
            queue.put((ship.position, 0, is_empty_ship, ship))

        budget_sufficient = (MAX_HALITE * 2 <= a.free_halite + self.budget)

        # 幅優先探索
        sup_d = 11
        best_score = 1.0
        best_position = None
        best_ship = None
        best_d = 0
        min_best_d = None
        while not queue.empty():
            position, d, is_empty_ship, ship = queue.get()
            if sup_d <= d:
                break
            if visited[is_empty_ship, position[0], position[1]]:
                continue
            visited[:is_empty_ship+1, position[0], position[1]] = True
            score = self.calculate_shipyard_position_score(position, min_best_d, d, is_empty_ship)
            if best_score < score:
                best_score = score
                best_position = position
                best_ship = ship
                best_d = d
                if min_best_d is None:
                    min_best_d = d
                a.log(f'prj={self.project_id} score_updated. d{best_d} score{score:.1f} {position} s{best_ship.id} h{best_ship.halite}')
            for x2, y2, d2 in neighbor_positions(1, position):
                d3 = d + d2
                if (sup_d <= d3) or visited[is_empty_ship, x2, y2]:
                    continue
                queue.put(((x2, y2), d3, is_empty_ship, ship))
        self.set_target_position(best_position)
        if best_ship:
            best_ship_id = best_ship.id
            self.join_project(staff_ids=[best_ship.id], role='leader', forced=True)
        else:
            best_ship_id = None
        self.d_leader = best_d
        a.log(s=f'search_target_position={best_position}, leader={best_ship_id} d={best_d} score={best_score}')

    def set_target_position(self, target_position=None):
        self.target_position = target_position

    def calculate_shipyard_position_score(self, position, min_best_d, d, is_empty_ship):
        a = self.agent
        i, j = position_to_ij(position)
        if (not is_empty_ship) or a.board.step < self.early_game_threshold:
            opponent_reach = int(1e-6 + a.scores[I_SCORE_OPPONENT_REACH, i, j])
            if opponent_reach - 2 <= d:
                return 0.0  # 敵に邪魔されず CONVERT, SPAWN できる距離限定
        if 0.5 < a.scores[I_SCORE_ALLY_SHIPYARD_D7, i, j]:
            return 0.0  # 近所に作るのはやめよう
        if 0 < a.len_shipyards:
            if 0.5 < a.scores[I_SCORE_OPPONENT_SHIPYARD_D6, i, j]:
                return 0.0  # 近所に作るのはやめよう
        score_candidates = a.scores[I_SCORE_SHIPYARD_CANDIDATES, i, j]
        future_hunt_zone = a.scores[I_SCORE_FUTURE_HUNT_ZONE, i, j]
        score_hunt_zone = 1.0 + 0.15 * future_hunt_zone
        if min_best_d is None:
            score_gamma = 1.0
        else:
            score_gamma = 0.9 ** (d - min_best_d)
        score_opponent_shipyard_d4 = 1 / max(1.0, a.scores[I_SCORE_OPPONENT_SHIPYARD_D4, i, j])
        score = score_candidates * score_hunt_zone * score_gamma * score_opponent_shipyard_d4
        # a.log(f'prj={self.project_id} {position} d{d} sc_c{score_candidates:.0f} sc_hz{score_hunt_zone:.1f} sc_g{score_gamma:.5f} sc_opyd{score_opponent_shipyard_d4:.0f} sc{score:.1f}')

        if score < 0.6 * self.shipyard_halite_threshold:
            return 0.0  # 周囲の halite が足りないなら避けよう
        elif score < self.shipyard_halite_threshold:
            # 微妙なライン
            if 200 < a.board.step:
                return 0.0
            if 0.5 < a.scores[I_SCORE_OPPONENT_SHIPYARD_D4, i, j]:
                return 0.0  # 敵の近所に作るのはやめよう
            if -10.5 < a.scores[I_SCORE_DETOUR_ADVANTAGE, i, j]:
                return 0.0  # 遠征だけにしておこう
        elif 0.5 < a.scores[I_SCORE_OPPONENT_SHIPYARD_D3, i, j]:
            return 0.0  # 敵の近所に作るのはやめよう
        return score

    def leader_ship_strategy(self, ship, d):
        a = self.agent
        a.log(id_=ship.id, s=f'prj={self.project_id} leader d{d}')
        condition = None

        if d == 0:  # convert予定地
            i_, j_ = position_to_ij(ship.position)
            opponent_reach = int(1e-6 + a.scores[I_SCORE_OPPONENT_REACH, i_, j_])
            safely_convert_without_spawn = 0
            for ship_id_c in self.ships.keys():
                if ship_id_c == ship.id:
                    continue
                d_c = a.ally_ship_distances[ship.id][ship_id_c]
                if (d_c < opponent_reach) or (
                    (d_c == opponent_reach) and (0 == a.board.ships[ship_id_c].halite)):
                    safely_convert_without_spawn = 1
                    break
            sufficient_budget = ((2 - safely_convert_without_spawn) * MAX_HALITE <= self.budget + ship.halite)
            if ((a.board.cells[ship.position].halite < self.mine_threshold_before_convert
                    or a.scores[I_SCORE_REACH_ADVANTAGE, i_, j_] < 2.5)
                    and sufficient_budget):
                self.reserve_budget(-max(0, MAX_HALITE - ship.halite))
                a.reserve_ship(ship, ShipAction.CONVERT)
                self.last_converted_position = ship.position
                condition = 'convert'
            else:
                a.moving_ship_strategy(ship, position=self.target_position, mode='mine', mine_threshold=4.0)
                condition = 'target_reached_mine'
        else:
            a.moving_ship_strategy(ship, position=self.target_position, mode='escape', mine_threshold=None)
            condition = 'move_to_target'
        if self.last_converted_position:
            i_action = I_CONVERT
        else:
            i_action = I_MINE
            self.ship_per_neighbor_positions[i_action] = ship
        a.log(id_=ship.id, s=f'prj={self.project_id} leader {self.target_position} a{i_action} cond({condition})')

    def coworker_ship_strategy(self, ship, d, leader_ship):
        a = self.agent
        best_position = None
        to_sort = []
        for k_action, cell_k in enumerate(a.neighbor_cells(a.board.cells[self.target_position])):
            if self.ship_per_neighbor_positions[k_action] is not None:
                continue
            d_k = calculate_distance(ship.position, cell_k.position)
            steps_k = d_k - d + 2  # 遠いほど掘れるstep数は減る
            mine_threshold_k = None
            halite_k = None
            priority = None
            if cell_k.halite < 1e-6:
                priority = 0.0
            else:
                mine_project = a.projects.get(f'mine_{cell_k.position[0]}_{cell_k.position[1]}', None)
                if mine_project:
                    mine_threshold_k = mine_project.halite_threshold
                    if mine_project.ships:
                        priority = -100.0
                        mine_threshold_k = None
                else:
                    mine_threshold_k = 40.0
                if mine_threshold_k:
                    halite_k = cell_k.halite - mine_threshold_k
                    priority = halite_k / steps_k
            if k_action == 0:
                priority = -1.0
                if self.last_converted_position:  # leaderはこのstepでconvertしてくる
                    if 0 < ship.halite:
                        priority = 10003.0  # convertと同時にdepositしたい
                    else:  # 同時ガード必要?
                        i_, j_ = position_to_ij(cell_k.position)
                        opponent_reach = int(1e-6 + a.scores[I_SCORE_OPPONENT_REACH, i_, j_])
                        if d == 1 and opponent_reach <= 1:
                            priority = 10001.0
                        elif d <= opponent_reach and (self.budget + leader_ship.halite < 2 * MAX_HALITE):
                            priority = 10002.0
            a.log(id_=ship.id, s=f'prj={self.project_id} coworker k{k_action}{cell_k.position} prio{priority} h{cell_k.halite} halite_k{halite_k} steps_k{steps_k} budget{self.budget}')
            to_sort.append([priority, k_action, mine_threshold_k, cell_k])
        if not to_sort:
            priority = None
            target_position = self.target_position
            k_action = I_MINE
            mine_threshold_k = None
        else:
            priority, k_action, mine_threshold_k, cell_k = sorted(to_sort, key=itemgetter(0, 1), reverse=True)[0]
            target_position = cell_k.position
        self.ship_per_neighbor_positions[k_action] = ship
        if (ship.position == target_position and mine_threshold_k is not None):
            mode = 'mine'
        else:
            mode = 'escape'
        preference, forced = a.calculate_moving_ship_preference(
                ship, position=target_position, mode=mode, mine_threshold=mine_threshold_k)
        q = np.ones(LEN_MOVE) * preference
        a.reserve_ship_by_q(ship=ship, q=q, forced=forced, depend_on=leader_ship)
        a.log(id_=ship.id, s=f'prj={self.project_id} coworker {self.target_position}{k_action}->{target_position} prio{priority}')

    
    def run(self):
        super().run()
        self.last_converted_position = None
        if self.priority < 0.0:
            return False
        a = self.agent
        # 予算(CONVERT & SPAWN)と人員の確保
        # if len(a.board.current_player.shipyards) <= 2:
        self.reserve_budget(min(a.free_halite, 2 * MAX_HALITE - self.budget))
        # else:  # 4軒目以降は保守的
            # self.reserve_budget(-self.budget)
            # if a.free_halite < 4 * MAX_HALITE:
                # return True
        self.dismiss_project(staff_ids=list(self.ships.keys()))
        self.search_target_position()
        if self.target_position is None:
            return False
        len_staffs = len(self.ships)
        to_sort = []
        for ship in self.ships_generator(with_free=True):
            d = calculate_distance(self.target_position, ship.position)
            if 0 < d and 0 < ship.halite and self.early_game_threshold <= a.board.step:
                continue  # 中盤以降は原則 empty_ship
            if self.d_leader + 5 < d:  # あんまり遠いならスルー
                continue
            to_sort.append((d, -ship.halite, ship.id, ship))
        max_staffs = 1 if a.board.step < 50 else 2
        to_sort = sorted(to_sort)[:max_staffs - len_staffs]
        staff_ids = list(map(itemgetter(2), to_sort))
        a.log(s=f'prj={self.project_id} target_position={self.target_position} ships={list(self.ships.keys())}+{staff_ids}')

        # ここから実際に作戦を実行
        leader_ship = None
        for ship_id, role in self.ships.items():
            if role == 'leader':
                leader_ship = a.board.ships.get(ship_id, None)
                self.leader_ship_strategy(ship=leader_ship, d=self.d_leader)
                break
        a.log(s=f'prj={self.project_id} leader{leader_ship.id} staff_ids{staff_ids}')
        self.join_project(staff_ids=staff_ids, role='coworker')
        self.ship_per_neighbor_positions = [None] * (LEN_MOVE + 1)
        for i, (d, negative_halite, ship_id, ship) in enumerate(to_sort):
            self.coworker_ship_strategy(ship=ship, d=d, leader_ship=leader_ship)
        return True

class HuntProject(Project):
    """狩り"""
    def __init__(self, target_ship_id, *args, **kwargs):
        super().__init__(*args, project_id=f'hunt{target_ship_id}', **kwargs)
        self.target_ship_id = target_ship_id
        self.max_staffs = 6
        self.max_d = 3
        self.center_direction = []

    def schedule(self):
        super().schedule()
        a = self.agent
        self.target_ship = a.board.ships.get(self.target_ship_id, None)
        if self.target_ship is None:  # もう死んだ
            return False
        self.priority = 1e4
        if self.target_ship.halite == 0:
            self.priority = -1.0  # 生きている限りproject自体は存続

        self.opponent_safe_direction = 0x1F
        p0 = self.target_ship.position
        cell = a.board.cells[p0]
        cells = a.neighbor_cells(cell)
        for k_action, cell_k in enumerate(cells):  # 敵にとって安全な移動はあるか?
            for l_action, cell_l in enumerate(a.neighbor_cells(cell_k)):
                ship_l = cell_l.ship
                if (ship_l is None) or ship_l.player_id == self.target_ship.player_id:
                    continue
                if ship_l.halite < self.target_ship.halite:
                    self.opponent_safe_direction &= ~(1 << k_action)
                    break
        if self.opponent_safe_direction < 0:
            defender_count = 0  # 周囲の護衛が多かったらあきらめる
            for x_k, y_k, d_k in neighbor_positions(d=2, p=self.target_ship.position):
                cell_k = a.board.cells[x_k, y_k]
                ship_k = cell_k.ship
                if (ship_k is None) or (ship_k.player_id != self.target_ship.player_id) or (0 < ship_k.halite):
                    continue
                defender_count += 1
            if 2 <= defender_count:
                self.priority = -1.0


        self.dx_limit = [-self.max_d, self.max_d]
        self.dy_limit = [-self.max_d, self.max_d]
        # indexを守るこちらのshipにとって、左右どちらへ行きがちかの情報
        # 例: self.opponent_tend_to_move[I_SOUTH] == I_WEST の場合、南南西に敵shipyardがあるイメージ
        self.opponent_tend_to_move = [None] * LEN_MOVE
        for shipyard in a.board.players[self.target_ship.player_id].shipyards:
            dx = rotated_diff_position(self.target_ship.position[0], shipyard.position[0]) 
            dy = rotated_diff_position(self.target_ship.position[1], shipyard.position[1]) 
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            if abs_dx <= abs_dy:
                if dy <= 0 and self.dy_limit[0] <= dy + 1:
                    # 南へ行きたい
                    self.dy_limit[0] = dy + 1
                    if dx < 0:
                        self.opponent_tend_to_move[I_SOUTH] = I_WEST
                    elif 0 < dx:
                        self.opponent_tend_to_move[I_SOUTH] = I_EAST
                if 0 <= dy and dy - 1 <= self.dy_limit[1]:
                    # 北へ行きたい
                    self.dy_limit[1] = dy - 1
                    if dx < 0:
                        self.opponent_tend_to_move[I_NORTH] = I_WEST
                    elif 0 < dx:
                        self.opponent_tend_to_move[I_NORTH] = I_EAST
            if abs_dy <= abs_dx:
                if dx <= 0 and self.dx_limit[0] <= dx + 1:
                    # 西へ行きたい
                    self.dx_limit[0] = dx + 1
                    if dy < 0:
                        self.opponent_tend_to_move[I_WEST] = I_SOUTH
                    elif 0 < dy:
                        self.opponent_tend_to_move[I_WEST] = I_NORTH
                if 0 <= dx and dx - 1 <= self.dx_limit[1]:
                    # 東へ行きたい
                    self.dx_limit[1] = dx - 1
                    if dy < 0:
                        self.opponent_tend_to_move[I_EAST] = I_SOUTH
                    elif 0 < dy:
                        self.opponent_tend_to_move[I_EAST] = I_NORTH
        if not self.assign_candidates():  # 再構成
            self.priority = -1.0

        if self.priority < 0.0:
            self.dismiss_project(staff_ids=list(self.ships.keys()))
        return True

    def assign_candidates(self):
        if self.priority < 0.0:
            return False
        previous_roles = copy.deepcopy(self.ships)

        def get_ship_info_for_debug(a_):
            return list(map(lambda u: f'{u.id}{u.position}', a_))

        def get_ship_info_for_debug_2(a_):
            a2 = []
            for t in a_:
                a2.append(get_ship_info_for_debug(t))
            return a2

        def get_ship_info_for_debug_3_1(a_):
            return list(map(lambda u: f'{u[0].id}{u[0].position}', a_))

        def get_ship_info_for_debug_3(a_):
            a2 = []
            for t in a_:
                a2.append(get_ship_info_for_debug_3_1(t))
            return a2

        a = self.agent
        self.dismiss_project(staff_ids=list(self.ships.keys()))

        p0 = self.target_ship.position
        count = 0
        dp_index = [0] * 9
        candidates = [[] for _ in range(9)]
        # limitターン敵が全力で一方向へ逃げた時に追いつくか
        north_position = Point(
                x=p0[0],
                y=mod_map_size_x(p0[1] + self.dy_limit[1]))
        east_position = Point(
                x=mod_map_size_x(p0[0] + self.dx_limit[1]),
                y=p0[1])
        south_position = Point(
                x=p0[0],
                y=mod_map_size_x(p0[1] + self.dy_limit[0]))
        west_position = Point(
                x=mod_map_size_x(p0[0] + self.dx_limit[0]),
                y=p0[1])
        a.log(id_=self.target_ship_id, s=f'{p0} n{north_position} e{east_position} s{south_position} w{west_position} dx_limit{self.dx_limit} dy_limit{self.dy_limit}')
        for ship in a.sorted_ships:
            # hunt assign が現状最優先なので、他のproject事情を無視する
            if a.determined_ships.get(ship.id, None):
                continue  # もう行動されていたらさすがに覆さない
            project_id = a.belonging_project.get(ship.id, None)
            if project_id and project_id != self.project_id and project_id[:4] == 'hunt':
                continue  # 他の hunt project
            if self.target_ship.halite <= ship.halite:
                continue
            dx = rotated_diff_position(p0[0], ship.position[0])
            dy = rotated_diff_position(p0[1], ship.position[1])
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            m = 0
            d_north = calculate_distance(north_position, ship.position)
            if d_north <= self.dy_limit[1]:
                m |= (1 << I_NORTH)
            d_east = calculate_distance(east_position, ship.position)
            if d_east <= self.dx_limit[1]:
                m |= (1 << I_EAST)
            d_south = calculate_distance(south_position, ship.position)
            if d_south <= abs(self.dy_limit[0]):
                m |= (1 << I_SOUTH)
            d_west = calculate_distance(west_position, ship.position)
            if d_west <= abs(self.dx_limit[0]):
                m |= (1 << I_WEST)
            previous_role = int(previous_roles.get(ship.id, -1))
            original_m = m
            im = None
            if 0 < m:
                # 前回やっていた守備範囲を尊重する
                if I_NORTH <= previous_role and (m & (1 << previous_role)):
                    m = (1 << previous_role)

                im = DIRECTION_MAPPING[m]
                if im < I_NORTH_EAST:
                    dp_index[im] = 1
                else:
                    dp_index[im] = min(2, dp_index[im] + 1)
                candidates[im].append(ship)
                count += 1
            # a.log(id_=self.target_ship_id, s=f'assign_candidates 0 s{ship.id}{ship.position} dn{d_north} de{d_east} ds{d_south} dw{d_west} m{m} origm{original_m} prerole{previous_role} im{im}')
        # a.log(id_=self.target_ship_id, s=f'assign_candidates 1 count{count} candidates{get_ship_info_for_debug_2(candidates)}')
        if count < 5:
            return False
        # 斜めshipの方角割り当ては超面倒なのでHUNT_DPに事前計算してある
        diag_distribution = HUNT_DP[dp_index[1], dp_index[2], dp_index[3], dp_index[4], dp_index[5], dp_index[6], dp_index[7], dp_index[8]]
        # a.log(id_=self.target_ship_id, s=f'assign_candidates 2 diag_distribution{diag_distribution}')
        if diag_distribution[0] == HUNT_IMPOSSIBLE:
            return False

        # 8方向から4方向へ削減する
        candidates2 = [[] for _ in range(LEN_MOVE)]
        for direction, ships in enumerate(candidates):
            if not ships:
                continue
            ships_d =  [(ship, calculate_distance(p0, ship.position)) for ship in ships]
            if direction < I_NORTH_EAST:
                candidates2[direction] += ships_d
                continue
            diag = direction - I_NORTH_EAST
            strategy = diag_distribution[diag]
            if strategy == 5:  # 双方向
                if len(ships) <= 1:
                    a.log(loglevel='warning', s=f'strategy == 5. len(ships)={len(ships)}')
                candidates2[DIAG_DIRECTIONS[diag][0]].append(ships_d[0])
                candidates2[DIAG_DIRECTIONS[diag][1]] += ships_d[1:]
            else:
                candidates2[strategy] += ships_d

        # [5, 8] ships 決定する
        key_fn = lambda t: t[1] * 10000 + t[0].halite
        min_d = 99999
        min_direction = None
        to_sort = []
        for direction in range(1, LEN_MOVE):
            c = sorted(candidates2[direction], key=key_fn)
            len_c = len(c)
            if 0 == len_c:
                c1 = get_ship_info_for_debug_2(candidates)
                c2 = get_ship_info_for_debug_3(candidates2)
                a.log(logloevel='warning', s=f'direction={direction} ship candidate not found target{self.target_ship.position} c1={c1} c2={c2} diag_distribution={diag_distribution} dp_index={dp_index}')
                continue
            for j in range(2):
                if len_c <= j:
                    break
                ship_j, d_j = c[j]
                pre_project = a.belonging_project.get(ship_j.id, None)
                self.join_project(staff_ids=[ship_j.id], role=str(direction), forced=True)
                post_project = a.belonging_project.get(ship_j.id, None)
                # a.log(id_=self.target_ship_id, s=f'assign_candidates 3 {ship_j.id}{ship_j.position} prerole{previous_roles.get(ship_j.id, -1)} role{self.ships.get(ship_j.id)} preprj={pre_project} postprj={post_project}')
                to_sort.append([ship_j, d_j, ship_j.id])
                if 0 < j:  # 遠いほうのshipがカバーできる方から中央へ攻める
                    if d_j < min_d:
                        min_d = d_j
                        min_direction = direction
        if min_direction is None:
            c1 = get_ship_info_for_debug_2(candidates)
            c2 = get_ship_info_for_debug_3(candidates2)
            a.log(loglevel='warning', s=f'min_ship not found. c1={c1} c2={c2}')
        self.center_direction = int(min_direction)
        self.sorted_ships = sorted(to_sort, key=itemgetter(1, 2))
        def ships_to_log():
            a_ = []
            for ship_id, role in self.ships.items():
                a_.append(f'{ship_id}{a.board.ships[ship_id].position}={role}')
            return a_
        a.log(id_=self.target_ship_id, s=f'assign_candidates 4 target{self.target_ship.id}{self.target_ship.position} ships{ships_to_log()} center{self.center_direction}')
        return True

    def run(self):
        super().run()
        if self.priority < 0.0:
            return True
        a = self.agent
        p0 = self.target_ship.position
        cell = a.board.cells[p0]
        cells = a.neighbor_cells(cell)
        center_ship_id = None
        opponent_safe_direction = self.opponent_safe_direction
        positions = []
        checkmate_count = 0
        for ship, d, ship_id in self.sorted_ships:
            role_s = self.ships.get(ship_id, None)
            if role_s is None:  # EscortProject final にとられていることがある
                continue
            role = int(role_s)
            i, j = position_to_ij(ship.position)
            cell_r = cells[role]
            get_halite = int((1e-6 + cell_r.halite) * 0.25)
            last_stop = self.target_ship.halite <= ship.halite + get_halite
            p1 = cell_r.position  # 1手詰めの目標地点
            p2 = p1  # それ以外の目標地点
            p_checkmate = cell_r.position  # p1 は片方軸合わせしただけのことがあるので checkmate 判定では使えない
            should_challenge = False
            # tend = self.opponent_tend_to_move[role]
            mask_ns = (1 << I_NORTH) | (1 << I_SOUTH)
            mask_ew = (1 << I_EAST) | (1 << I_WEST)
            if role == I_NORTH:
                if ship.position[0] == p1[0]:  # x座標はあっている
                    if ship.position[1] != p1[1] or 0 == get_halite or (0 < ship.halite and (not last_stop)):
                        pass  # 普通に目標地点向かって距離詰めたり停止すればOK
                    elif (opponent_safe_direction & mask_ew) == (1 << I_EAST):  # 東は安全だから東行くっしょ
                        p2 = cell_r.east.position
                    elif (opponent_safe_direction & mask_ew) == (1 << I_WEST):  # 西は安全だから西行くっしょ
                        p2 = cell_r.west.position
                    elif self.dy_limit[1] <= 1: # こちらが追い詰められている
                        should_challenge = True
                    else:  # 後退して時間稼ぎ
                        p2 = cell_r.north.position
                else:  # 先にx座標を合わせる
                    p1 = Point(x=p0[0], y=ship.position[1])
                    p2 = p1
            elif role == I_EAST:
                if ship.position[1] == p1[1]:  # y座標はあっている
                    if ship.position[0] != p1[0] or 0 == get_halite or (0 < ship.halite and (not last_stop)):
                        pass  # 普通に目標地点向かって距離詰めたり停止すればOK
                    elif (opponent_safe_direction & mask_ns) == (1 << I_NORTH):  # 北は安全だから北行くっしょ
                        p2 = cell_r.north.position
                    elif (opponent_safe_direction & mask_ns) == (1 << I_SOUTH):  # 南は安全だから南行くっしょ
                        p2 = cell_r.south.position
                    elif self.dx_limit[1] <= 1: # こちらが追い詰められている
                        should_challenge = True
                    else:  # 後退して時間稼ぎ
                        p2 = cell_r.east.position
                else:  # 先にy座標を合わせる
                    p1 = Point(x=ship.position[0], y=p0[1])
                    p2 = p1
            elif role == I_SOUTH:
                if ship.position[0] == p1[0]:  # x座標はあっている
                    if ship.position[1] != p1[1] or 0 == get_halite or (0 < ship.halite and (not last_stop)):
                        pass  # 普通に目標地点向かって距離詰めたり停止すればOK
                    elif (opponent_safe_direction & mask_ew) == (1 << I_EAST):  # 東は安全だから東行くっしょ
                        p2 = cell_r.east.position
                    elif (opponent_safe_direction & mask_ew) == (1 << I_WEST):  # 西は安全だから西行くっしょ
                        p2 = cell_r.west.position
                    elif -1 <= self.dy_limit[0]: # こちらが追い詰められている
                        should_challenge = True
                    else:  # 後退して時間稼ぎ
                        p2 = cell_r.south.position
                else:  # 先にx座標を合わせる
                    p1 = Point(x=p0[0], y=ship.position[1])
                    p2 = p1
            elif role == I_WEST:
                if ship.position[1] == p1[1]:  # y座標はあっている
                    if ship.position[0] != p1[0] or 0 == get_halite or (0 < ship.halite and (not last_stop)):
                        pass  # 普通に目標地点向かって距離詰めたり停止すればOK
                    if (opponent_safe_direction & mask_ns) == (1 << I_NORTH):  # 北は安全だから北行くっしょ
                        p2 = cell_r.north.position
                    elif (opponent_safe_direction & mask_ns) == (1 << I_SOUTH):  # 南は安全だから南行くっしょ
                        p2 = cell_r.south.position
                    elif -1 <= self.dx_limit[0]: # こちらが追い詰められている
                        should_challenge = True
                    else:  # 後退して時間稼ぎ
                        p2 = cell_r.west.position
                else:  # 先にy座標を合わせる
                    p1 = Point(x=ship.position[0], y=p0[1])
                    p2 = p1
            if (center_ship_id is None) and (role == self.center_direction):
                center_ship_id = ship.id
                if ship.position == p1:
                    p1 = p0
                    p2 = p0
                    p_checkmate = p0
                    self.ships[ship.id] = str(I_MINE)
            d_r = calculate_distance(ship.position, p_checkmate)
            if d_r <= 1:
                bit_flag = (1 << int(self.ships[ship.id]))
                checkmate_count |= bit_flag
                # a.log(id_=self.target_ship_id, s=f'ship{ship.id}{ship.position} bit_flag{bit_flag} checkmate{checkmate_count}')
            positions.append((ship, p1, p2, d_r, last_stop, should_challenge))
        for ship, p1, p2, d_r, last_stop, should_challenge in positions:
            if 0x1F == checkmate_count or should_challenge:
                p = p1
                mine_threshold = 3.99
            else:
                p = p2
                if d_r <= 1 and 0 < ship.halite and (not last_stop):
                    mine_threshold = 3.99
                else:
                    mine_threshold = None
            a.log(id_=self.target_ship_id, s=f'p0{p0} s{ship.id}{ship.position}->p1{p1}/p2{p2} checkmate{checkmate_count} p{p} role{self.ships[ship.id]} center{self.center_direction} center_ship{center_ship_id} should_challenge={should_challenge} safe_dir{opponent_safe_direction}')
            a.moving_ship_strategy(ship, position=p, mode='cancel_without_shipyard', mine_threshold=mine_threshold)
        return True




class MyAgent(object):
    def __init__(self, player_id, *args, verbose, **kwargs):
        self.player_id = player_id
        self.verbose = verbose
        self.flags = FlagsManager()
        self.scores = np.zeros((N_SCORE_TYPES, ROWS, COLS), dtype=np.float32)
        self.best_scores = np.zeros(N_SCORE_TYPES, dtype=np.float32)
        self.best_score_cells = [None] * N_SCORE_TYPES
        self.initial_phase_step = 20
        self.spawn_step_threshold = 200 # + 50 * self.player_id
        self.greedily_spawn_step_threshold = 50 # + 50 * self.player_id
        self.len_ships_threshold = 57
        self.len_ships_threshold_weak = 99 #3  # weak用
        self.opponent_history = [
                {
                    'defense_against_shipyard_attack': [0, 0],
                    'deposit_against_shipyard_attack': [0, 0],
                    'cancel_against_shipyard_attack': [0, 0],
                    'stop_shipyard_neighbor': [0, 0],
                    # cancel (相殺) 系は empty_ship だけ計上する FUGA
                    'cancel_any': [0, 0],
                    'cancel_both_move_to_mine': [0, 0],  # おいしい土地かつ敵が真上にいない所に突っ込むか
                    'cancel_move_to_mining_opponent': [0, 0],  # 敵が掘っているところへ突っ込む
                    'cancel_to_mine_here': [0, 0],  # 掘るために相殺覚悟
                    'cancel_with_rob_chance': [0, 0],  # rob優先した結果相殺したっぽい
                    'shipyard_attacked': [0, 0],  # 実際にshipyard破壊された回数
                    } for i in range(4)]
        self.last_shipyard_attacked_step = -999
        self.opponent_history_queue = []
        self.board = None
        self.previous_board = None
        self.previous_len_opponent_ships = 3
        self.belonging_project = {}  # key is ship_id or shipyard_id, value is project_id
        self.projects = {}  # key is project_id, value is Project
        self.log_step = -1
        self.logs = {}
        self.reserving_ships = {}

    def dismiss_project(self, project_id, staff_ids):
        assert isinstance(project_id, str)
        project = self.projects.get(project_id, None)
        if project is None:
            return
        # staffを解雇
        for staff_id in staff_ids:
            assert isinstance(staff_id, str)
            self.belonging_project[staff_id] = None
            if staff_id in project.ships:
                del project.ships[staff_id]
            if staff_id in project.shipyards:
                del project.shipyards[staff_id]

    def join_project(self, project_id, staff_ids, role='no_role', forced=False):
        assert isinstance(project_id, str)
        project = self.projects.get(project_id, None)
        if project is None:
            return
        for staff_id in staff_ids:
            assert isinstance(staff_id, str)
            previous_project_id = self.belonging_project.get(staff_id, None)
            if (previous_project_id is not None) and (project_id != previous_project_id):
                if not forced:
                    self.log(loglevel='warning', s=f'p{self.player_id} staff_id={staff_id} join to {project_id} but it already belongs to {self.belonging_project[staff_id]}')
                previous_project = self.projects.get(previous_project_id, None)
                if previous_project:
                    previous_project.dismiss_project(staff_ids=[staff_id])
            self.belonging_project[staff_id] = project_id
            maybe_ship = self.board.ships.get(staff_id, None)
            if maybe_ship:
                project.ships[staff_id] = role
            maybe_shipyard = self.board.shipyards.get(staff_id, None)
            if maybe_shipyard:
                project.shipyards[staff_id] = role

    def log(self, s, step=None, id_=None, indent=0, loglevel='DEBUG'):
        if not self.verbose:
            return
        level = getattr(logging, loglevel.upper())
        if level < logging.DEBUG:
            return
        prefix = ''
        if 0 < indent:
            prefix += ' ' * indent
        if step is None:
            step = self.board.step
        prefix += f'step{step} '
        if self.log_step != step:
            self.logs.clear()
            self.log_step = step
        if id_ is not None:
            prefix += f'id{id_} '
            if id_ not in self.logs:
                self.logs[id_] = []
            if (self.verbose & 4) == 4:
                self.logs[id_].append(f'{prefix}{s}')
        
        if ((self.verbose & 4) == 4) or ((self.verbose & 1) == 1 and logging.DEBUG < level):
            easy_log(f'{prefix}{s}', loglevel=loglevel)
            

    def update_opponent_history(self):
        if self.previous_board is None:
            self.log(step=self.board.step, id_=None, s=f'update_opponent_history: previous_board is None')
            return  # 初手は処理しない

        # defense_against_shipyard_attack, deposit_against_shipyard_attack
        # stop_shipyard_neighbor
        for previous_shipyard in self.previous_board.shipyards.values():
            player_id = previous_shipyard.player_id
            position = previous_shipyard.position
            previous_cell = self.previous_board.cells[position]
            cell = self.board.cells[position]
            shipyard = cell.shipyard
            result = None  # 0: 守ってない,  1: 守っている, None: 無効
            depositor_result = None  # 0: depositしなかった, 1: deposit強行した, None: 無効かdepositorいない
            cancel_result = None  # 0: 安全に相殺できるのにしなかった 1: 相殺した None: その他
            shipyard_attacked_result = None  # 0=1: shipyard_attackで破壊されたshipyard数
            attackers = []
            empty_attackers = []
            dead_empty_attackers = []
            defender_candidates = []
            defenders = []
            depositors = []
            dead_depositors = []
            min_attacker_halite = 99999
            for previous_cell_i in self.neighbor_cells(previous_cell):
                ship = previous_cell_i.ship
                if ship is None:
                    pass
                elif ship.player_id == player_id:
                    defender_candidates.append(ship)
                else:
                    attackers.append(ship.id)
                    current_ship = self.board.ships.get(ship.id, None)
                    if ship.halite == 0:
                        empty_attackers.append(ship.id)
                        if current_ship:
                            attacker_result = 1 if (current_ship.position == ship.position) else 0
                            self.opponent_history[ship.player_id]['stop_shipyard_neighbor'][0] += attacker_result
                            self.opponent_history[ship.player_id]['stop_shipyard_neighbor'][1] += 1
                        else:
                            dead_empty_attackers.append(ship.id)
                    min_attacker_halite = min(ship.halite, min_attacker_halite)
            if 0 == len(attackers):  # 脅威などなかった
                continue
            for candidate in defender_candidates:
                if candidate.halite <= min_attacker_halite:
                    defenders.append(candidate.id)
                else:
                    depositors.append(candidate.id)
                    if self.board.ships.get(candidate.id, None) is None:
                        dead_depositors.append(candidate.id)
            if shipyard is None:
                result = 0
                if dead_depositors:
                    depositor_result = 1  # 正確ではないが多分そうでしょ
                elif depositors:
                    depositor_result = 0
                shipyard_attacked_result = 1
            elif dead_empty_attackers:
                # 相殺した どちらが攻めたのかが問題となる
                if cell.ship:  # shipyard側が攻めたの確定
                    cancel_result = 1
                    # depositors が帰還しているなら守っているわけではない
                    if cell.ship.id in depositors:
                        result = 0
                        depositor_result = 1
                    else:  # 守りも盤石
                        result = 1
                        if depositors:
                            depositor_result = 0
                else:  # どこで相殺したか不明 意味合いも変わるので放置
                    pass
            elif cell.ship:
                # attackerに動きなし 膠着状態か, deposit強行したか
                if cell.ship.id in depositors:
                    result = 0
                    depositor_result = 1
                else:  # 戻ったのはdefender
                    result = 1
                    if depositors:  # deposit できるチャンスの時だけ統計加算
                        depositor_result = 0
            else: # attackerに動きなし shipyardは守られていない
                result = 0
                if depositors:
                    depositor_result = 0  # 少なくとも deposit はしてない
            if (len(defenders) == 0) and self.previous_board.players[player_id].halite < MAX_HALITE:
                # halite 足りないときは spawn できないに決まっているので無視
                result = None
            if result is not None:
                # 分子: 守った回数
                self.opponent_history[player_id]['defense_against_shipyard_attack'][0] += result
                # 分母: 試行回数
                self.opponent_history[player_id]['defense_against_shipyard_attack'][1] += 1
            if depositor_result is not None:
                self.opponent_history[player_id]['deposit_against_shipyard_attack'][0] += depositor_result
                self.opponent_history[player_id]['deposit_against_shipyard_attack'][1] += 1
            if cancel_result is not None:
                self.opponent_history[player_id]['cancel_against_shipyard_attack'][0] += cancel_result
                self.opponent_history[player_id]['cancel_against_shipyard_attack'][1] += 1
            if shipyard_attacked_result is not None:
                self.opponent_history[player_id]['shipyard_attacked'][0] += shipyard_attacked_result
                self.opponent_history[player_id]['shipyard_attacked'][1] += 1
                if player_id == self.player_id:
                    self.last_shipyard_attacked_step = self.board.step

        for ship_id, previous_ship in self.previous_board.ships.items():
            ship = self.board.ships.get(ship_id, None)
            player_id = previous_ship.player_id
            if ship is None:
                self.log(id_=ship_id, s=f'{previous_ship.position} h{previous_ship.halite} p{player_id} dead')
                pass
            if 0 < previous_ship.halite:
                continue  # ひとまず empty_ship だけ調べる
            # 分母は cancel おきえたか (merge, shipyard_attackは考えない)
            can_cancel = [False] * LEN_MOVE
            can_rob = [False] * LEN_MOVE
            ground_halite = np.zeros(LEN_MOVE, dtype=np.float32)
            empty_opponent = np.zeros(LEN_MOVE, dtype=np.bool)
            move_to_cancel_position = False
            move_to_rob_position = False
            mine_here = False
            i_action = None
            i_action_candidates = np.zeros(LEN_MOVE, dtype=np.bool)  # ship いなくなってしまった時に相殺対象がいうる方向
            for k_action, cell_k0 in enumerate(self.neighbor_cells(self.previous_board.cells[previous_ship.position])):
                ground_halite[k_action] = cell_k0.halite
                for l_action, cell_l0 in enumerate(self.neighbor_cells(self.previous_board.cells[cell_k0.position])):
                    if not cell_l0.ship:
                        continue
                    if cell_l0.ship.player_id == player_id:
                        continue
                    if cell_l0.ship.halite == 0:
                        can_cancel[k_action] = True
                        if l_action == 0:
                            empty_opponent[k_action] = True
                    else:
                        can_rob[k_action] = True
                    ship_l1 = self.board.ships.get(cell_l0.ship.id, None)
                    if ship_l1 is None:
                        i_action_candidates[k_action] = True

                # 現在board
                cell_k1 = self.board.cells[cell_k0.position]

                if k_action == 0:
                    if (not cell_k0.shipyard) and (cell_k1.shipyard):
                        i_action = I_CONVERT
                elif cell_k0.ship and cell_k0.ship.player_id != player_id and cell_k0.ship.halite == 0:
                    empty_opponent[k_action] = True

                if cell_k1.ship and cell_k1.ship.id == ship_id:
                    i_action = k_action

            if i_action == I_CONVERT:
                continue
            if ship is None:
                if not any(i_action_candidates):
                    continue  # mergeとかshipyard_attack(含spawn相殺)と思われる
                # 複数候補あり正確なところがわからないので、保守的な戦略をとっている
                # (なにかメリットあるところへ相殺しにいった) と仮定する
                move_to_cancel_position = True
                move_to_rob_position = any(can_rob)
            else:
                i_action_candidates[:] = False
                i_action_candidates[i_action] = True
                move_to_cancel_position = can_cancel[i_action]
                move_to_rob_position = can_rob[i_action]

            for k_action in range(LEN_MOVE):
                # cancel しうる場所にしか興味ない
                if not can_cancel[k_action]:
                    can_rob[k_action] = False
                    ground_halite[k_action] = 0.0

            if any(can_cancel):
                self.opponent_history[player_id]['cancel_any'][1] += 1
                if move_to_cancel_position:
                    # self.log(id_=ship_id, s=f'can_cancel{can_cancel} a{i_action}')
                    self.opponent_history[player_id]['cancel_any'][0] += 1

            if any(can_rob):
                self.opponent_history[player_id]['cancel_with_rob_chance'][1] += 1
                if move_to_cancel_position:
                    # self.log(id_=ship_id, s=f'can_rob{can_rob} a{i_action}')
                    self.opponent_history[player_id]['cancel_with_rob_chance'][0] += 1

            if 1e-6 < ground_halite[I_MINE]:
                self.opponent_history[player_id]['cancel_to_mine_here'][1] += 1
                if i_action_candidates[I_MINE]:
                    self.opponent_history[player_id]['cancel_to_mine_here'][0] += 1

            ground_halite[I_MINE] = 0.0  # あとは移動だけなので用済み
            # 分母増加の halite threshold は保守的にしておく
            has_ground_halite = 200.0 < np.array(ground_halite, dtype=np.float32)
            if np.any(has_ground_halite):
                # 敵が掘ろうとしているところへ突っ込む
                t = np.logical_and(has_ground_halite, empty_opponent)
                if np.any(t):
                    self.opponent_history[player_id]['cancel_move_to_mining_opponent'][1] += 1
                    if np.any(np.logical_and(t, i_action_candidates)):
                        self.opponent_history[player_id]['cancel_move_to_mining_opponent'][0] += 1

                # 敵はまだいないので先取りしようとした
                t = np.logical_and(has_ground_halite, np.logical_not(empty_opponent))
                if np.any(t):
                    self.opponent_history[player_id]['cancel_both_move_to_mine'][1] += 1
                    if np.any(np.logical_and(t, i_action_candidates)):
                        self.opponent_history[player_id]['cancel_both_move_to_mine'][0] += 1

    def update_projects(self):
        """
        前 step から継続している project の継続判断
        新規projectの立ち上げ
        """
        self.free_staff_ids = {}
        deleting_project_ids = []

        # まず予算だけ与えてしまう
        project_key_fn = attrgetter('priority', 'project_id')
        projects = sorted(list(self.projects.values()), key=project_key_fn, reverse=True)
        for project in projects:
            self.free_halite -= project.budget
            if self.free_halite < 0 or project.budget < 0:  # バグっていそう
                self.log(loglevel='warning', s=f'free_halite < 0 project_id={project.project_id}, project.budget={project.budget} halite={self.board.current_player.halite} free_halite={self.free_halite} discard')
                project.discard()

        # 新規projectの立ち上げ

        # HuntProject, EscortProject
        for ship in self.board.ships.values():
            if ship.player_id == self.player_id:
                project_id = f'escort{ship.id}'
                class_ = EscortProject
            else:
                project_id = f'hunt{ship.id}'
                class_ = HuntProject
            if project_id not in self.projects:
                project = class_(target_ship_id=ship.id, agent=self)
                self.projects[project_id] = project

        # MineProject
        if self.board.step == 0:
            for x in range(COLS):
                for y in range(ROWS):
                    cell = self.board.cells[x, y]
                    if cell.halite < 1e-6:
                        continue
                    project = MineProject(position=cell.position, agent=self)
                    self.projects[project.project_id] = project

        # ExpeditionProject
        project_id = 'expedition0'
        if project_id not in self.projects:
            project = ExpeditionProject(project_id=project_id, agent=self)
            self.projects[project_id] = project

        # DefenseShipyardProject, RestrainShipyardProject
        for shipyard_id, shipyard in self.board.shipyards.items():
            project_id = self.belonging_project.get(shipyard_id, None)
            if project_id is not None:
                continue
            if shipyard.player_id == self.player_id:
                project = DefenseShipyardProject(
                        agent=self,
                        shipyard_id=shipyard_id)
            else:
                continue  # 無いほうが相殺減って強そう
                project = RestrainShipyardProject(
                        agent=self,
                        shipyard_id=shipyard_id)
            project_id = project.project_id
            if self.projects.get(project_id, None):
                continue  # もうあった
            self.projects[project_id] = project

        # 一斉に schedule
        projects = sorted(list(self.projects.values()), key=project_key_fn, reverse=True)
        for project in projects:
            if not project.schedule():
                project.discard()
        return sorted(self.projects.values(), key=project_key_fn, reverse=True)

    def update_score_impl(self, score_type, *, p, d, max_score=1e9):
        i0, j0 = position_to_ij(p)
        for i1, j1, d1 in NEIGHBOR_POSITIONS[d - 1][i0, j0]:
            t = self.scores[score_type, i1, j1]
            self.scores[score_type, i1, j1] = min(max_score, t + d - d1)

    def split_ranges(self, i0, i1, di):
        results_out = self.split_ranges_impl(i0, i1, di)
        if 2 <= di:
            i0_ = mod_map_size_x(i0 + 1)
            i1_ = mod_map_size_x(i1 - 1)
            results_in = self.split_ranges_impl(i0_, i1_, di - 2)
        elif di <= -2:
            i0_ = mod_map_size_x(i0 - 1)
            i1_ = mod_map_size_x(i1 + 1)
            results_in = self.split_ranges_impl(i0_, i1_, di + 2)
        else:
            results_in = []
        return results_in, results_out
            
    def split_ranges_impl(self, i0, i1, di):
        """トーラスなのでスライスを分割しないといけないことがある"""
        results = []
        if 0 <= di:
            if i1 < i0:  # 画面端
                results.append(slice(i0, MAP_SIZE))
                results.append(slice(0, i1 + 1))
            else:
                results.append(slice(i0, i1 + 1))
        elif i0 < i1:  # 画面端
            results.append(slice(i1, MAP_SIZE))
            results.append(slice(0, i0 + 1))
        else:
            results.append(slice(i1, i0 + 1))
        return results

    def has_internal(self, i_k, j_k, i_l, j_l, di_kl, dj_kl, player_id_k):
        for shipyard_id_m, shipyard_m in self.board.shipyards.items():
            if shipyard_m.player_id == player_id_k:
                continue
            i_m, j_m = position_to_ij(shipyard_m.position)
            di_km = rotated_diff_position(i_k, i_m)
            dj_km = rotated_diff_position(j_k, j_m)
            ok = [False, False]
            if 0 <= di_kl:
                if 0 <= di_km <= di_kl:
                    ok[0] = True
            elif di_kl <= di_km <= 0:
                ok[0] = True
            if 0 <= dj_kl:
                if 0 <= dj_km <= dj_kl:
                    ok[1] = True
            elif dj_kl <= dj_km <= 0:
                ok[1] = True
            if np.all(ok):
                return True
        return False

    def update_score_danger_zone(self):
        """敵shipyard2箇所を頂点とする四角形内は挟み撃ちされる危険性が高い"""
        shipyards = self.board.shipyards
        self.scores[I_SCORE_DANGER_ZONE, ...] = 0.0
        d_threshold = 5
        for shipyard_id_k, shipyard_k in shipyards.items():
            i_k, j_k = position_to_ij(shipyard_k.position)
            for shipyard_id_l, shipyard_l in shipyards.items():
                if shipyard_id_l <= shipyard_id_k:
                    continue
                if shipyard_k.player_id != shipyard_l.player_id:
                    continue
                i_l, j_l = position_to_ij(shipyard_l.position)
                di_kl = rotated_diff_position(i_k, i_l)
                dj_kl = rotated_diff_position(j_k, j_l)
                if d_threshold < abs(di_kl) or d_threshold < abs(dj_kl):
                    continue
                if self.has_internal(i_k, j_k, i_l, j_l, di_kl, dj_kl, shipyard_k.player_id):
                    continue
                ranges_in_i, ranges_out_i = self.split_ranges(i_k, i_l, di_kl)
                ranges_in_j, ranges_out_j = self.split_ranges(j_k, j_l, dj_kl)
                # self.log(f'i_k{i_k} j_k{j_k} i_l{i_l} j_l{j_l} di_kl{di_kl} dj_kl{dj_kl} ranges_i{ranges_i} ranges_j{ranges_j}')
                if shipyard_k.player_id == self.player_id:
                    i_score = I_SCORE_HUNT_ZONE
                else:
                    i_score = I_SCORE_DANGER_ZONE
                for range_i in ranges_out_i:
                    for range_j in ranges_out_j:
                        self.scores[i_score, range_i, range_j] += 1.0
                if shipyard_k.player_id == self.player_id:
                    i_score = I_SCORE_HUNT_ZONE_IN
                else:
                    i_score = I_SCORE_DANGER_ZONE_IN
                for range_i in ranges_in_i:
                    for range_j in ranges_in_j:
                        self.scores[i_score, range_i, range_j] += 1.0
        self.scores_initialized[I_SCORE_DANGER_ZONE] = True
        self.scores_initialized[I_SCORE_HUNT_ZONE] = True
        self.scores_initialized[I_SCORE_DANGER_ZONE_IN] = True
        self.scores_initialized[I_SCORE_HUNT_ZONE_IN] = True
        self.scores_initialized[I_SCORE_FUTURE_HUNT_ZONE] = True
        self.scores[I_SCORE_FUTURE_HUNT_ZONE, ...] = 0.0
        my_shipyards = self.board.current_player.shipyards
        if 10 <= len(my_shipyards):
            return
        visited = np.zeros((ROWS, COLS), dtype=np.bool)
        valid1 = self.scores[I_SCORE_HUNT_ZONE] < 0.5
        valid2 = self.scores[I_SCORE_OPPONENT_SHIPYARD_D4] < 0.5
        valid = np.logical_and(valid1, valid2)
        # valid には IN でなく OUT 使ってみる
        for i_k in range(ROWS):
            for j_k in range(COLS):
                visited[...] = False
                cell_k = self.board.cells[ij_to_position(i_k, j_k)]
                if cell_k.shipyard:
                    continue  # 重複
                if 0.5 < self.scores[I_SCORE_OPPONENT_SHIPYARD_D4, i_k, j_k]:
                    continue  # 敵から近すぎ
                has_near_shipyard = False
                for shipyard_l in my_shipyards:
                    i_l, j_l = position_to_ij(shipyard_l.position)
                    di_kl = rotated_diff_position(i_k, i_l)
                    abs_di_kl = abs(di_kl)
                    if not (2 <= abs_di_kl <= d_threshold):
                        continue
                    dj_kl = rotated_diff_position(j_k, j_l)
                    abs_dj_kl = abs(dj_kl)
                    if not (2 <= abs_dj_kl <= d_threshold):
                        continue
                    if abs_di_kl + abs_dj_kl <= 3:
                        has_near_shipyard = True
                        break
                    if self.has_internal(i_k, j_k, i_l, j_l, di_kl, dj_kl, self.player_id):
                        continue
                    ranges_in_i, ranges_out_i = self.split_ranges(i_k, i_l, di_kl)
                    ranges_in_j, ranges_out_j = self.split_ranges(j_k, j_l, dj_kl)
                    for range_i in ranges_in_i:
                        for range_j in ranges_in_j:
                            visited[range_i, range_j] = 1
                if not has_near_shipyard:
                    self.scores[I_SCORE_FUTURE_HUNT_ZONE, i_k, j_k] = np.sum(np.logical_and(visited, valid))

    def update_score_detour_distance(self):
        """I_SCORE_OPPONENT_SHIPYARD_D2を利用する
        self.nearest_ally_shipyard,
        I_SCORE_DTOUR_REACH,
        I_FLAG_GO_HOME_STRAIGHT を更新
        """
        assert self.scores_initialized[I_SCORE_OPPONENT_SHIPYARD_D2]
        assert self.scores_initialized[I_SCORE_DANGER_ZONE]
        self.nearest_ally_shipyard = [[None for j in range(COLS)] for i in range(ROWS)]
        shipyards = self.board.shipyards
        queue = []
        opponent_queue = []
        for shipyard_k in shipyards.values():
            i_k, j_k = position_to_ij(shipyard_k.position)
            if shipyard_k.player_id == self.player_id:
                heapq.heappush(queue, (0, i_k, j_k, shipyard_k.id))
            else:
                heapq.heappush(opponent_queue, (0, i_k, j_k, shipyard_k.id))
        self.scores[I_SCORE_DETOUR_REACH, ...] = 99999.0
        self.scores[I_SCORE_OPPONENT_DETOUR_REACH, ...] = 99999.0
        self.scores[I_SCORE_DETOUR_ADVANTAGE, ...] = 99999.0
        self.flags.reset_all(I_FLAG_GO_HOME_STRAIGHT)
        while queue:
            d_k, i_k, j_k, shipyard_id_k = heapq.heappop(queue)
            if self.scores[I_SCORE_DETOUR_REACH, i_k, j_k] < 99999.:
                continue
            shipyard_k = shipyards[shipyard_id_k]
            if self.nearest_ally_shipyard[i_k][j_k] is None:
                self.nearest_ally_shipyard[i_k][j_k] = shipyard_k
            self.scores[I_SCORE_DETOUR_REACH, i_k, j_k] = d_k
            i_yd, j_yd = position_to_ij(shipyard_k.position)
            d_straight_k = calculate_distance((i_k, j_k), (i_yd, j_yd))
            if d_straight_k == d_k or d_straight_k <= 1:
                # self.log(f'i_k{i_k} j_k{j_k} d_k{d_k} d_straight_k{d_straight_k}')
                self.flags.set(I_FLAG_GO_HOME_STRAIGHT, i=i_k, j=j_k)

            for i_l, j_l, d_l in neighbor_positions(d=1, p=(i_k, j_k)):
                if d_l == 0:
                    continue
                if self.scores[I_SCORE_DETOUR_REACH, i_l, j_l] < 99990.:
                    continue
                new_d = d_k
                if 0.5 < self.scores[I_SCORE_OPPONENT_SHIPYARD_D2, i_l, j_l]:
                    new_d += 14  # nash均衡が大体7%で期待値このぐらい?
                else:
                    new_d += d_l
                danger_zone = int(1e-6 + self.scores[I_SCORE_DANGER_ZONE, i_l, j_l])
                if danger_zone:
                    new_d += 3 * danger_zone # 通る前提のルートどりをすべきでない
                # self.log(f'i_k{i_k} j_k{j_k} d_k{d_k} d_straight_k{d_straight_k} new_d{new_d} i_l{i_l} j_l{j_l}')
                heapq.heappush(queue, (new_d, i_l, j_l, shipyard_id_k))

        while opponent_queue:
            d_k, i_k, j_k, shipyard_id_k = heapq.heappop(opponent_queue)
            if self.scores[I_SCORE_OPPONENT_DETOUR_REACH, i_k, j_k] < 99999.:
                continue
            shipyard_k = shipyards[shipyard_id_k]
            self.scores[I_SCORE_OPPONENT_DETOUR_REACH, i_k, j_k] = d_k
            i_yd, j_yd = position_to_ij(shipyard_k.position)

            for i_l, j_l, d_l in neighbor_positions(d=1, p=(i_k, j_k)):
                if d_l == 0:
                    continue
                if self.scores[I_SCORE_OPPONENT_DETOUR_REACH, i_l, j_l] < 99990.:
                    continue
                if 0.5 < self.scores[I_SCORE_ALLY_SHIPYARD_D1, i_l, j_l]:
                    new_d = d_k + 14
                else:
                    new_d = d_k + d_l
                heapq.heappush(opponent_queue, (new_d, i_l, j_l, shipyard_id_k))
        self.scores[I_SCORE_DETOUR_ADVANTAGE, ...] = self.scores[I_SCORE_OPPONENT_DETOUR_REACH, ...] - self.scores[I_SCORE_DETOUR_REACH, ...]
        self.scores_initialized[I_SCORE_DETOUR_REACH] = True
        self.scores_initialized[I_SCORE_OPPONENT_DETOUR_REACH] = True
        self.scores_initialized[I_SCORE_DETOUR_ADVANTAGE] = True

        self.ships_by_shipyard = {shipyard.id: {} for shipyard in self.board.current_player.shipyards}
        for ship in self.board.current_player.ships:
            i_, j_ = position_to_ij(ship.position)
            nearest_ally_shipyard = self.nearest_ally_shipyard[i_][j_]
            if nearest_ally_shipyard:
                self.ships_by_shipyard[nearest_ally_shipyard.id][ship.id] = ship


    def update_score(self):
        self.flags.reset_all(slice(I_FLAG_NEXT_SHIP_POSITION, None))
        self.scores[I_SCORE_EMPTY_OPPONENT_D2:, ...] = 0.0
        self.best_scores[I_SCORE_EMPTY_OPPONENT_D2] = 0.0
        self.best_score_cells = [None] * N_SCORE_TYPES
        self.scores_initialized = np.zeros(N_SCORE_TYPES, dtype=np.bool)
        
        # 地面の halite

        # まずは普通に抽出
        for x in range(COLS):
            for y in range(ROWS):
                i, j = position_to_ij((x, y))
                self.scores[I_SCORE_HALITE, i, j] = self.board.cells[x, y].halite
        self.scores_initialized[I_SCORE_HALITE] = True
        self.world_halite = np.sum(self.scores[I_SCORE_HALITE])
        self.halite_per_ship = self.world_halite / max(1, len(self.board.ships))

        # I_SCORE_SURROUNDED_HALITE_GROUND を更新する
        self.scores[I_SCORE_SURROUNDED_HALITE_GROUND, ...] = 0.0
        b = (1e-9 < self.scores[I_SCORE_HALITE]).astype(np.float32)
        for di, dj, d in neighbor_d_positions(d=1):
            # 対称なので di, dj も dx, dy も変わらん
            if d == 0:
                continue
            self.scores[I_SCORE_SURROUNDED_HALITE_GROUND] += np.roll(
                np.roll(b, shift=di, axis=0),
                shift=dj, axis=1)
        self.scores_initialized[I_SCORE_SURROUNDED_HALITE_GROUND] = True

        self.scores[I_SCORE_SHIPYARD_CANDIDATES_SUB, ...] = self.scores[I_SCORE_HALITE]
        # 他の shipyard 影響範囲は複数shipyardsあると複数回乗算あり
        # 自分の shipyard は特に範囲広いとみなす
        i_scores = [None, None, None,
                np.full(len(neighbor_d_positions(d=3)), I_SCORE_SHIPYARD_CANDIDATES_SUB),
                None,
                np.full(len(neighbor_d_positions(d=5)), I_SCORE_SHIPYARD_CANDIDATES_SUB),
                ]
        ally_ratio = [0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
        opponent_ratio = [0.0, 0.1, 0.4, 0.7]
        for shipyard in self.board.shipyards.values():
            i, j = position_to_ij(shipyard.position)
            if shipyard.player_id == self.player_id:
                d = 5
                ratio = ally_ratio
            else:
                d = 3
                ratio = opponent_ratio
            ijd = neighbor_positions(d=d, p=(i, j))
            for i_k, j_k, d_k in ijd:
                self.scores[i_scores[d], i_k, j_k] *= ratio[d_k]
        self.scores_initialized[I_SCORE_SHIPYARD_CANDIDATES_SUB] = True

        # 周辺の halite の和
        # 変数名のD4は update_score_impl の d の意味で
        # d=4 の時スコア0 , d=3の時スコア非0
        # neighbor_d_positions については非0部分を指定するので d=3 でよい
        for di, dj, d in neighbor_d_positions(d=3):
            # 対称なので di, dj も dx, dy も変わらん
            self.scores[I_SCORE_HALITE_D4] += np.roll(
                    np.roll(self.scores[I_SCORE_HALITE], shift=di, axis=0),
                    shift=dj, axis=1)
            if d == 0:
                continue  # I_SCORE_SHIPYARD_CANDIDATES に関しては自分の足元はなくす
            self.scores[I_SCORE_SHIPYARD_CANDIDATES] += np.roll(
                    np.roll(self.scores[I_SCORE_SHIPYARD_CANDIDATES_SUB], shift=di, axis=0),
                    shift=dj, axis=1)
        self.scores_initialized[I_SCORE_HALITE_D4] = True
        self.scores_initialized[I_SCORE_SHIPYARD_CANDIDATES] = True
        # 地面の halite 関連終了

        self.scores[I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE, ...] = 99999.0
        for ship_id, ship in self.board.ships.items():
            i, j = position_to_ij(ship.position)
            if ship.player_id == self.player_id:
                self.update_score_impl(I_SCORE_ALLY_D2, p=ship.position, d=2, max_score=1.0)
                self.update_score_impl(I_SCORE_ALLY_D4, p=ship.position, d=4)
                if ship.halite == 0:
                    self.update_score_impl(I_SCORE_EMPTY_ALLY_D4, p=ship.position, d=4)
                else:
                    self.update_score_impl(I_SCORE_NON_EMPTY_ALLY_D4, p=ship.position, d=4)
            else:
                self.update_score_impl(I_SCORE_OPPONENT_D2, p=ship.position, d=2, max_score=1.0)
                if ship.halite == 0:
                    self.update_score_impl(I_SCORE_EMPTY_OPPONENT_D2, p=ship.position, d=2, max_score=1.0)
                else:
                    self.update_score_impl(I_SCORE_NON_EMPTY_OPPONENT_D2, p=ship.position, d=2, max_score=1.0)
                self.update_score_impl(I_SCORE_OPPONENT_D3, p=ship.position, d=3, max_score=1.0)
                for i2, j2, d2 in NEIGHBOR_POSITIONS[1][i, j]:
                    self.scores[I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE, i2, j2] = min(
                            ship.halite, self.scores[I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE, i2, j2])
        self.scores_initialized[I_SCORE_ALLY_D2] = True
        self.scores_initialized[I_SCORE_ALLY_D4] = True
        self.scores_initialized[I_SCORE_EMPTY_ALLY_D4] = True
        self.scores_initialized[I_SCORE_NON_EMPTY_ALLY_D4] = True
        self.scores_initialized[I_SCORE_OPPONENT_D2] = True
        self.scores_initialized[I_SCORE_OPPONENT_D3] = True
        self.scores_initialized[I_SCORE_EMPTY_OPPONENT_D2] = True
        self.scores_initialized[I_SCORE_NON_EMPTY_OPPONENT_D2] = True
        self.scores_initialized[I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE] = True

        # shipyard 関連
        for ship_id, shipyard in self.board.shipyards.items():
            if shipyard.player_id == self.player_id:
                self.update_score_impl(I_SCORE_ALLY_SHIPYARD_D1, p=shipyard.position, d=1)
                self.update_score_impl(I_SCORE_ALLY_SHIPYARD_D4, p=shipyard.position, d=4)
                self.update_score_impl(I_SCORE_ALLY_SHIPYARD_D7, p=shipyard.position, d=7)
            else:
                self.update_score_impl(I_SCORE_OPPONENT_SHIPYARD_D6, p=shipyard.position, d=6)
                self.update_score_impl(I_SCORE_OPPONENT_SHIPYARD_D2, p=shipyard.position, d=2)
                self.update_score_impl(I_SCORE_OPPONENT_SHIPYARD_D3, p=shipyard.position, d=3)
                self.update_score_impl(I_SCORE_OPPONENT_SHIPYARD_D4, p=shipyard.position, d=4)
        self.scores_initialized[I_SCORE_OPPONENT_SHIPYARD_D6] = True
        self.scores_initialized[I_SCORE_ALLY_SHIPYARD_D1] = True
        self.scores_initialized[I_SCORE_ALLY_SHIPYARD_D4] = True
        self.scores_initialized[I_SCORE_ALLY_SHIPYARD_D7] = True
        self.scores_initialized[I_SCORE_OPPONENT_SHIPYARD_D2] = True
        self.scores_initialized[I_SCORE_OPPONENT_SHIPYARD_D3] = True
        self.scores_initialized[I_SCORE_OPPONENT_SHIPYARD_D4] = True
        self.update_score_danger_zone()
        self.update_score_detour_distance()


        for i_score in range(N_SCORE_TYPES):
            ij = np.argmax(self.scores[i_score])
            i, j = ij // COLS, ij % COLS
            self.best_score_cells[i_score] = self.board.cells[ij_to_position(i, j)]
            self.best_scores[i_score] = self.scores[i_score, i, j]
        self.ally_ship_distances = {}  # dict dict, first key: 全staff_id, second key: 味方ship_idのみ
        self.ally_shipyard_distances = {}  # dict dict, first key: 全staff_id, second key: 味方shipyard_idのみ
        self.opponent_ship_distances = {}  # dict dict, first key: 全staff_id, second key: 敵ship_idのみ
        self.opponent_shipyard_distances = {}  # dict dict, first key: 全staff_id, second key: 敵shipyard_idのみ
        staffs = list(self.board.ships.values()) + list(self.board.shipyards.values())
        self.store_distance_into_dict(
                ha=self.ally_ship_distances,
                ho=self.opponent_ship_distances,
                staff_is=staffs,
                staff_js=self.board.ships.values())
        self.store_distance_into_dict(
                ha=self.ally_shipyard_distances,
                ho=self.opponent_shipyard_distances,
                staff_is=staffs,
                staff_js=self.board.shipyards.values())


        # I_SCORE_ALLY_REACH, I_SCORE_EMPTY_ALLY_REACH, I_SCORE_OPPONENT_REACH
        self.ally_ship_positions = []
        self.empty_ally_ship_positions = []
        self.non_empty_ally_ship_positions = []
        self.opponent_ship_positions = []
        self.empty_opponent_ship_positions = []
        self.non_empty_opponent_ship_positions = []
        self.ally_shipyard_positions = []
        self.opponent_shipyard_positions = []
        for ship in self.board.ships.values():
            if ship.player_id == self.player_id:
                self.ally_ship_positions.append(ship.position)
                if ship.halite == 0:
                    self.empty_ally_ship_positions.append(ship.position)
                else:
                    self.non_empty_ally_ship_positions.append(ship.position)
            else:
                self.opponent_ship_positions.append(ship.position)
                if ship.halite == 0:
                    self.empty_opponent_ship_positions.append(ship.position)
                else:
                    self.non_empty_opponent_ship_positions.append(ship.position)
        for shipyard in self.board.shipyards.values():
            if shipyard.player_id == self.player_id:
                self.ally_shipyard_positions.append(shipyard.position)
            else:
                self.opponent_shipyard_positions.append(shipyard.position)
        self.update_score_impl_pq(I_SCORE_ALLY_REACH, initial_ship_positions=self.ally_ship_positions, initial_shipyard_positions=self.ally_shipyard_positions, obstacle_score_type=I_SCORE_OPPONENT_SHIPYARD_D2)
        self.update_score_impl_pq(I_SCORE_EMPTY_ALLY_REACH, initial_ship_positions=self.empty_ally_ship_positions, initial_shipyard_positions=self.ally_shipyard_positions, obstacle_score_type=I_SCORE_OPPONENT_SHIPYARD_D2)
        self.update_score_impl_pq(I_SCORE_NON_EMPTY_ALLY_REACH, initial_ship_positions=self.non_empty_ally_ship_positions, initial_shipyard_positions=[], obstacle_score_type=I_SCORE_OPPONENT_SHIPYARD_D2)
        self.update_score_impl_pq(I_SCORE_OPPONENT_REACH, initial_ship_positions=self.opponent_ship_positions, initial_shipyard_positions=self.opponent_shipyard_positions, obstacle_score_type=I_SCORE_ALLY_SHIPYARD_D1)
        self.update_score_impl_pq(I_SCORE_EMPTY_OPPONENT_REACH, initial_ship_positions=self.empty_opponent_ship_positions, initial_shipyard_positions=self.opponent_shipyard_positions, obstacle_score_type=I_SCORE_ALLY_SHIPYARD_D1)
        self.update_score_impl_pq(I_SCORE_NON_EMPTY_OPPONENT_REACH, initial_ship_positions=self.non_empty_opponent_ship_positions, initial_shipyard_positions=[], obstacle_score_type=I_SCORE_ALLY_SHIPYARD_D1)
        self.scores[I_SCORE_REACH_ADVANTAGE, ...] = self.scores[I_SCORE_OPPONENT_REACH] - self.scores[I_SCORE_ALLY_REACH]
        self.scores_initialized[I_SCORE_ALLY_REACH] = True
        self.scores_initialized[I_SCORE_EMPTY_ALLY_REACH] = True
        self.scores_initialized[I_SCORE_NON_EMPTY_ALLY_REACH] = True
        self.scores_initialized[I_SCORE_OPPONENT_REACH] = True
        self.scores_initialized[I_SCORE_EMPTY_OPPONENT_REACH] = True
        self.scores_initialized[I_SCORE_NON_EMPTY_OPPONENT_REACH] = True
        self.scores_initialized[I_SCORE_REACH_ADVANTAGE] = True

        assert np.all(self.scores_initialized), self.scores_initialized

        # 衝突回避のため、周囲の人口密度をチェックし高いshipから行動する
        self.collision_threat = {}
        for ship in self.board.current_player.ships:
            count = -1  # 自分自身の分は減らす
            for x1, y1, d in neighbor_positions(d=2, p=ship.position):
                cell1 = self.board.cells[x1, y1]
                if cell1.shipyard and cell1.shipyard.player_id != self.player_id:
                    count += 1
                if cell1.ship:
                    if cell1.ship.player_id == self.player_id:
                        count += 1
                    elif cell1.ship.halite <= ship.halite:
                        count += 1
            self.collision_threat[ship.id] = count



    def update_score_impl_pq(self, i_score, initial_ship_positions, initial_shipyard_positions, obstacle_score_type):
        queue = []
        for position in initial_ship_positions:
            i, j = position_to_ij(position)
            heapq.heappush(queue, (0, i, j))
        for position in initial_shipyard_positions:
            i, j = position_to_ij(position)
            heapq.heappush(queue, (0, i, j))
        visited = np.zeros((ROWS, COLS), dtype=np.bool)
        while queue:
            d, i, j = heapq.heappop(queue)
            if visited[i, j]:
                continue
            visited[i, j] = True
            self.scores[i_score, i, j] = d
            for i2, j2, d2 in NEIGHBOR_POSITIONS[1][i, j]:
                if visited[i2, j2]:
                    continue
                new_d = d + 1
                if 0.5 < self.scores[obstacle_score_type, i2, j2]:
                    new_d = d + 14
                heapq.heappush(queue, (new_d, i2, j2))


    def store_distance_into_dict_impl(self, staff_i, staff_js):
        ha = {}
        ho = {}
        for staff_j in staff_js:
            d = calculate_distance(staff_i.position, staff_j.position)
            if staff_j.player_id == self.player_id:
                ha[staff_j.id] = d
            else:
                ho[staff_j.id] = d
        for h in (ha, ho):
            if h:
                h['min'] = min(h.values())
            else:
                h['min'] = 99999
        return ha, ho

    def store_distance_into_dict(self, ha, ho, staff_is, staff_js):
        for staff_i in staff_is:
            ha_i, ho_i = self.store_distance_into_dict_impl(
                    staff_i=staff_i,
                    staff_js=staff_js)
            ha[staff_i.id] = ha_i
            ho[staff_i.id] = ho_i

    def can_spawn(self, shipyard, budget=0, len_ships=None):
        condition = ''
        if len_ships is None:
            len_ships = len(self.board.current_player.ships)

        i, j = position_to_ij(shipyard.position)
        if self.flags.get(I_FLAG_NEXT_SHIP_POSITION, i=i, j=j):
            self.log(id_=shipyard.id, s=f'cannot spawn: I_FLAG_NEXT_SHIP_POSITION')
            return False  # 誰かが帰還する

        # 予算チェック
        budget_level = 0
        if MAX_HALITE <= budget:
            budget_level = 2
            condition += f' enough_budget'
        elif MAX_HALITE * 3 <= self.free_halite + budget:
            budget_level = 2
            condition += f' too_many_free+budget'
        elif MAX_HALITE <= self.free_halite + budget:
            budget_level = 1
            condition += f' enough_free+budget'
        if budget_level < 1:
            self.log(id_=shipyard.id, s=f'cannot spawn: budget_level{budget_level} freeh{self.free_halite} budget{budget}')
            return False

        if self.board.step < self.greedily_spawn_step_threshold:
            self.log(id_=shipyard.id, s=f'can spawn: greedily_spawn_step_threshold{self.greedily_spawn_step_threshold}')
            return True

        budget_threshold = 3

        remaining_steps = 399 - self.board.step
        if remaining_steps < 50:
            self.log(id_=shipyard.id, s=f'cannot spawn: remaining_steps{remaining_steps}')
            return False
        halite_per_ship_threshold = 1000.
        if remaining_steps < 100:
            halite_per_ship_threshold = [180., 220., 300.]
        elif remaining_steps < 150:
            halite_per_ship_threshold = [150., 180., 250.]
        elif remaining_steps < 200:
            halite_per_ship_threshold = [120., 150., 220.]
        else:
            halite_per_ship_threshold = [90., 120., 150.]
        if halite_per_ship_threshold[2] <= self.halite_per_ship:
            budget_threshold -= 1
        elif halite_per_ship_threshold[1] <= self.halite_per_ship:
            pass
        elif halite_per_ship_threshold[0] <= self.halite_per_ship:
            budget_threshold += 1
        else:
            budget_threshold += 2
        condition += f' hps_thre{halite_per_ship_threshold} hps{self.halite_per_ship}'

        len_shipyards = len(self.board.current_player.shipyards)

        score = self.scores[I_SCORE_HALITE_D4, i, j]
        if 2000. < score:
            budget_threshold -= 1
            condition += f' 2000<score{score:.1f}'
        if len_ships < 1:
            self.log(id_=shipyard.id, s=f'can spawn: len_ships{len_ships}')
            return True
        if self.len_ships_threshold <= len_ships:  # timeout対策
            self.log(id_=shipyard.id, s=f'cannot spawn: len_ships{len_ships}')
            return False
        len_ships_threshold = min(57, 5 + remaining_steps // 10)
        if len_ships < len_ships_threshold:
            budget_threshold -= 1
        condition += f' len_ships{len_ships}_thre{len_ships_threshold}'
        ok = budget_threshold <= budget_level
        self.log(id_=shipyard.id, s=f'can{"" if ok else "not"} spawn: budget_level{budget_level} budget_threshold{budget_threshold} {condition}')
        return ok


    def convert_strategy(self):
        """
        オイシイ土地ならconvertする
        緊急回避的にとるconvertは別
        1stepに1箇所限定
        """
        shipyards = self.board.shipyards.values()
        my_shipyards = self.board.players[self.player_id].shipyards  # player経由だとdictでなくlist
        len_my_shipyards = len(my_shipyards)
        # 1箇所目はできる時ならいつでも 2箇所目以降は SPAWN する余裕を持つ
        if len_my_shipyards == 0:
            halite_threshold = MAX_HALITE - self.free_halite
        else:
            # return  # weak用
            halite_threshold = MAX_HALITE * 2 - self.free_halite

        best_ship = None
        best_score = -1
        best_i_action = None
        best_position = None
        for ship in self.sorted_ships:
            if ship.id in self.determined_ships:
                continue
            if ship.halite < halite_threshold:
                continue  # halite 不足
            for i_action, cell in enumerate(self.neighbor_cells(self.board.cells[ship.position])):
                if cell.shipyard:
                    continue  # 既に shipyard がある
                if len_my_shipyards == 0 and 0 < i_action:
                    continue  # 即建てたい
                # if (0 < len_my_shipyards) and (100.0 < cell.halite):
                    # continue  # 先に掘るべし
                i, j = position_to_ij(cell.position)
                if i_action != 0 and self.flags.get(I_FLAG_NEXT_SHIP_POSITION, i=i, j=j):
                    continue  # ぶつかる
                i_score = I_SCORE_OPPONENT_D2 if i_action == 0 else I_SCORE_OPPONENT_D3
                if 0.5 < self.scores[i_score, i, j]:
                    continue  # 即 shipyard_attack されたらもったいない
                surrounding_halite = self.scores[I_SCORE_SHIPYARD_CANDIDATES, i, j]
                if (0 < len_my_shipyards) and (surrounding_halite < 1500.):
                    continue  # 周囲がおいしくない
                if best_score < surrounding_halite:
                    best_score = surrounding_halite
                    best_ship = ship
                    best_i_action = i_action
                    best_position = cell.position
        if not best_ship:
            return
        self.log(step=self.board.step, id_=best_ship.id, s=f'convert_strategy a{best_i_action}')
        if 0 == best_i_action:
            self.reserve_ship(ship=best_ship, next_action=ShipAction.CONVERT)
        else:
            self.moving_ship_strategy(ship=best_ship, position=best_position, mode='escape', mine_threshold=None)

    def reserve_ship(self, ship, next_action, forced=False):
        determined = self.determined_ships.get(ship.id, None)
        if ((ship is None) or (ship.player_id != self.player_id)
                or ((determined is not None) and (determined != 'reserved'))):
            return
        next_action = ShipAction(next_action) if next_action else None
        ship.next_action = next_action
        next_position = calculate_next_position(ship.position, next_action)
        self.determined_ships[ship.id] = (next_action, next_position)
        if next_action == ShipAction.CONVERT:
            self.free_halite -= max(0, MAX_HALITE - ship.halite)
            if self.free_halite < 0:
                self.log(loglevel='warning', s=f's{ship.id} attempt to convert without free_halite={self.free_halite}')
            # scores を更新
            self.update_score_impl(I_SCORE_ALLY_SHIPYARD_D1, p=ship.position, d=1)
            self.update_score_impl(I_SCORE_ALLY_SHIPYARD_D7, p=ship.position, d=7)
        else:
            m = self.flags.get(I_FLAG_NEXT_SHIP_POSITION, x=next_position[0], y=next_position[1])
            if m and (not forced):
                self.log(loglevel='warning', s=f's{ship.id} attempt to set reserved position. position={ship.position}, next_action={next_action}, next_position={next_position}, m={m}')
            else:
                self.log(id_=ship.id, s=f'reserve_ship {ship.position}->{next_position} next_action={next_action} m{m}')
            # self.log(id_=ship.id, s=f'{ship.position} next_action={next_action} reserve_ship_before_flags=\n{self.flags.flags[0]}')
            self.flags.set(I_FLAG_NEXT_SHIP_POSITION, x=next_position[0], y=next_position[1])
            # self.log(id_=ship.id, s=f'{next_position} reserve_ship_after_flags=\n{self.flags.flags[0]}')

    def find_nearest_ally_shipyard(self, position):
        """迂回考慮済み"""
        i_, j_ = position_to_ij(position)
        shipyard = self.nearest_ally_shipyard[i_][j_]
        d = int(1e-6 + self.scores[I_SCORE_DETOUR_REACH, i_, j_])
        return shipyard, d

    def find_leader_ship(self, position):
        if 0 == len(self.sorted_ships):
            return None, 99999
        ship = self.sorted_ships[0]
        d = calculate_distance(position, ship.position)
        return ship, d

    def find_default_cell(self, position):
        p = [(5, 15), (15, 15), (5, 5), (15, 5)][self.player_id]
        return self.board.cells[p], calculate_distance(p, position)

    def find_nearest_shipyard(self, position, player_ids=None):
        best_shipyard = None
        best_d = 99999
        if player_ids is None:
            player_ids = [self.player_id]
        for shipyard_id, shipyard in self.board.shipyards.items():
            if shipyard.player_id not in player_ids:
                continue
            d = calculate_distance(position, shipyard.position)
            if d < best_d:
                best_shipyard = shipyard
                best_d = d
        return best_shipyard, best_d

    def returning_preference(self, ship, shipyard=None):
        distance = None
        if shipyard is None:
            shipyard, distance = self.find_nearest_ally_shipyard(ship.position)
        if shipyard is not None:
            position = shipyard.position
        else:
            leader_ship, distance = self.find_leader_ship(ship.position)
            if leader_ship:
                position = leader_ship.position
            else:
                default_cell, distance = self.find_default_cell(ship.position)
                position = default_cell.position

        q = np.ones(LEN_MOVE, dtype=np.float32)
        q *= preference_move_to(ship.position, position)
        q *= self.calculate_escape_preference(ship)
        return q, shipyard, distance

    def moving_ship_strategy(self, ship, position, mode, mine_threshold=None, q=None):
        q, forced = self.calculate_moving_ship_preference(ship=ship, position=position, mode=mode, mine_threshold=mine_threshold, q=q)
        return self.reserve_ship_by_q(ship, q=q, forced=forced)

    def scheduled_next_action(self, ship_id):
        determined = self.determined_ships.get(ship_id, None)
        if not determined:
            return None  # 仮
        elif determined == 'reserved':
            reserving = self.reserving_ships.get(ship_id, None)
            if not reserving:
                self.log(loglevel='warning', s=f's{ship_id} determined==reserve but reserving not found')
                return None
            # [priority, ship.id, q, ship, forced, depend_on]
            q = np.copy(reserving[2])
            if len(q.shape) == 2:
                depend_on = reserving[5]
                next_action2 = self.scheduled_next_action(depend_on.id)
                i_next_action2 = ship_action_to_int(next_action2)
                i_next_action = np.argmax(q[i_next_action2])
            else:
                i_next_action = np.argmax(q)
                i_next_action2 = None
            assert 0 <= i_next_action < LEN_MOVE, f'i_next_action{i_next_action} q{q} i_next_action2={i_next_action2}'
            return MOVE[i_next_action]
        return determined[0]

    def calculate_moving_ship_preference(self, ship, position, mode, mine_threshold=None, q=None):
        """
        mode
            raw: 敵味方無視してシンプルに最速
            merge: 味方同士の合流上等 終盤用
            cancel: 敵との相殺上等で最速
            cancel_without_shipyard: 敵との相殺上等, shipyardは避ける
            escape: 敵との相殺避けながらできるだけ最速
            mine: 敵を避けながら道中掘れたら掘る
        mine_threshold:
            None: 掘れる位置の停止を回避する
            値: それ以上なら掘る
        return q, forced
        """
        if mine_threshold is None and mode == 'mine':
            self.log(loglevel='warning', s=f'moving_ship_strategy: mine_threshold is None and mode == "mine"')

        

        i_, j_ = position_to_ij(ship.position)
        opponent_reach = int(1e-6 + self.scores[I_SCORE_OPPONENT_REACH, i_, j_])
        d = calculate_distance(ship.position, position)
        preference = preference_move_to(ship.position, position)
        if q is None:
            q = np.ones(LEN_MOVE, dtype=np.float32)
        q *= preference
        if 0 < d and self.board.step % 5 != self.player_id:
            q[0] *= 0.7  # デッドロック対策
        if mode == 'raw':
            i_action = np.argmax(q)
            return q, True
        q_mine = np.ones(LEN_MOVE, dtype=np.float32)
        escape_mode = False
        condition = ''
        for i_action, cell in enumerate(self.neighbor_cells(self.board.cells[ship.position])):
            d_to_target_i = calculate_distance(cell.position, position)
            condition += f' [{i_action}]'
            i2, j2 = position_to_ij(cell.position)
            opponent_halite = int(1e-6 + self.scores[I_SCORE_MIN_NEIGHBOR_OPPONENT_HALITE, i2, j2])

            ally_mining = False
            if cell.ship and cell.ship.id != ship.id and 3.99 < cell.halite and cell.ship.player_id == self.player_id:
                # 味方の邪魔はしない
                next_action = self.scheduled_next_action(cell.ship.id)
                if next_action is None:  # 掘る / 不明
                    q[i_action] *= 0.9
                    condition += f' cellh{cell.halite}_but_ally_mining mine_thre{mine_threshold}'
                    ally_mining = True
                elif next_action == ShipAction.CONVERT and ship.halite == 0:
                    # 今見えているhaliteは消える
                    q_mine[i_action] *= 0.1
                    condition += f' cellh{cell.halite}_but_convert'

            if mine_threshold is None or (mine_threshold < 0.0):
                # 掘りたくない
                if (i_action == 0) and 3.99 < cell.halite:
                    q[i_action] *= 0.01
                    condition += f' evade_to_mine'
            elif (not ally_mining) and 3.99 <= cell.halite and 0.99 < preference[i_action]:
                if mine_threshold <= cell.halite:
                    if i_action == 0:
                        r = 2.0
                        condition += f' want_to_mine'
                    else:
                        r = 1.01
                        condition += f' want_to_move_to_mine'
                else:
                    r = 1.0
                r = r * min(2.0, max(mine_threshold, cell.halite) / max(mine_threshold, 4.0))
                q_mine[i_action] *= r
                condition += f' h{cell.halite}_found_r{r:.3f}_pref{preference}'
            if 1 < d and i_action == 0 and cell.halite < 3.99:
                condition += f' 1<{d}_stop_on_cellh0_sh{ship.halite}_opreach{opponent_reach}'
                if 0 < ship.halite or 1 < opponent_reach:
                    # 目標地点から遠いところの empty で待っても良いことない
                    q[i_action] *= 0.5
            if ((opponent_halite < ship.halite)
                    or (
                        opponent_halite == ship.halite  # 相殺の時の条件は複雑
                        and  # emptyから掘りたい時は相殺もやむなし
                        (not ((ship.halite == 0 and mode == 'mine' and mine_threshold <= cell.halite and i_action == 0) 
                            or (mode in ['cancel', 'cancel_without_shipyard'])))
                        )):
                # 倒されたくない
                q[i_action] *= 0.02
                condition += f' evade_cancel'
                if i_action == 0:
                    escape_mode = True
                    condition += f' escape_mode'
            if (mode != 'mine') and (1e-6 < cell.halite):
                # regenerateさせたいので通らないようにする
                q[i_action] *= 0.99
                condition += f' evade_to_regenerate'
            if cell.shipyard:
                if cell.shipyard.player_id == self.player_id:
                    if 0 < ship.halite:  # deposit しておきたい
                        q[i_action] *= 1.2
                        condition += f' want_to_deposit'
                    elif cell.position[0] != position[0] or cell.position[1] != position[1]:
                        # 目的地でないなら他 ship の deposit の邪魔はしたくない
                        q[i_action] *= 0.8
                        condition += f' evade_my_shipyard'
                elif (mode in ['cancel_without_shipyard', 'escape', 'mine']):
                    # shipyard_attack しない
                    q[i_action] *= 0.1
                    condition += f' no_ydatk'

            if 0 < i_action and 1 < d_to_target_i and self.previous_board:
                previous_cell = self.previous_board.cells[cell.position]
                if previous_cell.ship and previous_cell.ship.id == ship.id:
                    # デッドロックしそうなので避ける
                    q[i_action] *= 0.4
                    condition += f' evade_previous_position'
        self.log(id_=ship.id, s=f'mv prj={self.belonging_project.get(ship.id, None)} cond({condition})')
        if mode == 'merge':
            i_action = np.argmax(q)
            return q, True
        elif mode == 'mine' and (not escape_mode):
            q *= q_mine
        if escape_mode:
            q_escape = self.calculate_escape_preference(ship)
            q *= q_escape
        else:
            q_escape = None
        self.log(id_=ship.id, s=f'mv {ship.position}->{position} mode={mode} mine_th={mine_threshold} q={q}')

        if ship.position[0] == position[0] and ship.position[1] == position[1]:
            # 周囲のpreferenceは帰還を想定しておく
            q1, nearest_shipyard, nearest_shipyard_distance = self.returning_preference(ship)
            q2 = 0.25 * q1 + 0.75  # q値をマイルドに(onesとの内分点)
            q = q2 * q
            self.log(id_=ship.id, s=f'moving_ship_strategy target reached. q1={q1} q2={q2} q={q}')
        return q, False

    def mining_ship_strategy(self, ship, q=None, escape_mode=False, shipyard=None):
        cell = self.board.cells[ship.position]
        # 基本は帰還
        preference, nearest_shipyard, nearest_shipyard_distance = self.returning_preference(ship, shipyard=shipyard)
        if q is None:
            q = np.ones(LEN_MOVE, dtype=np.float32)
        q *= preference
        q_mine = np.ones(LEN_MOVE, dtype=np.float32)
        condition = ''
        for i_action, cell_i in enumerate(self.neighbor_cells(self.board.cells[ship.position])):
            condition += f' [{i_action}]'
            if i_action == 0:
                if 140.0 < cell.halite:  # 掘りたい
                    q_mine[i_action] *= 1.2
                    condition += ' mine'
                elif 1 < nearest_shipyard_distance and cell.halite < 20.0:
                    # 停止したくない
                    q[i_action] *= 0.5
                    condition += ' evade_stop'
            elif self.previous_board:
                previous_cell_i = self.previous_board.cells[cell_i.position]
                if previous_cell_i.ship and previous_cell_i.ship.id == ship.id:
                    # デッドロックしそうなので避ける
                    q[i_action] *= 0.4
                    condition += ' evade_previous_position'
        self.log(step=self.board.step, id_=ship.id, s=f'h{ship.halite} mining pref{preference} q{q} q_mine{q_mine} {condition}')
        return self.reserve_ship_by_q(ship, q=q)

    def reserve_ship_by_q(self, ship, q, forced=False, depend_on=None, priority=None):
        """reserving_shipsに無ければ登録し, determined_shipsにも登録する
        q.shape == (LEN_MOVE,) が通常だが、
        depend_on に ship が設定されている場合、その ship の移動方向に応じて
        q.shape が変化してもよい (LEN_MOVE, LEN_MOVE)
        q[i, j]: depend_on が move i したときにこのshipが move jする preference
        shape変化させず単に決定順を設定するためだけにdepend_onを利用する事も可能
        手順上依存元の方を先にreserve_shipしている前提である
        """
        if self.reserving_ships.get(ship.id, None) is not None:
            return
        if depend_on:
            assert 1 <= len(q.shape) <= 2, q
        else:
            assert len(q.shape) == 1, q
        if priority is None:
            priority = self.calculate_collision_priority(ship)
        self.reserving_ships[ship.id] = [
                priority, ship.id, q, ship, forced, depend_on]
        self.determined_ships[ship.id] = 'reserved'

        # 依存している ship の priority を場合によっては上げる
        t = ship
        priority_t = priority
        u = depend_on
        while u:
            if self.determined_ships.get(u.id, None) is None:
                self.log(loglevel='warning', s=f'{t.id} depend on {u.id}, but {u.id} was not determined')
                break
            # CONVERTだとreserving_shipsには登録されていないよ
            next_reserving_ship = self.reserving_ships.get(u.id, None)
            if not next_reserving_ship:
                break
            priority_u = self.calculate_collision_priority(u)
            if priority_t < priority_u:  # 序列は正常
                break
            priority_u = priority_t + 1
            self.reserving_ships[u.id][0] = priority_u
            next_u = self.reserving_ships[u.id][5]
            t = u
            priority_t = priority_u
            u = next_u

    def calculate_collision_priority(self, ship):
        t = self.reserving_ships.get(ship.id, None)
        if t:
            return t[0]
        return ship.halite + 100 * self.collision_threat[ship.id]

    def calculate_q_convert_threshold(self, h):
        if h < 250:
            q_convert_threshold = 0.0
        elif h < 333:
            q_convert_threshold = 0.09
        elif h < 444:
            q_convert_threshold = 0.19
        elif h < 500:
            q_convert_threshold = 0.29
        elif h < 750:
            q_convert_threshold = 0.39
        elif h < 1000:
            q_convert_threshold = 0.49
        else:
            q_convert_threshold = 0.79
        return q_convert_threshold

    def resolve_ship_by_q(self, ship, q, forced=False, depend_on=None):
        """
        q高い順に降順ソートして順番にreserve試みる
        depend_on に ship が指定されていて q.shapeが2次元の場合、depend_onの移動方向に応じて利用するqを変える
        """
        condition = ''
        if depend_on and len(q.shape) == 2:
            determined = self.determined_ships[depend_on.id]
            if determined is None or determined == 'reserved':
                self.log(loglevel='warning', s=f'reserve_ship_by_q s{ship.id} depend_on{depend_on.id} determined_depend_on={determined}')
                q = q[I_MINE]
            else:
                next_action1, next_position1 = determined
                if next_action1 == ShipAction.NORTH or next_action1 == I_NORTH:
                    q = q[I_NORTH]
                    condition += f' depN'
                elif next_action1 == ShipAction.EAST or next_action1 == I_EAST:
                    q = q[I_EAST]
                    condition += f' depE'
                elif next_action1 == ShipAction.SOUTH or next_action1 == I_SOUTH:
                    q = q[I_SOUTH]
                    condition += f' depS'
                elif next_action1 == ShipAction.WEST or next_action1 == I_WEST:
                    q = q[I_WEST]
                    condition += f' depW'
                else:
                    q = q[I_MINE]
                    condition += f' depM'
        else:
            condition += f' no_dep'
            assert len(q.shape) == 1, q
        i_actions = np.arange(LEN_MOVE)
        cells = self.neighbor_cells(self.board.cells[ship.position])
        ally_shipyard, d_ally_shipyard = self.find_nearest_ally_shipyard(ship.position)
        i_, j_ = position_to_ij(ship.position)
        q_convert_threshold = -99999.0
        danger_zone_in = int(1e-6 + self.scores[I_SCORE_DANGER_ZONE_IN, i_, j_])
        opponent_reach = int(1e-6 + self.scores[I_SCORE_OPPONENT_REACH, i_, j_])
        if (0 < ship.halite) and MAX_HALITE - ship.halite <= self.free_halite:
            h = ship.halite
            ally_shipyard_d4 = self.scores[I_SCORE_ALLY_SHIPYARD_D4, i_, j_]
            ally_shipyard_d7 = self.scores[I_SCORE_ALLY_SHIPYARD_D7, i_, j_]
            opponent_shipyard_d3 = self.scores[I_SCORE_OPPONENT_SHIPYARD_D3, i_, j_]
            t = 0.0
            if (ally_shipyard_d7 < 0.5):
                t += 1.0
            if (ally_shipyard_d4 < 0.5 and danger_zone_in):
                t += 1.5
            if 0.5 < t and opponent_shipyard_d3 < 1.5 and 2 <= opponent_reach:
                # 普通に戦略的にconvertできる
                score = self.scores[I_SCORE_SHIPYARD_CANDIDATES, i_, j_]
                h *= max(1.0, t * score / 1000.)
                condition += f' ydcan{int(score)} t{t:.1f}'
            condition += f' h{h}'
            q_convert_threshold = self.calculate_q_convert_threshold(h)
        condition += f' qconv_thre{q_convert_threshold:.3f} danger_zone{danger_zone_in} op_reach{opponent_reach} d_ally_yd{d_ally_shipyard}'
        if 3 < opponent_reach:  # 1手で死なないのならなにもしない
            q_convert_threshold = -99999.
            condition += f' 3<op_reach'
        elif d_ally_shipyard <= 1:
            q_convert_threshold *= 0.3
            condition += f'<=1'
        elif d_ally_shipyard <= 2:
            q_convert_threshold *= 0.7
            condition += f'==2'
        elif d_ally_shipyard <= 3:
            q_convert_threshold *= 0.9
            condition += f'==3'
        else:
            condition += f'>3'


        if cells[0].shipyard is None and np.max(q[I_NORTH:]) < q_convert_threshold:
            # 囲まれているなら停止が安全か否かにかかわらず早めにconvert
            condition += f' surrounded.'
            self.log(id_=ship.id, s=f'reserve_ship_by_q cond({condition})')
            return self.reserve_ship(ship, ShipAction.CONVERT, forced=forced)

        for i_action, q_i in sorted(zip(i_actions, q), key=itemgetter(1), reverse=True):
            condition += f' [{i_action}]'
            if q_i < q_convert_threshold:
                condition += f'q_i<q_conv_thre.'
                self.log(id_=ship.id, s=f'reserve_ship_by_q cond({condition})')
                return self.reserve_ship(ship, ShipAction.CONVERT, forced=forced)
            action = MOVE[i_action]
            cell = cells[i_action]
            m = self.flags.get(I_FLAG_NEXT_SHIP_POSITION, x=cell.position[0], y=cell.position[1])
            if forced or (not m):
                condition += f' forced{forced} m{m}'
                self.log(id_=ship.id, s=f'reserve_ship_by_q cond({condition})')
                return self.reserve_ship(ship, action, forced=forced)

        if self.board.step < 250:
            # 味方に挟まれたっぽいのでまだ中盤でconvertできるならしておく
            condition += f' no_position.'
            self.log(id_=ship.id, s=f'reserve_ship_by_q cond({condition})')
            return self.reserve_ship(ship, ShipAction.CONVERT, forced=forced)
        condition += 'do_nothing.'
        self.log(id_=ship.id, s=f'reserve_ship_by_q cond({condition})')

        
    def initial_phase_strategy(self, ship, q=None):
        """初動事前計算する場合"""
        return self.empty_ship_strategy(ship, q=q)

    def neighbor_cells(self, cell):
        cells = [cell]
        cells.append(cell.north)
        cells.append(cell.east)
        cells.append(cell.south)
        cells.append(cell.west)
        return cells

    def calculate_escape_preference(self, ship):
        """気合避け"""
        q = np.ones(LEN_MOVE, dtype=np.float32)
        cell = self.board.cells[ship.position]
        min_opponent_halite = np.full((LEN_MOVE, PLAYERS), 99999, dtype=np.int32)
        min_ally_halite = np.full(LEN_MOVE, 99999, dtype=np.int32)
        safe = np.zeros(LEN_MOVE, dtype=np.int32)  # 安全
        maybe_kill = np.zeros(LEN_MOVE, dtype=np.int32)  # この ship が opponent をkillできるかも
        maybe_cancel = np.zeros(LEN_MOVE, dtype=np.int32)  # この ship が opponent と相殺できるかも
        maybe_killed = np.zeros(LEN_MOVE, dtype=np.int32)  # この ship が opponent にkillされるかも
        maybe_kill_opponent = np.zeros((LEN_MOVE, PLAYERS), dtype=np.int32)  # opponent視点 他のopponentもkillできるので一石二鳥
        maybe_cancel_opponent = np.zeros((LEN_MOVE, PLAYERS), dtype=np.int32)  # opponent視点 opponent同士で相殺発生
        maybe_killed_by_opponent = np.zeros((LEN_MOVE, PLAYERS), dtype=np.int32)  # opponent視点 他のopponentにkillされるので行きたくない
        maybe_kill_ally = np.zeros((LEN_MOVE, PLAYERS), dtype=np.int32)  # opponent視点 他のallyもkillできるので一石二鳥
        maybe_cancel_ally = np.zeros((LEN_MOVE, PLAYERS), dtype=np.int32)  # opponent視点 味方が相殺できる
        maybe_killed_by_ally = np.zeros((LEN_MOVE, PLAYERS), dtype=np.int32)  # opponent視点 味方がkillできる
        has_ground_halite = np.zeros(LEN_MOVE, dtype=np.bool)
        killer_opponent_count = np.zeros(LEN_MOVE, dtype=np.float32)
        # opponent_next_positions = {}
        oids = []  # 自分自身を除外した player_id
        motivation = np.full((LEN_MOVE, PLAYERS), -1, dtype=np.int32)
        for player_id in range(PLAYERS):
            if player_id == self.player_id:
                continue
            oids.append(player_id)
        i_, j_ = position_to_ij(cell.position)
        detour_advantage = int(1e-6 + self.scores[I_SCORE_DETOUR_ADVANTAGE, i_, j_])
        neighbor_cells = self.neighbor_cells(cell)
        for k_action, cell_k in enumerate(neighbor_cells):
            if 3.99 < cell_k.halite:
                has_ground_halite[k_action] = True
            if cell_k.shipyard and cell_k.shipyard.player_id != self.player_id:
                # shipyard へ突っ込むのはやめておこう
                q[k_action] *= 0.1
            i_k, j_k = position_to_ij(cell_k.position)
            detour_advantage_k = int(1e-6 + self.scores[I_SCORE_DETOUR_ADVANTAGE, i_k, j_k])
            if detour_advantage_k < detour_advantage:
                q[k_action] *= 0.95  # advantage 増やしに行きたい
            elif detour_advantage < detour_advantage_k:
                q[k_action] *= 1.05  # advantage 増やしに行きたい
        for k_action, cell_k in enumerate(neighbor_cells):
            for l_action, cell_l in enumerate(self.neighbor_cells(cell_k)):
                ship_l = cell_l.ship
                if not ship_l:
                    continue
                elif ship_l.player_id == self.player_id:
                    # 敵から見たリスク評価なので自分自身 (ship_l == ship) も含む
                    min_ally_halite[k_action] = min(
                            min_ally_halite[k_action],
                            ship_l.halite)
                else:
                    # 停止するとhalite手に入るので中央へ突っ込んできがち
                    non_stop_chance_l = (l_action == 0 and ship_l.halite == 0 and has_ground_halite[k_action])
                    non_stop_threat_k = (k_action == 0 and ship_l.halite == 0 and has_ground_halite[l_action])
                    min_opponent_halite[k_action, ship_l.player_id] = min(
                            min_opponent_halite[k_action, ship_l.player_id],
                            ship_l.halite)
                    if ship_l.halite < ship.halite:  # 択状況に応じて期待値を分配
                        if k_action == l_action:  # 寄るなら1通り
                            count = 1.0
                        elif k_action == 0 or l_action == 0:  # 直接隣接している 2択
                            if non_stop_chance_l:
                                count = 0.25
                            elif non_stop_threat_k:
                                count = 0.75
                            else:
                                count = 0.5
                        else:  # ナナメ2択
                            count = 0.5
                        killer_opponent_count[k_action] += count
                        self.log(indent=2, step=self.board.step, id_=ship.id, s=f'  k{k_action} l{l_action} sl{ship_l.id}{ship_l.position} h{ship_l.halite} nscl{non_stop_chance_l} nstk{non_stop_threat_k} count{count:.2f}')
            min_opponent_halite_k = np.min(min_opponent_halite[k_action, ...])
            if ship.halite < min_opponent_halite_k:  # ド安全
                safe[k_action] += 1
                if min_opponent_halite_k < 99999:  # 実際に opponent がいる
                    maybe_kill[k_action] += 1
            elif ship.halite == min_opponent_halite_k:
                maybe_cancel[k_action] += 1
                # maybe_kill は考慮しない
            else:
                maybe_killed[k_action] += 1
                # maybe_kill, maybe_cancel は考慮しない
            for m, n in [[0, 1], [1, 2], [2, 0]]:  # クロス集計
                halite_m = min_opponent_halite[k_action, oids[m]]
                if 99999 <= halite_m:
                    continue  # opponent_ship などない
                # 味方(か自分自身)
                halite_ally = min_ally_halite[k_action]
                if halite_ally < 99999:  # ally は存在する
                    if halite_ally < halite_m:
                        maybe_killed_by_ally[k_action, oids[m]] += 1
                    elif halite_ally == halite_m:
                        maybe_cancel_ally[k_action, oids[m]] += 1
                    else:
                        maybe_kill_ally[k_action, oids[m]] += 1
                halite_n = min_opponent_halite[k_action, oids[n]]
                if halite_n < 99999:  # 別playerのopponentがいる
                    if halite_m < halite_n:  
                        maybe_kill_opponent[k_action, oids[m]] += 1
                        maybe_killed_by_opponent[k_action, oids[n]] += 1
                    elif halite_m == halite_n:
                        maybe_cancel_opponent[k_action, oids[m]] += 1
                        maybe_cancel_opponent[k_action, oids[n]] += 1
                    else:
                        maybe_kill_opponent[k_action, oids[n]] += 1
                        maybe_killed_by_opponent[k_action, oids[m]] += 1
            for opponent_id in oids:
                halite = min_opponent_halite[k_action, opponent_id]
                if 99999 <= halite:  # そもそもいない
                    continue
                killed_count = maybe_killed_by_opponent[k_action, opponent_id] + maybe_killed_by_ally[k_action, opponent_id]
                cancel_count = maybe_cancel_opponent[k_action, opponent_id] + maybe_cancel_ally[k_action, opponent_id]
                kill_count = maybe_kill_opponent[k_action, opponent_id] + maybe_kill_ally[k_action, opponent_id]
                self.log(indent=2, step=self.board.step, id_=ship.id, s=f'k{k_action} oid{opponent_id} h{halite} killed{killed_count} cancel{cancel_count} kill{kill_count}')
                if 0 < killed_count:  # 差し合い
                    motivation[k_action, opponent_id] = 0
                    continue
                if 0 < kill_count:  # このship以外にも複数ターゲットある
                    if 0 < cancel_count:
                        motivation[k_action, opponent_id] = 2
                    else:  # 攻め得
                        motivation[k_action, opponent_id] = 3
                elif 0 < cancel_count: # このship次第
                    motivation[k_action, opponent_id] = 1
                else:
                    self.log(loglevel='warning', s=f's{ship.id} h{ship.halite} calculate_evade_preference: unreachable code: k_action{k_action} opponent_id{opponent_id} counts[{killed_count},{cancel_count},{kill_count}]')
            max_motivation_k = np.max(motivation[k_action])
            # 情報収集終わり
            if safe[k_action]:
                pass
                # if maybe_kill[k_action]:
                    # q[k_action] *= 2.0
            elif maybe_cancel[k_action]:
                if max_motivation_k <= 1:  # 敵にうまみがないので安全
                    q[k_action] *= 0.8
                else:
                    q[k_action] *= 0.7
            elif maybe_killed[k_action]:
                # ここから相対的に有利な活路を見出していかねばならない
                # 敵の人数が薄いところを優先
                if max_motivation_k <= 0:  # 敵は一方的にやられるリスクあり
                    q[k_action] *= 0.7
                elif max_motivation_k <= 1:  # 相殺リスクあり
                    q[k_action] *= 0.6
                elif max_motivation_k <= 2:  # 相殺リスクはあるが, 複数の的がある
                    q[k_action] *= 0.5
                else:  # 敵がノーリスクなので, その中でも薄いところを狙う
                    q[k_action] *= 0.1 / max(0.25, killer_opponent_count[k_action])
        if np.any(q != np.ones(LEN_MOVE, dtype=np.float32)):
            # self.log(step=self.board.step, id_=ship.id, s=f'{ship.position} calculate_escape_preference min_oh=\n{min_opponent_halite} koc{killer_opponent_count} motiv=\n{motivation} q{q}')
            self.log(step=self.board.step, id_=ship.id, s=f'q_esc{q}')
        return q

    def ship_on_shipyard_strategy(self, ship, q=None):
        self.log(step=self.board.step, id_=ship.id, s=f'h{ship.halite} ship_on_shipyard_strategy')
        if q is None:
            q = np.ones(LEN_MOVE, dtype=np.float32)
            q[0] *= 0.6  # デフォルト設定では停止して spawn の邪魔をしたくない
        return self.empty_ship_strategy(ship, q=q)

    def empty_ship_strategy(self, ship, q=None, mine_threshold=1000.):
        min_i = 0
        if q is None:
            q = np.ones(LEN_MOVE, dtype=np.float32) + np.random.rand(LEN_MOVE) * 0.1
        preference, nearest_shipyard, nearest_shipyard_distance = self.returning_preference(ship)
        if nearest_shipyard:
            nearest_shipyard_position = nearest_shipyard.position
        else:
            nearest_shipyard_position = self.sorted_ships[0].position
        isolated_count = 0
        condition = ''
        for i_action, cell in enumerate(self.neighbor_cells(self.board.cells[ship.position])):
            i, j = position_to_ij(cell.position)
            my_empty_score_d4 = 4.0 if i_action == 0 else 3.0
            condition += f' [{i_action}]'
            if cell.shipyard:
                player_id = cell.shipyard.player_id
                if player_id == self.player_id:
                    score = self.scores[I_SCORE_OPPONENT_D2, i, j]
                    if 0.5 < score:  # 防衛しないと
                        q[i_action] *= 10.0
                        condition += ' defense'
                    else:  # 用事ないのに戻るのはよそう
                        q[i_action] *= 0.3
                        condition += ' unneeded_defense'
                elif self.board.players[player_id].halite < MAX_HALITE:
                    # shipyard_attack: spawn されない
                    q[i_action] *= 4.0
                    condition += ' sure_ydatk'
                else:  # shipyard_attack: spawn されうる
                    n0, n1 = self.opponent_history[player_id].get('defense_against_shipyard_attack', [0, 0])
                    r = max(0.02, (n1 - n0) / max(1, n1))
                    q[i_action] *= r
                    condition += ' bad_ydatk'

            score = self.scores[I_SCORE_EMPTY_ALLY_D4, i, j] + self.scores[I_SCORE_NON_EMPTY_ALLY_D4, i, j] - my_empty_score_d4
            if score < 0.5 and 6 < nearest_shipyard_distance:  # 孤立はよくないので回避
                q[i_action] *= 0.95
                isolated_count += 1
                condition += ' isolated'
            else:
                # 密もよくない
                r = 30.0 / (29.0 + score)
                q[i_action] *= r
                condition += ' dense'
            if mine_threshold <= cell.halite:
                if (i_action != 0 and cell.ship and cell.ship.player_id == self.player_id):
                    # 味方がすでにいる
                    r = 0.6
                    condition += ' ally_mine'
                else:
                    # 掘る / 堀りに行く
                    r = 2.0 if i_action == 0 else 1.0
                    r = min(4.0, r * cell.halite / 100.0)
                    condition += ' mine'
                q[i_action] *= r
            elif (3.99 < cell.halite):
                if (i_action == 0):
                    # 中途半端な halite 確保するぐらいなら empty でさまよっていたほうがマシ
                    q[i_action] *= 0.1
                    condition += ' evade_mine'
                elif cell.ship and cell.ship.player_id == self.player_id:
                    # 味方がすでにいる
                    q[i_action] *= 0.6
                    condition += ' ally_mine_B'
                else:  # 停止しづらい場所にはいきたくない
                    q[i_action] *= 0.8
                    condition += ' evade_ground_h'
            else:
                pass
            # 全 shipyard から離れておく
            # score = self.scores[I_SCORE_SHIPYARD_D7, i, j]
            # shipyard_score = 10.0 / (score * 0.125 + 9.0)
            # q[i_action] *= shipyard_score
            score = self.scores[I_SCORE_EMPTY_OPPONENT_D2, i, j]
            if 0.5 < score:
                # 相殺の可能性は避ける shipyardの近所は出待ち対策のため緩和するが, 敵が掘る権利持っているなら譲る
                if nearest_shipyard and calculate_distance(nearest_shipyard_position, cell.position) < 2:
                    # shipyard圏内
                    if 4.01 < cell.halite and cell.ship and cell.ship.halite == 0:
                        # 掘らせて追い返す
                        r = 0.05 / score
                        condition += ' evade_cancel_camper'
                    else:
                        r = 0.05 / score  # 1.0 やっぱり回避しておこう
                        condition += f' cancel_camper{q[i_action]:.3f}x{r:.3f}'
                else:
                    r = 0.05 / score 
                    condition += ' evade_cancel'
                q[i_action] *= r
            score = self.scores[I_SCORE_NON_EMPTY_OPPONENT_D2, i, j]
            # if 0.5 < score and (0 < i_action or cell.halite < 3.99):
                # kill とりに行く
                # r = 2.5 * score
                # q[i_action] *= r
                # condition += ' kill'
        if isolated_count == LEN_MOVE:  # 完全に孤立したので帰還
            q[:] *= preference
            condition += ' completely_isolated'

        self.log(step=self.board.step, id_=ship.id, s=f'h{ship.halite} empty_ship_strategy q{q} cond=({condition})')
        return self.reserve_ship_by_q(ship, q)

    def ship_strategy(self, ship, q=None):
        if ship.id in self.determined_ships:
            return
        if self.board.cells[ship.position].shipyard:
            return self.ship_on_shipyard_strategy(ship, q=q)
        if 0 < ship.halite:
            self.log(step=self.board.step, id_=ship.id, s=f'h{ship.halite} mining_ship_strategy')
            return self.mining_ship_strategy(ship, q=q)
        return self.empty_ship_strategy(ship, q=q)

    def shipyard_strategy(self):
        """不要不急のSPAWN"""
        # 周囲のhaliteが多いshipyardからspawnさせたい
        to_sort = []
        for shipyard in self.board.players[self.player_id].shipyards:
            if shipyard.id in self.determined_shipyards:
                continue  # 緊急対応済み
            i, j = position_to_ij(shipyard.position)
            if self.flags.get(I_FLAG_NEXT_SHIP_POSITION, i=i, j=j):
                self.reserve_shipyard(shipyard, None)  # 誰かが帰還する
                continue
            score = self.scores[I_SCORE_HALITE_D4, i, j]
            to_sort.append([shipyard, score, i, j])

        len_ships = len(self.board.current_player.ships)
        len_shipyards = len(self.board.current_player.shipyards)
        for shipyard, score, i, j in sorted(to_sort, key=itemgetter(1, 2, 3), reverse=True):
            if self.can_spawn(shipyard, len_ships=len_ships):
                self.reserve_shipyard(shipyard, ShipyardAction.SPAWN)
                len_ships += 1
            else:
                self.reserve_shipyard(shipyard, None)

    def reserve_shipyard(self, shipyard, next_action, forced=False):
        if shipyard.id in self.determined_shipyards:
            return
        next_action = ShipyardAction(next_action) if next_action else None
        self.determined_shipyards[shipyard.id] = next_action
        shipyard.next_action = next_action
        if next_action == ShipyardAction.SPAWN:
            if (not forced) and self.flags.get(I_FLAG_NEXT_SHIP_POSITION, x=shipyard.position[0], y=shipyard.position[1]):
                self.log(loglevel='warning', s=f'yd{shipyard.id} reserve_shipyard: try to spawn although merging')
            self.free_halite -= MAX_HALITE
            self.flags.set(I_FLAG_NEXT_SHIP_POSITION, x=shipyard.position[0], y=shipyard.position[1])

    def __call__(self, board):
        self.board = board
        assert self.board.current_player.id == self.player_id
        # self.log(loglevel='info', s=f'__call__ begin')
        self.len_ships = len(self.board.current_player.ships)
        self.len_shipyards = len(self.board.current_player.shipyards)
        np.random.seed(5431 + self.player_id * 10000 + self.board.step * 100000)
        self.update_opponent_history()
        self.free_halite = board.players[self.player_id].halite
        self.determined_ships = {}  # ship_id: (next_action, next_position)
        self.determined_shipyards = {}
        self.reserving_ships = {}
        self.configuration = board.configuration
        self.len_ships = len(self.board.current_player.ships)
        self.len_opponent_ships = len(self.board.ships) - self.len_ships
        self.sorted_ships = sorted(
                self.board.players[self.player_id].ships,
                key=attrgetter('halite'),
                reverse=True)
        self.non_empty_ships = []
        self.empty_ships = []
        for ship in self.sorted_ships:
            if ship.halite == 0:
                self.empty_ships.append(ship)
            else:
                self.non_empty_ships.append(ship)
        self.update_score()
        self.log(f'world_halite={self.world_halite}, halite_per_ship={self.halite_per_ship}, free_halite={self.free_halite}')

        # 前回からの続き
        sorted_projects = self.update_projects()
        self.log(step=self.board.step, s=f'len(projects)={len(self.projects)}, len(sorted_projects)={len(sorted_projects)}')

        for project in sorted_projects:
            if not project.run():  # 打ち切り
                project.discard()

        # ここからはフリーのship shipyard
        self.convert_strategy()
        for ship in self.sorted_ships:
            if ship.id in self.determined_ships:
                continue
            self.ship_strategy(ship)
        # ここまででq値が登録されたので、優先度高い順に実行
        reserving_ships = sorted(
                list(self.reserving_ships.values()), key=itemgetter(0, 1), reverse=True)
        for priority, ship_id, q, ship, forced, depend_on in reserving_ships:
            priority_depend_on = None if depend_on is None else self.calculate_collision_priority(depend_on)
            self.log(id_=ship_id, s=f'{ship.position} resolve_ship prio{priority} h{ship.halite} threat{self.collision_threat.get(ship_id, None)} q{q}, dep{depend_on.id if depend_on else None} prio{priority_depend_on}')
        for priority, ship_id, q, ship, forced, depend_on in reserving_ships:
            self.resolve_ship_by_q(ship=ship, q=q, forced=forced, depend_on=depend_on)
        # 最後に, SPAWN判定
        self.shipyard_strategy()

        self.previous_board = copy.deepcopy(board)
        self.previous_len_opponent_ships = self.len_opponent_ships
        s = 'ophist'
        for key in self.opponent_history[0].keys():
            a = []
            for player_id, history in enumerate(self.opponent_history):
                a += history[key]
            if 0 < np.sum(a):
                s += f' {key}{a}'
        self.log(loglevel='info', s=s)
        for ship in sorted(self.board.current_player.ships, key=attrgetter('id')):
            project_id = self.belonging_project.get(ship.id, None)
            project_ships = []
            if project_id:
                project = self.projects.get(project_id, None)
                if project:
                    project_ships = list(project.ships.keys())
            self.log(loglevel='info', id_=ship.id, s=f'h{ship.halite} prj={project_id}')
            if project_id and project:
                assert ship.id in project_ships
        for project_id, project in sorted(self.projects.items(), key=itemgetter(0)):
            if 0 < project.budget:
                self.log(loglevel='info', s=f'prj={project_id} budget={project.budget}')
        for shipyard_id, ships in self.ships_by_shipyard.items():
            self.log(loglevel='info', s=f'yd{shipyard_id} {list(ships.keys())}')
        # self.log(loglevel='info', s=f'__call__ end')
        return self.board.current_player.next_actions


g_agents = [None] * 4
@board_agent
def agent_fn(board, **kwargs):
    pid = board.current_player.id
    if (g_agents[pid] is None) or (board.step == 0):
        g_agents[pid] = MyAgent(pid, verbose=0)
    return g_agents[pid](board)

