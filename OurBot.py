#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position
# heap
from heapq import heappush, heappop
# This library allows you to generate random numbers.
import random
from collections import deque

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
import time

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.

ship_state = {}  # ship.id -> ship state
ship_path = {}  # ship.id -> directional path to ship_dest[ship.id]
ship_dest = {}  # ship.id -> destination
previous_position = {}  # ship.id-> previous pos
previous_state = {}  # ship.id -> previous state
ship_shipyards = {}  # ship.id -> shipyard.id
shipyard_halite = {}  # shipyard.id -> halite priority queue
shipyard_pos = {}  # shipyard.id -> shipyard position
shipyard_halite_pos = {}  # shipyard.id -> halite pos dictionary

VARIABLES = ["YEEHAW", 0, 50, 129, 0.87, 0.85, 290, 9,
             0.65, 0, 1, 0, 0.01, 0.98, 1.05, 0.85, 500, 450, 4, 0.25, 0.6]
VERSION = VARIABLES[1]
# search area for halite relative to shipyard
SCAN_AREA = int(VARIABLES[2])
# when switch collectable percentage of max halite
PERCENTAGE_SWITCH = int(VARIABLES[3])
SMALL_PERCENTAGE = float(VARIABLES[4])
BIG_PERCENTAGE = float(VARIABLES[5])
# definition of medium patch size for stopping and collecting patch if on
# the way
MEDIUM_HALITE = int(VARIABLES[6])
# halite left at patch to stop collecting at that patch
HALITE_STOP = int(VARIABLES[7])
# until which turn to spawn ships
SPAWN_TURN = int(VARIABLES[8] * constants.MAX_TURNS)
# Coefficients for halite heuristics
A = float(VARIABLES[9])
B = float(VARIABLES[10])
C = float(VARIABLES[11])
D = float(VARIABLES[12])
E = float(VARIABLES[13])
F = float(VARIABLES[14])
CRASH_TURN = constants.MAX_TURNS  #
CRASH_PERCENTAGE_TURN = float(VARIABLES[15])
CRASH_SELECTION_TURN = int(CRASH_PERCENTAGE_TURN * constants.MAX_TURNS)
SHIPYARD_VICINITY = 2
# If enemy ship has at least this halite kill it if near dropoff or shipyard
KILL_ENEMY_SHIP = int(VARIABLES[16])
# Minimum halite needed to join a halite cluster
HALITE_PATCH_THRESHOLD = int(VARIABLES[17])
MIN_CLUSTER_SIZE = int(VARIABLES[18])  # Minimum number of patches in a cluster
DETERMINE_CLUSTER_TURN = int(VARIABLES[19] * constants.MAX_TURNS)
CLUSTER_AMOUNT = float(VARIABLES[20])  # percentage of whole cluster amount

game.ready("Sea_Whackers {}".format(VERSION))


# h_amount <= 0 to run minheap as maxheap
def f(h_amount, h_distance):  # function for determining patch priority
    return (A * h_amount * h_amount + B * h_amount + C) / (D * h_distance * h_distance + E * h_distance + F)


def halite_priority_q(pos):
    h = []  # stores halite amount * -1 with its position in a minheap
    h_pos = {}
    top_left = Position(int(-1 * SCAN_AREA / 2),
                        int(-1 * SCAN_AREA / 2)) + pos  # top left of scan area
    for y in range(SCAN_AREA):
        for x in range(SCAN_AREA):
            p = Position((top_left.x + x) % game_map.width,
                         (top_left.y + y) % game_map.height)  # position of patch
            factor = f(game_map[p].halite_amount * -1, game_map.calculate_distance(p,
                                                                                   pos))  # f(negative halite amount,  distance from shipyard to patch)
            h_pos[factor] = p
            # add negative halite amounts so that would act as maxheap
            heappush(h, factor)
    return h, h_pos


def ship_priority_q(me, game_map):
    ships = []  # ship priority queue
    has_moved = {}
    for s in me.get_ships():
        has_moved[s.id] = False
        if s.id in ship_state:
            shipyard = shipyard_pos[ship_shipyards[
                ship.id]]  # its shipyard position
            # importance, the lower the number, bigger importance
            if ship_state[s.id] == "returning":
                importance = game_map.calculate_distance(s.position, shipyard) / (
                    game_map.width * 2)  # 0,1 range
            elif ship_state[s.id] == "exploring":
                importance = game_map.calculate_distance(
                    s.position, shipyard)  # normal distance
            else:  # collecting
                importance = game_map.calculate_distance(s.position,
                                                         shipyard) * game_map.width * 2  # normal distance * X since processing last
        else:
            importance = 0  # newly spawned ships max importance
        heappush(ships, (importance, s))
    return ships, has_moved


# selects turn when to crash
def select_crash_turn():
    distance = 0
    for ship in me.get_ships():
        shipyard = shipyard_pos[ship_shipyards[
            ship.id]]  # its shipyard position
        d = game_map.calculate_distance(shipyard, ship.position)
        if d > distance:  # get maximum distance away of shipyard
            distance = d
    crash_turn = constants.MAX_TURNS - distance - 5
    # set the crash turn to be turn s.t. all ships make it
    return max(crash_turn, CRASH_SELECTION_TURN)


def find_new_destination(h, ship_id, halite_pos):
    ''' h: priority queue of halite factors,
        halite_pos: dictionary of halite factor -> patch position '''
    biggest_halite = heappop(h)  # get biggest halite
    while halite_pos[
            biggest_halite] in ship_dest.values():  # get biggest halite while its a position no other ship goes to
        biggest_halite = heappop(h)
    ship_dest[ship_id] = halite_pos[biggest_halite]  # set the destination


def clear_dictionaries():
    # clear dictionaries of crushed ships
    for ship_id in list(ship_dest.keys()):
        if not me.has_ship(ship_id):
            del ship_dest[ship_id]
            del ship_state[ship_id]
            del ship_path[ship_id]
            del previous_state[ship_id]
            del previous_position[ship_id]
            del ship_shipyards[ship_id]


def get_dijkstra_move(current_position):
    """
    Gets a move from the map created by Dijkstra
    :return: The target position of the move and the direction of the move.
    """
    cell = game_map[current_position].parent
    new_pos = cell.position
    dirs = game_map.get_target_direction(ship.position, new_pos)
    new_dir = dirs[0] if dirs[0] is not None else dirs[1]
    return new_pos, new_dir


def make_returning_move(ship, has_moved, command_queue):
    """
    Makes a returning move based on Dijkstras and other ship positions.
    :return: has_moved, command_queue
    """
    # Get the cell and direction we want to go to from dijkstra
    target_pos, move = get_dijkstra_move(ship.position)

    # Target is occupied
    if game_map[target_pos].is_occupied:
        other_ship = game_map[target_pos].ship
        # Occupied by own ship that can move, perform swap
        if other_ship in me.get_ships() \
                and other_ship.halite_amount >= game_map[target_pos].halite_amount * 0.1 \
                and not has_moved[other_ship.id]:
            if ship_state[other_ship.id] == "returning":
                # Let this ship process first since it has not moved
                if target_pos not in get_dropoff_positions():
                    other_move = make_returning_move(
                        other_ship, has_moved, command_queue)
                if game_map[target_pos].is_occupied:
                    move = Direction.Still
            else:
                # Move other ship to this position
                command_queue.append(other_ship.move(Direction.invert(move)))
                game_map[ship.position].mark_unsafe(other_ship)
                has_moved[other_ship.id] = True
        # Occupied by enemy ship, try to go around
        elif other_ship not in me.get_ships():
            # logging.info("ship {} going around enemy ship".format(ship.id))
            # If ship was trying to go north or south, go east or west (to move
            # around)

            # produces both moves that are not the direction going and not the
            # inverse of that direction
            possible_move1 = (move[1], move[0])
            possible_move2 = (-1 * move[1], -1 * move[0])

            new_pos1 = ship.position.directional_offset(possible_move1)
            new_pos2 = ship.position.directional_offset(possible_move2)
            # select the move which leads to a patch that has the least halite
            if game_map[new_pos1].halite_amount < game_map[new_pos2].halite_amount:
                move = possible_move1
            else:
                move = possible_move2

        # Occupied by own unmovable ship
        else:
            move = Direction.Still

    has_moved[ship.id] = True
    return move


def create_halite_clusters(game_map):
    """
    Creates halite clusters of adjacent halite patches with halite > HALITE_PATCH_THRESHOLD
    :return: centers a list of positions for the centers of clusters, sorted by cluster value
    """
    # logging.info("Map size: {} by {}".format(game_map.width, game_map.height))
    # map of size MAPSIZE where -1 means not in a cluster else the value is
    # the ID of the cluster (starting at 1)
    cluster_map = [[0 for _ in range(game_map.width)]
                   for _ in range(game_map.height)]
    # cluster_map = [game_map.height][game_map.width]
    current_cluster_id = 1
    for i in range(game_map.height):
        for j in range(game_map.width):
            # Not in any cluster
            if game_map[Position(j, i)].halite_amount < HALITE_PATCH_THRESHOLD:
                cluster_map[i][j] = -1
            elif cluster_map[i][j] != 0:  # Already in cluster
                continue
            else:  # Create new cluster
                cluster_dfs(game_map, cluster_map, current_cluster_id, i, j)
                current_cluster_id += 1
    #  All patches on the map should be assigned -1 or a cluster ID
    #  Log the current cluster map
    res = ""
    for i in range(game_map.height):
        line = ""
        for j in range(game_map.width):
            line += str(cluster_map[i][j]) + " "
        res += line + '\n'

    # logging.info("Halite Cluster map:")
    # logging.info(res)

    clusters = [[] for _ in range(current_cluster_id - 1)]
    for i in range(game_map.height):
        for j in range(game_map.width):
            if cluster_map[i][j] != -1:
                clusters[cluster_map[i][j] - 1].append(Position(j, i))

    # cluster ID starts at 0
    cluster_value = [0 for _ in range(current_cluster_id - 1)]
    for i, clust in enumerate(clusters):
        for patch in clust:  # determine cluster value
            cluster_value[i] += game_map[patch].halite_amount / game_map.calculate_distance(patch, me.shipyard.position)

    # Sort by cluster value
    clusters = [c for _, c in sorted(
        zip(cluster_value, clusters), reverse=True)]

    # Remove small clusters
    clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]

    # Find centers in all usable clusters
    centers = [Position(0, 0) for _ in clusters]
    minx = [game_map.width for _ in clusters]
    miny = [game_map.height for _ in clusters]
    maxx = [0 for _ in clusters]
    maxy = [0 for _ in clusters]
    for i, c in enumerate(clusters):
        for patch in c:
            if patch.x > maxx[i]:
                maxx[i] = patch.x
            elif patch.x < minx[i]:
                minx[i] = patch.x
            if patch.y > maxy[i]:
                maxy[i] = patch.y
            elif patch.y < miny[i]:
                miny[i] = patch.y

    # logging.info("Found {} clusters".format(len(clusters)))
    center_info = ""
    for i, c in enumerate(clusters):
        xdiff = maxx[i] - minx[i]
        ydiff = maxy[i] - miny[i]
        if xdiff > game_map.width / 2:
            xcenter = (maxx[i] + ((game_map.width - xdiff) / 2)
                       ) % game_map.width
        else:
            xcenter = minx[i] + xdiff / 2
        xcenter = int(xcenter)
        if ydiff > game_map.height / 2:
            ycenter = (maxy[i] + ((game_map.height - ydiff) / 2)
                       ) % game_map.height
        else:
            ycenter = miny[i] + ydiff / 2
        ycenter = int(ycenter)
        centers[i] = Position(xcenter, ycenter)
        center_info += str(centers[i]) + " "

    # logging.info(center_info)

    # A list of centers where the first center has most halite and the last
    # has the least halite
    return centers


def cluster_dfs(game_map, cluster_map, id, i, j):
    # logging.info("Cluster DFS trying for ({},{})".format(i, j))
    if game_map[Position(j, i)].halite_amount < HALITE_PATCH_THRESHOLD:  # Not in any cluster
        cluster_map[i][j] = -1
        return
    if cluster_map[i][j] != 0:  # Already in cluster
        return

    cluster_map[i][j] = id
    dy = [-1, 0, 1, 0]
    dx = [0, -1, 0, 1]
    for k in range(4):
        newI = (i + dy[k]) % game_map.height
        newJ = (j + dx[k]) % game_map.width
        if newI < 0:
            newI += game_map.height
        if newJ < 0:
            newJ += game_map.width
        # logging.info("NewI,NewJ = {},{}".format(newI, newJ))
        cluster_dfs(game_map, cluster_map, id, newI, newJ)


def enemy_near_shipyard():
    """
    Returns a list of position objects of all enemy ships near the shipyard
    and enemy ships near dropoffs that carry more than KILL_ENEMY_SHIP halite
    """
    nearby_enemy_ships = []
    # Check shipyard vicinity
    for y in range(-1 * SHIPYARD_VICINITY + me.shipyard.position.y, SHIPYARD_VICINITY + 1 + me.shipyard.position.y):
        for x in range(-1 * SHIPYARD_VICINITY + me.shipyard.position.x, SHIPYARD_VICINITY + 1 + me.shipyard.position.x):
            if game_map[Position(x, y)].is_occupied and not game_map[Position(x, y)].ship in me.get_ships():
                nearby_enemy_ships.append(Position(x, y))
    dropoffs = me.get_dropoffs()
    # Check vicinity of all dropoffs
    for dropoff in dropoffs:
        for y in range(-1 * SHIPYARD_VICINITY + dropoff.position.y, SHIPYARD_VICINITY + 1 + dropoff.position.y):
            for x in range(-1 * SHIPYARD_VICINITY + dropoff.position.x, SHIPYARD_VICINITY + 1 + dropoff.position.x):
                if game_map[Position(x, y)].is_occupied and not game_map[Position(x, y)].ship in me.get_ships() and \
                        game_map[Position(x, y)].ship.halite_amount > KILL_ENEMY_SHIP:
                    nearby_enemy_ships.append(Position(x, y))
    return nearby_enemy_ships


def check_shipyard_blockade(enemies, ship_position):
    for enemy_position in enemies:
        if game_map.calculate_distance(ship_position, enemy_position) < 3:
            return enemy_position
    return None


def state_switch(ship_id, new_state):
    if not previous_state[ship_id] == new_state:
        previous_state[ship_id] = ship_state[ship_id]
    ship_state[ship_id] = new_state


def produce_move(ship):
    state = ship_state[ship.id]
    destination = ship_dest[ship.id]
    ''' produces move for ship '''

    if ship.halite_amount < game_map[ship.position].halite_amount / 10:
        return Direction.Still

    mover = {
        "collecting": collecting,
        "returning": returning,
        "harakiri": harakiri,
        "assassinate": assassinate,
        "exploring": exploring,
        "build": exploring,
    }

    return mover[state](ship, destination)


def collecting(ship, destination):
    return Direction.Still


def returning(ship, destination):
    return make_returning_move(ship, has_moved, command_queue)


def exploring(ship, destination):
    # next direction occupied, recalculate
    if ship.id not in ship_path or len(ship_path[ship.id]) == 0:
        ship_path[ship.id] = game_map.explore(ship, destination)
    else:
        direction = ship_path[ship.id][0][0]
        if game_map[ship.position.directional_offset(direction)].is_occupied and not direction == Direction.Still:
            ship_path[ship.id] = game_map.explore(ship, destination)

    # move in calculated direction
    ship_path[ship.id][0][1] -= 1  # take that direction
    direction = ship_path[ship.id][0][0]
    logging.info(ship_path[ship.id])
    if ship_path[ship.id][0][1] == 0:  # if no more left that direction remove it
        del ship_path[ship.id][0]
    logging.info(direction)
    return direction


def harakiri(ship, destination):
    shipyard = shipyard_pos[ship_shipyards[ship.id]]
    if ship.position == shipyard:  # if at shipyard
        return Direction.Still  # let other ships crash in to you
    else:  # otherwise move to the shipyard
        target_dir = game_map.get_target_direction(
            ship.position, shipyard)
        return target_dir[0] if target_dir[0] is not None else target_dir[1]


def assassinate(ship, destination):
    state_switch(ship.id, previous_state[ship.id])
    if game_map.calculate_distance(ship.position, destination) == 1:
        target_direction = game_map.get_target_direction(
            ship.position, destination)
        return target_direction[0] if target_direction[0] is not None else target_direction[1]
    else:
        return exploring(ship, destination)


def state_transition(ship):
    # transition
    new_state = None
    shipyard_id = ship_shipyards[ship.id]
    if ship_state[ship.id] == "returning" and game.turn_number >= CRASH_TURN and game_map.calculate_distance(
            ship.position, shipyard_pos[shipyard_id]) < 2:
        # if returning after crash turn, suicide
        new_state = "harakiri"

    elif (ship_state[ship.id] == "collecting" or ship_state[
            ship.id] == "exploring") and game.turn_number >= CRASH_TURN:
        # return if at crash turn
        new_state = "returning"

    elif ship_state[ship.id] == "exploring" and (ship.position == ship_dest[ship.id]
                                                 or game_map[ship.position].halite_amount > MEDIUM_HALITE):
        # collect if reached destination or on medium sized patch
        new_state = "collecting"

    elif ship_state[ship.id] == "exploring" and ship.halite_amount >= constants.MAX_HALITE * return_percentage:
        # return if ship is 70+% full
        new_state = "returning"

    elif ship_state[ship.id] == "collecting" and game_map[ship.position].halite_amount < HALITE_STOP:
        # Keep exploring if current halite patch is empty
        ship_h, ship_h_positions = halite_priority_q(ship.position)
        find_new_destination(ship_h, ship.id, ship_h_positions)
        new_state = "exploring"

    elif ship_state[
            ship.id] == "collecting" and ship.halite_amount >= constants.MAX_HALITE * return_percentage:  # return to shipyard if enough halite
        # return ship is 70% full
        new_state = "returning"

    elif ship_state[ship.id] == "returning" and ship.position in get_dropoff_positions():
        # explore again when back in shipyard
        new_state = "exploring"
        find_new_destination(
            shipyard_halite[shipyard_id], ship.id, shipyard_halite_pos[shipyard_id])

    # if someone already build dropoff there before us
    elif ship_state[ship.id] == "build" and game_map[ship_dest[ship.id]].has_structure:
        # explore
        ship_h, ship_h_positions = halite_priority_q(ship.position)
        find_new_destination(ship_h, ship.id, ship_h_positions)
        new_state = "exploring"

    if new_state is not None:
        state_switch(ship.id, new_state)


def do_halite_priorities():
    ''' determines halite priority queues
    and positions for all dropoffs, shipyards '''
    shipyard_halite[me.shipyard.id], shipyard_halite_pos[
        me.shipyard.id] = halite_priority_q(me.shipyard.position)
    shipyard_pos[me.shipyard.id] = me.shipyard.position
    for dropoff in me.get_dropoffs():
        shipyard_halite[dropoff.id], shipyard_halite_pos[
            dropoff.id] = halite_priority_q(dropoff.position)
        shipyard_pos[dropoff.id] = dropoff.position


def closest_shipyard_id(ship):
    ''' returns shipyard id that is closest to ship '''
    distance = game_map.calculate_distance(ship.position, me.shipyard.position)
    shipyard_id = me.shipyard.id
    for dropoff in me.get_dropoffs():
        new_distance = game_map.calculate_distance(
            ship.position, dropoff.position)
        if new_distance < distance:
            distance = new_distance
            shipyard_id = dropoff.id
    return shipyard_id


def get_dropoff_positions():
    """ Returns a list of all positions of dropoffs and the shipyard """
    positions = [me.shipyard.position]
    for dropoff in me.get_dropoffs():
        positions.append(dropoff.position)
    return positions


def get_fleet(position, fleet_size):
    ''' returns list of fleet_size amount
        of ships closest to the position'''
    distances = []
    for s in me.get_ships():
        if s.id not in ship_shipyards.keys() or ship_shipyards[s.id] == me.shipyard.id:
            distances.append(
                (game_map.calculate_distance(position, s.position), s))
    distances.sort(key=lambda x: x[0])
    return [t[1] for t in distances[:fleet_size]]


def bfs_unoccupied(position):
    # bfs for closest cell
    Q = deque([])
    Q.append(position)
    while Q:
        cur = Q.popleft()
        if not (game_map[cur].is_occupied and game_map[cur].has_structure):
            return cur
        for neighbour in game_map.get_neighbours(game_map[cur]):
            if neighbour.position not in Q:
                Q.append(neighbour.position)


def is_builder(ship):
    ''' checks if this ship is a builder '''
    return ship_state[ship.id] == "waiting" or (ship_state[ship.id] == "build" and ship_dest[ship.id] == ship.position)


def should_build():
    # if clusters determined, more than 13 ships, we have clusters and nobody
    # is building at this turn (in order to not build too many)
    return clusters_determined and len(me.get_ships()) > 10 and len(cluster_centers) > 0 and not "build" in ship_state.values()


def send_ships(pos, ship_amount):
    '''sends a fleet of size ship_amount to build a dropoff and explore in dropoff_pos'''
    fleet = get_fleet(pos, ship_amount)
    # for rest of the fleet to explore
    h, h_pos = halite_priority_q(pos)
    for fleet_ship in fleet:  # for other fleet members
        if fleet_ship.id not in previous_state.keys():  # if new ship
            previous_state[fleet_ship.id] = "exploring"
            has_moved[fleet_ship.id] = False
        # explore in area of the new dropoff
        state_switch(fleet_ship.id, "exploring")
        find_new_destination(h, fleet_ship.id, h_pos)

clusters_determined = False
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    # Dijkstra the graph based on all dropoffs
    game_map.create_graph(get_dropoff_positions())
    # Calculate halite clusters
    if game.turn_number == DETERMINE_CLUSTER_TURN:
        clusters_determined = True
        cluster_centers = create_halite_clusters(game_map)
        save_amount = int(len(cluster_centers) * CLUSTER_AMOUNT)
        cluster_centers = cluster_centers[:save_amount]
    # determine the turn for harikiri state
    if game.turn_number == CRASH_SELECTION_TURN:
        CRASH_TURN = select_crash_turn()

    return_percentage = BIG_PERCENTAGE if game.turn_number < PERCENTAGE_SWITCH else SMALL_PERCENTAGE
    command_queue = []
    # priority Q of patch function values of function f(halite, distance)
    do_halite_priorities()

    if should_build():
        dropoff_pos = cluster_centers.pop(0)  # remove from list
        # sends ships to position where closest will build dropoff
        fleet = get_fleet(dropoff_pos, 1)
        closest_ship = fleet.pop(0)  # remove and get closest ship

        state_switch(closest_ship.id, "build")  # will build dropoff
        # if dropoffs position already has a structure (e.g. other dropoff) or
        # somebody is going there already
        if game_map[dropoff_pos].has_structure or dropoff_pos in ship_dest.values():
            # bfs for closer valid unoccupied position
            dropoff_pos = bfs_unoccupied(dropoff_pos)
        ship_dest[closest_ship.id] = dropoff_pos  # go to the dropoff

    # has_moved ID->True/False, moved or not
    # ships priority queue of (importance, ship)
    ships, has_moved = ship_priority_q(me, game_map)
    start = time.time()
    # True if a ship moves into the shipyard this turn
    move_into_shipyard = False
    # whether a dropoff has been built this turn so that wouldnt use too much
    # halite
    dropoff_built = False

    nearby_enemy_ships = enemy_near_shipyard()

    while ships:  # go through all ships
        ship = heappop(ships)[1]
        if has_moved[ship.id]:
            continue
        if ship.id not in previous_position:  # if new ship the
            previous_position[ship.id] = me.shipyard.position

        # set the closest shipyard
        ship_shipyards[ship.id] = closest_shipyard_id(ship)
        shipyard_id = ship_shipyards[ship.id]

        # setup state
        if ship.id not in ship_dest:  # if ship hasnt received a destination yet
            find_new_destination(
                shipyard_halite[shipyard_id], ship.id, shipyard_halite_pos[shipyard_id])
            previous_state[ship.id] = "exploring"
            ship_state[ship.id] = "exploring"  # explore

        enemy_position = check_shipyard_blockade(
            nearby_enemy_ships, ship.position)
        if enemy_position is not None:
            state_switch(ship.id, "assassinate")
            # logging.info("state: {}".format(ship_state[ship.id]))
            ship_dest[ship.id] = enemy_position
            nearby_enemy_ships.remove(enemy_position)

        # transition
        state_transition(ship)

        # logging.info("ship:{} , state:{} ".format(ship.id, ship_state[ship.id]))
        # logging.info("destination: {}, {} ".format(ship_dest[ship.id].x,
        # ship_dest[ship.id].y))

        # if ship is dropoff builder
        if is_builder(ship):
            # if enough halite and havent built a dropoff this turn
            if me.halite_amount >= constants.DROPOFF_COST and not dropoff_built:
                send_ships(ship.position, 7)
                command_queue.append(ship.make_dropoff())
                do_halite_priorities()  # recalc all dictionaries
                dropoff_built = True
            else:  # cant build
                ship_state[ship.id] = "waiting"  # wait in the position
                game_map[ship.position].mark_unsafe(ship)
                command_queue.append(ship.move(Direction.Still))
        else:  # not associated with building a dropoff, so move regularly
            move = produce_move(ship)
            logging.info(move)
            command_queue.append(ship.move(move))
            previous_position[ship.id] = ship.position
            
            game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
            if move != Direction.Still and game_map[ship.position].ship == ship:
               game_map[ship.position].ship = None

        clear_dictionaries()  # of crashed or transformed ships
        # This ship has made a move
        has_moved[ship.id] = True

    logging.info(time.time() - start)
    surrounded_shipyard = game_map.is_surrounded(me.shipyard.position)

    if not dropoff_built and game.turn_number <= SPAWN_TURN and me.halite_amount >= constants.SHIP_COST \
            and not (game_map[me.shipyard].is_occupied or surrounded_shipyard or "waiting" in ship_state.values()):
        command_queue.append(me.shipyard.spawn())
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
