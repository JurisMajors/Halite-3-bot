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

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
import time

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
VARIABLES = ["YEEHAW", 0, 50, 129, 0.87, 0.85, 290, 9, 221, 0.01, 0.98, 1.05, 0.92]
VERSION = VARIABLES[1]
ship_state = {}
ship_dest = {}  # ship.id -> destination
halite_positions = {}  # halite -> position
previous_position = {}  # ship.id-> previous pos
previous_state = {}  # ship.id -> previous state
# search area for halite relative to shipyard
SCAN_AREA = int(VARIABLES[2])
PERCENTAGE_SWITCH = int(VARIABLES[3])  # when switch collectable percentage of max halite
SMALL_PERCENTAGE = float(VARIABLES[4])
BIG_PERCENTAGE = float(VARIABLES[5])
MEDIUM_HALITE = int(VARIABLES[6])  # definition of medium patch size for stopping and collecting patch if on the way
HALITE_STOP = int(VARIABLES[7])  # halite left at patch to stop collecting at that patch
SPAWN_TURN = int(VARIABLES[8])  # until which turn to spawn ships
A = float(VARIABLES[9])
B = float(VARIABLES[10])
C = float(VARIABLES[11])
CRASH_TURN = constants.MAX_TURNS
CRASH_PERCENTAGE_TURN = float(VARIABLES[12])
CRASH_SELECTION_TURN = int(CRASH_PERCENTAGE_TURN * constants.MAX_TURNS)
SHIPYARD_VICINITY = 2
KILL_ENEMY_SHIP = 500  # If enemy ship has at least this halite kill it if near dropoff or shipyard
HALITE_PATCH_THRESHOLD = 400  # Minimum halite needed to join a halite cluster
MIN_CLUSTER_SIZE = 4  # Minimum number of patches in a cluster
game.ready("Sea_Whackers {}".format(VERSION))


# h_amount <= 0 to run minheap as maxheap
def f(h_amount, h_distance):  # function for determining patch priority
    return h_amount / (A * h_distance * h_distance + B * h_distance + C)


def halitePriorityQ(pos, game_map, h_pos):
    h = []  # stores halite amount * -1 with its position in a minheap
    top_left = Position(int(-1 * SCAN_AREA / 2), int(-1 * SCAN_AREA / 2)) + pos  # top left of scan area
    for y in range(SCAN_AREA):
        for x in range(SCAN_AREA):
            p = Position((top_left.x + x) % game_map.width, (top_left.y + y) % game_map.height)  # position of patch
            factor = f(game_map[p].halite_amount * -1, game_map.calculate_distance(p,
                                                                                   pos))  # f(negative halite amount,  distance from shipyard to patch)
            h_pos[factor] = p
            heappush(h, factor)  # add negative halite amounts so that would act as maxheap
    return h, h_pos


def shipPriorityQ(me, game_map):
    ships = []  # ship priority queue
    has_moved = {}
    for s in me.get_ships():
        has_moved[s.id] = False
        if s.id in ship_state:
            # importance, the lower the number, bigger importance
            if ship_state[s.id] == "returning":
                importance = game_map.calculate_distance(s.position, me.shipyard.position) / (
                        game_map.width * 2)  # 0,1 range
            elif ship_state[s.id] == "exploring":
                importance = game_map.calculate_distance(s.position, me.shipyard.position)  # normal distance
            else:  # collecting
                importance = game_map.calculate_distance(s.position,
                                                         me.shipyard.position) * game_map.width * 2  # normal distance * X since processing last
        else:
            importance = 0  # newly spawned ships max importance
        heappush(ships, (importance, s))
    return ships, has_moved


def returnShip(ship_id, ship_dest, ship_state):
    ship_dest[ship_id] = me.shipyard.position
    ship_state[ship_id] = "returning"
    return ship_state, ship_dest


# selects turn when to crash
def selectCrashTurn():
    distance = 0
    for ship in me.get_ships():
        d = game_map.calculate_distance(me.shipyard.position, ship.position)
        if d > distance:  # get maximum distance away of shipyard
            distance = d
    crash_turn = constants.MAX_TURNS - distance - 4
    # set the crash turn to be turn s.t. all ships make it
    return crash_turn if crash_turn > CRASH_SELECTION_TURN else CRASH_SELECTION_TURN


def findNewDestination(h, ship_id, halite_pos):
    ''' h: priority queue of halite factors,
        halite_pos: dictionary of halite factor -> patch position '''
    biggest_halite = heappop(h)  # get biggest halite
    while halite_pos[
        biggest_halite] in ship_dest.values():  # get biggest halite while its a position no other ship goes to
        biggest_halite = heappop(h)
    ship_dest[ship_id] = halite_pos[biggest_halite]  # set the destination


def clearDictionaries():
    # clear dictionaries of crushed ships
    for ship_id in list(ship_dest.keys()):
        if not me.has_ship(ship_id):
            del ship_dest[ship_id]
            del ship_state[ship_id]


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


def make_returning_move(game_map, ship, me, has_moved, command_queue):
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
                and not has_moved[other_ship.id] and not ship_state[other_ship.id] == "returning":
            # Move other ship to this position
            command_queue.append(other_ship.move(Direction.invert(move)))
            game_map[ship.position].mark_unsafe(other_ship)
            has_moved[other_ship.id] = True

        # Occupied by enemy ship, try to go around
        elif other_ship not in me.get_ships():
            logging.info("ship {} going around enemy ship".format(ship.id))
            # If ship was trying to go north or south, go east or west (to move around)
            if move == Direction.South or Direction.North:
                # Go to the patch with the least halite.
                if game_map[ship.position.directional_offset(Direction.East)].halite_amount < \
                        game_map[ship.position.directional_offset(Direction.West)].halite_amount:
                    move = Direction.East
                else:
                    move = Direction.West
            # Same as previous but directions reversed.
            else:
                if game_map[ship.position.directional_offset(Direction.South)].halite_amount < \
                        game_map[ship.position.directional_offset(Direction.North)].halite_amount:
                    move = Direction.South
                else:
                    move = Direction.North

            # move = Direction.Still
            target_pos = ship.position.directional_offset(move)
        # Occupied by own unmovable ship
        else:
            move = Direction.Still
            target_pos = ship.position

    return move, target_pos, has_moved, command_queue


def create_halite_clusters(game_map):
    """
    Creates halite clusters of adjacent halite patches with halite > HALITE_PATCH_THRESHOLD
    :return: centers a list of positions for the centers of clusters, sorted by cluster value
    """
    logging.info("Map size: {} by {}".format(game_map.width, game_map.height))
    # map of size MAPSIZE where -1 means not in a cluster else the value is the ID of the cluster (starting at 1)
    cluster_map = [[0 for _ in range(game_map.width)] for _ in range(game_map.height)]
    # cluster_map = [game_map.height][game_map.width]
    current_cluster_id = 1
    for i in range(game_map.height):
        for j in range(game_map.width):
            if game_map[Position(j, i)].halite_amount < HALITE_PATCH_THRESHOLD:  # Not in any cluster
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

    logging.info("Halite Cluster map:")
    logging.info(res)

    clusters = [[] for _ in range(current_cluster_id - 1)]
    for i in range(game_map.height):
        for j in range(game_map.width):
            if cluster_map[i][j] != -1:
                clusters[cluster_map[i][j] - 1].append(Position(j, i))

    cluster_value = [0 for _ in range(current_cluster_id - 1)]  # cluster ID starts at 0
    for i, clust in enumerate(clusters):
        for patch in clust:
            cluster_value[i] += game_map[patch].halite_amount

    # Sort by cluster value
    clusters = [c for _, c in sorted(zip(cluster_value, clusters), reverse=True)]

    # Remove small clusters
    clusters = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]

    # Find centers in all usable clusters
    centers = [Position(0, 0) for _ in clusters]
    xsum = [0 for _ in clusters]
    ysum = [0 for _ in clusters]
    for i, c in enumerate(clusters):
        for patch in c:
            xsum[i] += patch.x
            ysum[i] += patch.y

    center_info = ""
    for i, c in enumerate(clusters):
        centers[i] = Position(int(xsum[i] / len(c)), int(ysum[i] / len(c)))
        center_info += str(centers[i]) + " "

    logging.info(center_info)

    # A list of centers where the first center has most halite and the last has the least halite
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


while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    # Dijkstra the graph
    game_map.dijkstra(me.shipyard)
    # Calculate halite clusters
    if game.turn_number == 1:
        create_halite_clusters(game_map)

    return_percentage = BIG_PERCENTAGE if game.turn_number < PERCENTAGE_SWITCH else SMALL_PERCENTAGE

    command_queue = []
    # priority Q of patch function values of function f(halite, distance)
    h, halite_positions = halitePriorityQ(me.shipyard.position, game_map, halite_positions)
    # has_moved ID->True/False, moved or not
    # ships priority queue of (importance, ship) 
    ships, has_moved = shipPriorityQ(me, game_map)
    start = time.time()
    # True if a ship moves into the shipyard this turn
    move_into_shipyard = False

    if game.turn_number == CRASH_SELECTION_TURN:
        CRASH_TURN = selectCrashTurn()

    nearby_enemy_ships = enemy_near_shipyard()

    while ships:  # go through all ships
        ship = heappop(ships)[1]
        if has_moved[ship.id]:
            continue
        if ship.id not in previous_position:  # if new ship the
            previous_position[ship.id] = me.shipyard.position
        find_new_dest = False
        possible_moves = []

        # setup state
        if ship.id not in ship_dest:  # if ship hasnt received a destination yet
            findNewDestination(h, ship.id, halite_positions)
            ship_state[ship.id] = "exploring"  # explore

        # transition
        if ship_state[ship.id] == "returning" and game.turn_number >= CRASH_TURN and game_map.calculate_distance(
                ship.position, me.shipyard.position) < 2:
            # if returning after crash turn, suicide
            previous_state[ship.id] = ship_state[ship.id]
            ship_state[ship.id] = "harakiri"
        elif (ship_state[ship.id] == "collecting" or ship_state[
            ship.id] == "exploring") and game.turn_number >= CRASH_TURN:
            # return if at crash turn
            ship_state, ship_dest = returnShip(ship.id, ship_dest, ship_state)
        elif ship_state[ship.id] == "exploring" and (
                ship.position == ship_dest[ship.id] or game_map[ship.position].halite_amount > MEDIUM_HALITE):
            # collect if reached destination or on medium sized patch
            previous_state[ship.id] = ship_state[ship.id]
            ship_state[ship.id] = "collecting"
        elif ship_state[ship.id] == "exploring" and ship.halite_amount >= constants.MAX_HALITE * return_percentage:
            # return if ship is 70+% full
            ship_state, ship_dest = returnShip(ship.id, ship_dest, ship_state)
        elif ship_state[ship.id] == "collecting" and game_map[ship.position].halite_amount < HALITE_STOP:
            # Keep exploring if current halite patch is empty
            ship_h_positions = {}
            ship_h, ship_h_positions = halitePriorityQ(ship.position, game_map, ship_h_positions)
            findNewDestination(ship_h, ship.id, ship_h_positions)
            previous_state[ship.id] = ship_state[ship.id]
            ship_state[ship.id] = "exploring"
        elif ship_state[
            ship.id] == "collecting" and ship.halite_amount >= constants.MAX_HALITE * return_percentage:  # return to shipyard if enough halite
            # return ship is 70% full
            ship_state, ship_dest = returnShip(ship.id, ship_dest, ship_state)
        elif ship_state[ship.id] == "returning" and ship.position == ship_dest[ship.id]:
            # explore again when back in shipyard
            previous_state[ship.id] = ship_state[ship.id]
            ship_state[ship.id] = "exploring"
            findNewDestination(h, ship.id, halite_positions)
        elif ship_state[ship.id] == "KillEnemyNearDropoff":
            if game_map.calculate_distance(ship.position, ship_dest[ship.id]) == 1:
                target_direction = game_map.get_target_direction(ship.position, ship_dest[ship.id])
                move = target_direction[0] if target_direction[0] is not None else target_direction[1]
            else:
                move = game_map.explore(ship, ship_dest[ship.id])
            game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
            command_queue.append(ship.move(move))
            ship_state[ship.id] = previous_state[ship.id]
            previous_state[ship.id] = ship_state[ship.id]

        logging.info("ship:{} , state:{} ".format(ship.id, ship_state[ship.id]))
        logging.info("destination: {}, {} ".format(ship_dest[ship.id].x, ship_dest[ship.id].y))

        clearDictionaries()  # of crashed ships

        # make move
        if ship.halite_amount < game_map[ship.position].halite_amount / 10:  # Cannot move, stay stil
            move = Direction.Still
            command_queue.append(ship.move(move))

        elif ship_state[ship.id] == "exploring":  # if exploring move to its destinition in ship_dest dictionary
            move = game_map.explore(ship, ship_dest[ship.id])
            game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
            command_queue.append(ship.move(move))

        elif ship_state[ship.id] == "returning":  # if returning
            move, target_pos, has_moved, command_queue = make_returning_move(game_map, ship, me, has_moved,
                                                                             command_queue)
            # Track if we are moving a ship into the shipyard
            if target_pos == me.shipyard.position:
                move_into_shipyard = True

            # Keep track of unsafe position and make move
            game_map[target_pos].mark_unsafe(ship)
            command_queue.append(ship.move(move))

        elif ship_state[ship.id] == "collecting":
            move = Direction.Still  # collect
            command_queue.append(ship.move(move))

        elif ship_state[ship.id] == "harakiri":
            if ship.position == me.shipyard.position:  # if at shipyard
                move = Direction.Still  # let other ships crash in to you
            else:  # otherwise move to the shipyard
                target_pos = me.shipyard.position
                target_dir = game_map.get_target_direction(ship.position, target_pos)
                move = target_dir[0] if target_dir[0] is not None else target_dir[1]
            command_queue.append(ship.move(move))

        previous_position[ship.id] = ship.position
        # This ship has made a move
        has_moved[ship.id] = True
    logging.info(time.time() - start)

    # check if shipyard is surrounded by ships
    shipyard_surrounded = True
    for direction in Direction.get_all_cardinals():
        position = me.shipyard.position.directional_offset(direction)
        if not game_map[me.shipyard.position.directional_offset(direction)].is_occupied:
            shipyard_surrounded = False
            break

    if game.turn_number <= SPAWN_TURN and me.halite_amount >= constants.SHIP_COST \
            and not (game_map[me.shipyard].is_occupied or shipyard_surrounded or move_into_shipyard):
        command_queue.append(me.shipyard.spawn())
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
