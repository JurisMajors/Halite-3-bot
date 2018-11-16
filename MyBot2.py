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
VERSION = 2
game.ready("Sea_Whackers {}".format(VERSION))

""" <<<Game Loop>>> """
ship_state = {}
ship_dest = {}  # destination -> ship.id
halite_positions = {}  # halite -> position
previous_position = {}  # ship.id-> previous pos
# search area for halite relative to shipyard
SCAN_AREA = 30
PERCENTAGE_SWITCH = 50  # when switch collectable percentage of max halite
SMALL_PERCENTAGE = 0.7
BIG_PERCENTAGE = 0.95
MEDIUM_HALITE = 300  # definition of medium patch size for stopping and collecting patch if on the way
HALITE_STOP = 10  # halite left at patch to stop collecting at that patch
SPAWN_TURN = 220  # until which turn to spawn ships
A = 0
B = 2
C = 1
CRASH_TURN = constants.MAX_TURNS
CRASH_SELECTION_TURN = int(0.8 * constants.MAX_TURNS)


# h_amount <= 0 to run minheap as maxheap
def f(h_amount, h_distance):  # function for determining patch priority
    return h_amount / (A * h_distance * h_distance + B * h_distance + C)


def halitePriorityQ(shipyard_pos, game_map):
    h = []  # stores halite amount * -1 with its position in a minheap
    top_left = Position(int(-1 * SCAN_AREA / 2), int(-1 * SCAN_AREA / 2)) + shipyard_pos  # top left of scan area
    for y in range(SCAN_AREA):
        for x in range(SCAN_AREA):
            p = Position((top_left.x + x) % game_map.width, (top_left.y + y) % game_map.height)  # position of patch
            factor = f(game_map[p].halite_amount * -1, game_map.calculate_distance(p,
                                                                                   shipyard_pos))  # f(negative halite amount,  distance from shipyard to patch)
            halite_positions[factor] = p
            heappush(h, factor)  # add negative halite amounts so that would act as maxheap
    return h


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


def findNewDestination(h, ship_id):
    biggest_halite = heappop(h)  # get biggest halite
    while halite_positions[
        biggest_halite] in ship_dest.values():  # get biggest halite while its a position no other ship goes to
        biggest_halite = heappop(h)
    ship_dest[ship_id] = halite_positions[biggest_halite]  # set the destination


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
                and not has_moved[other_ship.id]:
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


while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    # Djikstra the graph
    game_map.dijkstra(me.shipyard)
    return_percentage = BIG_PERCENTAGE if game.turn_number < PERCENTAGE_SWITCH else SMALL_PERCENTAGE

    command_queue = []
    # priority Q of patch function values of function f(halite, distance)
    h = halitePriorityQ(me.shipyard.position, game_map)
    # has_moved ID->True/False, moved or not
    # ships priority queue of (importance, ship) 
    ships, has_moved = shipPriorityQ(me, game_map)

    # True if a ship moves into the shipyard this turn
    move_into_shipyard = False

    if game.turn_number == CRASH_SELECTION_TURN:
        CRASH_TURN = selectCrashTurn()

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
            findNewDestination(h, ship.id)
            ship_state[ship.id] = "exploring"  # explore

        # transition
        if ship_state[ship.id] == "returning" and game.turn_number >= CRASH_TURN and game_map.calculate_distance(
                ship.position, me.shipyard.position) < 2:
            # if returning after crash turn, suicide
            ship_state[ship.id] = "harakiri"
        elif (ship_state[ship.id] == "collecting" or ship_state[
            ship.id] == "exploring") and game.turn_number >= CRASH_TURN:
            # return if at crash turn
            ship_state, ship_dest = returnShip(ship.id, ship_dest, ship_state)
        elif ship_state[ship.id] == "exploring" and (
                ship.position == ship_dest[ship.id] or game_map[ship.position].halite_amount > MEDIUM_HALITE):
            # collect if reached destination or on medium sized patch
            ship_state[ship.id] = "collecting"
        elif ship_state[ship.id] == "exploring" and ship.halite_amount >= constants.MAX_HALITE * return_percentage:
            # return if ship is 70+% full
            ship_state, ship_dest = returnShip(ship.id, ship_dest, ship_state)
        elif ship_state[ship.id] == "collecting" and (game_map[
                                                          ship.position].halite_amount < HALITE_STOP or ship.halite_amount >= constants.MAX_HALITE * return_percentage):  # return to shipyard if enough halite
            # return if patch has little halite or ship is 70% full
            ship_state, ship_dest = returnShip(ship.id, ship_dest, ship_state)
        elif ship_state[ship.id] == "returning" and ship.position == ship_dest[ship.id]:
            # explore again when back in shipyard
            ship_state[ship.id] = "exploring"
            findNewDestination(h, ship.id)

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
