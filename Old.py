#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt
import pickle
# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position
from hlt.game_map import MapCell
# heap
from heapq import heappush, heappop, merge
# This library allows you to generate random numbers.
import random
from collections import deque

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
import time
import sys
import os
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import numpy as np
from math import ceil
from copy import deepcopy

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

dropoff_clf = pickle.load(open('mlp.sav', 'rb'))
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
# heap for halite positions with their factors for all shipyards in one
game_map = []
shipyard_pos = {}  # shipyard.id -> shipyard position
# ship.id (s.t. fleet state) -> fleet leader ship object, s.t. build state
fleet_leader = {}
ship_obj = {}  # ship.id to ship obj for processing crashed ship stuff
crashed_positions = []  # heap of (-1 * halite, crashed position )
crashed_ship_positions = []  # list of crashed ship positions


heuristic_variables = [0, 0.8, 0, 0.01, 0.98, 1.05]
VARIABLES = ["YEEHAW", 1285, 0.4, 0.9, 0.95, 500, 50, 0.55] + \
    heuristic_variables + [0.9, 0.15, 0.15, 8, 8]
VERSION = VARIABLES[1]
# when switch collectable percentage of max halite
PERCENTAGE_SWITCH = int(float(VARIABLES[2]) * constants.MAX_TURNS)
SMALL_PERCENTAGE = float(VARIABLES[3])
BIG_PERCENTAGE = float(VARIABLES[4])
# definition of medium patch size for stopping and collecting patch if on
# the way
MEDIUM_HALITE = int(VARIABLES[5])
# halite left at patch to stop collecting at that patch
HALITE_STOP = int(VARIABLES[6])
# until which turn to spawn ships
SPAWN_TURN = int(float(VARIABLES[7]) * constants.MAX_TURNS)
# Coefficients for halite heuristics
A = float(VARIABLES[8])
B = float(VARIABLES[9])
C = float(VARIABLES[10])
D = float(VARIABLES[11])
E = float(VARIABLES[12])
F = float(VARIABLES[13])
CRASH_TURN = constants.MAX_TURNS
CRASH_SELECTION_TURN = int(float(VARIABLES[14]) * constants.MAX_TURNS)
# Minimum halite needed to join a halite cluster

DETERMINE_CLUSTER_TURN = int(float(VARIABLES[15]) * constants.MAX_TURNS)
CLUSTER_TOO_CLOSE = float(VARIABLES[16])  # distance two clusters can be within
MAX_CLUSTERS = int(VARIABLES[17])  # max amount of clusters: currently 8
FLEET_SIZE = int(VARIABLES[18])  # fleet size to send for new dropoff

CLOSE_TO_SHIPYARD = 0.18  # close to main shipyard
ENEMY_SHIPYARD_CLOSE = 0.15  # close to enemy dropoff/shipyard
SHIP_SCAN_AREA = 20  # area to scan around ship when changing destinations
EXTRA_FLEET_MAP_SIZE = 32  # which maps >= to send fleets on
# % of patches that have a ship on them for ships to return earlier
BUSY_PERCENTAGE = 0.15
BUSY_RETURN_AMOUNT = 0.5 * constants.MAX_HALITE

MAX_SHIP_DROPOFF_RATIO = 40  # for each 40 ships there should be a dropoff

game.ready("v55")
NR_OF_PLAYERS = len(game.players.keys())
# for dropoff filtering, what should be minimum value of halite around the
# dropoff in 5x5 area
MIN_CLUSTER_VALUE = 6000 if game.game_map.width >= 64 else 8000
if game.game_map.width == 64:
    CRASH_SELECTION_TURN -= 5
ENABLE_COMBAT = True  # this gets changed in the loop, initally true though
TURN_START = 0  # for timing
UNSAFE_AREA = 4
=======
DETERMINE_CLUSTER_TURN = int(float(VARIABLES[17]) * constants.MAX_TURNS)
CLUSTER_TOO_CLOSE = float(VARIABLES[18])  # distance two clusters can be within
MAX_CLUSTERS = int(VARIABLES[19])  # max amount of clusters
FLEET_SIZE = int(VARIABLES[20])  # fleet size to send for new dropoff
TURN_START = 0  # for timing
CLOSE_TO_SHIPYARD = 0.25
ENEMY_SHIPYARD_CLOSE = 0.15
SHIP_SCAN_AREA = 10
EXTRA_FLEET_MAP_SIZE = 32
CHANGE_HEURISTIC_TURN = int(0.3 * constants.MAX_TURNS)
game.ready("OLD")
NR_OF_PLAYERS = len(game.players.keys())

SAVIOR_FLEET_SIZE = 0.1 if NR_OF_PLAYERS == 2 else 0.05
ENABLE_COMBAT = True


def f(h_amount, h_distance):  # function for determining patch priority
    return (A * h_amount * h_amount + B * h_amount + C) / (D * h_distance * h_distance + E * h_distance + F)


def halite_priority_q(pos, area):

    # h_amount <= 0 to run minheap as maxheap
    h = []  # stores halite amount * -1 with its position in a minheap
    top_left = Position(int(-1 * area / 2),
                        int(-1 * area / 2)) + pos  # top left of scan area
    for y in range(area):
        for x in range(area):
            p = Position((top_left.x + x) % game_map.width,
                         (top_left.y + y) % game_map.height)  # position of patch
            if not (p == pos or p in get_dropoff_positions()):
                cell = game_map[p]
                if 0 < cell.halite_amount <= game_map.HALITE_STOP:
                    ratio = cell.halite_amount / \
                        (2 * game_map.calculate_distance(p, pos))
                    heappush(h, (-1 * ratio, p))
                elif cell.halite_amount > 0:
                    factor = cell_factor(pos, cell)
                    # add negative halite amounts so that would act as maxheap
                    heappush(h, (factor, p))
    return h


def cell_factor(cntr, cell):
    if cell.position in get_dropoff_positions():
        return 1000

    neighbours = game_map.get_cells_in_area(cell, 2)
    n_factor_sum = 0
    # get rid of dropoffs
    for neighbour in neighbours[:]:
        inspire_multiplier = get_inspire_multiplier(cntr, neighbour)
        if neighbour.position in get_dropoff_positions():
            neighbours.remove(neighbour)
        elif neighbour.halite_amount <= game_map.HALITE_STOP:
            n_factor_sum += f(neighbour.halite_amount * inspire_multiplier,
                              game_map.calculate_distance(neighbour.position, cntr))
        else:
            n_factor_sum += f(neighbour.halite_amount * inspire_multiplier,
                              game_map.calculate_distance(neighbour.position, cntr))

    multiplier = get_inspire_multiplier(cntr, cell)

    return -1 * (len(neighbours) * f(cell.halite_amount * multiplier, game_map.calculate_distance(cell.position, cntr)) + n_factor_sum)


def should_inspire():
    return (NR_OF_PLAYERS == 2 and game_map.width in [32, 40, 48]) or\
        (NR_OF_PLAYERS == 4 and game_map.width in [32, 40])


def get_inspire_multiplier(cntr, cell):
    if cell.inspired and (should_inspire() or game_map.calculate_distance(cntr, cell.position) < 6):
        return 3
    else:
        return 1


def ship_priority_q(me, game_map):
    ships = []  # ship priority queue
    has_moved = {}
    for s in me.get_ships():
        ship_obj[s.id] = s
        has_moved[s.id] = False
        if s.id in ship_state:
            # get ships shipyard
            shipyard = get_shipyard(s.position)
            # importance, the lower the number, bigger importance
            if s.position == shipyard:
                importance = -10000
            elif ship_state[s.id] in ["returning", "harikiri"]:
                importance = round((game_map.width * -2) / game_map[
                    s.position].dijkstra_distance, 2)
            elif ship_state[s.id] in ["exploring", "build", "backup", "fleet"]:
                if s.id in ship_dest:
                    destination = ship_dest[s.id]
                else:
                    destination = shipyard
                importance = game_map.calculate_distance(
                    s.position, destination) * game_map.width + 1
            else:  # other
                importance = game_map.calculate_distance(
                    s.position, shipyard) * game_map.width ** 2
        else:
            importance = -10000  # newly spawned ships max importance
        heappush(ships, (importance, s))
    return ships, has_moved


def get_shipyard(position):
    """ gives shipyard that the ship would return to """
    # return game_map[position].dijkstra_dest
    return min([(game_map.euclidean_distance(position, d), d) for d in get_dropoff_positions()], key=lambda x: x[0])[1]

def select_crash_turn():
    '''selects turn when to crash'''
    distance = 0
    for ship in me.get_ships():
        shipyard = get_shipyard(ship.position)  # its shipyard position
        d = game_map.calculate_distance(shipyard, ship.position)
        if d > distance:  # get maximum distance away of shipyard
            distance = d
    crash_turn = constants.MAX_TURNS - distance - 5
    # set the crash turn to be turn s.t. all ships make it
    crash_turn = max(crash_turn, CRASH_SELECTION_TURN)
    if game_map.width == 64:
        crash_turn -= 5
    return crash_turn


def dist_to_enemy_doff(pos):
    ''' determines how close to an enemy dropoff is the position'''
    if NR_OF_PLAYERS < 2:
        return 1000
    return min([game_map.euclidean_distance(pos, d) for d in get_enemy_dropoff_positions()])


def halite_priority_q(pos, area):
    # h_amount <= 0 to run minheap as maxheap
    h = []  # stores halite amount * -1 with its position in a minheap
    top_left = Position(int(-1 * area / 2),
                        int(-1 * area / 2)) + pos  # top left of scan area
    for y in range(area):
        for x in range(area):
            p = Position((top_left.x + x) % game_map.width,
                         (top_left.y + y) % game_map.height)  # position of patch
            # we dont consider the position of the centre or dropoffs
            if not p in get_dropoff_positions():
                cell = game_map[p]
                # we ignore cells who have 0 halite.
                # if that cell has small amount of halite, just take a ratio
                # with 2x distance to lesser the priority
                if 0 < cell.halite_amount <= game_map.HALITE_STOP:
                    ratio = cell.halite_amount / \
                        (2 * game_map.calculate_distance(p, pos) + 1)
                    heappush(h, (round(1 / ratio, 2), p))
                elif cell.halite_amount > 0:
                    factor = game_map.cell_factor(pos, cell, me, enable_backup)
                    # add negative halite amounts so that would act as maxheap
                    heappush(h, (factor, p))
    return h



def find_new_destination(h, this_ship):
    ''' h: priority queue of halite factors,
    halite_pos: dictionary of halite factor -> patch position '''

    ship_id = this_ship.id
    biggest_halite, position = heappop(h)  # get biggest halite
    destination = game_map.normalize(position)
    not_first_dest = ship_id in ship_dest

    # repeat while not a viable destination, or enemies around the position or
    # too many ships going to that dropoff area
    # or the same destination as before
    while bad_destination(this_ship, destination) or (NR_OF_PLAYERS == 4 and game_map[destination].enemy_neighbouring > 0):
        # if no more options, return
        if len(h) == 0:
            state_switch(this_ship.id, "returning")
            return
        biggest_halite, position = heappop(h)
        destination = game_map.normalize(position)
    ship_dest[ship_id] = destination  # set the destination
    reassign_duplicate_dests(destination, ship_id) # deal with duplicate destinations
    # if another ship had the same destination
    s = get_ship_w_destination(destination, this_id)
    if s:  # find a new destination for it
        for other in s:
            process_new_destination(other)


def process_new_destination(this_ship):
    """ Processes new destination for ship """ 
    ship_path[this_ship.id] = []
    if this_ship.position in get_dropoff_positions() or this_ship.id not in ship_dest:
        find_new_destination(game_map.halite_priority, this_ship)
    else:
        source = this_ship.position
        ship_h = halite_priority_q(source, SHIP_SCAN_AREA)
        find_new_destination(ship_h, this_ship)

def too_many_near_dropoff(this_ship, destination):
    ''' checks if ship distribution over dropoffs is balanced '''
    if get_shipyard(this_ship.position) == get_shipyard(destination):
        return False
    else:
        return prcntg_ships_returning_to_doff(get_shipyard(destination)) > (1 / len(get_dropoff_positions()))


def bad_destination(this_ship, destination):
    """ definition of unviable destination for ship """
    if game_map[destination].halite_amount == 0:
        return True
    if game.turn_number <= SPAWN_TURN:
        return not dest_viable(destination, this_ship) or game_map[destination].enemy_amount >= UNSAFE_AREA\
            or too_many_near_dropoff(this_ship, destination)
    else:
        return not dest_viable(destination, this_ship) or game_map[destination].enemy_amount >= UNSAFE_AREA


def dest_viable(position, this_ship):
    """ is a destination viable for ship, i.e. 
    if any other ship is going there, if thats the case
    then if that ship is further away than ship """
    if position in ship_dest.values():
        # get another ship with same destination
        inspectable_ships = get_ship_w_destination(position, this_ship.id)
        if not inspectable_ships:  # shouldnt happen but for safety
            # if this ship doesnt exist for some reason
            return True

        my_dist = game_map.calculate_distance(position, this_ship.position)
        their_dist = min([game_map.calculate_distance(
            position, inspectable_ship.position) for inspectable_ship in inspectable_ships])

        # if im closer to destination, assign it to me.
        return my_dist < their_dist
    else:
        return True  # nobody has the best patch, all good


def get_ship_w_destination(dest, this_id):
    """ gets a ship with dest, s.t. that ship is not this_id """
    other_ships = []
    if dest in ship_dest.values():
        for s in ship_dest.keys():  # get ship with the same destination
            if s != this_id and ship_dest[s] == dest and me.has_ship(s):
                other_ships.append(me.get_ship(s))
    return other_ships


def clear_dictionaries():
    # clear dictionaries of crushed ships
    for ship_id in list(ship_dest.keys()):
        if not me.has_ship(ship_id):
            crashed_ship_positions.append(ship_obj[ship_id].position)
            del ship_dest[ship_id]
            del ship_state[ship_id]
            del previous_state[ship_id]
            del previous_position[ship_id]
            if ship_id in ship_path:
                del ship_path[ship_id]


def add_crashed_position(pos):
    """ adds a carshed position ot the crashed positions list '"""
    neighbours = game_map.get_neighbours(game_map[pos])
    h_amount = -1
    distance_to_enemy_dropoff = dist_to_enemy_doff(pos)
    for n in neighbours:
        h_amount = max(h_amount, n.halite_amount)
    if h_amount > 800:
        heappush(crashed_positions, (-1 * h_amount, pos))


def get_dijkstra_move(current_position):
    """
    Gets a move from the map created by Dijkstra
    :return: The target position of the move and the direction of the move.
    """
    cell = game_map[current_position].parent
    new_pos = cell.position
    dirs = game_map.get_target_direction(ship.position, new_pos)
    new_dir = dirs[0] if dirs[0] is not None else dirs[
        1] if dirs[1] is not None else Direction.Still

    return new_pos, new_dir


def make_returning_move(this_ship, has_moved, command_queue):
    """
    Makes a returning move based on Dijkstras and other ship positions.
    """
    if ship_path[this_ship.id]:
        direction = get_step(ship_path[this_ship.id])
        to_go = this_ship.position.directional_offset(direction)
        if direction == Direction.Still or not game_map[to_go].is_occupied:
            return direction

    # Get the cell and direction we want to go to from dijkstra
    target_pos, move = get_dijkstra_move(this_ship.position)
    # Target is occupied
    if game_map[target_pos].is_occupied:
        other_ship = game_map[target_pos].ship
        # target position occupied by own ship
        if me.has_ship(other_ship.id):

            if other_ship.id not in ship_state or ship_state[other_ship.id] in ["exploring", "build", "fleet",
                                                                                "backup"]:
                can_move = other_ship.halite_amount >= game_map[
                           other_ship.position].halite_amount / constants.MOVE_COST_RATIO

                if other_ship.id in ship_dest:
                    can_move = can_move and ship_dest[other_ship.id] != other_ship.position
                # if other ship has enough halite to move, hasnt made a move
                # yet, and if it would move in the ship
                if not has_moved[other_ship.id] and \
                        (can_move or other_ship.position in get_dropoff_positions()):
                    if (other_ship.id not in ship_path or (not ship_path[other_ship.id] or
                                                                other_ship.position.directional_offset(ship_path[other_ship.id][0][0]) == ship.position)):
                        # move stays the same target move
                        # move other_ship to ship.position
                        # hence swapping ships
                        logging.info(f"SWAPPING {this_ship.id} with {other_ship.id}")
                        move_ship_to_position(other_ship, this_ship.position)
                    elif other_ship.id in ship_path and ship_path[other_ship.id] and ship_path[other_ship.id][0][0] == Direction.Still:
                        move = a_star_move(this_ship)
                else: # wait until can move
                    move = Direction.Still

            elif ship_state[other_ship.id] in ["returning", "harakiri"]:  # suiciding or queue
                if has_moved[other_ship.id] or (game.turn_number <= CRASH_TURN and other_ship.position in get_dropoff_positions()):
                    move = Direction.Still
                elif Direction.Still == simulate_make_returning_move(other_ship, has_moved, command_queue): # should never happen but just in case :D
                    move = Direction.Still
            elif ship_state[other_ship.id] in ["collecting", "waiting"]:  # move around these ships
                if ship_state[other_ship.id] == "collecting" and \
                game_map[other_ship.position].halite_amount - game_map[other_ship.position].halite_amount / constants.EXTRACT_RATIO <= game_map.HALITE_STOP:
                    move = Direction.Still
                else:
                    move = a_star_move(this_ship)

        else:  # target position occupied by enemy ship
            move = a_star_move(this_ship)
    return move

def simulate_make_returning_move(other_ship, has_moved, command_queue):
    other_move = produce_move(other_ship)
    command_queue.append(other_ship.move(other_move))
    previous_position[other_ship.id] = other_ship.position
    game_map[other_ship.position.directional_offset(other_move)].mark_unsafe(other_ship)
    if other_move != Direction.Still and game_map[other_ship.position].ship == other_ship:
        game_map[other_ship.position].ship = None
    has_moved[other_ship.id] = True
    return other_move



def a_star_move(this_ship, dest=None):
    if dest is None:  # if returning
        cell = game_map[this_ship.position]
        d_to_dijkstra_dest = game_map.calculate_distance(
            cell.position, cell.dijkstra_dest)
        dest = interim_djikstra_dest(cell).position
    return exploring(this_ship, dest)


def interim_djikstra_dest(source_cell):
    ''' finds the intermediate djikstra destination that is not occupied '''
    cell = source_cell.parent
    while cell.is_occupied and not cell.position in get_dropoff_positions():
        cell = cell.parent
        if time_left() < 0.3:
            logging.info(
                "INTERIM DIJKSTRA DESTINATION: STANDING STIsLL TOO SLOW")
            return source_cell
    return cell


def move_ship_to_position(this_ship, destination):
    ''' moves ship to destination
    precondition: destination one move away'''
    if this_ship.id in ship_path and ship_path[this_ship.id]:
        move = get_step(ship_path[this_ship.id])
    else:
        move = dir_to_dest(this_ship.position, destination)

    has_moved[this_ship.id] = True
    command_queue.append(this_ship.move(move))
    game_map[this_ship.position].ship = None
    game_map[destination].mark_unsafe(this_ship)

def dir_to_dest(pos, dest):
    """ Precondition: 
        position one move away """
    normalized_dest = game_map.normalize(dest)
    for d in Direction.get_all_cardinals():
        new_pos = game_map.normalize(pos.directional_offset(d))
        if new_pos == normalized_dest:
            return d
    return Direction.Still # should never happen

def state_switch(ship_id, new_state):
    if ship_id not in previous_state:
        previous_state[ship_id] = "exploring"
    if ship_id not in ship_state:
        ship_state[ship_id] = previous_state[ship_id]
    if not new_state == "exploring":  # reset path to empty list
        ship_path[ship_id] = []
    if new_state == "returning":
        ship_dest[ship.id] = game_map[ship.position].dijkstra_dest
    previous_state[ship_id] = ship_state[ship_id]
    ship_state[ship_id] = new_state


def prcntg_ships_returning_to_doff(d_pos):
    amount = 0
    for s in me.get_ships():
        eval_pos = s.position if s.id not in ship_dest else ship_dest[s.id]
        if get_shipyard(eval_pos) == d_pos:
            amount += 1
    return amount / (len(me.get_ships()) + 1)


def produce_move(ship):
    if ship.id not in ship_obj:
        ship_obj[ship.id] = ship
    state = ship_state[ship.id]
    destination = ship_dest[ship.id]
    ''' produces move for ship '''

    if ship.halite_amount < game_map[ship.position].halite_amount / constants.MOVE_COST_RATIO:
        logging.info("not enough halite standing still")
        return Direction.Still

    mover = {
        "collecting": collecting,
        "returning": returning,
        "harakiri": harakiri,
        "assassinate": assassinate,
        "exploring": exploring,
        "build": exploring,
        "fleet": exploring,
        "backup": exploring,
    }

    return mover[state](ship, destination)


def collecting(ship, destination):
    return Direction.Still


def returning(ship, destination):
    return make_returning_move(ship, has_moved, command_queue)


def interim_exploring_dest(position, path):
    ''' finds intermediate destination from a direction path that is not occupied '''
    to_go = get_step(path)
    next_pos = game_map.normalize(position.directional_offset(to_go))
    while game_map[next_pos].is_occupied:
        if time_left() < 0.3:
            logging.info("INTERIM EXPLORING STANDING STILL")
            return position
        if not path:
            return next_pos
        to_go = get_step(path)
        next_pos = game_map.normalize(next_pos.directional_offset(to_go))
    return next_pos


def exploring(this_ship, destination):
    if this_ship.position == destination:
        ship_path[this_ship.id] = []
        return Direction.Still
    elif time_left() < 0.1:
        logging.info(f"Exploring ship standing still, {time_left()} left")
        return Direction.Still
    elif this_ship.position in get_dropoff_positions() and game_map.is_surrounded(this_ship.position): # if in a dropoff and surrounded
        # find closest neighbour
        closest_n = None
        closest_dist = None
        for n in game_map.get_neighbours(game_map[this_ship.position]):
            if n.ship.id in ship_state and ship_state[n.ship.id] != "returning": # only swap with returning baybes 
                continue
            dist = game_map.calculate_distance(n.position, destination)
            if closest_dist is None or dist < closest_dist: # set smallest dist and neighbour
                closest_dist = dist
                closest_n = n

        if closest_n is None: # wait for exploring ships to go away
            return Direction.Still

        # if enemy ship
        if not me.has_ship(closest_n.ship.id):
            # kill him
            ship_state[this_ship.id] = "assassinate"
            ship_dest[this_ship.id] = closest_n.position
            return dir_to_dest(this_ship.position, ship_dest[this_ship.id]) 
        # else its our ship
        # so swap with it
        move_ship_to_position(closest_n.ship, this_ship.position)
        move_ship_to_position(this_ship, closest_n.position)
        game_map[this_ship.position].ship = closest_n.ship
        ship_path[this_ship.id] = []
        return None
    elif game_map.is_surrounded(this_ship.position):
        return Direction.Still

    # next direction occupied, recalculate
    if this_ship.id not in ship_path or not ship_path[this_ship.id]:
        ship_path[this_ship.id] = game_map.explore(this_ship, destination)
    else:
        direction = ship_path[this_ship.id][0][0]
        next_pos = this_ship.position.directional_offset(direction)
        if game_map[next_pos].is_occupied and not direction == Direction.Still:
            other_ship = game_map[next_pos].ship
            # move to intermediate destination, aka move around
            new_dest = interim_exploring_dest(
                this_ship.position, ship_path[this_ship.id])
            # use intermediate unoccpied position instead of actual dest
            ship_path[this_ship.id] = game_map.explore(
                this_ship, new_dest) + ship_path[this_ship.id]
                # add rest of the path, interim path + rest of path
    # move in calculated direction
    return get_step(ship_path[this_ship.id])


def get_step(path):
    path[0][1] -= 1  # take that direction
    direction = path[0][0]
    if path[0][1] == 0:  # if no more left that direction remove it
        del path[0]
    return direction


def harakiri(ship, destination):
    """ pre: next to or on shipyard """
    shipyard = get_shipyard(ship.position)
    ship_pos = game_map.normalize(ship.position)
    if ship.position == shipyard:  # if at shipyard
        return Direction.Still  # let other ships crash in to you
    else:  # otherwise move to the shipyard
        target_dir = dir_to_dest(ship.position, shipyard)
        return target_dir


def assassinate(ship, destination):
    state_switch(ship.id, previous_state[ship.id])
    if game_map.calculate_distance(ship.position, destination) == 1:
        target_direction = game_map.get_target_direction(
            ship.position, destination)
        return target_direction[0] if target_direction[0] is not None else target_direction[1]
    else:
        return exploring(ship, destination)

def get_best_neighbour(position):
    ''' gets best neighbour at the ships positioņ
    returns a cell'''
    current = game_map[position]
    neighbours = game_map.get_neighbours(current)
    max_halite = current.halite_amount * \
        game_map.get_inspire_multiplier(position, current, enable_backup)
    best = current
    for n in neighbours:
        n_halite = n.halite_amount * \
            game_map.get_inspire_multiplier(position, n, enable_backup)
        if n.enemy_amount < UNSAFE_AREA and n_halite > max_halite:
            best = n
            max_halite = n_halite

    return best


def exists_better_in_area(cntr, current, area):
    top_left = Position(int(-1 * area / 2),
                        int(-1 * area / 2)) + cntr  # top left of scan area
    current_factor = game_map.cell_factor(
        cntr, game_map[current], me, NR_OF_PLAYERS)
    for y in range(area):
        for x in range(area):
            p = Position((top_left.x + x) % game_map.width,
                         (top_left.y + y) % game_map.height)  # position of patch
            cell = game_map[p]
            if cell.halite_amount >= game_map.HALITE_STOP:
                other_factor = game_map.cell_factor(
                    cntr, cell, me, enable_backup)
                if not cell.is_occupied and other_factor < current_factor:
                    return True
    return False


def better_patch_neighbouring(ship, big_diff):
    ''' returns true if there is a lot better patch right next to it'''
    current = game_map[ship.position]
    neighbours = game_map.get_neighbours(current)
    current_h = current.halite_amount * \
        game_map.get_inspire_multiplier(
            ship.position, game_map[ship.position], enable_backup)

    for n in neighbours:
        neighbour_h = n.halite_amount * \
            game_map.get_inspire_multiplier(
                ship.position, game_map[n.position], enable_backup)
        if n.enemy_amount < UNSAFE_AREA and neighbour_h >= current_h + big_diff:
            return True

    return False


def get_enemy_dropoff_positions():
    ''' returns a list of enemy dropoffs, including shipyards '''
    positions = []
    for player in game.players.values():  # for each player in game
        if not player.id == me.id:  # if not me
            positions.append(player.shipyard.position)
            for d_off in player.get_dropoffs():
                positions.append(d_off.position)
    return positions


def exploring_transition(ship):
    new_state = None
    distance_to_dest = game_map.calculate_distance(
        ship.position, ship_dest[ship.id])
    euclid_to_dest = game_map.euclidean_distance(
        ship.position, ship_dest[ship.id])

    if ship.position == ship_dest[ship.id]:
        # collect if reached destination or on medium sized patch
        new_state = "collecting"

    elif game_map[ship.position].halite_amount >= MEDIUM_HALITE: 
        new_state = "collecting"
        ship_dest[ship.id] = ship.position
        ship_path[ship.id] = []
        reassign_duplicate_dests(ship_dest[ship.id], ship.id)

    elif game_map[ship.position].inspired and game_map[ship.position].enemy_amount <= UNSAFE_AREA:

        # for inspiring
        ship_dest[ship.id] = get_best_neighbour(ship.position).position
        reassign_duplicate_dests(ship_dest[ship.id], ship.id)


    elif euclid_to_dest <= 2\
            and exists_better_in_area(ship.position, ship_dest[ship.id], 4):
        process_new_destination(ship)

    elif ENABLE_COMBAT and (distance_to_dest > CLOSE_TO_SHIPYARD * game_map.width or distance_to_dest == 1)\
            and dist_to_enemy_doff(ship.position) >= ENEMY_SHIPYARD_CLOSE * game_map.width:
        new_state = attempt_switching_assasinate(ship)

    return new_state


def collecting_transition(ship):
    new_state = None
    inspire_multiplier = game_map.get_inspire_multiplier(
        ship.position, game_map[ship.position], enable_backup)
    cell_halite = game_map[ship.position].halite_amount * inspire_multiplier

    if ship.is_full:
        new_state = "returning"

    elif NR_OF_PLAYERS == 4 and game_map.percentage_occupied >= BUSY_PERCENTAGE and ship.halite_amount >= BUSY_RETURN_AMOUNT:
        # in 4 player games return earlier if many ships on map
        new_state = "returning"

    elif ship.halite_amount >= constants.MAX_HALITE * return_percentage and \
            not (cell_halite > MEDIUM_HALITE * inspire_multiplier and not ship.is_full):
        # return to shipyard if enough halite
        new_state = "returning"

    elif ship.halite_amount < constants.MAX_HALITE * return_percentage * 0.9 and game_map[ship.position].halite_amount <= 100 and \
            better_patch_neighbouring(ship, game_map[ship.position].halite_amount):
        # explore to best neighbour if current cell has low halite, and there is a 2x patch next to it
        ship_dest[ship.id] = get_best_neighbour(ship.position).position
        new_state = "exploring"
        reassign_duplicate_dests(ship_dest[ship.id], ship.id)

    elif (cell_halite < game_map.HALITE_STOP * inspire_multiplier and ship.halite_amount < constants.MAX_HALITE * return_percentage * .9):
        # Keep exploring if halite patch low on halite and ship doesnt have close to returning percentage
        process_new_destination(ship)
        new_state = "exploring"

    elif cell_halite < game_map.HALITE_STOP * inspire_multiplier:
        # return if cell has low halite
        new_state = "returning"

    elif ship.halite_amount <= constants.MAX_HALITE * 0.4 and ENABLE_COMBAT\
            and dist_to_enemy_doff(ship.position) >= CLOSE_TO_SHIPYARD * game_map.width and game_map[ship.position].enemy_neighbouring:
        # if not that mch halite and not too close to assasinate and enemy is neighbouring, attempt to kill sm1 
        new_state = attempt_switching_assasinate(ship)

    elif game_map[ship.position].enemy_neighbouring > 0 and ship.halite_amount >= MEDIUM_HALITE and game_map[ship.position].halite_amount <= MEDIUM_HALITE:  # if enemy right next to it
        # move to neighbour that has minimal enemies, but more halite
        ratio = cell_halite / (game_map[ship.position].enemy_neighbouring + 1)
        next_dest = ship.position
        for n in game_map.get_neighbours(game_map[ship.position]):
            n_ratio = n.halite_amount * game_map.get_inspire_multiplier(
                ship.position, n, enable_backup) / (n.enemy_neighbouring + 1)
            if n_ratio > ratio:
                ratio = n_ratio
                next_dest = n.position
        ship_path[ship.id] = []
        new_state = "exploring"
        ship_dest[ship.id] = n.position
        reassign_duplicate_dests(ship_dest[ship.id], ship.id)
            not (cell_halite * inspire_multiplier > MEDIUM_HALITE and not ship.is_full):
                             # return to shipyard if enough halite
        new_state = "returning"

    elif cell_halite < game_map.HALITE_STOP * inspire_multiplier:
        # Keep exploring if current halite patch is empty

        process_new_destination(ship)
        new_state = "exploring"

    if ship.halite_amount <= constants.MAX_HALITE * 0.5 and NR_OF_PLAYERS == 2 and ENABLE_COMBAT:
        # if not so close
        # check if neighbours have an enemy nearby with 2x more halite
        # if so, kill him
        for n in game_map.get_neighbours(game_map[ship.position]):
            if n.is_occupied and not me.has_ship(n.ship.id):
                if n.ship.halite_amount >= 1.8 * ship.halite_amount:
                    logging.info("ASSASINATING")
                    new_state = "assassinate"
                    ship_dest[ship.id] = n.position
    return new_state


def returning_transition(ship):
    new_state = None
    if game.turn_number >= CRASH_TURN and\
            game_map[ship.position].parent.is_occupied and not me.has_ship(game_map[ship.position].parent.ship.id):
        new_state = "assassinate"
        ship_dest[ship.id] = game_map[ship.position].parent.position

    elif ship.position in get_dropoff_positions():
        # explore again when back in shipyard
        new_state = "exploring"
        process_new_destination(ship)

    elif game_map.calculate_distance(ship.position, game_map[ship.position].dijkstra_dest) == 1:
        # if next to a dropoff
        cell = game_map[game_map[ship.position].dijkstra_dest]
        if cell.is_occupied and not me.has_ship(cell.ship.id) and "harakiri" not in ship_state.values():
            new_state = "harakiri"

    return new_state


def fleet_transition(ship):
    new_state = None
    destination = ship_dest[ship.id]
    if ship.position == destination:  # if arrived
        ship_path[ship.id] = []
        new_state = "collecting"
    elif game_map.calculate_distance(ship.position, destination) == 1:
        if game_map[destination].is_occupied:
            ship_dest[ship.id] = get_best_neighbour(destination).position
            reassign_duplicate_dests(ship_dest[ship.id], ship.id)

    elif ship.id in fleet_leader:
        leader = fleet_leader[ship.id]
        if me.has_ship(leader.id) and ship_state[leader.id] not in ["waiting", "build"]:
            process_new_destination(ship)
            new_state = "exploring"
    return new_state


def builder_transition(ship):
    new_state = None
    # if someone already built dropoff there before us
    future_dropoff_cell = game_map[ship_dest[ship.id]]
    distance_to_dest = game_map.euclidean_distance(
        ship.position, ship_dest[ship.id])
    if future_dropoff_cell.has_structure:
        if distance_to_dest <= CLOSE_TO_SHIPYARD * game_map.width:
            ship_dest[ship.id] = bfs_unoccupied(future_dropoff_cell.position)
            reassign_duplicate_dests(ship_dest[ship.id], ship.id)
        else:
            process_new_destination(ship)
            new_state = "exploring"

    elif game_map[future_dropoff_cell.position].enemy_amount >= UNSAFE_AREA:
        new_state = "exploring"
        process_new_destination(ship)

    elif distance_to_dest <= 2:
        neighbours = game_map.get_neighbours(game_map[ship.position])
        for n in neighbours:
            if n.is_occupied and not me.has_ship(n.ship.id):
                new_state = "build"
                ship_dest[ship.id] = get_best_neighbour(ship.position).position
                reassign_duplicate_dests(ship_dest[ship.id], ship.id)
                break
    elif NR_OF_PLAYERS == 4 or game_map.width >= 56: # for 4 players and large maps 1v1
        smallest_dist = dist_to_enemy_doff(ship_dest[ship.id])
        if smallest_dist <= game_map.width * ENEMY_SHIPYARD_CLOSE:
            process_new_destination(ship)
            new_state = "exploring"
    return new_state


def waiting_transition(ship):
    new_state = None
    neighbours = game_map.get_neighbours(game_map[ship.position])
    cell = game_map[ship.position]
    for n in neighbours:
        if n.is_occupied and not me.has_ship(n.ship.id):
            new_state = "build"
            ship_dest[ship.id] = get_best_neighbour(ship.position).position
            reassign_duplicate_dests(ship_dest[ship.id], ship.id)
            break
    else:
        if cell.halite_amount <= 500 or (ship.is_full and cell.halite_amount <= 800):
            new_state = "build"
            ship_dest[ship.id] = get_best_neighbour(ship.position).position
            reassign_duplicate_dests(ship_dest[ship.id], ship.id)

    return new_state


def backup_transition(ship):
    new_state = None
    destination = ship_dest[ship.id]
    if ship.position == destination:  # if arrived
        ship_path[ship.id] = []
        new_state = "collecting"
    elif ship.halite_amount >= constants.MAX_HALITE * return_percentage:
        new_state = "returning"
    elif game_map.calculate_distance(ship.position, destination) == 1:
        if game_map[destination].is_occupied and me.has_ship(game_map[destination].ship.id):
            ship_dest[ship.id] = get_best_neighbour(destination).position
            reassign_duplicate_dests(ship_dest[ship.id], ship.id)
        elif game_map[destination].is_occupied:  # not our ship
            new_state = attempt_switching_assasinate(ship)
    elif game_map[destination].enemy_amount >= UNSAFE_AREA:
        new_state = "exploring"
        process_new_destination(ship)
    elif ENABLE_COMBAT and dist_to_enemy_doff(ship.position) >= CLOSE_TO_SHIPYARD * game_map.width:
        new_state = attempt_switching_assasinate(ship)

    return new_state


def attempt_switching_assasinate(ship):
    new_state = None
    for n in game_map.get_neighbours(game_map[ship.position]):
        if n.is_occupied and not me.has_ship(n.ship.id):
            # if that ship has 2x the halite amount, decent amount of halite,
            # and if tehre are ships to send as backup
            if (n.ship.halite_amount + n.halite_amount) >= 2 * (ship.halite_amount + game_map[ship.position].halite_amount) and \
                    n.ship.halite_amount >= MEDIUM_HALITE and\
                    enough_backup_nearby(n.position, int(0.15 * game_map.width), 2):
                # assasinate that mofo
                logging.info("ASSASINATING")
                new_state = "assassinate"
                ship_dest[ship.id] = n.position
                break
    return new_state



def enough_backup_nearby(pos, distance, amount):
    """ boolean function for determining whether there are 
    amount of friendly ships within distance from pos """
    actual_amount = 0
    for my_ship in me.get_ships():
        distance_to_pos = game_map.euclidean_distance(pos, my_ship.position)
        if distance_to_pos <= distance and is_savior(my_ship):
            actual_amount += 1
    return actual_amount >= amount


def state_transition(ship):
    # transition
    new_state = None
    shipyard = get_shipyard(ship.position)

    if game.turn_number >= CRASH_TURN and game_map.calculate_distance(
            ship.position, shipyard) < 2:
        # if next to shipyard after crash turn, suicide
        new_state = "harakiri"

    elif game.turn_number >= CRASH_TURN:
        # return if at crash turn
        new_state = "returning"

    elif ship.position in get_dropoff_positions():
        process_new_destination(ship)
        new_state = "exploring"

    # decent halite and close to enemy dropoff, return
    elif ship.halite_amount >= 0.5 * constants.MAX_HALITE and dist_to_enemy_doff(ship.position) < 0.1 * game_map.width:
        new_state = "returning"

    elif ship.halite_amount >= constants.MAX_HALITE * return_percentage and ship_state[ship.id] not in ["build", "waiting", "collecting", "returning"]:
        new_state = "returning"

    elif ship_state[ship.id] == "exploring":
        new_state = exploring_transition(ship)

    elif ship_state[ship.id] == "collecting":
        new_state = collecting_transition(ship)

    elif ship_state[ship.id] == "returning":
        new_state = returning_transition(ship)

    elif ship_state[ship.id] == "fleet":
        new_state = fleet_transition(ship)

    elif ship_state[ship.id] == "build":
        new_state = builder_transition(ship)

    elif ship_state[ship.id] == "waiting":
        new_state = waiting_transition(ship)

    elif ship_state[ship.id] == "backup":
        new_state = backup_transition(ship)

    if new_state is not None:
        state_switch(ship.id, new_state)


def get_dropoff_positions():
    """ Returns a list of all positions of dropoffs and the shipyard """
    return [dropoff.position for dropoff in me.get_dropoffs()] + [me.shipyard.position]


def is_savior(ship):
    return me.has_ship(ship.id) and ship.halite_amount <= return_percentage * 0.5 * constants.MAX_HALITE \
        and (ship.id not in ship_state or not (ship_state[ship.id] in ["waiting", "returning", "build"]))

def is_fleet(ship):
    ''' returns if a ship is good for adding it to a fleet '''
    return me.has_ship(ship.id) and (
        ship.id not in ship_state or not (ship_state[ship.id] in ["fleet", "waiting", "returning", "build"]))


def get_fleet(position, fleet_size, condition=is_fleet):
    ''' returns list of fleet_size amount
    of ships closest to the position'''
    distances = []
    for s in me.get_ships():
        if condition(s) and not s.position == position:
            distances.append(
                (game_map.calculate_distance(position, s.position), s))
    distances.sort(key=lambda x: x[0])
    fleet_size = min(len(distances), fleet_size)
    return [t[1] for t in distances[:fleet_size]]


def bfs_unoccupied(position):
    # bfs for closest cell
    Q = deque([])
    visited = set()
    Q.append(position)
    while Q:
        cur = Q.popleft()
        visited.add(cur)
        if not (game_map[cur].is_occupied or game_map[cur].has_structure):

            return cur
        for neighbour in game_map.get_neighbours(game_map[cur]):
            if not neighbour.position in visited:
                Q.append(neighbour.position)
                visited.add(neighbour.position)


def builder_at_destination(ship):
    ''' checks if this ship is a builder '''
    return ship_state[ship.id] == "waiting" or (ship_state[ship.id] == "build" and ship_dest[ship.id] == ship.position)


def fleet_availability():
    ''' returns how many ships are available atm'''
    amount = 0
    for s in me.get_ships():
        if is_fleet(s):
            amount += 1
    if amount >= len(list(me.get_ships())) * 0.9:
        amount /= 2
    return int(amount)


def should_build():
    """ definition of sending a ship to build """
    if len(me.get_ships()) / len(get_dropoff_positions()) >= MAX_SHIP_DROPOFF_RATIO and not any_builders():
        # there are more than 40 ships per dropoff
        if cluster_centers:  # there is already a dropoff position
            return True
        # there is no good dropoff position yet, make one
        logging.info("determining extra cluster centers")
        game_map.set_close_friendly_ships(me)
        pos = game_map.get_most_dense_dropoff_position(get_dropoff_positions())
        cluster_centers.append((10000, pos))  # fake 10000 halite for new needed cluster
        logging.info(cluster_centers)
        return True

    # Original dropoff code
    return clusters_determined and game.turn_number >= dropoff_last_built + 15 and cluster_centers\
        and len(me.get_ships()) > (len(get_dropoff_positions()) + 1) * FLEET_SIZE\
        and fleet_availability() >= 1.5 * FLEET_SIZE and not any_builders()


def any_builders():
    return "waiting" in ship_state.values() or "build" in ship_state.values()

def send_ships(pos, ship_amount, new_state, condition=is_fleet, leader=None):
    '''sends a fleet of size ship_amount to explore around pos
    new_state : state to switch the fleet members
    condition : boolean function that qualifies a ship to send'''
    fleet = get_fleet(game_map.normalize(pos), ship_amount, condition)
    # for rest of the fleet to explore
    h = halite_priority_q(pos, SHIP_SCAN_AREA)

    for fleet_ship in fleet:  # for other fleet members
        if len(h) == 0:
            break
        if fleet_ship.id not in previous_state.keys():  # if new ship
            previous_state[fleet_ship.id] = "exploring"
            has_moved[fleet_ship.id] = False
        # explore in area of the new dropoff
        if leader is not None:
            fleet_leader[fleet_ship.id] = leader

        state_switch(fleet_ship.id, new_state)
        find_new_destination(h, fleet_ship)


def get_cell_data(x, y, center):
    cell = game_map[Position(x, y)]
    # normalized data of cell: halite amount and distance to shipyard
    return [round(cell.halite_amount / 1000, 2),
            round(game_map.calculate_distance(cell.position, center) / game_map.width, 2)]


def get_patch_data(x, y, center):
    # pool + 1 x pool + 1 size square inspected for data (classifier trained
    # on 5x5)
    pool = 4
    # add center info
    total_halite = 0  # total 5x5 patch halite
    cntr_cell_data = get_cell_data(x, y, center)
    biggest_cell = Position(x, y)
    biggest_halite = cntr_cell_data[0]
    # data must contain normalized game_size
    area_d = [round(game_map.width / 64, 2)] + cntr_cell_data

    for diff_x in range(-1 * int(pool / 2), int(pool / 2) + 1):
        for diff_y in range(-1 * int(pool / 2), int(pool / 2) + 1):

            new_coord_x, new_coord_y = x - diff_x, y - \
                diff_y  # get patch coordinates from centr
            total_halite += game_map[Position(new_coord_x,
                                              new_coord_y)].halite_amount  # add to total halite
            c_data = get_cell_data(new_coord_x, new_coord_y, center)

            if biggest_halite < c_data[0]:  # determine cell with most halite
                biggest_halite = c_data[0]
                biggest_cell = Position(new_coord_x, new_coord_y)

            area_d += c_data

    return [area_d], total_halite, biggest_cell


def find_center():
    ''' finds center of our part of the map '''
    travel = int(game_map.width / NR_OF_PLAYERS)
    # get all the centers depending on the amount of players
    if NR_OF_PLAYERS == 4:
        cntrs = [Position(travel, travel), Position(travel * 3, travel),
                 Position(travel * 3, travel * 3), Position(travel, travel * 3)]
    elif NR_OF_PLAYERS == 2:
        cntrs = [Position(int(travel / 2), travel),
                 Position(travel + int(travel / 2), travel)]
    else:
        cntrs = [me.shipyard.position]

    min_dist = 1000
    # find the center thats the closes to the shipyard
    for pos in cntrs:
        dist = game_map.calculate_distance(pos, me.shipyard.position)
        if dist < min_dist:
            cntr = pos
            min_dist = dist
    return cntr


def predict_centers():
    cntr = find_center()

    # get area around our cntr
    x_size = int(game_map.width /
                 2) if NR_OF_PLAYERS in [2, 4] else game_map.width
    y_size = game_map.height if NR_OF_PLAYERS in [2, 1] else int(
        game_map.height / 2)
    diff1, diff2, diff3, diff4 = (0, 0, 0, 0)

    # in 4 player maps limit the scanning area so that we dont classify centers of map
    # or too close to enemies
    if NR_OF_PLAYERS == 4:
        if cntr.x > x_size:  # if right side of map
            diff1 = 2
            diff2 = 3
        else:
            diff2 = -2

        if cntr.y > y_size:
            diff3 = 2

    cluster_centers = []
    # classify areas of the map
    for x in range(cntr.x - int(x_size / 2) + diff1,
                   cntr.x + int(x_size / 2) + diff2, 5):
        for y in range(cntr.y - int(y_size / 2) + diff3,
                       cntr.y + int(y_size / 2) + diff4, 5):
            p_data, total_halite, p_center = get_patch_data(
                x, y, cntr)  # get the data
            prediction = dropoff_clf.predict(p_data)[0]  # predict on it
            p_center = game_map.normalize(p_center)
            if prediction == 1:  # if should be dropoff
                # add node with most halite to centers
                for _, c in cluster_centers:
                    if c == p_center:
                        break
                else:
                    cluster_centers.append((total_halite, p_center))
    return cluster_centers


def clusters_with_classifier():
    ''' uses classifier to determine clusters for dropoff '''
    cluster_centers = predict_centers()
    logging.info(f"found {len(cluster_centers)} clusters from classifier")
    # do filtering
    cluster_centers = filter_clusters(cluster_centers, MAX_CLUSTERS)
    logging.info(f"found {len(cluster_centers)} clusters after filtering")
    logging.info("Finally")
    logging.info(cluster_centers)
    return cluster_centers


def filter_clusters(centers, max_centers):
    '''filters cluster centres on some human logic '''
    centers.sort(key=lambda x: x[0], reverse=True)  # sort by halite amount
    if len(centers) > max_centers:  # if more than max centres specified
        centers = centers[:max_centers]  # get everything until that index
    centers_copy = centers[:]  # copy to remove stuff from original

    logging.info("additional filtering")
    for d in centers_copy:
        halite, pos = d

        if halite < MIN_CLUSTER_VALUE:  # if area contains less than 8k then remove it
            if d in centers:  # if not removed arldy
                centers.remove(d)

    logging.info(centers)
    # do clustering algorithm on classified patches
    if len(centers) > 2:
        centers = merge_clusters(centers)
    logging.info(centers)

    centers_copy = centers[:]
    # remove points that are too close to each other or the shipyard
    # in priority of points that are have the largest amount of points in area
    for i, d in enumerate(centers_copy, start=0):
        halite, pos = d
        diff = game_map.euclidean_distance(pos, me.shipyard.position)
        if diff < CLOSE_TO_SHIPYARD * game_map.width\
                or dist_to_enemy_doff(pos) < CLOSE_TO_SHIPYARD * game_map.width\
                or halite < 0.7 * MIN_CLUSTER_VALUE:
            if d in centers:
                centers.remove(d)
            continue

        if i < len(centers_copy) - 1:  # if not out of bounds
            # get list of centers too close
            r = too_close(centers_copy[i + 1:], pos)
            for t in r:
                if t in centers:
                    centers.remove(t)  # remove those centers

    return centers


def merge_clusters(centers):
    ''' merges clusters using clustering in 3D where
    x: x
    y: y
    z: halite amount '''

    logging.info("Merging clusters")
    area = CLUSTER_TOO_CLOSE * game_map.width
    metric = distance_metric(type_metric.USER_DEFINED, func=custom_dist)
    normalizer = 1
    X = []  # center coordinates that are merged in an iteration
    tmp_centers = []  # to not modify the list looping through
    history = []  # contains all already merged centers
    # for each center
    for c1 in centers:
        # add the center itself
        X.append([c1[1].x, c1[1].y, round(c1[0] / normalizer, 2)])
        for c2 in centers:  # for other centers
            # if not merged already
            if not c2 == c1:
                dist = game_map.euclidean_distance(c1[1], c2[1])
                # if close enough for merging
                if dist <= area and not dist <= 4:
                    X.append([c2[1].x, c2[1].y, round(c2[0] / normalizer, 2)])

        # get initialized centers for the algorithm
        init_centers = kmeans_plusplus_initializer(X, 1).initialize()
        median = kmedians(X, init_centers, metric=metric)
        median.process()  # do clustering
        # get clustered centers
        tmp_centers += [(x[2] * normalizer, game_map.normalize(Position(int(x[0]), int(x[1]))))
                        for x in median.get_medians() if
                        (x[2], game_map.normalize(Position(int(x[0]), int(x[1])))) not in tmp_centers]
        if len(X) > 1:
            history += X[1:]
        X = []

    centers = tmp_centers
    centers.sort(key=lambda x: x[0], reverse=True)  # sort by best patches
    return centers


def custom_dist(p1, p2):
    ''' distance function for a clustering algorithm,
    manh dist + the absolute difference in halite amount '''
    if len(p1) < 3:
        p1 = p1[0]
        p2 = p2[0]
    euclid_dist = game_map.euclidean_distance(
        Position(p1[0], p1[1]), Position(p2[0], p2[1]))
    return euclid_dist + abs(p1[2] - p2[2])


def too_close(centers, position):
    ''' removes clusters that are too close to each other '''
    to_remove = []
    for d in centers:
        _, other = d
        distance = game_map.euclidean_distance(position, other)
        if distance < CLUSTER_TOO_CLOSE * game_map.width:
            to_remove.append(d)
    return to_remove


def time_left():
    return round(2 - (time.time() - TURN_START), 2)


def have_less_ships(ratio):
    for player in game.players.values():
        if len(me.get_ships()) < ratio * len(player.get_ships()):
            return True
    return False


def process_building(cluster_centers):
    dropoff_val, dropoff_pos = cluster_centers.pop(0)  # remove from list
    # sends ships to position where closest will build dropoff
    dropoff_pos = game_map.normalize(dropoff_pos)
    fleet = get_fleet(dropoff_pos, 1)
    if fleet:
        closest_ship = fleet.pop(0)  # remove and get closest ship
        state_switch(closest_ship.id, "build")  # will build dropoff
        # if dropoffs position already has a structure (e.g. other dropoff) or
        # somebody is going there already
        if game_map[dropoff_pos].has_structure or dropoff_pos in ship_dest.values():
            # bfs for closer valid unoccupied position
            dropoff_pos = bfs_unoccupied(dropoff_pos)

        ship_dest[closest_ship.id] = dropoff_pos  # go to the dropoff
        if game_map.width >= EXTRA_FLEET_MAP_SIZE \
                and game_map.euclidean_distance(get_shipyard(dropoff_pos), dropoff_pos) >= 0.2 * game_map.width:
            value_ratio = dropoff_val / 12_000
            send_ships(dropoff_pos, int(FLEET_SIZE * 0.6 *
                                        value_ratio), "fleet", leader=closest_ship)
    else:  # if builder not available
        cluster_centers.insert(0, (dropoff_val, dropoff_pos))


def get_ship_amount(playerID):
    return len(game.players[playerID].get_ships())


def max_enemy_ships():
    if NR_OF_PLAYERS not in [2, 4]:  # for testing solo games
        return 100000
    ships = 0
    for player_id in game.players.keys():
        if not player_id == me.id:
            ships = max(ships, get_ship_amount(player_id))
    return ships


def process_backup_sending():
    """ Processes sending backup ships to a position
    where a ship crashed previously """
    to_remove = []
    for pos in crashed_ship_positions:
        if dist_to_enemy_doff(pos) > 4:
            add_crashed_position(pos)
        to_remove.append(pos)
    for s in to_remove:  # remove the crashed positions
        if s in crashed_ship_positions:
            crashed_ship_positions.remove(s)
    if crashed_positions:  # if there are any crashed positions to process
        hal, crashed_pos = heappop(crashed_positions)  # get pos info
        # if there are little enemies in that area
        if game_map[crashed_pos].enemy_amount <= UNSAFE_AREA and game_map[crashed_pos].halite_amount >= constants.MAX_HALITE:
            # send a backup fleet there
            send_ships(crashed_pos, 2, "backup", is_savior)



def get_ship_amount(playerID):
    return len(game.players[playerID].get_ships())


def max_enemy_ships():
    if NR_OF_PLAYERS not in [2, 4]:  # for testing solo games
        return 100000
    ships = 0
    for player_id in game.players.keys():
        if not player_id == me.id:
            ships = max(ships, get_ship_amount(player_id))
    return ships


clusters_determined = False
enable_backup = True
INITIAL_HALITE_STOP = HALITE_STOP
dropoff_last_built = 0
prcntg_halite_left = 1
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    command_queue = []
    swapped = set() # for swapping destinations
    clear_dictionaries()  # of crashed or transformed ships
    if game.turn_number == 1:
        game_map.HALITE_STOP = INITIAL_HALITE_STOP
        game_map.c = [A, B, C, D, E, F]  # set the heuristic constants
    for s in me.get_ships():
        if s.id not in ship_state:
            ship_state[s.id] = "exploring"

    ENABLE_COMBAT = not have_less_ships(0.9) and NR_OF_PLAYERS == 2
    TURN_START = time.time()

    enable_inspire = not have_less_ships(0.8)
    enable_backup = ENABLE_COMBAT
    # initialize shipyard halite, inspiring stuff and other
    game_map.init_map(me, list(game.players.values()), enable_inspire, enable_backup)
    logging.info(f"map init took {time.time() - TURN_START} time")
    if game.turn_number == 1:
        TOTAL_MAP_HALITE = game_map.total_halite

    prcntg_halite_left = game_map.total_halite / TOTAL_MAP_HALITE
    # if clusters_determined and not cluster_centers:
    if game.turn_number >= SPAWN_TURN:
        game_map.HALITE_STOP = prcntg_halite_left * INITIAL_HALITE_STOP

    if crashed_ship_positions and game.turn_number < CRASH_TURN and enable_backup:
        process_backup_sending()

    # Dijkstra the graph based on all dropoffs
    game_map.create_graph(get_dropoff_positions())

    if game.turn_number == DETERMINE_CLUSTER_TURN:
        clusters_determined = True
        cluster_centers = clusters_with_classifier()

    if game.turn_number == CRASH_SELECTION_TURN:
        CRASH_TURN = select_crash_turn()
    if prcntg_halite_left > 0.2:
        return_percentage = BIG_PERCENTAGE if game.turn_number < PERCENTAGE_SWITCH else SMALL_PERCENTAGE
    else:  # if low percentage in map, return when half full
        return_percentage = 0.6
    if should_build():
        process_building(cluster_centers)

    # has_moved ID->True/False, moved or not
    # ships priority queue of (importance, ship)
    ships, has_moved = ship_priority_q(me, game_map)
    # True if a ship moves into the shipyard this turn
    move_into_shipyard = False
    # whether a dropoff has been built this turn so that wouldnt use too much
    # halite
    dropoff_built = False

    while ships:  # go through all ships
        ship = heappop(ships)[1]
        if has_moved[ship.id]:
            continue
        if time_left() < 0.15:
            logging.info("STANDING STILL TOO SLOW")
            command_queue.append(ship.stay_still())
            ship_state[ship.id] = "collecting"
            continue
        if ship.id not in previous_position:  # if new ship the
            previous_position[ship.id] = me.shipyard.position

        # setup state
        # if ship hasnt received a destination yet
        if ship.id not in ship_dest or not ship.id in ship_state:
            process_new_destination(ship)
            previous_state[ship.id] = "exploring"
            ship_state[ship.id] = "exploring"  # explore
        # transition
        state_transition(ship)

        logging.info("SHIP {}, STATE {}, DESTINATION {}".format(
            ship.id, ship_state[ship.id], ship_dest[ship.id]))

        # if ship is dropoff builder
        if builder_at_destination(ship):
            # if enough halite and havent built a dropoff this turn
            if (ship.halite_amount + me.halite_amount + game_map[ship.position].halite_amount) >= constants.DROPOFF_COST and not dropoff_built:
                command_queue.append(ship.make_dropoff())
                dropoff_last_built = game.turn_number
                SPAWN_TURN += 10
                dropoff_built = True
            else:  # cant build
                ship_state[ship.id] = "waiting"  # wait in the position
                game_map[ship.position].mark_unsafe(ship)
                command_queue.append(ship.move(Direction.Still))
        else:  # not associated with building a dropoff, so move regularly
            move = produce_move(ship)
            if move is not None:
                command_queue.append(ship.move(move))
                previous_position[ship.id] = ship.position
                game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
                if move != Direction.Still and game_map[ship.position].ship == ship:
                    game_map[ship.position].ship = None

        clear_dictionaries()  # of crashed or transformed ships

        # This ship has made a move
        has_moved[ship.id] = True

    surrounded_shipyard = game_map.is_surrounded(me.shipyard.position)
    logging.info(time_left())
    if not dropoff_built and game.turn_number <= SPAWN_TURN and me.halite_amount >= constants.SHIP_COST\
        and 2 * (max_enemy_ships() + 1) > len(me.get_ships()) and prcntg_halite_left > (1 - 0.65) and \
            not (game_map[me.shipyard].is_occupied or surrounded_shipyard or "waiting" in ship_state.values()):

    if not dropoff_built and 2.5 * max_enemy_ships() > len(me.get_ships()) and game.turn_number <= SPAWN_TURN\
            and me.halite_amount >= constants.SHIP_COST and prcntg_halite_left > (1 - 0.65) and\
            not (game_map[me.shipyard].is_occupied or surrounded_shipyard or "waiting" in ship_state.values()):
        if not ("build" in ship_state.values() and me.halite_amount <= (constants.SHIP_COST + constants.DROPOFF_COST)):
            command_queue.append(me.shipyard.spawn())
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
