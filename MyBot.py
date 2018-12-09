#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt
import pickle
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
import sys
import os
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import numpy as np

# stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

""" <<<Game Begin>>> """
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
shipyard_halite = {}  # shipyard.id -> halite priority queue
shipyard_pos = {}  # shipyard.id -> shipyard position
shipyard_halite_pos = {}  # shipyard.id -> halite pos dictionary

VARIABLES = ["YEEHAW", 1285, 50, 0.3, 0.94, 0.85, 300, 50, 0.65,
			 0, 0.8, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.5, 6, 8]
VERSION = VARIABLES[1]
# search area for halite relative to shipyard
SCAN_AREA = int(VARIABLES[2])
# when switch collectable percentage of max halite
PERCENTAGE_SWITCH = int(float(VARIABLES[3]) * constants.MAX_TURNS)
SMALL_PERCENTAGE = float(VARIABLES[4])
BIG_PERCENTAGE = float(VARIABLES[5])
# definition of medium patch size for stopping and collecting patch if on
# the way
MEDIUM_HALITE = int(VARIABLES[6])
# halite left at patch to stop collecting at that patch
HALITE_STOP = int(VARIABLES[7])
# until which turn to spawn ships
SPAWN_TURN = int(float(VARIABLES[8]) * constants.MAX_TURNS)
# Coefficients for halite heuristics
A = float(VARIABLES[9])
B = float(VARIABLES[10])
C = float(VARIABLES[11])
D = float(VARIABLES[12])
E = float(VARIABLES[13])
F = float(VARIABLES[14])
CRASH_TURN = constants.MAX_TURNS
CRASH_PERCENTAGE_TURN = float(VARIABLES[15])
CRASH_SELECTION_TURN = int(float(CRASH_PERCENTAGE_TURN) * constants.MAX_TURNS)
SHIPYARD_VICINITY = 2
# If enemy ship has at least this halite kill it if near dropoff or shipyard
KILL_ENEMY_SHIP = int(VARIABLES[16])
# Minimum halite needed to join a halite cluster
DETERMINE_CLUSTER_TURN = int(float(VARIABLES[17]) * constants.MAX_TURNS)
CLUSTER_TOO_CLOSE = float(VARIABLES[18])  # distance two clusters can be within
MAX_CLUSTERS = int(VARIABLES[19])  # max amount of clusters
FLEET_SIZE = int(VARIABLES[20])  # fleet size to send for new dropoff
TURN_START = 0  # for timing
CLOSE_TO_SHIPYARD = 0.2
SHIP_SCAN_AREA = 10
game.ready("MLP")
NR_OF_PLAYERS = len(game.players.keys())


def f(h_amount, h_distance):  # function for determining patch priority
	return (A * h_amount * h_amount + B * h_amount + C) / (D * h_distance * h_distance + E * h_distance + F)


def halite_priority_q(pos, area):
	# h_amount <= 0 to run minheap as maxheap
	h = []  # stores halite amount * -1 with its position in a minheap
	h_pos = {}
	top_left = Position(int(-1 * area / 2),
						int(-1 * area / 2)) + pos  # top left of scan area
	for y in range(area):
		for x in range(area):
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
			if ship_state[s.id] in ["returning", "harikiri"]:
				importance = -1 * game_map[
					ship.position].dijkstra_distance / (game_map.width * 2)
			elif ship_state[s.id] in ["exploring", "fleet", "build"]:
				importance = game_map.calculate_distance(
					s.position, shipyard)  # normal distance
			else:  # collecting
				importance = game_map.calculate_distance(s.position,
														 shipyard) * game_map.width * 2  # normal distance * X since processing last
		else:
			importance = -100  # newly spawned ships max importance
		heappush(ships, (importance, s))
	return ships, has_moved


def select_crash_turn():
	'''selects turn when to crash'''
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


def find_new_destination(h, ship, halite_pos):
	''' h: priority queue of halite factors,
		halite_pos: dictionary of halite factor -> patch position '''
	ship_id = ship.id
	biggest_halite = heappop(h)  # get biggest halite
	destination = game_map.normalize(halite_pos[biggest_halite])
	# get biggest halite while its a position no other ship goes to
	while not dest_viable(destination, ship):
		biggest_halite = heappop(h)
		destination = game_map.normalize(halite_pos[biggest_halite])

	ship_dest[ship_id] = destination  # set the destination

	# if another ship had the same destination
	s = get_ship_w_destination(destination, ship_id)
	if s is not None:  # find a new destination for it
		prio, h_pos = halite_priority_q(s.position, 10)
		find_new_destination(prio, s, h_pos)


def dest_viable(position, ship):
	if position in ship_dest.values():
		inspectable_ship = get_ship_w_destination(position, ship.id)
		if inspectable_ship is None:
			# if this ship doesnt exist for some reason
			return True

		my_dest = game_map.calculate_distance(position, ship.position)
		their_dest = game_map.calculate_distance(
			position, inspectable_ship.position)

		if my_dest < their_dest:  # if im closer give it to me
			return True
		return False  # if not then recalculate
	else:
		return True  # nobody has the best patch, all good


def get_ship_w_destination(dest, this_id):
	if dest in ship_dest.values():
		for s in ship_dest.keys():  # get ship with the same destination
			if not s == this_id and ship_dest[s] == dest and me.has_ship(s):
				for the_s in me.get_ships():
					if the_s.id == s:
						return the_s
	return None


def clear_dictionaries():
	# clear dictionaries of crushed ships
	for ship_id in list(ship_dest.keys()):
		if not me.has_ship(ship_id):
			del ship_dest[ship_id]
			del ship_state[ship_id]
			del previous_state[ship_id]
			del previous_position[ship_id]
			del ship_shipyards[ship_id]
			if ship_id in ship_path:
				del ship_path[ship_id]


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


def make_returning_move(ship, has_moved, command_queue):
	"""
	Makes a returning move based on Dijkstras and other ship positions.
	"""
	if ship_path[ship.id]:
		direction = get_step(ship_path[ship.id])
		if not (game_map[ship.position.directional_offset(direction)].is_occupied or direction == Direction.Still):
			return direction

	if game.turn_number >= CRASH_TURN and should_better_dropoff(ship):
		return a_star_move(ship, better_dropoff_pos(ship))

	# Get the cell and direction we want to go to from dijkstra
	target_pos, move = get_dijkstra_move(ship.position)

	# Target is occupied
	if game_map[target_pos].is_occupied:
		other_ship = game_map[target_pos].ship
		# target position occupied by own ship
		if me.has_ship(other_ship.id):

			if ship_state[other_ship.id] in ["exploring", "build", "fleet"]:
				# if other ship has enough halite and hasnt made a move yet:
				if not has_moved[other_ship.id] and \
						(other_ship.halite_amount > game_map[
							other_ship.position].halite_amount / 10 or other_ship.position in get_dropoff_positions()):
					# move stays the same target move
					# move other_ship to ship.destination
					# hence swapping ships
					move_ship_to_position(other_ship, ship.position)
				else:
					move = a_star_move(ship)

			elif ship_state[other_ship.id] in ["returning", "harakiri"]:
				move = Direction.Still
			elif ship_state[other_ship.id] in ["collecting", "waiting"]:
				move = a_star_move(ship)

		else:  # target position occupied by enemy ship
			move = a_star_move(ship)

	return move


def should_better_dropoff(ship):
	''' determines whether there are too many ships going to the same dropoff as ship'''
	current = game_map[ship.position]
	if len(me.get_dropoffs()) >= 1: # if there are multiple dropoffs then check for other options
		my_dest = current.dijkstra_dest
		amount = 0 # ship amount going to the same dropoff
		for other in me.get_ships():
			other_cell = game_map[other.position]
			other_dest = other_cell.dijkstra_dest
			# count other ships that are returning, with the same destinationa and are within 80% of ships dijkstra distance
			if ship_state[other.id] == "returning" and\
					other_cell.dijkstra_dest == my_dest and other_cell.dijkstra_distance < 0.8 * current.dijkstra_distance:
				amount += 1
		# if more than 30 percent of the ships are very close to the shipyard
		if amount > 0.3 * len(me.get_ships()):
			return True
	return False

def better_dropoff_pos(ship):
	''' find the better dropoff position
	precondition: should_better_dropoff(ship) and
	game.turn_number >= CRASH_TURN
	'''
	current_dest = game_map[ship.position].dijkstra_dest
	current_distance = None
	new_dest = None
	for d in get_dropoff_positions(): # for all dropoffs
		dist = game_map.calculate_distance(d, ship.position)
		# if nto the same dropoff and still can reach it by the end of game
		if not d == current_dest and dist < (constants.MAX_TURNS - game.turn_number):
			# check if distance is better than other dropoffs
			if current_distance is None or current_distance < dist:
				current_distance = dist
				new_dest = d
	# return best option
	if new_dest is None:
		return current_dest
	return new_dest


def a_star_move(ship, dest=None):
	if dest is None:
		cell = game_map[ship.position]
		d_to_dijkstra_dest = game_map.calculate_distance(
			cell.position, cell.dijkstra_dest)
		dest = interim_djikstra_dest(
			cell).position if d_to_dijkstra_dest > 10 else cell.dijkstra_dest
	return exploring(ship, dest)


def interim_djikstra_dest(source_cell):
	''' finds the intermediate djikstra destination that is not occupied '''
	cell = source_cell.parent
	while cell.is_occupied:
		cell = cell.parent
		if time_left() < 0.5:
			logging.info("STANDING STILL TOO SLOW")
			return source_cell
	return cell


def move_ship_to_position(ship, destination):
	''' moves ship to destination
	precondition: destination one move away'''
	normalized_dest = game_map.normalize(destination)
	for d in Direction.get_all_cardinals():
		new_pos = game_map.normalize(ship.position.directional_offset(d))
		if new_pos == normalized_dest:
			move = d
			break

	has_moved[ship.id] = True
	command_queue.append(ship.move(move))
	game_map[destination].mark_unsafe(ship)
	game_map[ship.position].ship = None


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
	if ship_id not in previous_state:
		previous_state[ship_id] = "exploring"
	if ship_id not in ship_state:
		ship_state[ship_id] = previous_state[ship_id]
	if not new_state == "exploring":  # reset path to empty list
		ship_path[ship_id] = []

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
		"fleet": exploring,
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
			logging.info("STANDING STILL")
			return position
		if not path:
			return next_pos
		to_go = get_step(path)
		next_pos = game_map.normalize(next_pos.directional_offset(to_go))
	return next_pos


def exploring(ship, destination):
	# next direction occupied, recalculate
	if ship.id not in ship_path or not ship_path[ship.id]:
		ship_path[ship.id] = game_map.explore(ship, destination)
	else:
		direction = ship_path[ship.id][0][0]
		if game_map[ship.position.directional_offset(direction)].is_occupied and not direction == Direction.Still:
			if game_map.calculate_distance(destination, ship.position) > 10:
				new_dest = interim_exploring_dest(
					ship.position, ship_path[ship.id])
				# use intermediate unoccpied position instead of actual
				# destination
				ship_path[ship.id] = game_map.explore(
					ship, new_dest) + ship_path[ship.id]
			else:
				ship_path[ship.id] = game_map.explore(ship, destination)

	# move in calculated direction
	return get_step(ship_path[ship.id])


def get_step(path):
	path[0][1] -= 1  # take that direction
	direction = path[0][0]
	if path[0][1] == 0:  # if no more left that direction remove it
		del path[0]
	return direction


def harakiri(ship, destination):
	shipyard = shipyard_pos[ship_shipyards[ship.id]]
	ship_pos = game_map.normalize(ship.position)
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


def better_patch_neighbouring(ship, big_diff):
	''' returns true if there is a lot better patch right next to it'''
	current = game_map[ship.position]
	neighbours = game_map.get_neighbours(current)
	for n in neighbours:
		if n.halite_amount >= current.halite_amount + big_diff:
			return True
	return False


def get_best_neighbour(position):
	''' gets best neighbour at the ships positioņ
	returns a cell'''
	current = game_map[position]
	neighbours = game_map.get_neighbours(current)
	max_halite = current.halite_amount
	best = current
	for n in neighbours:
		if n.halite_amount > max_halite:
			best = n
			max_halite = n.halite_amount
	return best


def state_transition(ship):
	# transition
	new_state = None
	shipyard_id = ship_shipyards[ship.id]

	if game.turn_number >= CRASH_TURN and game_map.calculate_distance(
			ship.position, shipyard_pos[ship_shipyards[ship.id]]) < 2:
		# if next to shipyard after crash turn, suicide
		ship_path[ship.id] = []
		new_state = "harakiri"

	elif (ship_state[ship.id] == "collecting" or ship_state[
			ship.id] == "exploring") and game.turn_number >= CRASH_TURN:
		# return if at crash turn
		ship_path[ship.id] = []
		new_state = "returning"

	elif ship_state[ship.id] == "exploring" and (ship.position == ship_dest[ship.id]
												 or game_map[ship.position].halite_amount > MEDIUM_HALITE):
		# collect if reached destination or on medium sized patch
		ship_path[ship.id] = []
		new_state = "collecting"

	elif ship_state[ship.id] == "exploring" and ship.halite_amount >= constants.MAX_HALITE * return_percentage:
		# return if ship is 70+% full
		ship_path[ship.id] = []
		new_state = "returning"

	elif ship_state[ship.id] == "collecting" and ship.halite_amount <= constants.MAX_HALITE * (return_percentage * 0.5) and better_patch_neighbouring(ship, 250):
		# if collecting and ship is half full but next to it there is a really good patch, explore to that patch
		neighbour = get_best_neighbour(ship.position)
		ship_dest[ship.id] = neighbour.position

		for sh in me.get_ships():
			# if somebody else going there recalc the destination
			if not sh.id == ship.id and ship_dest[sh.id] == neighbour.position:
				ship_h, ship_h_positions = halite_priority_q(
					sh.position, SHIP_SCAN_AREA)
				find_new_destination(ship_h, sh, ship_h_positions)

		new_state = "exploring"

	elif ship_state[
		ship.id] == "collecting" and ship.halite_amount >= constants.MAX_HALITE * return_percentage and \
			not (game_map[ship.position].halite_amount > MEDIUM_HALITE and not ship.halite_amount >= constants.MAX_HALITE):
		 # return to shipyard if enough halite
		new_state = "returning"

	elif ship_state[ship.id] == "collecting" and game_map[ship.position].halite_amount < HALITE_STOP:
		# Keep exploring if current halite patch is empty
		ship_h, ship_h_positions = halite_priority_q(
			ship.position, SHIP_SCAN_AREA)
		find_new_destination(ship_h, ship, ship_h_positions)
		new_state = "exploring"

	elif ship_state[ship.id] == "returning" and ship.position in get_dropoff_positions():
		# explore again when back in shipyard
		new_state = "exploring"
		find_new_destination(
			shipyard_halite[shipyard_id], ship, shipyard_halite_pos[shipyard_id])

	# if someone already build dropoff there before us
	elif ship_state[ship.id] == "build" and game_map[ship_dest[ship.id]].has_structure:
		# explore
		ship_h, ship_h_positions = halite_priority_q(
			ship.position, SHIP_SCAN_AREA)
		find_new_destination(ship_h, ship, ship_h_positions)
		ship_path[ship.id] = []
		new_state = "exploring"

	elif ship_state[ship.id] == "fleet" and ship_dest[ship.id] == ship.position:
		ship_path[ship.id] = []
		new_state = "collecting"

	if new_state is not None:
		state_switch(ship.id, new_state)


def do_halite_priorities():
	''' determines halite priority queues
	and positions for all dropoffs, shipyards '''
	shipyard_halite[me.shipyard.id], shipyard_halite_pos[
		me.shipyard.id] = halite_priority_q(me.shipyard.position, SCAN_AREA)
	shipyard_pos[me.shipyard.id] = me.shipyard.position
	for dropoff in me.get_dropoffs():
		shipyard_halite[dropoff.id], shipyard_halite_pos[
			dropoff.id] = halite_priority_q(dropoff.position, SCAN_AREA)
		shipyard_pos[dropoff.id] = dropoff.position


def closest_shipyard_id(ship_pos):
	''' returns shipyard id that is closest to ship '''
	distance = game_map.calculate_distance(ship_pos, me.shipyard.position)
	shipyard_id = me.shipyard.id
	for dropoff in me.get_dropoffs():
		new_distance = game_map.calculate_distance(
			ship_pos, dropoff.position)
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
		if is_fleet(s) and not s.position == position:
			distances.append(
				(game_map.calculate_distance(position, s.position), s))
	distances.sort(key=lambda x: x[0])
	fleet_size = min(len(distances), fleet_size)
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


def is_fleet(ship):
	''' returns if a ship is good for adding it to a fleet '''
	return closest_shipyard_id(ship.position) == me.shipyard.id and me.has_ship(ship.id) and (
		ship.id not in ship_state or not (ship_state[ship.id] in ["fleet", "waiting", "returning", "build"]))


def is_builder(ship):
	''' checks if this ship is a builder '''
	return ship_state[ship.id] == "waiting" or (ship_state[ship.id] == "build" and ship_dest[ship.id] == ship.position)


def fleet_availability():
	''' returns how many ships are available atm'''
	amount = 0
	for s in me.get_ships():
		if is_fleet(s):
			amount += 1
	if amount >= len(list(me.get_ships())) * 0.75:
		amount /= 2
	return int(amount)


def should_build():
	# if clusters determined, more than 13 ships, we have clusters and nobody
	# is building at this turn (in order to not build too many)
	return clusters_determined and len(me.get_ships()) > FLEET_SIZE and cluster_centers \
		and fleet_availability() > 0.75 * FLEET_SIZE and not any_builders()


def send_ships(pos, ship_amount):
	'''sends a fleet of size ship_amount to explore around pos'''
	fleet = get_fleet(game_map.normalize(pos), ship_amount)
	# for rest of the fleet to explore
	h, h_pos = halite_priority_q(pos, SHIP_SCAN_AREA)
	for fleet_ship in fleet:  # for other fleet members
		if fleet_ship.id not in previous_state.keys():  # if new ship
			previous_state[fleet_ship.id] = "exploring"
			has_moved[fleet_ship.id] = False
		# explore in area of the new dropoff

		state_switch(fleet_ship.id, "fleet")
		find_new_destination(h, fleet_ship, h_pos)


def get_cell_data(x, y, center):
	cell = game_map[Position(x, y)]
	# normalized data of cell: halite amount and distance to shipyard
	return [round(cell.halite_amount / 1000, 2),
			round(game_map.calculate_distance(cell.position, center) / game_map.width, 2)]


def any_builders():
	return "waiting" in ship_state.values() or "build" in ship_state.values()


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


def clusters_with_classifier():
	''' uses classifier to determine clusters for dropoff '''
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
				   cntr.x + int(x_size / 2) + diff2, 4):
		for y in range(cntr.y - int(y_size / 2) + diff3,
					   cntr.y + int(y_size / 2) + diff4, 4):
			p_data, total_halite, p_center = get_patch_data(
				x, y, cntr)  # get the data
			prediction = dropoff_clf.predict(p_data)[0]  # predict on it
			if prediction == 1:  # if should be dropoff
				# add node with most halite to centers
				cluster_centers.append((total_halite, p_center))
	# do filtering
	cluster_centers = filter_clusters(cluster_centers, MAX_CLUSTERS)
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
	for i, d in enumerate(centers_copy):
		halite, pos = d

		if halite < 8000:  # if area contains less than 8k then remove it
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
		diff = game_map.calculate_distance(pos, me.shipyard.position)
		if diff < CLOSE_TO_SHIPYARD * game_map.width:
			if d in centers:
				logging.info(d)
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
	z: halite amount / 8000 '''

	logging.info("Merging clusters")
	area = int(CLOSE_TO_SHIPYARD * game_map.width)
	metric = distance_metric(type_metric.USER_DEFINED, func=custom_dist)
	X = [] # center coordinates that are merged in an iteration
	tmp_centers = [] # to not modify the list looping through
	history = [] # contains all already merged centers
	# for each center
	for c1 in centers:
		# add the center itself
		X.append([c1[1].x, c1[1].y, c1[0] / 8000])
		for c2 in centers: # for other centers
			# if not merged already
			if [c2[1].x, c2[1].y, c2[0] / 8000] not in history:
				dist = game_map.calculate_distance(c1[1], c2[1])
				# if close enough for merging
				if dist <= area:
					X.append([c2[1].x, c2[1].y, c2[0] / 8000])
		# get initialized centers for the algorithm
		init_centers = kmeans_plusplus_initializer(X, 1).initialize()
		median = kmedians(X, init_centers, metric=metric)
		median.process() # do clustering
		# get clustered centers
		tmp_centers += [(x[2], game_map.normalize(Position(int(x[0]), int(x[1]))))
						for x in median.get_medians()]
		history += X
		X = []

	centers = tmp_centers
	centers.sort(key=lambda x: x[0], reverse=True) # sort by best patches
	return centers


def custom_dist(p1, p2):
	''' distance function for a clustering algorithm, 
	manh dist + the absolute difference in halite amount '''
	if len(p1) < 3:
		p1 = p1[0]
		p2 = p2[0]
	manh_dist = game_map.calculate_distance(
		Position(p1[0], p1[1]), Position(p2[0], p2[1]))
	return manh_dist + abs(p1[2] - p2[2])


def too_close(centers, position):
	''' removes clusters that are too close to each other '''
	to_remove = []
	for d in centers:
		_, other = d
		distance = game_map.calculate_distance(position, other)
		shipyard_distance = game_map.calculate_distance(
			me.shipyard.position, other)
		if distance < CLUSTER_TOO_CLOSE * game_map.width or shipyard_distance < CLOSE_TO_SHIPYARD * game_map.width:
			to_remove.append(d)
	return to_remove


def time_left():
	return 2 - (time.time() - TURN_START)

clusters_determined = False
INITIAL_HALITE_STOP = HALITE_STOP
while True:
	game.update_frame()
	me = game.me
	game_map = game.game_map
	TURN_START = time.time()
	game_map.set_total_halite()

	if game.turn_number == 1:
		TOTAL_MAP_HALITE = game_map.total_halite

	prcntg_halite_left = game_map.total_halite / TOTAL_MAP_HALITE
	HALITE_STOP = prcntg_halite_left * INITIAL_HALITE_STOP

	# Dijkstra the graph based on all dropoffs
	game_map.create_graph(get_dropoff_positions())

	if game.turn_number == DETERMINE_CLUSTER_TURN:
		clusters_determined = True
		cluster_centers = clusters_with_classifier()

	if game.turn_number == CRASH_SELECTION_TURN:
		CRASH_TURN = select_crash_turn()

	return_percentage = BIG_PERCENTAGE if game.turn_number < PERCENTAGE_SWITCH else SMALL_PERCENTAGE
	command_queue = []
	# priority Q of patch function values of function f(halite, distance)
	do_halite_priorities()

	if should_build():
		_, dropoff_pos = cluster_centers.pop(0)  # remove from list
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
			if game_map.width >= 56:
				send_ships(dropoff_pos, int(FLEET_SIZE / 2))

		else:  # if builder not available
			cluster_centers.insert(0, (_, dropoff_pos))

	# has_moved ID->True/False, moved or not
	# ships priority queue of (importance, ship)
	ships, has_moved = ship_priority_q(me, game_map)
	# True if a ship moves into the shipyard this turn
	move_into_shipyard = False
	# whether a dropoff has been built this turn so that wouldnt use too much
	# halite
	dropoff_built = False

	nearby_enemy_ships = enemy_near_shipyard()

	while ships:  # go through all ships
		ship = heappop(ships)[1]
		if time_left() < 0.3:
			logging.info("STANDING STILL TOO SLOW")
			command_queue.append(ship.stay_still())
			ship_state[ship.id] = "collecting"
			continue
		if has_moved[ship.id]:
			continue
		if ship.id not in previous_position:  # if new ship the
			previous_position[ship.id] = me.shipyard.position
		# set the closest shipyard
		ship_shipyards[ship.id] = closest_shipyard_id(ship.position)
		shipyard_id = ship_shipyards[ship.id]

		# setup state
		if ship.id not in ship_dest:  # if ship hasnt received a destination yet
			find_new_destination(
				shipyard_halite[shipyard_id], ship, shipyard_halite_pos[shipyard_id])
			previous_state[ship.id] = "exploring"
			ship_state[ship.id] = "exploring"  # explore

		enemy_position = check_shipyard_blockade(
			nearby_enemy_ships, ship.position)
		if enemy_position is not None:
			state_switch(ship.id, "assassinate")
			ship_dest[ship.id] = game_map.normalize(enemy_position)
			nearby_enemy_ships.remove(enemy_position)

		# transition
		state_transition(ship)

		# if ship is dropoff builder
		if is_builder(ship):
			# if enough halite and havent built a dropoff this turn
			if (ship.halite_amount + me.halite_amount) >= constants.DROPOFF_COST and not dropoff_built:
				send_ships(ship.position, int(4*FLEET_SIZE/5))
				command_queue.append(ship.make_dropoff())
				do_halite_priorities()  # recalc all dictionaries
				dropoff_built = True
			else:  # cant build
				ship_state[ship.id] = "waiting"  # wait in the position
				game_map[ship.position].mark_unsafe(ship)
				command_queue.append(ship.move(Direction.Still))
		else:  # not associated with building a dropoff, so move regularly
			move = produce_move(ship)
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

	if not dropoff_built and len(me.get_ships()) < 70 and game.turn_number <= SPAWN_TURN and me.halite_amount >= constants.SHIP_COST \
			and prcntg_halite_left > (1 - 0.65) and not (game_map[me.shipyard].is_occupied or surrounded_shipyard or "waiting" in ship_state.values()):
		if not ("build" in ship_state.values() and me.halite_amount <= (constants.SHIP_COST + constants.DROPOFF_COST)):
			command_queue.append(me.shipyard.spawn())
	# Send your moves back to the game environment, ending this turn.
	game.end_turn(command_queue)
