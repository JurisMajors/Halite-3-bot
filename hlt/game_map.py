import queue
import random

from . import constants
from .entity import Entity, Shipyard, Ship, Dropoff
from .player import Player
from .positionals import Direction, Position
from .common import read_input
from heapq import heappush, heappop
import logging
import time
from collections import deque


class MapCell:
	"""A cell on the game map."""

	def __init__(self, position, halite_amount):
		self.position = position
		self.halite_amount = halite_amount
		self.ship = None
		self.structure = None
		self.inspired = False

		# Parameters for Dijkstra (to nearest dropoff/shipyard)
		self.weight_to_shipyard = 0
		self.parent = None
		self.dijkstra_distance = None
		self.dijkstra_dest = None

		# Parameters for AStar
		self.visited = None
		self.a_star_parent = None
		self.cost = None

	@property
	def is_empty(self):
		"""
		:return: Whether this cell has no ships or structures
		"""
		return self.ship is None and self.structure is None

	@property
	def is_occupied(self):
		"""
		:return: Whether this cell has any ships
		"""
		return self.ship is not None

	@property
	def has_structure(self):
		"""
		:return: Whether this cell has any structures
		"""
		return self.structure is not None

	@property
	def structure_type(self):
		"""
		:return: What is the structure type in this cell
		"""
		return None if not self.structure else type(self.structure)
	

	def mark_unsafe(self, ship):
		"""
		Mark this cell as unsafe (occupied) for navigation.

		Use in conjunction with GameMap.naive_navigate.
		"""
		self.ship = ship

	def __eq__(self, other):
		return self.position == other.position

	def __ne__(self, other):
		return not self.__eq__(other)

	def __lt__(self, other):
		return self.position.x < other.position.x

	def __str__(self):
		return 'MapCell({}, halite={})'.format(self.position, self.halite_amount)


class GameMap:
	"""
	The game map.

	Can be indexed by a position, or by a contained entity.
	Coordinates start at 0. Coordinates are normalized for you
	"""

	def __init__(self, cells, width, height):
		self.width = width
		self.height = height
		self._cells = cells

	def __getitem__(self, location):
		"""
		Getter for position object or entity objects within the game map
		:param location: the position or entity to access in this map
		:return: the contents housing that cell or entity
		"""
		if isinstance(location, Position):
			location = self.normalize(location)
			return self._cells[location.y][location.x]
		elif isinstance(location, Entity):
			return self._cells[location.position.y][location.position.x]
		return None

	def calculate_distance(self, source, target):
		"""
		Compute the Manhattan distance between two locations.
		Accounts for wrap-around.
		:param source: The source from where to calculate
		:param target: The target to where calculate
		:return: The distance between these items
		"""
		source = self.normalize(source)
		target = self.normalize(target)
		resulting_position = abs(source - target)
		return min(resulting_position.x, self.width - resulting_position.x) + \
			min(resulting_position.y, self.height - resulting_position.y)

	def normalize(self, position):
		"""
		Normalized the position within the bounds of the toroidal map.
		i.e.: Takes a point which may or may not be within width and
		height bounds, and places it within those bounds considering
		wraparound.
		:param position: A position object.
		:return: A normalized position object fitting within the bounds of the map
		"""
		return Position(position.x % self.width, position.y % self.height)

	def get_target_direction(self, source, target):
		"""
		Returns where in the cardinality spectrum the target is from source. e.g.: North, East; South, West; etc.
		NOTE: Ignores toroid
		:param source: The source position
		:param target: The target position
		:return: A tuple containing the target Direction. A tuple item (or both) could be None if within same coords
		"""
		# Horizontal without using the edges
		horizontal = Direction.South if target.y > source.y else Direction.North if target.y < source.y else None
		# Use edge if its faster than normal
		if abs(target.y - source.y) > (self.height - abs(target.y - source.y)) and horizontal is not None:
			horizontal = Direction.invert(horizontal)

		vertical = Direction.East if target.x > source.x else Direction.West if target.x < source.x else None

		if abs(target.x - source.x) > (self.width - abs(target.x - source.x)) and vertical is not None:
			vertical = Direction.invert(vertical)
		return (horizontal,
				vertical)

	def get_unsafe_moves(self, source, destination):
		"""
		Return the Direction(s) to move closer to the target point, or empty if the points are the same.
		This move mechanic does not account for collisions. The multiple directions are if both directional movements
		are viable.
		:param source: The starting position
		:param destination: The destination towards which you wish to move your object.
		:return: A list of valid (closest) Directions towards your target.
		"""
		source = self.normalize(source)
		destination = self.normalize(destination)
		possible_moves = []
		distance = abs(destination - source)
		y_cardinality, x_cardinality = self.get_target_direction(
			source, destination)

		if distance.x != 0:
			possible_moves.append(x_cardinality if distance.x < (self.width / 2)
								  else Direction.invert(x_cardinality))
		if distance.y != 0:
			possible_moves.append(y_cardinality if distance.y < (self.height / 2)
								  else Direction.invert(y_cardinality))
		return possible_moves

	def naive_navigate(self, ship, destination):
		"""
		Returns a singular safe move towards the destination.

		:param ship: The ship to move.
		:param destination: Ending position
		:return: A direction.
		"""
		# No need to normalize destination, since get_unsafe_moves
		# does that
		final_dir = Direction.Still
		if ship.position == destination:
			return Direction.Still
		for direction in self.get_unsafe_moves(ship.position, destination):
			target_pos = ship.position.directional_offset(direction)
			if not self[target_pos].is_occupied:
				self[target_pos].mark_unsafe(ship)
				return direction
		all_moves = Direction.get_all_cardinals()
		# pick random move that is safe
		move = random.choice(all_moves)
		while self[ship.position.directional_offset(move)].is_occupied:
			all_moves.remove(move)
			if len(all_moves) == 0:  # if stuck in all directions then stay still
				return Direction.Still
			move = random.choice(all_moves)
		self[ship.position.directional_offset(move)].mark_unsafe(ship)
		return move

	def select_best_direction(self, prev_position, ship, destination):
		# selects best direction that is not occupied but with smallest
		# distance to destination
		# (direction, distance to destination if taken that direction)
		possible_moves = []
		for direction in Direction.get_all_cardinals():  # for 4 directions
			next_position = ship.position.directional_offset(direction)
			if not (self[next_position].is_occupied or self[prev_position] == self[
					next_position]):  # check if the position is valid
				# that is.. if its not occupied and not previous position (so
				# the ship doesnt wiggle back and forth)
				possible_moves.append(
					(direction, self.calculate_distance(next_position, destination)))

		if len(possible_moves) == 0:
			return Direction.Still
		# select direction with best distance
		final_direction = min(possible_moves, key=lambda t: t[1])[0]
		return final_direction

	def create_graph(self, dropoff_list):
		"""
		Assigns the correct value to weight_to_shipyard, and parent for each cell in this map
		by using Dijkstra using all different dropoffs as source and prioritizing the closest one.
		dropoff_list is a list that holds the position for all dropoffs and the shipyard
		"""
		# set distance to infinity on all nodes
		for i in range(self.height):
			for j in range(self.width):
				self._cells[i][j].weight_to_shipyard = 1_000_000

		# Do Dijkstra from all different sources
		for dropoff in dropoff_list:
			self.dijkstra(self[dropoff])

	def dijkstra(self, source_cell):
		"""
		Does Dijkstra from Cell source_cell
		:param source_cell: the source cell.
		"""
		# Priority Queue with map cells
		PQ = []

		source_cell.weight_to_shipyard = 0
		node_distance = 0
		source_cell.dijkstra_distance = node_distance
		source_cell.dijkstra_dest = source_cell.position
		heappush(PQ, (source_cell.weight_to_shipyard, source_cell))
		while PQ:
			dist_cell = heappop(PQ)
			dist = dist_cell[0]
			cell = dist_cell[1]
			# node_distance += 1
			if cell.weight_to_shipyard < dist:
				continue
			for neighbour in self.get_neighbours(cell):
				new_dist = dist + neighbour.halite_amount + \
					70  # Maybe divide halite amount by 10 here
				if new_dist < neighbour.weight_to_shipyard:
					neighbour.weight_to_shipyard = new_dist
					neighbour.parent = cell
					neighbour.dijkstra_dest = source_cell.position
					neighbour.dijkstra_distance = cell.dijkstra_distance + 1
					heappush(PQ, (new_dist, neighbour))

	def get_neighbours(self, source_cell):
		"""
		Returns a list of all neighbouring cells
		"""
		dy = [-1, 0, 1, 0]
		dx = [0, -1, 0, 1]
		neighbours = []
		for i in range(len(dy)):
			neighbours.append(
				self[source_cell.position + Position(dy[i], dx[i])])
		return neighbours

	def do_a_star(self, ship, target):
		"""
		param: source & target are cell positions
		Sets parents for making path from source to target
		"""
		PQ = []
		source = ship.position
		target = self.normalize(target)
		heappush(PQ, (0, self[source]))
		# determine whether reachable

		reachable = self.is_reachable(ship, self[source], self[
									  target]) and self.is_reachable(ship, self[target], self[source])
		self[source].cost = 0
		lowest_distance = self.calculate_distance(source, target)  # init
		closest_pos = source
		visited = []  # visited nodes

		while PQ:
			current = heappop(PQ)[1]
			current.visited = ship  # visit cell
			visited.append(current)
			distance_to_target = self.calculate_distance(
				current.position, target)

			if distance_to_target < lowest_distance:  # get lowest
				closest_pos = current.position
				lowest_distance = distance_to_target

			if reachable:  # look for the actual target
				if current.position == target:  # if end goal, we done
					break
			else:  # if moving very far away from end point terminate
				if lowest_distance + int(self.width / 6) <= distance_to_target:
					break

			for neighbour in self.get_neighbours(current):
				# if obstacle or we already visited, skip
				if neighbour.is_occupied or (neighbour.visited is not None and neighbour.visited == ship):
					continue
				# node.cost + distance from neighbour to node ( 1 )
				new_cost = current.cost + 1

				if neighbour.visited is None or new_cost < neighbour.cost:
					# new cost
					neighbour.cost = new_cost
					# distance is the heuristic
					priority = new_cost + \
						self.calculate_distance(neighbour.position, target)
					neighbour.a_star_parent = current
					visited.append(neighbour)
					heappush(PQ, (priority, neighbour))

		return (visited, target) if reachable else (visited, closest_pos)

	def explore(self, ship, destination):
		if self.is_surrounded(ship.position):
			return [[Direction.Still, 1]]

		visited, new_destination = self.do_a_star(ship, destination)
		path = self.find_path(ship.position, new_destination)

		for node in visited:  # reset
			node.cost = None
			node.a_star_parent = None
			node.visited = None

		return path

	def find_path(self, position, destination):
		start = self[position]
		end = self[destination]
		path = []

		if end == start:
			return [[Direction.Still, 1]]

		while not end == start:  # move down the path until in neighbours of initial cell
			next_cell = end.a_star_parent
			t_direction = self.get_target_direction(
				next_cell.position, end.position)
			direction = t_direction[0] if t_direction[
				0] is not None else t_direction[1]
			direction = direction if direction is not None else Direction.Still

			if path:  # if not length zero check last element
				previous = path[0]  # last element
				if previous[0] == direction:  # if same direction
					# increment the amount of times to move that direction
					previous[1] += 1
				else:  # new direction
					path.insert(0, [direction, 1])
			else:  # empty
				path.insert(0, [direction, 1])
			end = next_cell

		return path  # return the neighbour we arrived

	def is_reachable(self, ship, start_cell, end_cell):
		''' Flood fill algorithm from end cell.
		 Returns True if start_cell is reachable from end_cell'''
		if end_cell.is_occupied and not end_cell.ship == ship:
			return False
		Q = deque([])
		Q.append(end_cell)
		visited = []
		while Q:
			cur = Q.popleft()
			visited.append(cur)

			if self.calculate_distance(cur.position, end_cell.position) > 10:
				return True
			if cur == start_cell:
				return True

			for neighbour in self.get_neighbours(cur):
				if neighbour == start_cell:  # needed because start will be occupied by our ship
					return True
				if not neighbour.is_occupied and neighbour not in Q and neighbour not in visited:
					Q.append(neighbour)
		return False

	def is_surrounded(self, position):
		neighbours = position.get_surrounding_cardinals()
		for neighbour in neighbours:
			if not self[neighbour].is_occupied:
				return False
		return True

	def init_map(self, me):
		self.total_halite = 0
		for y in range(self.height):
			for x in range(self.width):
				cell = self[Position(x, y)]
				self.total_halite += cell.halite_amount
				cell.inspired = self.is_cell_inspired(cell, me)



	def get_cells_in_area(self, cell, area):
		top_left = Position(int(-1 * area / 2),
					int(-1 * area / 2)) + cell.position  # top left of scan area
		cells = []
		for y in range(area):
			for x in range(area):
				pos = Position((top_left.x + x) % self.width,
							 (top_left.y + y) % self.height)  # position of patch
				if not pos == cell.position:
					cells.append(self[pos])
		return cells

	# Boolean function returning whether a cell is inspired for a player
	def is_cell_inspired(self, cell, me):
		area = constants.INSPIRATION_RADIUS
		top_left = Position(int(-1 * area),
			int(-1 * area)) + cell.position  # top left of scan area
		counter = 0
		for y in range(area):
			for x in range(area):
				other_cell = self[Position(x, y)]
				if self.calculate_distance(cell.position, other_cell.position) <= constants.INSPIRATION_RADIUS:
					if other_cell.is_occupied and not me.has_ship(other_cell.ship.id):
						counter += 1
				if counter >= 2:
					return True
		return False


	@staticmethod
	def _generate():
		"""
		Creates a map object from the input given by the game engine
		:return: The map object
		"""
		map_width, map_height = map(int, read_input().split())
		game_map = [[None for _ in range(map_width)]
					for _ in range(map_height)]
		for y_position in range(map_height):
			cells = read_input().split()
			for x_position in range(map_width):
				game_map[y_position][x_position] = MapCell(Position(x_position, y_position),
														   int(cells[x_position]))
		return GameMap(game_map, map_width, map_height)

	def _update(self):
		"""
		Updates this map object from the input given by the game engine
		:return: nothing
		"""
		# Mark cells as safe for navigation (will re-mark unsafe cells
		# later)
		for y in range(self.height):
			for x in range(self.width):
				self[Position(x, y)].ship = None

		for _ in range(int(read_input())):
			cell_x, cell_y, cell_energy = map(int, read_input().split())
			self[Position(cell_x, cell_y)].halite_amount = cell_energy
