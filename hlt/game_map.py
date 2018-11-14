import queue
import random

from . import constants
from .entity import Entity, Shipyard, Ship, Dropoff
from .player import Player
from .positionals import Direction, Position
from .common import read_input
from heapq import heappush, heappop
import logging


class MapCell:
    """A cell on the game map."""

    def __init__(self, position, halite_amount):
        self.position = position
        self.halite_amount = halite_amount
        self.ship = None
        self.structure = None

        # Parameters for Dijkstra
        self.weight_to_shipyard = 0
        self.parent = None

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
        y_cardinality, x_cardinality = self.get_target_direction(source, destination)

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
        # selects best direction that is not occupied but with smallest distance to destination
        possible_moves = []  # (direction, distance to destination if taken that direction)
        for direction in Direction.get_all_cardinals():  # for 4 directions
            next_position = ship.position.directional_offset(direction)
            if not (self[next_position].is_occupied or self[prev_position] == self[
                next_position]):  # check if the position is valid
                # that is.. if its not occupied and not previous position (so the ship doesnt wiggle back and forth)
                possible_moves.append((direction, self.calculate_distance(next_position, destination)))

        if len(possible_moves) == 0:
            return Direction.Still
        # select direction with best distance
        final_direction = min(possible_moves, key=lambda t: t[1])[0]
        return final_direction

    def smart_navigate(self, previous, ship, destination):
        if ship.position == destination:
            return Direction.Still
        if self.calculate_distance(ship.position,
                                   destination) == 1:  # move in the needed direction or stay still if a ship is in the destination at this moment
            surrounded = True  # assumption
            for direction in Direction.get_all_cardinals():
                if not self[destination.directional_offset(direction)].is_occupied:
                    # if any position around destination is not occupied it is not surrounded
                    surrounded = False
                    break
            
            if surrounded:  # move in circle to not surround
                target_direction = self.get_target_direction(destination, ship.position)  # direction destination->ship
                relative_direction = target_direction[0] if target_direction[0] is not None else target_direction[1]  # get the direction
                movement = (relative_direction[1], relative_direction[0])  # rotate by 90 degrees
                new_position = ship.position.directional_offset(movement)  # get possible future position
                if self[new_position].is_occupied:  # if its occupied
                    movement = relative_direction  # go outwards
                self[new_position].mark_unsafe(ship)  # mark the new position unsafe
                return movement
            else:
                direction = self.get_target_direction(ship.position, destination)
                target_direction = direction[0] if direction[0] is not None else direction[1]  # get the direction
                new_position = ship.position.directional_offset(target_direction)
                if self[new_position].is_occupied:
                    return Direction.Still
                self[new_position].mark_unsafe(ship)
                return target_direction
        # else select best move 
        # and from that mark them
        final_direction = self.select_best_direction(previous, ship, destination)
        new_position = ship.position.directional_offset(final_direction)
        self[new_position].mark_unsafe(ship)
        # now artifically move and mark future spots unsafe too
        distance = self.calculate_distance(ship.position, destination)
        # select how many moves we mark in future
        nr_future_moves = 1
        position = new_position
        prev_position = ship.position
        # simulate moves
        for _ in range(nr_future_moves):
            direction = self.select_best_direction(prev_position, ship, destination)
            if direction == Direction.Still:  # if standing still stop cuz it will continue standing still
                self[position].mark_unsafe(ship)
                break
            prev_position = position
            position = position.directional_offset(direction)  # new position
            self[position].mark_unsafe(ship)

        return final_direction

    def create_graph(self):

        """
        Assigns the correct value to weight_to_shipyard, and direction_to_shipyard for each cell in this map
        by using Dijkstra using the shipyard as source.
        """
        pass

    def dijkstra(self, source_cell):
        """
        Does Dijkstra from Cell source_cell
        :param source_cell: the source cell.
        """
        # Priority Queue with map cells
        PQ = []
        # set distance to infinity on all nodes
        for i in range(self.height):
            for j in range(self.width):
                self._cells[i][j].weight_to_shipyard = 1_000_000
        # for cell in self._cells:
        #     cell.weight_to_shipyard = 1_000_000
        source_cell.weight_to_shipyard = 0
        heappush(PQ, (source_cell.weight_to_shipyard, source_cell))
        while PQ:
            dist_cell = heappop(PQ)
            dist = dist_cell[0]
            cell = dist_cell[1]
            if cell.weight_to_shipyard < dist: continue
            for neighbour in self.get_neighbours(cell):
                new_dist = dist + neighbour.halite_amount + 50
                if new_dist < neighbour.weight_to_shipyard:
                    neighbour.weight_to_shipyard = new_dist
                    neighbour.parent = cell
                    heappush(PQ, (new_dist, neighbour))

    def get_neighbours(self, source_cell):
        """
        Returns a list of all neighbouring cells
        """
        dy = [-1, 0, 1, 0]
        dx = [0, -1, 0, 1]
        neighbours = []
        for i in range(len(dy)):
            neighbours.append(self[source_cell.position + Position(dy[i], dx[i])])
        return neighbours

    @staticmethod
    def _generate():
        """
        Creates a map object from the input given by the game engine
        :return: The map object
        """
        map_width, map_height = map(int, read_input().split())
        game_map = [[None for _ in range(map_width)] for _ in range(map_height)]
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
