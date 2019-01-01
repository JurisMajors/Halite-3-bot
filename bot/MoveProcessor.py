import os
import sys
import inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import bot.GlobalConstants as GC
import logging
from hlt.positionals import Direction, Position
from bot.GlobalFunctions import GlobalFunctions
from bot.GlobalVariablesSingleton import GlobalVariablesSingleton


class MoveProcessor():

    def __init__(self, game, has_moved, command_queue, statemachine):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton.getInstance()
        self.ship_obj = GV.ship_obj
        self.ship_dest = GV.ship_dest
        self.previous_state = GV.previous_state
        self.ship_path = GV.ship_path
        self.ship_state = GV.ship_state
        self.previous_position = GV.previous_position
        self.has_moved = has_moved
        self.command_queue = command_queue
        self.NR_OF_PLAYERS = GV.NR_OF_PLAYERS
        self.GF = GlobalFunctions(self.game)
        self.SM = statemachine

    def produce_move(self, ship):
        if ship.id not in self.ship_obj:
            self.ship_obj[ship.id] = ship
        state = self.ship_state[ship.id]
        destination = self.ship_dest[ship.id]
        ''' produces move for ship '''

        if ship.halite_amount < self.game_map[ship.position].halite_amount / constants.MOVE_COST_RATIO:
            return Direction.Still

        mover = {
            "collecting": self.collecting,
            "returning": self.returning,
            "harakiri": self.harakiri,
            "assassinate": self.assassinate,
            "exploring": self.exploring,
            "build": self.exploring,
            "fleet": self.exploring,
            "backup": self.exploring,
        }

        return mover[state](ship, destination)

    def collecting(self, ship, destination):
        return Direction.Still

    def returning(self, ship, destination):
        return self.make_returning_move(ship)

    def harakiri(self, ship, destination):
        """ pre: next to or on shipyard """
        shipyard = self.GF.get_shipyard(ship.position)
        ship_pos = self.game_map.normalize(ship.position)
        if ship.position == shipyard:  # if at shipyard
            return Direction.Still  # let other ships crash in to you
        else:  # otherwise move to the shipyard
            target_dir = self.dir_to_dest(ship.position, shipyard)
            return target_dir

    def dir_to_dest(self, pos, dest):
        """ Precondition: 
            position one move away """
        normalized_dest = self.game_map.normalize(dest)
        for d in Direction.get_all_cardinals():
            new_pos = self.game_map.normalize(pos.directional_offset(d))
            if new_pos == normalized_dest:
                return d
        return Direction.Still  # should never happen

    def assassinate(self, ship, destination):
        self.GF.state_switch(ship.id, self.previous_state[ship.id])
        if self.game_map.calculate_distance(ship.position, destination) == 1:
            target_direction = self.game_map.get_target_direction(
                ship.position, destination)
            return target_direction[0] if target_direction[0] is not None else target_direction[1]
        else:
            return self.exploring(ship, destination)

    def exploring(self, ship, destination):
        if ship.position == destination:
            self.ship_path[ship.id] = []
            return Direction.Still
        elif self.GF.time_left() < 0.1:
            logging.info(f"Exploring ship standing still, {self.GF.time_left()} left")
            return Direction.Still
        elif ship.position in self.GF.get_dropoff_positions() and self.game_map.is_surrounded(ship.position):  # if in a dropoff and surrounded
            # find closest neighbour
            closest_n = None
            closest_dist = None
            for n in self.game_map.get_neighbours(self.game_map[ship.position]):
                # only swap with returning baybes
                if n.ship.id in self.ship_state and self.ship_state[n.ship.id] != "returning":
                    continue
                dist = self.game_map.calculate_distance(
                    n.position, destination)
                if closest_dist is None or dist < closest_dist:  # set smallest dist and neighbour
                    closest_dist = dist
                    closest_n = n
            # swap with the ship there
            if closest_n is None:  # wait for exploring ships to go away
                return Direction.Still
            if not self.me.has_ship(closest_n.ship.id):
                # kill him
                self.ship_state[ship.id] = "assassinate"
                self.ship_dest[ship.id] = closest_n.position
                return self.dir_to_dest(ship.position, self.ship_dest[ship.id])
            # else its our ship
            # so swap with it
            self.move_ship_to_position(closest_n.ship, ship.position)
            self.move_ship_to_position(ship, closest_n.position)
            self.game_map[ship.position].ship = closest_n.ship
            self.ship_path[ship.id] = []
            return None
        elif self.game_map.is_surrounded(ship.position):
            return Direction.Still

        if ship.id not in self.ship_path or not self.ship_path[ship.id]:
            self.ship_path[ship.id] = self.game_map.explore(ship, destination)
        else:
            direction = self.ship_path[ship.id][0][0]
            next_pos = ship.position.directional_offset(direction)
            # next direction occupied, recalculate
            if self.game_map[next_pos].is_occupied and not direction == Direction.Still:
                other_ship = self.game_map[next_pos].ship
                # move to intermediate destination, aka move around
                new_dest = self.interim_exploring_dest(
                    ship.position, self.ship_path[ship.id])
                # use intermediate unoccpied position instead of actual dest
                self.ship_path[ship.id] = self.game_map.explore(
                    ship, new_dest) + self.ship_path[ship.id]
                # add rest of the path, interim path + rest of path
        # move in calculated direction
        return self.get_step(self.ship_path[ship.id])

    def interim_exploring_dest(self, position, path):
        ''' finds intermediate destination from a direction path that is not occupied '''
        to_go = self.get_step(path)
        next_pos = self.game_map.normalize(position.directional_offset(to_go))
        while self.game_map[next_pos].is_occupied:
            if self.GF.time_left() < 0.3:
                logging.info("INTERIM EXPLORING STANDING STILL")
                return position
            if not path:
                return next_pos
            to_go = self.get_step(path)
            next_pos = self.game_map.normalize(
                next_pos.directional_offset(to_go))
        return next_pos

    @staticmethod
    def get_step(path):
        path[0][1] -= 1  # take that direction
        direction = path[0][0]
        if path[0][1] == 0:  # if no more left that direction remove it
            del path[0]
        return direction

    def make_returning_move(self, ship):
        """
        Makes a returning move based on Dijkstras and other ship positions.
        """
        if self.ship_path[ship.id]:
            direction = self.get_step(self.ship_path[ship.id])
            to_go = ship.position.directional_offset(direction)
            if direction == Direction.Still or not self.game_map[to_go].is_occupied:
                return direction

        # Get the cell and direction we want to go to from dijkstra
        target_pos, move = self.get_dijkstra_move(ship)
        # Target is occupied
        if self.game_map[target_pos].is_occupied:
            other_ship = self.game_map[target_pos].ship
            # target position occupied by own ship
            if self.me.has_ship(other_ship.id):

                if other_ship.id not in self.ship_state or self.ship_state[other_ship.id] in ["exploring", "build", "fleet", "backup"]:
                    can_move = other_ship.halite_amount >= self.game_map[
                        other_ship.position].halite_amount / constants.MOVE_COST_RATIO

                    if other_ship.id in self.ship_dest:
                        can_move = can_move and self.ship_dest[
                            other_ship.id] != other_ship.position
                    # if other ship has enough halite to move, hasnt made a move
                    # yet, and if it would move in the ship
                    if not self.has_moved[other_ship.id] and \
                            (can_move or other_ship.position in self.GF.get_dropoff_positions()):
                        if (other_ship.id not in self.ship_path or (not self.ship_path[other_ship.id] or
                                                                    other_ship.position.directional_offset(self.ship_path[other_ship.id][0][0]) == ship.position)):
                            # move stays the same target move
                            # move other_ship to ship.position
                            # hence swapping ships
                            logging.info(f"SWAPPING {ship.id} with {other_ship.id}")
                            self.move_ship_to_position(
                                other_ship, ship.position)
                        elif other_ship.id in self.ship_path and self.ship_path[other_ship.id] and self.ship_path[other_ship.id][0][0] == Direction.Still:
                            move = self.a_star_move(ship)
                        else:
                            self.SM.state_transition(other_ship) # state transition the ship
                            # if will be standing still
                            if self.SM.ship_state[other_ship.id] == "collecting" or other_ship.position == self.SM.ship_dest[other_ship.id]:
                                move = self.a_star_move(ship) # move around
                            # else its regular ship move, standard flow.
                    else:  # wait until can move
                        move = Direction.Still

                # suiciding or queue
                elif self.ship_state[other_ship.id] in ["returning", "harakiri"]:
                    if self.has_moved[other_ship.id] or (self.game.turn_number <= GC.CRASH_TURN and other_ship.position in self.GF.get_dropoff_positions()):
                        move = Direction.Still
                    # should never happen but just in case :D
                    elif Direction.Still == self.simulate_make_returning_move(other_ship):
                        move = Direction.Still
                # move around these ships
                elif self.ship_state[other_ship.id] in ["collecting", "waiting"]:
                    if self.ship_state[other_ship.id] == "collecting" and \
                            self.game_map[other_ship.position].halite_amount - self.game_map[other_ship.position].halite_amount / constants.EXTRACT_RATIO <= self.game_map.HALITE_STOP:
                        move = Direction.Still
                    else:
                        move = self.a_star_move(ship)

            else:  # target position occupied by enemy ship
                move = self.a_star_move(ship)
        return move

    def simulate_make_returning_move(self, other_ship):
        other_move = self.produce_move(other_ship)
        self.move_ship(other_ship, other_move)
        return other_move

    def get_dijkstra_move(self, ship):
        """
        Gets a move from the map created by Dijkstra
        :return: The target position of the move and the direction of the move.
        """
        current_position = ship.position
        cell = self.game_map[current_position].parent
        new_pos = cell.position
        dirs = self.game_map.get_target_direction(ship.position, new_pos)
        new_dir = dirs[0] if dirs[0] is not None else dirs[
            1] if dirs[1] is not None else Direction.Still

        return new_pos, new_dir

    def a_star_move(self, ship, dest=None):
        if dest is None:
            cell = self.game_map[ship.position]
            d_to_dijkstra_dest = self.game_map.calculate_distance(
                cell.position, cell.dijkstra_dest)
            dest = self.interim_dijkstra_dest(cell).position
        return self.exploring(ship, dest)

    def interim_dijkstra_dest(self, source_cell):
        ''' finds the intermediate dijkstra destination that is not occupied '''
        cell = source_cell.parent
        while cell.is_occupied and cell.position not in self.GF.get_dropoff_positions():
            cell = cell.parent
            if self.GF.time_left() < 0.3:
                logging.info(
                    "INTERIM DIJKSTRA DESTINATION STANDING STILL TOO SLOW")
                return source_cell
        return cell

    def move_ship_to_position(self, ship, destination):
        ''' moves ship to destination
        precondition: destination one move away'''
        if ship.id in self.ship_path and self.ship_path[ship.id]:
            move = self.get_step(self.ship_path[ship.id])
        else:
            move = self.dir_to_dest(ship.position, destination)
        self.move_ship(ship, move)

    def move_ship(self, ship, direction):
        """ Updates the command_queue
        and all necessary data structures about a ship moving """ 
        self.has_moved[ship.id] = True # ship has moved
        self.previous_position[ship.id] = ship.position # previous position is the current position
        self.command_queue.append(ship.move(direction)) # add the move to cmnd queue
        self.game_map[ship.position.directional_offset(direction)].mark_unsafe(ship) # mark next position unsafe
        # update cell occupation
        if direction != Direction.Still and self.game_map[ship.position].ship == ship:
            self.game_map[ship.position].ship = None
