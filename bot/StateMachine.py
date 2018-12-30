import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import bot.GlobalConstants as GC
import logging
from hlt.positionals import Direction, Position
from bot.DestinationProcessor import DestinationProcessor
from bot.GlobalFunctions import GlobalFunctions
from bot.GlobalVariablesSingleton import GlobalVariablesSingleton

class StateMachine():
    def __init__(self, game, return_percentage, prcntg_halite_left):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.return_percentage = return_percentage
        
        self.prcntg_halite_left = prcntg_halite_left
        GV = GlobalVariablesSingleton.getInstance()
        self.ENABLE_BACKUP = GV.ENABLE_BACKUP
        self.ENABLE_COMBAT = GV.ENABLE_COMBAT
        self.ship_path = GV.ship_path
        self.ship_state = GV.ship_state
        self.ship_dest = GV.ship_dest
        self.fleet_leader = GV.fleet_leader
        self.previous_state = GV.previous_state
        self.NR_OF_PLAYERS = GV.NR_OF_PLAYERS
        self.GF = GlobalFunctions(self.game)


    def state_transition(self, ship):
        # transition
        self.ship = ship
        new_state = None
        shipyard = self.GF.get_shipyard(self.ship.position)
        DP = DestinationProcessor(self.game)

        if self.game.turn_number >= GC.CRASH_TURN and self.game_map.calculate_distance(
                self.ship.position, shipyard) < 2:
            # if next to shipyard after crash turn, suicide
            new_state = "harakiri"

        elif self.game.turn_number >= GC.CRASH_TURN:
            # return if at crash turn
            new_state = "returning"

        elif self.ship.position in self.GF.get_dropoff_positions():
            DP.process_new_destination(self.ship)
            new_state = "exploring"

        # decent halite and close to enemy dropoff, return
        elif self.ship.halite_amount >= 0.5 * constants.MAX_HALITE and self.GF.dist_to_enemy_doff(self.ship.position) < 0.1 * self.game_map.width:
            new_state = "returning"

        elif self.ship.halite_amount >= constants.MAX_HALITE * self.return_percentage and self.ship_state[self.ship.id] not in ["build", "waiting", "collecting", "returning"]:
            new_state = "returning"

        elif self.ship_state[self.ship.id] == "exploring":
            new_state = self.exploring_transition()

        elif self.ship_state[self.ship.id] == "collecting":
            new_state = self.collecting_transition()

        elif self.ship_state[self.ship.id] == "returning":
            new_state = self.returning_transition()

        elif self.ship_state[self.ship.id] == "fleet":
            new_state = self.fleet_transition()

        elif self.ship_state[self.ship.id] == "build":
            new_state = self.builder_transition()

        elif self.ship_state[self.ship.id] == "waiting":
            new_state = self.waiting_transition()

        elif self.ship_state[self.ship.id] == "backup":
            new_state = self.backup_transition()

        if new_state is not None:
            self.GF.state_switch(self.ship.id, new_state)


    """Exploring_transition"""
    def exploring_transition(self):
        distance_to_dest = self.game_map.calculate_distance(self.ship.position, self.ship_dest[self.ship.id])
        euclid_to_dest = self.game_map.euclidean_distance(self.ship.position, self.ship_dest[self.ship.id])
        DP = DestinationProcessor(self.game)
        if self.ship.position == self.ship_dest[self.ship.id]:
            # collect if reached destination or on medium sized patch
            return "collecting"

        elif self.game_map[self.ship.position].halite_amount >= GC.MEDIUM_HALITE: 
            self.ship_dest[self.ship.id] = self.ship.position
            self.ship_path[self.ship.id] = []
            DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            return "collecting"

        elif self.game_map[self.ship.position].inspired and self.game_map[self.ship.position].enemy_amount <= GC.UNSAFE_AREA:
            # for inspiring
            self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
            DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)

        elif euclid_to_dest <= 2\
                and self.exists_better_in_area(self.ship.position, self.ship_dest[self.ship.id], 4):
            DP.process_new_destination(self.ship)

        elif self.ENABLE_COMBAT and (distance_to_dest > GC.CLOSE_TO_SHIPYARD * self.game_map.width or distance_to_dest == 1)\
                and self.GF.dist_to_enemy_doff(self.ship.position) >= GC.ENEMY_SHIPYARD_CLOSE * self.game_map.width:
            return self.attempt_switching_assasinate()
        return None


    def attempt_switching_assasinate(self):
        for n in self.game_map.get_neighbours(self.game_map[self.ship.position]):
            if n.is_occupied and not self.me.has_ship(n.ship.id):
                # if that ship has 2x the halite amount, decent amount of halite,
                # and if tehre are ships to send as backup
                if (n.ship.halite_amount + n.halite_amount) >= 2 * (self.ship.halite_amount + self.game_map[self.ship.position].halite_amount) and \
                        n.ship.halite_amount >= GC.MEDIUM_HALITE and\
                        self.enough_backup_nearby(n.position, int(0.15 * self.game_map.width), 2):
                    # assasinate that mofo
                    logging.info("ASSASSINATING")
                    self.ship_dest[self.ship.id] = n.position
                    return "assassinate"
        return None


    def enough_backup_nearby(self, pos, distance, amount):
        """ boolean function for determining whether there are 
        amount of friendly ships within distance from pos """
        actual_amount = 0
        for my_ship in self.me.get_ships():
            distance_to_pos = self.game_map.euclidean_distance(pos, my_ship.position)
            if distance_to_pos <= distance and self.is_savior(my_ship):
                actual_amount += 1
        return actual_amount >= amount


    def is_savior(self, ship):
        return self.me.has_ship(ship.id) and ship.halite_amount <= self.return_percentage * 0.5 * constants.MAX_HALITE \
                and (ship.id not in self.ship_state or not (self.ship_state[ship.id] in ["waiting", "returning", "build"]))


    def exists_better_in_area(self, cntr, current, area):
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + cntr  # top left of scan area
        current_factor = self.game_map.cell_factor(cntr, self.game_map[current], self.me, self.NR_OF_PLAYERS)
        for y in range(area):
            for x in range(area):
                p = Position((top_left.x + x) % self.game_map.width,
                             (top_left.y + y) % self.game_map.height)  # position of patch
                cell = self.game_map[p]
                if cell.halite_amount >= self.game_map.HALITE_STOP:
                    other_factor = self.game_map.cell_factor(cntr, cell, self.me, self.ENABLE_BACKUP)
                    if not cell.is_occupied and other_factor < current_factor:
                        return True
        return False


    def collecting_transition(self):
        inspire_multiplier = self.game_map.get_inspire_multiplier(
            self.ship.position, self.game_map[self.ship.position], self.ENABLE_BACKUP)
        cell_halite = self.game_map[self.ship.position].halite_amount * inspire_multiplier
        DP = DestinationProcessor(self.game)

        if self.ship.is_full:
            return "returning"

        elif self.NR_OF_PLAYERS == 4 and self.game_map.percentage_occupied >= GC.BUSY_PERCENTAGE and self.ship.halite_amount >= GC.BUSY_RETURN_AMOUNT:
            return "returning"

        elif self.ship.halite_amount >= constants.MAX_HALITE * self.return_percentage and \
                not (cell_halite > GC.MEDIUM_HALITE * inspire_multiplier and not self.ship.is_full):
            # return to shipyard if enough halite
            return "returning"

        elif self.ship.halite_amount < constants.MAX_HALITE * self.return_percentage * 0.9 and self.game_map[self.ship.position].halite_amount <= 100 and \
                self.better_patch_neighbouring(self.game_map[self.ship.position].halite_amount):
            # explore to best neighbour if current cell has low halite, and there is a 2x patch next to it
            self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
            DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            return "exploring"

        elif (cell_halite < self.game_map.HALITE_STOP * inspire_multiplier and self.ship.halite_amount < constants.MAX_HALITE * self.return_percentage * .9):
            # Keep exploring if halite patch low on halite and ship doesnt have close to returning percentage
            DP.process_new_destination(self.ship)
            return "exploring"

        elif cell_halite < self.game_map.HALITE_STOP * inspire_multiplier:
            # return if cell has low halite
            return "returning"

        elif self.ship.halite_amount <= constants.MAX_HALITE * 0.4 and self.ENABLE_COMBAT\
                and self.GF.dist_to_enemy_doff(self.ship.position) >= GC.CLOSE_TO_SHIPYARD * self.game_map.width and self.game_map[self.ship.position].enemy_neighbouring:
            # if not that mch halite and not too close to assasinate and enemy is neighbouring, attempt to kill sm1 
            return self.attempt_switching_assasinate()

        elif self.game_map[self.ship.position].enemy_neighbouring > 0 and self.ship.halite_amount >= GC.MEDIUM_HALITE and self.game_map[self.ship.position].halite_amount <= GC.MEDIUM_HALITE:  # if enemy right next to it
            # move to neighbour that has minimal enemies, but more halite
            ratio = cell_halite / (self.game_map[self.ship.position].enemy_neighbouring + 1)
            next_dest = self.ship.position
            for n in self.game_map.get_neighbours(self.game_map[self.ship.position]):
                n_ratio = n.halite_amount * self.game_map.get_inspire_multiplier(
                    self.ship.position, n, self.ENABLE_BACKUP) / (n.enemy_neighbouring + 1)
                if n_ratio > ratio:
                    ratio = n_ratio
                    next_dest = n.position
            self.ship_path[self.ship.id] = []
            self.ship_dest[self.ship.id] = n.position
            DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            return "exploring"

        return None


    def better_patch_neighbouring(self, big_diff):
        ''' returns true if there is a lot better patch right next to it'''
        current = self.game_map[self.ship.position]
        neighbours = self.game_map.get_neighbours(current)
        current_h = current.halite_amount * \
            self.game_map.get_inspire_multiplier(
                self.ship.position, self.game_map[self.ship.position], self.ENABLE_BACKUP)

        for n in neighbours:
            neighbour_h = n.halite_amount * \
                self.game_map.get_inspire_multiplier(
                    self.ship.position, self.game_map[n.position], self.ENABLE_BACKUP)
            if n.enemy_amount < GC.UNSAFE_AREA and neighbour_h >= current_h + big_diff:
                return True

        return False


    def returning_transition(self):
        DP = DestinationProcessor(self.game)

        if self.game.turn_number >= GC.CRASH_TURN and self.game_map[self.ship.position].parent.is_occupied\
            and not self.me.has_ship(self.game_map[self.ship.position].parent.ship.id):
            self.ship_dest[self.ship.id] = self.game_map[self.ship.position].parent.position
            return "assassinate"

        elif self.ship.position in self.GF.get_dropoff_positions():
            # explore again when back in shipyard
            DP.process_new_destination(self.ship)
            return "exploring"

        elif self.game_map.calculate_distance(self.ship.position, self.GF.get_shipyard(self.ship.position)) == 1:
            # if next to a dropoff 
            cell = self.game_map[self.GF.get_shipyard(self.ship.position)]
            if cell.is_occupied and not self.me.has_ship(cell.ship.id) and "harakiri" not in self.ship_state.values():
                return "harakiri"
        return None


    def fleet_transition(self):
        DP = DestinationProcessor(self.game)
        destination = self.ship_dest[self.ship.id]
        if self.ship.position == destination:  # if arrived
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied:
                self.ship_dest[self.ship.id] = self.get_best_neighbour(destination).position
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)

        elif self.ship.id in self.fleet_leader:
            leader = self.fleet_leader[self.ship.id]
            if self.me.has_ship(leader.id) and self.ship_state[leader.id] not in ["waiting", "build"]:
                DP.process_new_destination(self.ship)
                return "exploring"
        return None


    def builder_transition(self):
        # if someone already built dropoff there before us
        DP = DestinationProcessor(self.game)
        future_dropoff_cell = self.game_map[self.ship_dest[self.ship.id]]
        distance_to_dest = self.game_map.euclidean_distance(self.ship.position, self.ship_dest[self.ship.id])

        if future_dropoff_cell.has_structure:
            if distance_to_dest <= GC.CLOSE_TO_SHIPYARD * self.game_map.width:
                self.ship_dest[self.ship.id] = self.GF.bfs_unoccupied(future_dropoff_cell.position)
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            else:
                DP.process_new_destination(self.ship)
                return "exploring"

        elif self.game_map[future_dropoff_cell.position].enemy_amount >= GC.UNSAFE_AREA:
            DP.process_new_destination(self.ship)
            return "exploring"

        elif distance_to_dest <= 2:
            neighbours = self.game_map.get_neighbours(self.game_map[self.ship.position])
            for n in neighbours:
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
                    DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
                    return "build"
        elif self.NR_OF_PLAYERS == 4 or self.game_map.width >= 56: # for 4 players and large maps 1v1
            smallest_dist = self.GF.dist_to_enemy_doff(self.ship_dest[self.ship.id])
            if smallest_dist <= self.game_map.width * GC.ENEMY_SHIPYARD_CLOSE:
                DP.process_new_destination(self.ship)
                return "exploring"
        return None


    def waiting_transition(self):
        DP = DestinationProcessor(self.game)
        neighbours = self.game_map.get_neighbours(self.game_map[self.ship.position])
        cell = self.game_map[self.ship.position]
        for n in neighbours:
            if n.is_occupied and not self.me.has_ship(n.ship.id):
                self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
                return "build"
        else:
            if cell.halite_amount <= 500 or (self.ship.is_full and cell.halite_amount <= 800):
                self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
                return "build"
        return None


    def backup_transition(self):
        DP = DestinationProcessor(self.game)
        destination = self.ship_dest[self.ship.id]
        if self.ship.position == destination:  # if arrived
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif self.ship.halite_amount >= constants.MAX_HALITE * self.return_percentage:
            return "returning"
        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied and self.me.has_ship(self.game_map[destination].ship.id):
                self.ship_dest[self.ship.id] = self.get_best_neighbour(destination).position
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            elif self.game_map[destination].is_occupied:  # not our ship
                return self.attempt_switching_assasinate()
        elif self.game_map[destination].enemy_amount >= GC.UNSAFE_AREA:
            DP.process_new_destination(self.ship)
            return "exploring"
        elif self.ENABLE_COMBAT and self.GF.dist_to_enemy_doff(self.ship.position) >= GC.CLOSE_TO_SHIPYARD * self.game_map.width:
            return self.attempt_switching_assasinate()
        return None


    def get_best_neighbour(self, position):
        ''' gets best neighbour at the ships positioņ
        returns a cell'''
        current = self.game_map[position]
        neighbours = self.game_map.get_neighbours(current)
        max_halite = current.halite_amount * \
            self.game_map.get_inspire_multiplier(position, current, self.ENABLE_BACKUP)
        best = current
        for n in neighbours:
            n_halite = n.halite_amount * \
                self.game_map.get_inspire_multiplier(position, n, self.ENABLE_BACKUP)
            if n.enemy_amount < GC.UNSAFE_AREA and n_halite > max_halite:
                best = n
                max_halite = n_halite
        return best
