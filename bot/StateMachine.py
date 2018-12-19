import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import GlobalConstants as GC


'''
Needed:
  functions:
    state_switch
    get_shipyard
    halite_priority_q
    amount_of_enemies
    get_best_neighbour
    better_patch_neighbouring
    process_new_destination
    find_new_destination
'''

class StateMachine():

    def __init__(self, game, ship, ship_path, ship_state, ship_dest, fleet_leader):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship = ship
        self.ship_path = ship_path
        self.ship_state = ship_state
        self.ship_dest = ship_dest
        self.fleet_leader = fleet_leader


    def state_transition(self):
        # transition
        new_state = None
        shipyard = GF.get_shipyard(self.ship.position)

        if self.game.turn_number >= GC.CRASH_TURN and self.game_map.calculate_distance(
                self.ship.position, shipyard) < 2:
            # if next to shipyard after crash turn, suicide
            self.ship_path[self.ship.id] = []
            new_state = "harakiri"

        elif game.turn_number >= GC.CRASH_TURN:
            # return if at crash turn
            self.ship_path[self.ship.id] = []
            new_state = "returning"

        elif self.ship.position in GF.get_dropoff_positions():
            self.ship_path[self.ship.id] = []
            GF.find_new_destination(
                self.game_map.halite_priority, self.ship)
            new_state = "exploring"

        else: new_state = new_state_switch.get(self.ship_state[self.ship.id], None)

        new_state_switch = {
            "exploring": self.exploring_transition(self.ship),
            "collecting": self.collecting_transition(self.ship),
            "returning": self.returning_transition(self.ship),
            "fleet": self.fleet_transition(self.ship),
            "builder": self.builder_transition(self.ship),
            "waiting": self.waiting_transition(self.ship),
            "backup": self.backup_transition(self.ship),
        }

        if new_state is not None:
            GF.state_switch(self.ship.id, new_state)


    """Exploring_transition"""
    def exploring_transition(self):
        distance_to_dest = self.game_map.calculate_distance(self.ship.position, self.ship_dest[self.ship.id])
        euclid_to_dest = self.game_map.euclidean_distance(self.ship.position, self.ship_dest[self.ship.id])
        if self.ship.position == self.ship_dest[self.ship.id]:
            # collect if reached destination or on medium sized patch
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif GF.amount_of_enemies(self.ship.position, 4) >= 2:
            # for inspiring
            self.ship_dest[self.ship.id] = GF.get_best_neighbour(self.ship.position).position

        elif euclid_to_dest <= 5 and GF.exists_better_in_area(self.ship.position, self.ship_dest[self.ship.id], 4):
            ship_h = GF.halite_priority_q(self.ship.position, GC.SHIP_SCAN_AREA)
            GF.find_new_destination(ship_h, self.ship)
            self.ship_path[self.ship.id] = []

        elif NR_OF_PLAYERS == 2 and distance_to_dest > GC.CLOSE_TO_SHIPYARD * self.game_map.width and ENABLE_COMBAT:
            # if not so close
            # check if neighbours have an enemy nearby with 2x more halite
            # if so, kill him
            for n in self.game_map.get_neighbours(self.game_map[self.ship.position]):
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    if n.ship.halite_amount >= 1.5 * self.ship.halite_amount:
                        logging.info("ASSASINATING")
                        self.ship_dest[self.ship.id] = n.position
                        return "assassinate"
        return None


    def collecting_transition(self):
        inspire_multiplier = 3 if self.game_map[self.ship.position].inspired else 1
        cell_halite = self.game_map[self.ship.position].halite_amount * inspire_multiplier
        if ship.is_full:
            return "returning"
        elif self.game_map.percentage_occupied >= GC.BUSY_PERCENTAGE and self.ship.halite_amount >= GC.BUSY_RETURN_AMOUNT:
            new_state = "returning"
        elif self.ship.halite_amount >= constants.MAX_HALITE * (return_percentage * 0.8) \
                and GF.better_patch_neighbouring(self.ship, GC.MEDIUM_HALITE):
            # if collecting and ship is half full but next to it there is a really
            # good patch, explore to that patch
            neighbour = GF.get_best_neighbour(self.ship.position)
            if neighbour.position == self.ship.position:
                new_state = "returning"
            else:
                self.ship_dest[self.ship.id] = neighbour.position

                for sh in self.me.get_ships():
                    # if somebody else going there recalc the destination
                    if not sh.id == self.ship.id and sh.id in self.ship_dest and self.ship_dest[sh.id] == neighbour.position:
                        GF.process_new_destination(sh)

                new_state = "exploring"

        elif self.ship.halite_amount >= constants.MAX_HALITE * return_percentage and \
                not (cell_halite * inspire_multiplier > GC.MEDIUM_HALITE and not self.ship.is_full):
            # return to shipyard if enough halite
            new_state = "returning"

        elif cell_halite < self.game_map.HALITE_STOP * inspire_multiplier:
            # Keep exploring if current halite patch is empty

            GF.process_new_destination(ship)
            new_state = "exploring"

        if self.ship.halite_amount <= constants.MAX_HALITE * 0.5 and NR_OF_PLAYERS == 2 and ENABLE_COMBAT:
            # if not so close
            # check if neighbours have an enemy nearby with 2x more halite
            # if so, kill him
            for n in self.game_map.get_neighbours(self.game_map[self.ship.position]):
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    if n.ship.halite_amount >= 1.8 * self.ship.halite_amount:
                        logging.info("ASSASINATING")
                        new_state = "assassinate"
                        self.ship_dest[self.ship.id] = n.position
        return new_state


    def returning_transition(self):
        if self.ship.position in GF.get_dropoff_positions():
            # explore again when back in shipyard
            return "exploring"
            GF.find_new_destination(
                self.game_map.halite_priority, self.ship)
        elif self.game_map.calculate_distance(self.ship.position, self.game_map[self.ship.position].dijkstra_dest) == 1:
            # if next to a dropoff
            cell = self.game_map[self.game_map[self.ship.position].dijkstra_dest]
            if cell.is_occupied and not me.has_ship(cell.ship.id) and "harakiri" not in self.ship_state.values():
                return "harakiri"
        return None


    def fleet_transition(self):
        destination = self.ship_dest[self.ship.id]
        if self.ship.position == destination:  # if arrived
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied:
                self.ship_dest[self.ship.id] = GF.get_best_neighbour(destination).position

        elif self.ship.id in self.fleet_leader:
            leader = self.fleet_leader[self.ship.id]
            if self.me.has_ship(leader.id) and self.ship_state[leader.id] not in ["waiting", "build"]:
                GF.process_new_destination(self.ship)
                return "exploring"
        return None


    def builder_transition(self):
        # if someone already built dropoff there before us
        future_dropoff_cell = self.game_map[self.ship_dest[self.ship.id]]
        distance_to_dest = self.game_map.euclidean_distance(self.ship.position, self.ship_dest[self.ship.id])

        if future_dropoff_cell.has_structure:
            if distance_to_dest <= GC.CLOSE_TO_SHIPYARD * self.game_map.width:
                self.ship_dest[self.ship.id] = GF.bfs_unoccupied(future_dropoff_cell.position)
            else:
                GF.process_new_destination(self.ship)
                return "exploring"

        elif GF.amount_of_enemies(future_dropoff_cell.position, 4) >= 4:
            return "exploring"

        elif distance_to_dest <= 2:
            neighbours = self.game_map.get_neighbours(self.game_map[ship.position])
            for n in neighbours:
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    self.ship_dest[self.ship.id] = GF.get_best_neighbour(self.ship.position).position
                    return "build"

        elif len(self.game.players.keys()) >= 2:
            smallest_dist = GF.dist_to_enemy_doff(self.ship_dest[self.ship.id])
            if smallest_dist <= (self.game_map.width * GC.ENEMY_SHIPYARD_CLOSE + 1):
                GF.process_new_destination(self.ship)
                return "exploring"
        return None


    def waiting_transition(self):
        neighbours = self.game_map.get_neighbours(self.game_map[self.ship.position])
        for n in neighbours:
            if n.is_occupied and not self.me.has_ship(n.ship.id):
                self.ship_dest[self.ship.id] = GF.get_best_neighbour(self.ship.position).position
                return "build"
        return None


    def backup_transition(self):
        destination = self.ship_dest[self.ship.id]
        if self.ship.position == destination:  # if arrived
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied:
                self.ship_dest[ship.id] = GF.get_best_neighbour(destination).position

        elif GF.amount_of_enemies(destination, 4) >= 4:
            GF.process_new_destination(self.ship)
            return "exploring"

        elif NR_OF_PLAYERS == 2 and ENABLE_COMBAT:
            # if not so close
            # check if neighbours have an enemy nearby with 2x more halite
            # if so, kill him
            for n in self.game_map.get_neighbours(self.game_map[self.ship.position]):
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    if n.ship.halite_amount >= 2 * self.ship.halite_amount:
                        logging.info("ASSASINATING")
                        self.ship_dest[self.ship.id] = n.position
                        return "assassinate"
        return None