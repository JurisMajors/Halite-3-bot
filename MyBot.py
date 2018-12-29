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

#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')

dropoff_clf = pickle.load(open('mlp.sav', 'rb'))
# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
import bot.GlobalConstants as GC

from bot.ClusterProcessor import ClusterProcessor
from bot.DestinationProcessor import DestinationProcessor
from bot.GlobalFunctions import GlobalFunctions
from bot.GlobalVariablesSingleton import GlobalVariablesSingleton
from bot.MoveProcessor import MoveProcessor
from bot.StateMachine import StateMachine

game.ready("v56")


class main():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton(game)

        self.ENABLE_BACKUP = GV.ENABLE_BACKUP
        self.ENABLE_COMBAT = GV.ENABLE_COMBAT
        self.ship_state = GV.ship_state
        self.ship_path = GV.ship_path
        self.ship_dest = GV.ship_dest
        self.previous_position = GV.previous_position
        self.previous_state = GV.previous_state
        self.fleet_leader = GV.fleet_leader
        self.ship_obj = GV.ship_obj
        self.NR_OF_PLAYERS = GV.NR_OF_PLAYERS

        self.crashed_positions = []
        self.cluster_centers = []
        self.clusters_determined = False
        self.crashed_ship_positions = []
        self.dropoff_last_built = 0


    def mainloop(self):
        while True:
            self.game.update_frame()
            self.game_map = self.game.game_map
            self.me = self.game.me
            self.clear_dictionaries()  # of crashed or transformed ships
            command_queue = []

            if self.game.turn_number == 1:
                self.game_map.HALITE_STOP = GC.INITIAL_HALITE_STOP
                self.game_map.c = [GC.A, GC.B, GC.C, GC.D, GC.E, GC.F]  # set the heuristic constants
            for s in self.me.get_ships():
                if s.id not in self.ship_state:
                    self.ship_state[s.id] = "exploring"

            GlobalVariablesSingleton.getInstance().ENABLE_COMBAT = not self.have_less_ships(0.8) and self.NR_OF_PLAYERS == 2
            GlobalVariablesSingleton.getInstance().turn_start = time.time()
            self.ENABLE_COMBAT = GlobalVariablesSingleton.getInstance().ENABLE_COMBAT

            enable_inspire = not self.have_less_ships(0.8)
            GlobalVariablesSingleton.getInstance().ENABLE_BACKUP = self.ENABLE_COMBAT
            self.ENABLE_BACKUP = GlobalVariablesSingleton.getInstance().ENABLE_BACKUP
            # initialize shipyard halite, inspiring stuff and other
            self.game_map.init_map(self.me, list(self.game.players.values()), enable_inspire, self.ENABLE_BACKUP)
            if self.game.turn_number == 1:
                TOTAL_MAP_HALITE = self.game_map.total_halite

            self.prcntg_halite_left = self.game_map.total_halite / TOTAL_MAP_HALITE
            # if clusters_determined and not cluster_centers:
            if self.game.turn_number >= GC.SPAWN_TURN:
                self.game_map.HALITE_STOP = self.prcntg_halite_left * GC.INITIAL_HALITE_STOP

            if self.crashed_ship_positions and self.game.turn_number < GC.CRASH_TURN and self.ENABLE_BACKUP:
                self.process_backup_sending()

            # Dijkstra the graph based on all dropoffs
            self.game_map.create_graph(GlobalFunctions(self.game).get_dropoff_positions())

            if self.game.turn_number == GC.DETERMINE_CLUSTER_TURN:
                self.clusters_determined = True
                self.cluster_centers = ClusterProcessor(game).clusters_with_classifier()

            if self.game.turn_number == GC.CRASH_SELECTION_TURN:
                GC.CRASH_TURN = self.select_crash_turn()

            if self.prcntg_halite_left > 0.2:
                self.return_percentage = GC.BIG_PERCENTAGE if self.game.turn_number < GC.PERCENTAGE_SWITCH else GC.SMALL_PERCENTAGE
            else:  # if low percentage in map, return when half full
                self.return_percentage = 0.6

            if self.should_build():
                self.process_building(self.cluster_centers)

            # has_moved ID->True/False, moved or not
            # ships priority queue of (importance, ship)
            ships, self.has_moved = self.ship_priority_q()
            # True if a ship moves into the shipyard this turn
            move_into_shipyard = False
            # whether a dropoff has been built this turn so that wouldnt use too much
            # halite
            dropoff_built = False


            while ships:  # go through all ships
                ship = heappop(ships)[1]
                if self.has_moved[ship.id]:
                    continue
                if GlobalFunctions(self.game).time_left() < 0.15:
                    logging.info("STANDING STILL TOO SLOW")
                    command_queue.append(ship.stay_still())
                    self.ship_state[ship.id] = "collecting"
                    continue
                if ship.id not in self.previous_position:  # if new ship the
                    self.previous_position[ship.id] = self.me.shipyard.position

                # setup state
                # if ship hasnt received a destination yet
                if ship.id not in self.ship_dest or not ship.id in self.ship_state:
                    DestinationProcessor(self.game).find_new_destination(
                        self.game_map.halite_priority, ship)
                    self.previous_state[ship.id] = "exploring"
                    self.ship_state[ship.id] = "exploring"  # explore


                # transition
                SM = StateMachine(self.game, self.return_percentage, self.prcntg_halite_left)
                SM.state_transition(ship)

                # logging.info("SHIP {}, STATE {}, DESTINATION {}".format(
                #     ship.id, self.ship_state[ship.id], self.ship_dest[ship.id]))
                MP = MoveProcessor(self.game, self.has_moved, command_queue)
                # if ship is dropoff builder
                if self.is_builder(ship):
                    # if enough halite and havent built a dropoff this turn
                    if (ship.halite_amount + self.me.halite_amount) >= constants.DROPOFF_COST and not dropoff_built:
                        command_queue.append(ship.make_dropoff())
                        self.dropoff_last_built = game.turn_number
                        GC.SPAWN_TURN += 10
                        dropoff_built = True
                    else:  # cant build
                        self.ship_state[ship.id] = "waiting"  # wait in the position
                        self.game_map[ship.position].mark_unsafe(ship)
                        command_queue.append(ship.move(Direction.Still))
                else:  # not associated with building a dropoff, so move regularly
                    move = MP.produce_move(ship)
                    if move is not None:
                        command_queue.append(ship.move(move))
                        self.previous_position[ship.id] = ship.position
                        self.game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
                        if move != Direction.Still and self.game_map[ship.position].ship == ship:
                            self.game_map[ship.position].ship = None

                self.clear_dictionaries()  # of crashed or transformed ships

                # This ship has made a move
                self.has_moved[ship.id] = True

            surrounded_shipyard = self.game_map.is_surrounded(self.me.shipyard.position)
            logging.info(GlobalFunctions(self.game).time_left())
            if not dropoff_built and 2 * (self.max_enemy_ships() + 1) > len(self.me.get_ships()) and self.game.turn_number <= GC.SPAWN_TURN \
                    and self.me.halite_amount >= constants.SHIP_COST and self.prcntg_halite_left > (1 - 0.65) and \
                    not (self.game_map[self.me.shipyard].is_occupied or surrounded_shipyard or "waiting" in self.ship_state.values()):
                if not ("build" in self.ship_state.values() and self.me.halite_amount <= (constants.SHIP_COST + constants.DROPOFF_COST)):
                    command_queue.append(self.me.shipyard.spawn())
            # Send your moves back to the game environment, ending this turn.
            self.game.end_turn(command_queue)


    def max_enemy_ships(self):
        if self.NR_OF_PLAYERS not in [2, 4]:  # for testing solo games
            return 100000
        ships = 0
        for player_id in self.game.players.keys():
            if not player_id == self.me.id:
                ships = max(ships, self.get_ship_amount(player_id))
        return ships


    def get_ship_amount(self, playerID):
        return len(self.game.players[playerID].get_ships())


    def process_building(self, cluster_centers):
        dropoff_val, dropoff_pos = cluster_centers.pop(0)  # remove from list
        # sends ships to position where closest will build dropoff
        dropoff_pos = self.game_map.normalize(dropoff_pos)
        fleet = self.get_fleet(dropoff_pos, 1)
        if fleet:
            closest_ship = fleet.pop(0)  # remove and get closest ship
            GlobalFunctions(self.game).state_switch(closest_ship.id, "build")  # will build dropoff
            # if dropoffs position already has a structure (e.g. other dropoff) or
            # somebody is going there already
            if self.game_map[dropoff_pos].has_structure or dropoff_pos in self.ship_dest.values():
                # bfs for closer valid unoccupied position
                dropoff_pos = GlobalFunctions(self.game).bfs_unoccupied(dropoff_pos)
            self.ship_dest[closest_ship.id] = dropoff_pos  # go to the dropoff
            if self.game_map.width >= GC.EXTRA_FLEET_MAP_SIZE:
                self.send_ships(dropoff_pos, int(GC.FLEET_SIZE / 2), "fleet", leader=closest_ship)
        else:  # if builder not available
            cluster_centers.insert(0, (dropoff_val, dropoff_pos))


    def send_ships(self, pos, ship_amount, new_state, condition=None, leader=None):
        '''sends a fleet of size ship_amount to explore around pos
        new_state : state to switch the fleet members
        condition : boolean function that qualifies a ship to send'''
        fleet = self.get_fleet(self.game_map.normalize(pos), ship_amount, condition)
        # for rest of the fleet to explore
        h = GlobalFunctions(self.game).halite_priority_q(pos, GC.SHIP_SCAN_AREA)

        for fleet_ship in fleet:  # for other fleet members
            if len(h) == 0:
                break
            if fleet_ship.id not in self.previous_state.keys():  # if new ship
                self.previous_state[fleet_ship.id] = "exploring"
                self.has_moved[fleet_ship.id] = False
            # explore in area of the new dropoff
            if leader is not None:
                self.fleet_leader[fleet_ship.id] = leader

            GlobalFunctions(self.game).state_switch(fleet_ship.id, new_state)
            DestinationProcessor(self.game).find_new_destination(h, fleet_ship)


    def get_fleet(self, position, fleet_size, condition=None):
        ''' returns list of fleet_size amount
        of ships closest to the position'''
        if condition is None: condition = self.is_fleet
        distances = []
        for s in self.me.get_ships():
            if condition(s) and not s.position == position:
                distances.append(
                    (self.game_map.calculate_distance(position, s.position), s))
        distances.sort(key=lambda x: x[0])
        fleet_size = min(len(distances), fleet_size)
        return [t[1] for t in distances[:fleet_size]]


    def have_less_ships(self, ratio):
        for player in self.game.players.values():
            if len(self.me.get_ships()) < ratio * len(player.get_ships()):
                return True
        return False

    def should_build(self):
        # if clusters determined, more than 13 ships, we have clusters and nobody
        # is building at this turn (in order to not build too many)
        if len(self.me.get_ships()) / len(GlobalFunctions(self.game).get_dropoff_positions()) >= GC.MAX_SHIP_DROPOFF_RATIO and not self.any_builders():
            # there are more than 40 ships per dropoff
            if self.cluster_centers:  # there is already a dropoff position
                return True
            # there is no good dropoff position yet, make one
            self.game_map.set_close_friendly_ships(self.me)
            pos = self.game_map.get_most_dense_dropoff_position(GlobalFunctions(self.game).get_dropoff_positions())
            self.cluster_centers.append((10000, pos))  # fake 10000 halite for new needed cluster
            return True

    # Original dropoff code

        return self.clusters_determined and self.game.turn_number >= self.dropoff_last_built + 15 and self.cluster_centers \
            and len(self.me.get_ships()) > (len(GlobalFunctions(self.game).get_dropoff_positions()) + 1) * GC.FLEET_SIZE\
            and self.fleet_availability() >= 1.5 * GC.FLEET_SIZE and not self.any_builders()


    def any_builders(self):
        return "waiting" in self.ship_state.values() or "build" in self.ship_state.values()


    def fleet_availability(self):
        ''' returns how many ships are available atm'''
        amount = 0
        for s in self.me.get_ships():
            if self.is_fleet(s):
                amount += 1
        if amount >= len(list(self.me.get_ships())) * 0.9:
            amount /= 2
        return int(amount)


    def is_fleet(self, ship):
        ''' returns if a ship is good for adding it to a fleet '''
        return self.me.has_ship(ship.id) and (
                    ship.id not in self.ship_state or not (self.ship_state[ship.id] in ["fleet", "waiting", "returning", "build"]))


    def is_builder(self, ship):
        ''' checks if this ship is a builder '''
        return self.ship_state[ship.id] == "waiting" or (self.ship_state[ship.id] == "build" and self.ship_dest[ship.id] == ship.position)


    def select_crash_turn(self):
        '''selects turn when to crash'''
        distance = 0
        for ship in self.me.get_ships():
            shipyard = GlobalFunctions(self.game).get_shipyard(ship.position)  # its shipyard position
            d = self.game_map.calculate_distance(shipyard, ship.position)
            if d > distance:  # get maximum distance away of shipyard
                distance = d
        crash_turn = constants.MAX_TURNS - distance - 5
        # set the crash turn to be turn s.t. all ships make it
        crash_turn = max(crash_turn, GC.CRASH_SELECTION_TURN)
        if self.game_map.width == 64:
            crash_turn -= 5
        return crash_turn


    def ship_priority_q(self):
        ships = []  # ship priority queue
        has_moved = {}
        for s in self.me.get_ships():
            self.ship_obj[s.id] = s
            has_moved[s.id] = False
            if s.id in self.ship_state:
                # get ships shipyard
                shipyard = GlobalFunctions(self.game).get_shipyard(s.position)
                # importance, the lower the number, bigger importance
                if s.position == shipyard:
                    importance = -10000
                elif self.ship_state[s.id] in ["returning", "harikiri"]:
                    importance = round((self.game_map.width * -2) / self.game_map[
                        s.position].dijkstra_distance, 2)
                elif self.ship_state[s.id] in ["exploring", "build", "backup", "fleet"]:
                    if s.id in self.ship_dest:
                        destination = self.ship_dest[s.id]
                    else:
                        destination = shipyard
                    importance = self.game_map.calculate_distance(
                        s.position, destination) * self.game_map.width + 1
                else:  # other
                    importance = self.game_map.calculate_distance(
                        s.position, shipyard) * self.game_map.width ** 2
            else:
                importance = -1000  # newly spawned ships max importance
            heappush(ships, (importance, s))
        return ships, has_moved


    def clear_dictionaries(self):
        # clear dictionaries of crushed ships
        for ship_id in list(self.ship_dest.keys()):
            if not self.me.has_ship(ship_id):
                self.crashed_ship_positions.append(self.ship_obj[ship_id].position)
                del self.ship_dest[ship_id]
                del self.ship_state[ship_id]
                del self.previous_state[ship_id]
                del self.previous_position[ship_id]
                if ship_id in self.ship_path:
                    del self.ship_path[ship_id]



    def add_crashed_position(self, pos):
        """ adds a carshed position ot the crashed positions list '"""
        neighbours = self.game_map.get_neighbours(self.game_map[pos])
        h_amount = -1
        distance_to_enemy_dropoff = GlobalFunctions(self.game).dist_to_enemy_doff(pos)
        for n in neighbours:
            h_amount = max(h_amount, n.halite_amount)
        if h_amount > 800:
            heappush(self.crashed_positions, (-1 * h_amount, pos))


    def process_backup_sending(self):
        """ Processes sending backup ships to a position
        where a ship crashed previously """
        to_remove = []
        for pos in self.crashed_ship_positions:
            if GlobalFunctions(self.game).dist_to_enemy_doff(pos) > 4:
                self.add_crashed_position(pos)
            to_remove.append(pos)
        for s in to_remove:  # remove the crashed positions
            if s in self.crashed_ship_positions:
                self.crashed_ship_positions.remove(s)
        if self.crashed_positions:  # if there are any crashed positions to process
            hal, crashed_pos = heappop(self.crashed_positions)  # get pos info
            # if there are little enemies in that area
            if self.game_map[crashed_pos].enemy_amount <= GC.UNSAFE_AREA and self.game_map[crashed_pos].halite_amount >= constants.MAX_HALITE:
                # send a backup fleet there
                self.send_ships(crashed_pos, 2, "backup", StateMachine(self.game, self.return_percentage, self.prcntg_halite_left).is_savior)

    def is_savior(self, ship):
        return self.me.has_ship(ship.id) and ship.halite_amount <= self.return_percentage * 0.5 * constants.MAX_HALITE \
                and (ship.id not in self.ship_state or not (self.ship_state[ship.id] in ["waiting", "returning", "build"]))


backuped_dropoffs = []
maingameloop = main(game)
maingameloop.mainloop()
