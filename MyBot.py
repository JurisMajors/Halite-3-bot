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

#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')

""" <<<Game Begin>>> """
dropoff_clf = pickle.load(open('mlp.sav', 'rb'))
# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.

import bot.GlobalConstants as GC

game.ready("MLP")
NR_OF_PLAYERS = len(game.players.keys())

SAVIOR_FLEET_SIZE = 0.1 if NR_OF_PLAYERS == 2 else 0.05

class GlobalFunctions():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton.getInstance()
        self.ENABLE_BACKUP = GV.ENABLE_BACKUP
        self.ENABLE_COMBAT = GV.ENABLE_COMBAT
        self.previous_state = GV.previous_state
        self.ship_state = GV.ship_state
        self.ship_path = GV.ship_path
        self.ship_dest = GV.ship_dest


    def halite_priority_q(self, pos, area):
        # h_amount <= 0 to run minheap as maxheap
        h = []  # stores halite amount * -1 with its position in a minheap
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + pos  # top left of scan area
        for y in range(area):
            for x in range(area):
                p = Position((top_left.x + x) % self.game_map.width,
                             (top_left.y + y) % self.game_map.height)  # position of patch
                if not (p == pos or p in self.get_dropoff_positions()):  # we dont consider the position of the centre or dropoffs
                    cell = self.game_map[p]
                    # we ignore cells who have 0 halite.
                    # if that cell has small amount of halite, just take a ratio with 2x distance to lesser the priority
                    if 0 < cell.halite_amount <= self.game_map.HALITE_STOP:
                        ratio = cell.halite_amount / \
                                (2 * self.game_map.calculate_distance(p, pos))
                        heappush(h, (-1 * ratio, p))
                    elif cell.halite_amount > 0:
                        factor = self.game_map.cell_factor(pos, cell, self.me, self.ENABLE_BACKUP)
                        # add negative halite amounts so that would act as maxheap
                        heappush(h, (factor, p))
        return h


    def dist_to_enemy_doff(self, pos):
        ''' determines how close to an enemy dropoff is the position'''
        if NR_OF_PLAYERS < 2:
            return 1000
        return min([self.game_map.euclidean_distance(pos, d) for d in self.get_enemy_dropoff_positions()])


    def get_shipyard(self, position):
        """ gives shipyard that the ship would return to """
        # return game_map[position].dijkstra_dest
        return min([(self.game_map.euclidean_distance(position, d), d) for d in self.get_dropoff_positions()], key=lambda x: x[0])[1]


    def state_switch(self, ship_id, new_state):
        if ship_id not in self.previous_state:
            self.previous_state[ship_id] = "exploring"
        if ship_id not in self.ship_state:
            self.ship_state[ship_id] = self.previous_state[ship_id]
        if not new_state == "exploring":  # reset path to empty list
            self.ship_path[ship_id] = []
        if new_state == "returning":
            self.ship_dest[ship_id] = self.game_map[self.me.get_ship(ship_id).position].dijkstra_dest

        self.previous_state[ship_id] = self.ship_state[ship_id]
        self.ship_state[ship_id] = new_state


    def get_enemy_dropoff_positions(self):
        ''' returns a list of enemy dropoffs, including shipyards '''
        positions = []
        for player in self.game.players.values():  # for each player in game
            if not player.id == self.me.id:  # if not me
                positions.append(player.shipyard.position)
                for d_off in player.get_dropoffs():
                    positions.append(d_off.position)
        return positions


    def get_dropoff_positions(self):
        """ Returns a list of all positions of dropoffs and the shipyard """
        return [dropoff.position for dropoff in self.me.get_dropoffs()] + [self.me.shipyard.position]


    @staticmethod
    def time_left():
        return 2 - (time.time() - GlobalVariablesSingleton.getInstance().turn_start)


class ClusterProcessor():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton.getInstance()


    def clusters_with_classifier(self):
        ''' uses classifier to determine clusters for dropoff '''
        cluster_centers = self.predict_centers()
        # do filtering
        cluster_centers = self.filter_clusters(cluster_centers, GC.MAX_CLUSTERS)
        logging.info("Finally")
        logging.info(cluster_centers)
        return cluster_centers


    def predict_centers(self):
        cntr = self.find_center()

        # get area around our cntr
        x_size = int(self.game_map.width /
                     2) if NR_OF_PLAYERS in [2, 4] else self.game_map.width
        y_size = self.game_map.height if NR_OF_PLAYERS in [2, 1] else int(
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
                p_data, total_halite, p_center = self.get_patch_data(
                    x, y, cntr)  # get the data
                prediction = dropoff_clf.predict(p_data)[0]  # predict on it
                p_center = self.game_map.normalize(p_center)
                if prediction == 1:  # if should be dropoff
                    # add node with most halite to centers
                    for _, c in cluster_centers:
                        if c == p_center:
                            break
                    else:
                        cluster_centers.append((total_halite, p_center))
        return cluster_centers


    def find_center(self):
        ''' finds center of our part of the map '''
        travel = int(self.game_map.width / NR_OF_PLAYERS)
        # get all the centers depending on the amount of players
        if NR_OF_PLAYERS == 4:
            cntrs = [Position(travel, travel), Position(travel * 3, travel),
                     Position(travel * 3, travel * 3), Position(travel, travel * 3)]
        elif NR_OF_PLAYERS == 2:
            cntrs = [Position(int(travel / 2), travel),
                     Position(travel + int(travel / 2), travel)]
        else:
            cntrs = [self.me.shipyard.position]

        min_dist = 1000
        # find the center thats the closes to the shipyard
        for pos in cntrs:
            dist = self.game_map.calculate_distance(pos, self.me.shipyard.position)
            if dist < min_dist:
                cntr = pos
                min_dist = dist
        return cntr


    def filter_clusters(self, centers, max_centers):
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
            centers = self.merge_clusters(centers)
        logging.info(centers)

        centers_copy = centers[:]
        # remove points that are too close to each other or the shipyard
        # in priority of points that are have the largest amount of points in area
        for i, d in enumerate(centers_copy, start=0):
            halite, pos = d
            diff = self.game_map.euclidean_distance(pos, self.me.shipyard.position)
            if diff < GC.CLOSE_TO_SHIPYARD * self.game_map.width or GlobalFunctions(self.game).dist_to_enemy_doff(pos) < GC.CLOSE_TO_SHIPYARD * self.game_map.width:
                if d in centers:
                    centers.remove(d)
                continue

            if i < len(centers_copy) - 1:  # if not out of bounds
                # get list of centers too close
                r = self.too_close(centers_copy[i + 1:], pos)
                for t in r:
                    if t in centers:
                        centers.remove(t)  # remove those centers

        return centers


    def too_close(self, centers, position):
        ''' removes clusters that are too close to each other '''
        to_remove = []
        for d in centers:
            _, other = d
            distance = self.game_map.euclidean_distance(position, other)
            if distance < GC.CLUSTER_TOO_CLOSE * self.game_map.width:
                to_remove.append(d)
        return to_remove


    def merge_clusters(self, centers):
        ''' merges clusters using clustering in 3D where
        x: x
        y: y
        z: halite amount / 8000 '''

        logging.info("Merging clusters")
        normalizer = 1
        area = GC.CLUSTER_TOO_CLOSE * self.game_map.width
        metric = distance_metric(type_metric.USER_DEFINED, func=self.custom_dist)
        X = []  # center coordinates that are merged in an iteration
        tmp_centers = []  # to not modify the list looping through
        history = []  # contains all already merged centers
        # for each center
        for c1 in centers:
            # add the center itself
            X.append([c1[1].x, c1[1].y, c1[0] / normalizer])
            for c2 in centers:  # for other centers
                # if not merged already
                if not c2 == c1:
                    dist = self.game_map.euclidean_distance(c1[1], c2[1])
                    # if close enough for merging
                    if dist <= area:
                        X.append([c2[1].x, c2[1].y, c2[0] / normalizer])

            # get initialized centers for the algorithm
            init_centers = kmeans_plusplus_initializer(X, 1).initialize()
            median = kmedians(X, init_centers, metric=metric)
            median.process()  # do clustering
            # get clustered centers
            tmp_centers += [(x[2], self.game_map.normalize(Position(int(x[0]), int(x[1]))))
                            for x in median.get_medians() if
                            (x[2], self.game_map.normalize(Position(int(x[0]), int(x[1])))) not in tmp_centers]
            if len(X) > 1:
                history += X[1:]
            X = []

        centers = tmp_centers
        centers.sort(key=lambda x: x[0], reverse=True)  # sort by best patches
        return centers


    def custom_dist(self, p1, p2):
        ''' distance function for a clustering algorithm,
        manh dist + the absolute difference in halite amount '''
        if len(p1) < 3:
            p1 = p1[0]
            p2 = p2[0]
        manh_dist = self.game_map.calculate_distance(
            Position(p1[0], p1[1]), Position(p2[0], p2[1]))
        return manh_dist + abs(p1[2] - p2[2])


    def get_patch_data(self, x, y, center):
        # pool + 1 x pool + 1 size square inspected for data (classifier trained
        # on 5x5)
        pool = 4
        # add center info
        total_halite = 0  # total 5x5 patch halite
        cntr_cell_data = self.get_cell_data(x, y, center)
        biggest_cell = Position(x, y)
        biggest_halite = cntr_cell_data[0]
        # data must contain normalized game_size
        area_d = [round(self.game_map.width / 64, 2)] + cntr_cell_data

        for diff_x in range(-1 * int(pool / 2), int(pool / 2) + 1):
            for diff_y in range(-1 * int(pool / 2), int(pool / 2) + 1):

                new_coord_x, new_coord_y = x - diff_x, y - \
                                           diff_y  # get patch coordinates from centr
                total_halite += self.game_map[Position(new_coord_x,
                                                  new_coord_y)].halite_amount  # add to total halite
                c_data = self.get_cell_data(new_coord_x, new_coord_y, center)

                if biggest_halite < c_data[0]:  # determine cell with most halite
                    biggest_halite = c_data[0]
                    biggest_cell = Position(new_coord_x, new_coord_y)

                area_d += c_data

        return [area_d], total_halite, biggest_cell


    def get_cell_data(self, x, y, center):
        cell = self.game_map[Position(x, y)]
        # normalized data of cell: halite amount and distance to shipyard
        return [round(cell.halite_amount / 1000, 2),
                round(self.game_map.calculate_distance(cell.position, center) / self.game_map.width, 2)]


class DestinationProcessor():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton.getInstance()
        self.ship_dest = GV.ship_dest
        self.ship_state = GV.ship_state
        self.ship_path = GV.ship_path
        self.previous_state = GV.previous_state


    def find_new_destination(self, h, ship):
        ''' h: priority queue of halite factors,
                                        halite_pos: dictionary of halite factor -> patch position '''
        ship_id = ship.id
        biggest_halite, position = heappop(h)  # get biggest halite
        destination = self.game_map.normalize(position)
        not_first_dest = ship_id in self.ship_dest
        # repeat while not a viable destination, or enemies around the position or 
        # too many ships going to that dropoff area
        # or the same destination as before
        while self.bad_destination(ship, destination) or (NR_OF_PLAYERS == 4 and self.game_map[destination].enemy_neighbouring > 0):
            # if no more options, return
            if len(h) == 0:
                GlobalFunctions(self.game).state_switch(ship.id, "returning")
                return
            biggest_halite, position = heappop(h)
            destination = self.game_map.normalize(position)
        self.ship_dest[ship_id] = destination  # set the destination
        self.reassign_duplicate_dests(destination, ship_id) # deal with duplicate destinations


    def bad_destination(self, ship, destination):
        """ definition of unviable destination for ship """
        if self.game_map[destination].halite_amount == 0:
            return True
        if self.game.turn_number <= GC.SPAWN_TURN:
            return not self.dest_viable(destination, ship) or self.game_map[destination].enemy_amount >= GC.UNSAFE_AREA\
                or self.too_many_near_dropoff(ship, destination)
        else:
            return not self.dest_viable(destination, ship) or self.game_map[destination].enemy_amount >= GC.UNSAFE_AREA


    def reassign_duplicate_dests(self, destination, this_id):
        # if another ship had the same destination
        s = self.get_ship_w_destination(destination, this_id)
        if s:  # find a new destination for it
            for other in s:
                self.process_new_destination(other)


    def process_new_destination(self, ship):
        self.ship_path[ship.id] = []
        if ship.position in GlobalFunctions(self.game).get_dropoff_positions() or ship.id not in self.ship_dest:
            self.find_new_destination(self.game_map.halite_priority, ship)
        else:
            source = ship.position
            ship_h = GlobalFunctions(game).halite_priority_q(source, GC.SHIP_SCAN_AREA)
            self.find_new_destination(ship_h, ship)


    def dest_viable(self, position, ship):
        """ is a destination viable for ship, i.e. 
            if any other ship is going there, if thats the case
            then if that ship is further away than ship """
        if position in self.ship_dest.values():
            # get another ship with same destination
            inspectable_ships = self.get_ship_w_destination(position, ship.id)
            if not inspectable_ships: # shouldnt happen but for safety
                # if this ship doesnt exist for some reason
                return True

            my_dist = self.game_map.calculate_distance(position, ship.position)
            their_dist = min([self.game_map.calculate_distance(
                position, inspectable_ship.position) for inspectable_ship in inspectable_ships])

            return my_dist < their_dist # if im closer to destination, assign it to me.
        else:
            return True  # nobody has the best patch, all good


    def too_many_near_dropoff(self, ship, destination):
        # if no more options, use the same destination
        if GlobalFunctions(self.game).get_shipyard(ship.position) == GlobalFunctions(self.game).get_shipyard(destination):
            return False
        else:
            return self.prcntg_ships_returning_to_doff(GlobalFunctions(self.game).get_shipyard(destination)) > (1 / len(GlobalFunctions(self.game).get_dropoff_positions()))


    def prcntg_ships_returning_to_doff(self, d_pos):
        amount = 0
        for s in self.me.get_ships():
            eval_pos = s.position if s.id not in self.ship_dest else self.ship_dest[s.id]
            if GlobalFunctions(self.game).get_shipyard(eval_pos) == d_pos:
                amount += 1
        return amount / len(self.me.get_ships())


    def get_ship_w_destination(self, dest, this_id):
        """ gets a ship with dest, s.t. that ship is not this_id """
        other_ships = []
        if dest in self.ship_dest.values():
            for s in self.ship_dest.keys():  # get ship with the same destination
                if s != this_id and self.ship_dest[s] == dest and self.me.has_ship(s):
                    other_ships.append(self.me.get_ship(s))
        return other_ships


class MoveProcessor():

    def __init__(self, game, has_moved, command_queue):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton.getInstance()
        self.ship_obj = GV.ship_obj
        self.ship_dest = GV.ship_dest
        self.previous_state = GV.previous_state
        self.ship_path = GV.ship_path
        self.ship_state = GV.ship_state
        self.has_moved = has_moved
        self.command_queue = command_queue



    def produce_move(self, ship):
        if ship.id not in self.ship_obj:
            self.ship_obj[ship.id] = ship
        state = self.ship_state[ship.id]
        destination = self.ship_dest[ship.id]
        ''' produces move for ship '''

        if ship.halite_amount < self.game_map[ship.position].halite_amount / 10:
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

        logging.info(state)
        return mover[state](ship, destination)


    def collecting(self, ship, destination):
        return Direction.Still


    def returning(self, ship, destination):
        return self.make_returning_move(ship, self.has_moved)


    def harakiri(self, ship, destination):
        """ pre: next to or on shipyard """
        shipyard = GlobalFunctions(self.game).get_shipyard(ship.position)
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
        return Direction.Still # should never happen


    def assassinate(self, ship, destination):
        GlobalFunctions(self.game).state_switch(ship.id, self.previous_state[ship.id])
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
        elif GlobalFunctions(self.game).time_left() < 0.1:
            logging.info(f"Exploring ship standing still, {GlobalFunctions(self.game).time_left()} left")
            return Direction.Still
        elif ship.position in GlobalFunctions(self.game).get_dropoff_positions() and self.game_map.is_surrounded(ship.position): # if in a dropoff and surrounded
            # find closest neighbour
            closest_n = None
            closest_dist = None
            for n in self.game_map.get_neighbours(self.game_map[ship.position]):
                if n.ship.id in self.ship_state and self.ship_state[n.ship.id] != "returning": # only swap with returning baybes 
                    continue
                dist = self.game_map.calculate_distance(n.position, destination)
                if closest_dist is None or dist < closest_dist: # set smallest dist and neighbour
                    closest_dist = dist
                    closest_n = n
            # swap with the ship there
            if closest_n is None: # wait for exploring ships to go away
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

        # next direction occupied, recalculate
        if ship.id not in self.ship_path or not self.ship_path[ship.id]:
            self.ship_path[ship.id] = self.game_map.explore(ship, destination)
        else:
            direction = self.ship_path[ship.id][0][0]
            next_pos = ship.position.directional_offset(direction)
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
            if GlobalFunctions(self.game).time_left() < 0.3:
                logging.info("INTERIM EXPLORING STANDING STILL")
                return position
            if not path:
                return next_pos
            to_go = self.get_step(path)
            next_pos = self.game_map.normalize(next_pos.directional_offset(to_go))
        return next_pos


    @staticmethod
    def get_step(path):
        path[0][1] -= 1  # take that direction
        direction = path[0][0]
        if path[0][1] == 0:  # if no more left that direction remove it
            del path[0]
        return direction


    def make_returning_move(self, ship, command_queue):
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

                if other_ship.id not in self.ship_state or self.ship_state[other_ship.id] in ["exploring", "build", "fleet","backup"]:
                    can_move = other_ship.halite_amount >= self.game_map[
                               other_ship.position].halite_amount / constants.MOVE_COST_RATIO

                    if other_ship.id in self.ship_dest:
                        can_move = can_move and self.ship_dest[other_ship.id] != other_ship.position
                    # if other ship has enough halite to move, hasnt made a move
                    # yet, and if it would move in the ship
                    if not self.has_moved[other_ship.id] and \
                            (can_move or other_ship.position in GlobalFunctions(self.game).get_dropoff_positions()):
                        if (other_ship.id not in self.ship_path or (not self.ship_path[other_ship.id] or
                                                                    other_ship.position.directional_offset(self.ship_path[other_ship.id][0][0]) == ship.position)):
                            # move stays the same target move
                            # move other_ship to ship.position
                            # hence swapping ships
                            logging.info(f"SWAPPING {ship.id} with {other_ship.id}")
                            self.move_ship_to_position(other_ship, ship.position)
                        elif other_ship.id in self.ship_path and self.ship_path[other_ship.id] and self.ship_path[other_ship.id][0][0] == Direction.Still:
                            move = self.a_star_move(ship)
                    else: # wait until can move
                        move = Direction.Still

                elif self.ship_state[other_ship.id] in ["returning", "harakiri"]:  # suiciding or queue
                    if self.has_moved[other_ship.id] or (self.game.turn_number <= GC.CRASH_TURN and other_ship.position in GlobalFunctions(self.game).get_dropoff_positions()):
                        move = Direction.Still
                    elif Direction.Still == self.simulate_make_returning_move(other_ship, command_queue): # should never happen but just in case :D
                        move = Direction.Still
                elif self.ship_state[other_ship.id] in ["collecting", "waiting"]:  # move around these ships
                    if self.ship_state[other_ship.id] == "collecting" and \
                    self.game_map[other_ship.position].halite_amount - self.game_map[other_ship.position].halite_amount / constants.EXTRACT_RATIO <= self.game_map.HALITE_STOP:
                        move = Direction.Still
                    else:
                        move = self.a_star_move(ship)

            else:  # target position occupied by enemy ship
                move = self.a_star_move(ship)
        return move


    def simulate_make_returning_move(self, other_ship, command_queue):
        other_move = self.produce_move(other_ship)
        command_queue.append(other_ship.move(other_move))
        self.previous_position[other_ship.id] = other_ship.position
        self.game_map[other_ship.position.directional_offset(other_move)].mark_unsafe(other_ship)
        if other_move != Direction.Still and self.game_map[other_ship.position].ship == other_ship:
            self.game_map[other_ship.position].ship = None
        self.has_moved[other_ship.id] = True
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
            dest = self.interim_djikstra_dest(cell).position
        return self.exploring(ship, dest)


    def interim_djikstra_dest(self, source_cell):
        ''' finds the intermediate djikstra destination that is not occupied '''
        cell = source_cell.parent
        while cell.is_occupied:
            cell = cell.parent
            if GlobalFunctions(self.game).time_left() < 0.5:
                logging.info("INTERIM DIJKSTRA DESTINATION STANDING STILL TOO SLOW")
                return source_cell
        return cell


    def move_ship_to_position(self, ship, destination):
        ''' moves ship to destination
        precondition: destination one move away'''
        if ship.id in self.ship_path and self.ship_path[ship.id]:
            move = self.get_step(self.ship_path[ship.id])
        else:
            move = self.dir_to_dest(ship.position, destination)

        self.has_moved[ship.id] = True
        self.command_queue.append(ship.move(move))
        self.game_map[destination].mark_unsafe(ship)
        self.game_map[ship.position].ship = None


class StateMachine():
    def __init__(self, game, ship, return_percentage, prcntg_halite_left):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship = ship
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


    def state_transition(self):
        # transition
        new_state = None
        shipyard = GlobalFunctions(self.game).get_shipyard(self.ship.position)
        DP = DestinationProcessor(self.game)
        GF = GlobalFunctions(self.game)

        if self.game.turn_number >= GC.CRASH_TURN and self.game_map.calculate_distance(
                self.ship.position, shipyard) < 2:
            # if next to shipyard after crash turn, suicide
            new_state = "harakiri"

        elif self.game.turn_number >= GC.CRASH_TURN:
            # return if at crash turn
            new_state = "returning"

        elif self.ship.position in GF.get_dropoff_positions():
            DP.process_new_destination(self.ship)
            new_state = "exploring"

        # decent halite and close to enemy dropoff, return
        elif self.ship.halite_amount >= 0.5 * constants.MAX_HALITE and GF.dist_to_enemy_doff(self.ship.position) < 0.1 * self.game_map.width:
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
            GF.state_switch(self.ship.id, new_state)


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
                and GlobalFunctions(self.game).dist_to_enemy_doff(self.ship.position) >= GC.ENEMY_SHIPYARD_CLOSE * self.game_map.width:
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
                    logging.info("ASSASINATING")
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
        '''Duplicate function ATM'''
        return self.me.has_ship(ship.id) and ship.halite_amount <= self.return_percentage * 0.5 * constants.MAX_HALITE \
                and (ship.id not in self.ship_state or not (self.ship_state[ship.id] in ["waiting", "returning", "build"]))


    def exists_better_in_area(self, cntr, current, area):
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + cntr  # top left of scan area
        current_factor = self.game_map.cell_factor(cntr, self.game_map[current], self.me, NR_OF_PLAYERS)
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
        new_state = None
        inspire_multiplier = self.game_map.get_inspire_multiplier(
            self.ship.position, self.game_map[self.ship.position], self.ENABLE_BACKUP)
        cell_halite = self.game_map[self.ship.position].halite_amount * inspire_multiplier
        DP = DestinationProcessor(self.game)

        if self.ship.is_full:
            return "returning"

        elif NR_OF_PLAYERS == 4 and self.game_map.percentage_occupied >= GC.BUSY_PERCENTAGE and self.ship.halite_amount >= GC.BUSY_RETURN_AMOUNT:
            new_state = "returning"

        elif self.ship.halite_amount >= constants.MAX_HALITE * self.return_percentage and \
                not (cell_halite > GC.MEDIUM_HALITE * inspire_multiplier and not self.ship.is_full):
            # return to shipyard if enough halite
            new_state = "returning"

        elif self.ship.halite_amount < constants.MAX_HALITE * self.return_percentage * 0.9 and self.game_map[self.ship.position].halite_amount <= 100 and \
                self.better_patch_neighbouring(self.game_map[self.ship.position].halite_amount):
            # explore to best neighbour if current cell has low halite, and there is a 2x patch next to it
            self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
            new_state = "exploring"
            DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)

        elif (cell_halite < self.game_map.HALITE_STOP * inspire_multiplier and self.ship.halite_amount < constants.MAX_HALITE * self.return_percentage * .9):
            # Keep exploring if halite patch low on halite and ship doesnt have close to returning percentage
            DP.process_new_destination(self.ship)
            new_state = "exploring"

        elif cell_halite < self.game_map.HALITE_STOP * inspire_multiplier:
            # return if cell has low halite
            new_state = "returning"

        elif self.ship.halite_amount <= constants.MAX_HALITE * 0.4 and self.ENABLE_COMBAT\
                and GlobalFunctions(game).dist_to_enemy_doff(self.ship.position) >= GC.CLOSE_TO_SHIPYARD * self.game_map.width and self.game_map[self.ship.position].enemy_neighbouring:
            # if not that mch halite and not too close to assasinate and enemy is neighbouring, attempt to kill sm1 
            new_state = self.attempt_switching_assasinate()

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
            new_state = "exploring"
            self.ship_dest[self.ship.id] = n.position
            DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)

        return new_state


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
        if self.ship.position in GlobalFunctions(self.game).get_dropoff_positions():
            # explore again when back in shipyard
            return "exploring"
            DP.find_new_destination(self.game_map.halite_priority, self.ship)

        elif self.ship.position in GlobalFunctions(self.game).get_dropoff_positions():
            # explore again when back in shipyard
            DP.process_new_destination(self.ship)
            return "exploring"

        elif self.game_map.calculate_distance(self.ship.position, GlobalFunctions(self.game).get_shipyard(self.ship.position)) == 1:
            # if next to a dropoff 
            cell = self.game_map[GlobalFunctions(self.game).get_shipyard(self.ship.position)]
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
                self.ship_dest[self.ship.id] = main(self.game).bfs_unoccupied(future_dropoff_cell.position)
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            else:
                DP.process_new_destination(self.ship)
                return "exploring"

        elif self.game_map[future_dropoff_cell.position].enemy_amount >= GC.UNSAFE_AREA:
            new_state = "exploring"
            DP.process_new_destination(self.ship)

        elif distance_to_dest <= 2:
            neighbours = self.game_map.get_neighbours(self.game_map[self.ship.position])
            for n in neighbours:
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
                    DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
                    return "build"
        elif NR_OF_PLAYERS == 4 or self.game_map.width >= 56: # for 4 players and large maps 1v1
            smallest_dist = GlobalFunctions(self.game).dist_to_enemy_doff(self.ship_dest[self.ship.id])
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
            new_state = "returning"
        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied and self.me.has_ship(self.game_map[destination].ship.id):
                self.ship_dest[self.ship.id] = self.get_best_neighbour(destination).position
                DP.reassign_duplicate_dests(self.ship_dest[self.ship.id], self.ship.id)
            elif self.game_map[destination].is_occupied:  # not our ship
                return self.attempt_switching_assasinate()
        elif self.game_map[destination].enemy_amount >= GC.UNSAFE_AREA:
            DP.process_new_destination(ship)
            return "exploring"
        elif self.ENABLE_COMBAT and GlobalFunctions(self.game).dist_to_enemy_doff(self.ship.position) >= GC.CLOSE_TO_SHIPYARD * self.game_map.width:
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
            if n.enemy_amount < GC. UNSAFE_AREA and n_halite > max_halite:
                best = n
                max_halite = n_halite
        return best


class main():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        GV = GlobalVariablesSingleton.getInstance()

        self.ENABLE_BACKUP = GV.ENABLE_BACKUP
        self.ENABLE_COMBAT = GV.ENABLE_COMBAT
        self.ship_state = GV.ship_state
        self.ship_path = GV.ship_path
        self.ship_dest = GV.ship_dest
        self.previous_position = GV.previous_position
        self.previous_state = GV.previous_state
        self.fleet_leader = GV.fleet_leader
        self.ship_obj = GV.ship_obj

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
            swapped = set() # for swapping destinations
            self.clear_dictionaries()  # of crashed or transformed ships
            command_queue = []

            if self.game.turn_number == 1:
                self.game_map.HALITE_STOP = GC.INITIAL_HALITE_STOP
                self.game_map.c = [GC.A, GC.B, GC.C, GC.D, GC.E, GC.F]  # set the heuristic constants
            for s in self.me.get_ships():
                if s.id not in self.ship_state:
                    self.ship_state[s.id] = "exploring"

            GlobalVariablesSingleton.getInstance().ENABLE_COMBAT = not self.have_less_ships(0.8) and NR_OF_PLAYERS == 2
            GlobalVariablesSingleton.getInstance().turn_start = time.time()
            self.ENABLE_COMBAT = GlobalVariablesSingleton.getInstance().ENABLE_COMBAT

            enable_inspire = not self.have_less_ships(0.8)
            GlobalVariablesSingleton.getInstance().ENABLE_BACKUP = self.ENABLE_COMBAT
            self.ENABLE_BACKUP = GlobalVariablesSingleton.getInstance().ENABLE_BACKUP
            # initialize shipyard halite, inspiring stuff and other
            self.game_map.init_map(self.me, list(self.game.players.values()), enable_inspire, self.ENABLE_BACKUP)
            if self.game.turn_number == 1:
                TOTAL_MAP_HALITE = self.game_map.total_halite

            prcntg_halite_left = self.game_map.total_halite / TOTAL_MAP_HALITE
            # if clusters_determined and not cluster_centers:
            if self.game.turn_number >= GC.SPAWN_TURN:
                self.game_map.HALITE_STOP = prcntg_halite_left * GC.INITIAL_HALITE_STOP

            if self.crashed_ship_positions and self.game.turn_number < GC.CRASH_TURN and self.ENABLE_BACKUP:
                self.process_backup_sending()

            # Dijkstra the graph based on all dropoffs
            self.game_map.create_graph(GlobalFunctions(self.game).get_dropoff_positions())

            if self.game.turn_number == GC.DETERMINE_CLUSTER_TURN:
                self.clusters_determined = True
                self.cluster_centers = ClusterProcessor(game).clusters_with_classifier()

            if self.game.turn_number == GC.CRASH_SELECTION_TURN:
                GC.CRASH_TURN = self.select_crash_turn()

            if prcntg_halite_left > 0.2:
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

                # logging.info("SHIP {}, STATE {}, DESTINATION {}".format(
                #     ship.id, self.ship_state[ship.id], self.ship_dest[ship.id]))

                # transition
                SM = StateMachine(self.game, ship, self.return_percentage, prcntg_halite_left)
                SM.state_transition()


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
                        logging.info(move)
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
            if not dropoff_built and 2.5 * (self.max_enemy_ships() + 1) > len(self.me.get_ships()) and self.game.turn_number <= GC.SPAWN_TURN \
                    and self.me.halite_amount >= constants.SHIP_COST and prcntg_halite_left > (1 - 0.65) and \
                    not (self.game_map[self.me.shipyard].is_occupied or surrounded_shipyard or "waiting" in self.ship_state.values()):
                if not ("build" in self.ship_state.values() and self.me.halite_amount <= (constants.SHIP_COST + constants.DROPOFF_COST)):
                    command_queue.append(self.me.shipyard.spawn())
            # Send your moves back to the game environment, ending this turn.
            self.game.end_turn(command_queue)


    def max_enemy_ships(self):
        if NR_OF_PLAYERS not in [2, 4]:  # for testing solo games
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
                dropoff_pos = self.bfs_unoccupied(dropoff_pos)
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


    def bfs_unoccupied(self, position):
        # bfs for closest cell
        Q = deque([])
        visited = set()
        Q.append(position)
        while Q:
            cur = Q.popleft()
            visited.add(cur)
            if not (self.game_map[cur].is_occupied and self.game_map[cur].has_structure):
                return cur
            for neighbour in self.game_map.get_neighbours(self.game_map[cur]):
                if not neighbour.position in visited:
                    Q.append(neighbour.position)
                    visited.add(neighbour.position)


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


    def is_savior(self, ship):
        return self.me.has_ship(ship.id) and ship.halite_amount <= self.return_percentage * 0.5 * constants.MAX_HALITE \
                and (ship.id not in self.ship_state or not (self.ship_state[ship.id] in ["waiting", "returning", "build"]))


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
                self.send_ships(crashed_pos, 2, "backup", is_savior)


class GlobalVariablesSingleton():
    # Here the instance will be stored
    __instance = None

    @staticmethod
    def getInstance():
        ''' static access method '''
        if GlobalVariablesSingleton.__instance == None:
            GlobalVariablesSingleton()
        return GlobalVariablesSingleton.__instance

    def __init__(self):
        ''' virtually private constructor. '''
        if GlobalVariablesSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            GlobalVariablesSingleton.__instance = self
            self.ENABLE_BACKUP = True
            self.ENABLE_COMBAT = True
            self.ship_state = {} # ship.id -> ship state
            self.ship_path = {} # ship.id -> directional path to ship_dest[ship.id]
            self.ship_dest = {} # ship.id -> destination
            self.previous_position = {} # ship.id-> previous pos
            self.previous_state = {} # ship.id -> previous state
            self.fleet_leader = {}
            self.ship_obj = {} # ship.id to ship obj for processing crashed ship stuff
            self.turn_start = 0 # for timing


backuped_dropoffs = []
maingameloop = main(game)
maingameloop.mainloop()