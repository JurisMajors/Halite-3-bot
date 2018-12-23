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

ship_state = {}  # ship.id -> ship state
ship_path = {}  # ship.id -> directional path to ship_dest[ship.id]
ship_dest = {}  # ship.id -> destination
previous_position = {}  # ship.id-> previous pos
previous_state = {}  # ship.id -> previous state
fleet_leader = {}
ship_obj = {}  # ship.id to ship obj for processing crashed ship stuff
crashed_positions = []  # heap of (-1 * halite, crashed position )
crashed_ships = []
turn_start = 0  # for timing


game.ready("MLP")
NR_OF_PLAYERS = len(game.players.keys())

SAVIOR_FLEET_SIZE = 0.1 if NR_OF_PLAYERS == 2 else 0.05
ENABLE_COMBAT = True

class GlobalFunctions():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me


    def halite_priority_q(self, pos, area):
        # h_amount <= 0 to run minheap as maxheap
        h = []  # stores halite amount * -1 with its position in a minheap
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + pos  # top left of scan area
        for y in range(area):
            for x in range(area):
                p = Position((top_left.x + x) % self.game_map.width,
                             (top_left.y + y) % self.game_map.height)  # position of patch
                if not (p == pos or p in self.get_dropoff_positions()):
                    cell = self.game_map[p]
                    if 0 < cell.halite_amount <= self.game_map.HALITE_STOP:
                        ratio = cell.halite_amount / \
                                (2 * self.game_map.calculate_distance(p, pos))
                        heappush(h, (-1 * ratio, p))
                    elif cell.halite_amount > 0:
                        factor = self.game_map.cell_factor(pos, cell, self.me)
                        # add negative halite amounts so that would act as maxheap
                        heappush(h, (factor, p))
        return h


    def dist_to_enemy_doff(self, pos):
        if NR_OF_PLAYERS < 2:
            return 1000
        return min([self.game_map.euclidean_distance(pos, d) for d in self.get_enemy_dropoff_positions()])


    def state_switch(self, ship_id, new_state, previous_state, ship_state, ship_path):
        if ship_id not in previous_state:
            previous_state[ship_id] = "exploring"
        if ship_id not in ship_state:
            ship_state[ship_id] = previous_state[ship_id]
        if not new_state == "exploring":  # reset path to empty list
            ship_path[ship_id] = []

        previous_state[ship_id] = ship_state[ship_id]
        ship_state[ship_id] = new_state


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
    def time_left(turn_start):
        return 2 - (time.time() - turn_start)


    def amount_of_enemies(self, pos, area):
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + pos  # top left of scan area
        amount = 0
        for y in range(area):
            for x in range(area):
                p = Position((top_left.x + x) % self.game_map.width,
                             (top_left.y + y) % self.game_map.height)  # position of patch
                if self.game_map[p].is_occupied and not self.me.has_ship(self.game_map[p].ship.id):
                    amount += 1
        return amount


class ClusterProcessor():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me


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
                if prediction == 1:  # if should be dropoff
                    # add node with most halite to centers
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
            shipyard_distance = self.game_map.euclidean_distance(
                self.me.shipyard.position, other)
            if distance < GC.CLUSTER_TOO_CLOSE * self.game_map.width or \
                    shipyard_distance < GC.CLOSE_TO_SHIPYARD * self.game_map.width:
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
        logging.info("Done merging")
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

    def __init__(self, game, ship_dest, ship_state):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship_dest = ship_dest
        self.ship_state = ship_state


    def find_new_destination(self, h, ship):
        ''' h: priority queue of halite factors,
                                        halite_pos: dictionary of halite factor -> patch position '''
        ship_id = ship.id
        biggest_halite, position = heappop(h)  # get biggest halite
        destination = self.game_map.normalize(position)
        not_first_dest = ship_id in self.ship_dest
        # get biggest halite while its a position no other ship goes to
        while not self.dest_viable(destination, ship) or GlobalFunctions(self.game).amount_of_enemies(destination, 4) >= 4 \
                or self.too_many_near_dropoff(ship, destination) \
                or (not_first_dest and destination == self.ship_dest[ship_id]):
            if len(h) == 0:
                logging.info("ran out of options")
                return
            biggest_halite, position = heappop(h)
            destination = self.game_map.normalize(position)

        self.ship_dest[ship_id] = destination  # set the destination
        # if another ship had the same destination
        s = self.get_ship_w_destination(destination, ship_id)
        if s is not None:  # find a new destination for it
            self.process_new_destination(s)


    def process_new_destination(self, ship):
        ship_path[ship.id] = []
        if 0 < self.game_map.calculate_distance(ship.position, self.ship_dest[ship.id]) <= GC.SHIP_SCAN_AREA:
            source = ship.position
        else:
            source = self.ship_dest[ship.id]

        ship_h = GlobalFunctions(self.game).halite_priority_q(source, GC.SHIP_SCAN_AREA)
        self.find_new_destination(ship_h, ship)


    def dest_viable(self, position, ship):
        if position in self.ship_dest.values():
            inspectable_ship = self.get_ship_w_destination(position, ship.id)
            if inspectable_ship is None:
                # if this ship doesnt exist for some reason
                return True

            my_dist = self.game_map.calculate_distance(position, ship.position)
            their_dist = self.game_map.calculate_distance(
                position, inspectable_ship.position)

            return my_dist < their_dist
        else:
            return True  # nobody has the best patch, all good


    def too_many_near_dropoff(self, ship, destination):
        if self.game_map[ship.position].dijkstra_dest == self.game_map[destination].dijkstra_dest:
            return False
        else:
            return self.prcntg_ships_returning_to_doff(self.game_map[destination].dijkstra_dest) > (1 / len(GlobalFunctions(self.game).get_dropoff_positions()))


    def prcntg_ships_returning_to_doff(self, d_pos):
        amount = 0
        for s in self.me.get_ships():
            if self.game_map[s.position].dijkstra_dest == d_pos:
                amount += 1
        return amount / len(self.me.get_ships())


    def get_ship_w_destination(self, dest, this_id):
        if dest in self.ship_dest.values():
            for s in self.ship_dest.keys():  # get ship with the same destination
                if not s == this_id and self.ship_state[s] == "exploring" and self.ship_dest[s] == dest and self.me.has_ship(s):
                    return self.me.get_ship(s)
        return None


class MoveProcessor():

    def __init__(self, game, ship_obj, ship_dest, previous_state, ship_path, ship_state, turn_start, has_moved, command_queue):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship_obj = ship_obj
        self.ship_dest = ship_dest
        self.previous_state = previous_state
        self.ship_path = ship_path
        self.ship_state = ship_state
        self.turn_start = turn_start
        self.has_moved = has_moved
        self.command_queue = command_queue



    def produce_move(self, ship):
        if ship.id not in ship_obj:
            self.ship_obj[ship.id] = ship
        state = ship_state[ship.id]
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

        return mover[state](ship, destination)


    def collecting(self, ship, destination):
        return Direction.Still


    def returning(self, ship, destination):
        return self.make_returning_move(ship, self.has_moved)


    def harakiri(self, ship, destination):
        shipyard = self.game_map[ship.position].dijkstra_dest
        ship_pos = self.game_map.normalize(ship.position)
        if ship.position == shipyard:  # if at shipyard
            return Direction.Still  # let other ships crash in to you
        else:  # otherwise move to the shipyard
            target_dir = self.game_map.get_target_direction(
                ship.position, shipyard)
            return target_dir[0] if target_dir[0] is not None else target_dir[1]


    def assassinate(self, ship, destination):
        GlobalFunctions(self.game).state_switch(ship.id, self.previous_state[ship.id], self.previous_state, self.ship_state, self.ship_path)
        if self.game_map.calculate_distance(ship.position, destination) == 1:
            target_direction = self.game_map.get_target_direction(
                ship.position, destination)
            return target_direction[0] if target_direction[0] is not None else target_direction[1]
        else:
            return self.exploring(ship, destination)


    def exploring(self, ship, destination):
        # next direction occupied, recalculate
        if ship.id not in self.ship_path or not self.ship_path[ship.id]:
            self.ship_path[ship.id] = self.game_map.explore(ship, destination)
        else:
            direction = self.ship_path[ship.id][0][0]
            if self.game_map[ship.position.directional_offset(direction)].is_occupied and not direction == Direction.Still:
                if self.game_map.calculate_distance(destination, ship.position) > 10:
                    new_dest = self.interim_exploring_dest(
                        ship.position, self.ship_path[ship.id])
                    # use intermediate unoccpied position instead of actual
                    # destination
                    self.ship_path[ship.id] = self.game_map.explore(
                        ship, new_dest) + self.ship_path[ship.id]
                else:
                    self.ship_path[ship.id] = self.game_map.explore(ship, destination)
        # move in calculated direction
        return self.get_step(self.ship_path[ship.id])


    def interim_exploring_dest(self, position, path):
        ''' finds intermediate destination from a direction path that is not occupied '''
        to_go = self.get_step(path)
        next_pos = self.game_map.normalize(position.directional_offset(to_go))
        while self.game_map[next_pos].is_occupied:
            if GlobalFunctions(self.game).time_left(self.turn_start) < 0.3:
                logging.info("STANDING STILL")
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


    def make_returning_move(self, ship, has_moved):
        """
        Makes a returning move based on Dijkstras and other ship positions.
        """
        if self.ship_path[ship.id]:
            direction = self.get_step(self.ship_path[ship.id])
            to_go = ship.position.directional_offset(direction)
            if direction == Direction.Still or not self.game_map[to_go].is_occupied:
                return direction

        if self.game.turn_number >= GC.CRASH_TURN:
            ''' THIS DOESNT WORK FOR SOME REASON,
            NO BOTS GET IN THIS STAGE TO CHANGE SUICIDE DROPOFFS
            THE IDEA IS TO BALANCE OUT THE ENDING SO THAT TOO MANY DONT GO TO SAME
            DROPOFF '''
            if self.should_better_dropoff(ship):
                other_dropoff = GlobalFunctions(self.game).better_dropoff_pos(ship)
                # if not the same distance
                if not other_dropoff == self.game_map[ship.position].dijkstra_dest:
                    return self.a_star_move(ship)
        # Get the cell and direction we want to go to from dijkstra
        target_pos, move = self.get_dijkstra_move(ship)
        # Target is occupied
        if self.game_map[target_pos].is_occupied:
            other_ship = self.game_map[target_pos].ship
            # target position occupied by own ship
            if self.me.has_ship(other_ship.id):

                if other_ship.id not in ship_state or ship_state[other_ship.id] in ["exploring", "build", "fleet",
                                                                                    "backup"]:
                    # if other ship has enough halite and hasnt made a move yet:
                    if not has_moved[other_ship.id] and \
                            (other_ship.halite_amount > self.game_map[
                                other_ship.position].halite_amount / 10 or other_ship.position in GlobalFunctions(self.game).get_dropoff_positions()):
                        # move stays the same target move
                        # move other_ship to ship.destination
                        # hence swapping ships
                        self.move_ship_to_position(other_ship, ship.position)
                    else:
                        move = self.a_star_move(ship)

                elif self.ship_state[other_ship.id] in ["returning", "harakiri"]:
                    move = Direction.Still
                elif self.ship_state[other_ship.id] in ["collecting", "waiting"]:
                    move = self.a_star_move(ship)

            else:  # target position occupied by enemy ship
                move = self.a_star_move(ship)

        return move


    def better_dropoff_pos(self, ship):
        ''' find the better dropoff position
        precondition: should_better_dropoff(ship) and
        game.turn_number >= CRASH_TURN
        '''
        current_dest = self.game_map[ship.position].dijkstra_dest
        current_distance = None
        new_dest = None
        for d in GlobalFunctions(self.game).get_dropoff_positions():  # for all dropoffs
            dist = self.game_map.calculate_distance(d, ship.position)
            # if nto the same dropoff and still can reach it by the end of game
            if not d == current_dest and dist < (constants.MAX_TURNS - self.game.turn_number):
                # check if distance is better than other dropoffs
                if current_distance is None or current_distance < dist:
                    current_distance = dist
                    new_dest = d
        # return best option
        if new_dest is None:
            return current_dest
        return new_dest


    def should_better_dropoff(self, ship):
        ''' determines whether there are too many ships going to the same dropoff as ship'''
        current = self.game_map[ship.position]
        ratio = 1 / len(GlobalFunctions(self.game).get_dropoff_positions())
        if len(self.me.get_dropoffs()) >= 1:  # if there are multiple dropoffs then check for other options
            my_dest = current.dijkstra_dest
            amount = 0  # ship amount going to the same dropoff
            for other in self.me.get_ships():
                other_cell = self.game_map[other.position]
                other_dest = other_cell.dijkstra_dest
                # count other ships that are returning, with the same destinationa
                # and are within 80% of ships dijkstra distance
                if other.id in self.ship_state and self.ship_state[other.id] == "returning" and \
                        other_cell.dijkstra_dest == my_dest and other_cell.dijkstra_distance < 0.7 * current.dijkstra_distance:
                    amount += 1
                # if more than 30 percent of the ships are very close to the
                # shipyard
                if amount > ratio * len(self.me.get_ships()):
                    return True
        return False


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
            dest = self.interim_djikstra_dest(
                cell).position if d_to_dijkstra_dest > 10 else cell.dijkstra_dest  
        return self.exploring(ship, dest)


    def interim_djikstra_dest(self, source_cell):
        ''' finds the intermediate djikstra destination that is not occupied '''
        cell = source_cell.parent
        while cell.is_occupied:
            cell = cell.parent
            if GlobalFunctions(self.game).time_left(self.turn_start) < 0.5:
                logging.info("STANDING STILL TOO SLOW")
                return source_cell
        return cell


    def move_ship_to_position(self, ship, destination):
        ''' moves ship to destination
        precondition: destination one move away'''
        normalized_dest = self.game_map.normalize(destination)
        for d in Direction.get_all_cardinals():
            new_pos = self.game_map.normalize(ship.position.directional_offset(d))
            if new_pos == normalized_dest:
                move = d
                break

        self.has_moved[ship.id] = True
        self.command_queue.append(ship.move(move))
        self.game_map[destination].mark_unsafe(ship)
        self.game_map[ship.position].ship = None


class StateMachine():
    def __init__(self, game, ship, ship_path, ship_state, ship_dest, fleet_leader, return_percentage, previous_state):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship = ship
        self.ship_path = ship_path
        self.ship_state = ship_state
        self.ship_dest = ship_dest
        self.fleet_leader = fleet_leader
        self.return_percentage = return_percentage
        self.previous_state = previous_state


    def state_transition(self):
        # transition
        new_state = None
        shipyard = self.game_map[self.ship.position].dijkstra_dest

        new_state_switch = {
            "exploring": self.exploring_transition(),
            "collecting": self.collecting_transition(),
            "returning": self.returning_transition(),
            "fleet": self.fleet_transition(),
            "builder": self.builder_transition(),
            "waiting": self.waiting_transition(),
            "backup": self.backup_transition(),
        }

        if self.game.turn_number >= GC.CRASH_TURN and self.game_map.calculate_distance(
                self.ship.position, shipyard) < 2:
            # if next to shipyard after crash turn, suicide
            self.ship_path[self.ship.id] = []
            new_state = "harakiri"

        elif game.turn_number >= GC.CRASH_TURN:
            # return if at crash turn
            self.ship_path[self.ship.id] = []
            new_state = "returning"

        elif self.ship.position in GlobalFunctions(self.game).get_dropoff_positions():
            self.ship_path[self.ship.id] = []
            DestinationProcessor(self.game, self.ship_dest, self.ship_state).find_new_destination(
                self.game_map.halite_priority, self.ship)
            new_state = "exploring"

        else: new_state = new_state_switch.get(self.ship_state[self.ship.id], None)

        if new_state is not None:
            GlobalFunctions(self.game).state_switch(self.ship.id, new_state, self.previous_state, self.ship_state, self.ship_path)


    """Exploring_transition"""
    def exploring_transition(self):
        distance_to_dest = self.game_map.calculate_distance(self.ship.position, self.ship_dest[self.ship.id])
        euclid_to_dest = self.game_map.euclidean_distance(self.ship.position, self.ship_dest[self.ship.id])
        if self.ship.position == self.ship_dest[self.ship.id]:
            # collect if reached destination or on medium sized patch
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif GlobalFunctions(self.game).amount_of_enemies(self.ship.position, 4) >= 2:
            # for inspiring
            self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position

        elif euclid_to_dest <= 5 and self.exists_better_in_area(self.ship.position, self.ship_dest[self.ship.id], 4):
            ship_h = GlobalFunctions(self.game).halite_priority_q(self.ship.position, GC.SHIP_SCAN_AREA)
            DestinationProcessor(self.game, self.ship_dest, self.ship_state).find_new_destination(ship_h, self.ship)
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


    def exists_better_in_area(self, cntr, current, area):
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + cntr  # top left of scan area
        current_factor = self.game_map.cell_factor(cntr, self.game_map[current], self.me)
        for y in range(area):
            for x in range(area):
                p = Position((top_left.x + x) % self.game_map.width,
                             (top_left.y + y) % self.game_map.height)  # position of patch
                cell = self.game_map[p]
                if cell.halite_amount >= self.game_map.HALITE_STOP:
                    other_factor = self.game_map.cell_factor(cntr, cell, self.me)
                    if not cell.is_occupied and other_factor < current_factor:
                        return True
        return False


    def collecting_transition(self):
        new_state = None
        inspire_multiplier = 3 if self.game_map[self.ship.position].inspired else 1
        cell_halite = self.game_map[self.ship.position].halite_amount * inspire_multiplier
        if self.ship.is_full:
            return "returning"
        elif self.game_map.percentage_occupied >= GC.BUSY_PERCENTAGE and self.ship.halite_amount >= GC.BUSY_RETURN_AMOUNT:
            new_state = "returning"
        elif self.ship.halite_amount >= constants.MAX_HALITE * (self.return_percentage * 0.8) \
                and self.better_patch_neighbouring(GC.MEDIUM_HALITE):
            # if collecting and ship is half full but next to it there is a really
            # good patch, explore to that patch
            neighbour = self.get_best_neighbour(self.ship.position)
            if neighbour.position == self.ship.position:
                new_state = "returning"
            else:
                self.ship_dest[self.ship.id] = neighbour.position

                for sh in self.me.get_ships():
                    # if somebody else going there recalc the destination
                    if not sh.id == self.ship.id and sh.id in self.ship_dest and self.ship_dest[sh.id] == neighbour.position:
                        DestinationProcessor(self.game, self.ship_dest, self.ship_state).process_new_destination(sh)

                new_state = "exploring"

        elif self.ship.halite_amount >= constants.MAX_HALITE * self.return_percentage and \
                not (cell_halite * inspire_multiplier > GC.MEDIUM_HALITE and not self.ship.is_full):
            # return to shipyard if enough halite
            new_state = "returning"

        elif cell_halite < self.game_map.HALITE_STOP * inspire_multiplier:
            # Keep exploring if current halite patch is empty

            DestinationProcessor(self.game, self.ship_dest, self.ship_state).process_new_destination(self.ship)
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


    def better_patch_neighbouring(self, big_diff):
        ''' returns true if there is a lot better patch right next to it'''
        current = self.game_map[self.ship.position]
        neighbours = self.game_map.get_neighbours(current)
        current_h = current.halite_amount

        if current.inspired:
            current_h *= 3
        for n in neighbours:
            neighbour_h = n.halite_amount
            if n.inspired:
                neighbour_h *= 3
            if not n.is_occupied and neighbour_h >= current_h + big_diff:
                return True

        return False


    def returning_transition(self):
        if self.ship.position in GlobalFunctions(self.game).get_dropoff_positions():
            # explore again when back in shipyard
            return "exploring"
            DestinationProcessor(self.game, self.ship_dest, self.ship_state).find_new_destination(
                self.game_map.halite_priority, self.ship)
        elif self.game_map.calculate_distance(self.ship.position, self.game_map[self.ship.position].dijkstra_dest) == 1:
            # if next to a dropoff
            cell = self.game_map[self.game_map[self.ship.position].dijkstra_dest]
            if cell.is_occupied and not self.me.has_ship(cell.ship.id) and "harakiri" not in self.ship_state.values():
                return "harakiri"
        return None


    def fleet_transition(self):
        destination = self.ship_dest[self.ship.id]
        if self.ship.position == destination:  # if arrived
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied:
                self.ship_dest[self.ship.id] = self.get_best_neighbour(destination).position

        elif self.ship.id in self.fleet_leader:
            leader = self.fleet_leader[self.ship.id]
            if self.me.has_ship(leader.id) and self.ship_state[leader.id] not in ["waiting", "build"]:
                DestinationProcessor(self.game, self.ship_dest, self.ship_state).process_new_destination(self.ship)
                return "exploring"
        return None


    def builder_transition(self):
        # if someone already built dropoff there before us
        future_dropoff_cell = self.game_map[self.ship_dest[self.ship.id]]
        distance_to_dest = self.game_map.euclidean_distance(self.ship.position, self.ship_dest[self.ship.id])

        if future_dropoff_cell.has_structure:
            if distance_to_dest <= GC.CLOSE_TO_SHIPYARD * self.game_map.width:
                self.ship_dest[self.ship.id] = self.bfs_unoccupied(future_dropoff_cell.position)
            else:
                DestinationProcessor(self.game, self.ship_dest, self.ship_state).process_new_destination(self.ship)
                return "exploring"

        elif GlobalFunctions(self.game).amount_of_enemies(future_dropoff_cell.position, 4) >= 4:
            return "exploring"

        elif distance_to_dest <= 2:
            neighbours = self.game_map.get_neighbours(self.game_map[self.ship.position])
            for n in neighbours:
                if n.is_occupied and not self.me.has_ship(n.ship.id):
                    self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
                    return "build"

        elif len(self.game.players.keys()) >= 2:
            smallest_dist = GlobalFunctions(self.game).dist_to_enemy_doff(self.ship_dest[self.ship.id])
            if smallest_dist <= (self.game_map.width * GC.ENEMY_SHIPYARD_CLOSE + 1):
                DestinationProcessor(self.game, self.ship_dest, self.ship_state).process_new_destination(self.ship)
                return "exploring"
        return None


    def waiting_transition(self):
        neighbours = self.game_map.get_neighbours(self.game_map[self.ship.position])
        for n in neighbours:
            if n.is_occupied and not self.me.has_ship(n.ship.id):
                self.ship_dest[self.ship.id] = self.get_best_neighbour(self.ship.position).position
                return "build"
        return None


    def backup_transition(self):
        destination = self.ship_dest[self.ship.id]
        if self.ship.position == destination:  # if arrived
            self.ship_path[self.ship.id] = []
            return "collecting"

        elif self.game_map.calculate_distance(self.ship.position, destination) == 1:
            if self.game_map[destination].is_occupied:
                self.ship_dest[self.ship.id] = self.get_best_neighbour(destination).position

        elif GlobalFunctions(self.game).amount_of_enemies(destination, 4) >= 4:
            DestinationProcessor(self.game, self.ship_dest, self.ship_state).process_new_destination(self.ship)
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


    def get_best_neighbour(self, position):
        ''' gets best neighbour at the ships positioņ
        returns a cell'''
        current = self.game_map[position]
        neighbours = self.game_map.get_neighbours(current)
        if current.inspired:
            max_halite = 3 * current.halite_amount
        else:
            max_halite = current.halite_amount
        best = current
        for n in neighbours:
            n_halite = n.halite_amount
            if n.inspired:
                n_halite *= 3
            if not n.is_occupied and n_halite > max_halite:
                best = n
                max_halite = n_halite
        return best


class main():

    def __init__(self, game, ship_state, ship_path, ship_dest, previous_position, previous_state, fleet_leader, ship_obj, crashed_positions, crashed_ships, turn_start):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.ship_state = ship_state
        self.ship_path = ship_path
        self.ship_dest = ship_dest
        self.previous_position = previous_position
        self.previous_state = previous_state
        self.fleet_leader = fleet_leader
        self.ship_obj = ship_obj
        self.crashed_positions = crashed_positions
        self.crashed_ships = crashed_ships
        self.turn_start = turn_start
        self.cluster_centers = []
        self.clusters_determined = False


    def mainloop(self):
        while True:
            self.game.update_frame()
            self.game_map = self.game.game_map
            self.em = self.game.me

            if self.game.turn_number == 1:
                self.game_map.HALITE_STOP = GC.INITIAL_HALITE_STOP
                self.game_map.c = [GC.A, GC.B, GC.C, GC.D, GC.E, GC.F]  # set the heuristic constants
            for s in self.me.get_ships():
                if s.id not in self.ship_state:
                    self.ship_state[s.id] = "exploring"

            ENABLE_COMBAT = not self.have_less_ships(0.8) and NR_OF_PLAYERS == 2

            self.turn_start = time.time()
            logging.info(self.turn_start)
            # initialize shipyard halite, inspiring stuff and other
            self.game_map.init_map(self.me)
            if self.game.turn_number == 1:
                TOTAL_MAP_HALITE = self.game_map.total_halite

            prcntg_halite_left = self.game_map.total_halite / TOTAL_MAP_HALITE
            self.game_map.HALITE_STOP = prcntg_halite_left * GC.INITIAL_HALITE_STOP

            if len(self.crashed_ships) > 0 and not self.game.turn_number >= GC.CRASH_TURN and ENABLE_COMBAT:
                to_remove = []
                for pos in self.crashed_ships:
                    self.add_crashed_position(pos)
                    to_remove.append(pos)
                for s in to_remove:
                    if s in self.crashed_ships:
                        self.crashed_ships.remove(s)

                if len(self.crashed_positions) > 0:
                    hal, crashed_pos = heappop(self.crashed_positions)
                    nr_enemies = GlobalFunctions(self.game).amount_of_enemies(crashed_pos, 4)
                    if 6 >= nr_enemies and not self.have_less_ships(0.8):
                        self.send_ships(crashed_pos, int(
                            ceil(SAVIOR_FLEET_SIZE * len(self.me.get_ships()))), "backup", self.is_savior)

            # Dijkstra the graph based on all dropoffs
            self.game_map.create_graph(GlobalFunctions(self.game).get_dropoff_positions())
            if self.game.turn_number == GC.DETERMINE_CLUSTER_TURN:
                self.clusters_determined = True
                self.cluster_centers = ClusterProcessor(self.game).clusters_with_classifier()

            if self.game.turn_number == GC.CRASH_SELECTION_TURN:
                GC.CRASH_TURN = self.select_crash_turn()

            return_percentage = GC.BIG_PERCENTAGE if self.game.turn_number < GC.PERCENTAGE_SWITCH else GC.SMALL_PERCENTAGE
            command_queue = []

            if self.should_build(self.clusters_determined, self.cluster_centers):
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
                if GlobalFunctions(self.game).time_left(self.turn_start) < 0.3:
                    logging.info("STANDING STILL TOO SLOW")
                    command_queue.append(ship.stay_still())
                    ship_state[ship.id] = "collecting"
                    continue
                if ship.id not in self.previous_position:  # if new ship the
                    self.previous_position[ship.id] = self.me.shipyard.position

                # setup state
                # if ship hasnt received a destination yet
                if ship.id not in self.ship_dest or not ship.id in self.ship_state:
                    DestinationProcessor(self.game, self.ship_dest, self.ship_state).find_new_destination(
                        self.game_map.halite_priority, ship)
                    self.previous_state[ship.id] = "exploring"
                    self.ship_state[ship.id] = "exploring"  # explore

                # logging.info("SHIP {}, STATE {}, DESTINATION {}".format(
                #     ship.id, self.ship_state[ship.id], self.ship_dest[ship.id]))

                # transition
                StateMachine(self.game, ship, self.ship_path, self.ship_state, self.ship_dest, self.fleet_leader, return_percentage, self.previous_state).state_transition()


                # if ship is dropoff builder
                if self.is_builder(ship):
                    # if enough halite and havent built a dropoff this turn
                    if (ship.halite_amount + self.me.halite_amount) >= constants.DROPOFF_COST and not dropoff_built:
                        command_queue.append(ship.make_dropoff())
                        self.game_map.HALITE_STOP = GC.INITIAL_HALITE_STOP
                        dropoff_built = True
                    else:  # cant build
                        self.ship_state[ship.id] = "waiting"  # wait in the position
                        self.game_map[ship.position].mark_unsafe(ship)
                        command_queue.append(ship.move(Direction.Still))
                else:  # not associated with building a dropoff, so move regularly
                    MP = MoveProcessor(self.game, self.ship_obj, self.ship_dest, self.previous_state, self.ship_path, self.ship_state, self.turn_start, self.has_moved, command_queue)
                    move = MP.produce_move(ship)
                    command_queue.append(ship.move(move))
                    self.previous_position[ship.id] = ship.position
                    self.game_map[ship.position.directional_offset(move)].mark_unsafe(ship)
                    if move != Direction.Still and self.game_map[ship.position].ship == ship:
                        self.game_map[ship.position].ship = None

                self.clear_dictionaries()  # of crashed or transformed ships


                # This ship has made a move
                self.has_moved[ship.id] = True

            surrounded_shipyard = self.game_map.is_surrounded(self.me.shipyard.position)
            logging.info(GlobalFunctions(self.game).time_left(self.turn_start))
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
            GlobalFunctions(self.game).state_switch(closest_ship.id, "build", self.previous_state, self.ship_state, self.ship_path)  # will build dropoff
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

            GlobalFunctions(self.game).state_switch(fleet_ship.id, new_state, self.previous_state, self.ship_state, self.ship_path)
            DestinationProcessor(self.game, self.ship_dest, self.ship_state).find_new_destination(h, fleet_ship)


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


    def should_build(self, clusters_determined, cluster_centers):
        # if clusters determined, more than 13 ships, we have clusters and nobody
        # is building at this turn (in order to not build too many)
        return clusters_determined and len(self.me.get_ships()) > (len(self.me.get_dropoffs()) + 1) * GC.FLEET_SIZE and cluster_centers \
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
        return self.me.has_ship(ship.id) and ship.halite_amount < 0.5 * constants.MAX_HALITE \
               and (ship.id not in self.ship_state or not (self.ship_state[ship.id] in ["fleet", "waiting", "returning", "build"]))


    def select_crash_turn(self):
        '''selects turn when to crash'''
        distance = 0
        for ship in self.me.get_ships():
            shipyard = self.game_map[ship.position].dijkstra_dest  # its shipyard position
            d = self.game_map.calculate_distance(shipyard, ship.position)
            if d > distance:  # get maximum distance away of shipyard
                distance = d
        final_crash_turn = constants.MAX_TURNS - distance - 5
        # set the crash turn to be turn s.t. all ships make it
        return max(final_crash_turn, GC.CRASH_SELECTION_TURN)


    def ship_priority_q(self):
        ships = []  # ship priority queue
        has_moved = {}
        for s in self.me.get_ships():
            self.ship_obj[s.id] = s
            has_moved[s.id] = False
            if s.id in ship_state:
                # get ships shipyard
                shipyard = self.game_map[s.position].dijkstra_dest
                # importance, the lower the number, bigger importance
                if self.ship_state[s.id] in ["returning", "harikiri"]:
                    importance = -1 * self.game_map[
                        s.position].dijkstra_distance * self.game_map.width
                elif self.ship_state[s.id] in ["exploring", "build", "backup"]:
                    importance = self.game_map.calculate_distance(
                        s.position, shipyard)  # normal distance
                else:  # other
                    importance = self.game_map.calculate_distance(
                        s.position, shipyard) * self.game_map.width * 2
            else:
                importance = -1000  # newly spawned ships max importance
            heappush(ships, (importance, s))
        return ships, has_moved


    def clear_dictionaries(self):
        # clear dictionaries of crushed ships
        for ship_id in list(self.ship_dest.keys()):
            if not self.me.has_ship(ship_id):
                crashed_ships.append(self.ship_obj[ship_id].position)
                del self.ship_dest[ship_id]
                del self.ship_state[ship_id]
                del self.previous_state[ship_id]
                del self.previous_position[ship_id]
                if ship_id in self.ship_path:
                    del self.ship_path[ship_id]



    def add_crashed_position(self, pos):
        neighbours = self.game_map.get_neighbours(self.game_map[pos])
        h_amount = -1
        distance_to_enemy_dropoff = GlobalFunctions(self.game).dist_to_enemy_doff(pos)
        for n in neighbours:
            h_amount = max(h_amount, n.halite_amount)
        if h_amount > 800:
            heappush(self.crashed_positions, (-1 * h_amount, pos))


backuped_dropoffs = []
maingameloop = main(game, ship_state, ship_path, ship_dest, previous_position, previous_state, fleet_leader, ship_obj, crashed_positions, crashed_ships, turn_start)
maingameloop.mainloop()