import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
import pickle
from hlt import constants
import logging
import bot.GlobalConstants as GC
from bot.GlobalVariablesSingleton import GlobalVariablesSingleton
from bot.GlobalFunctions import GlobalFunctions
from hlt.positionals import Direction, Position
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import numpy as np
from math import ceil

dropoff_clf = pickle.load(open('mlp.sav', 'rb'))

class ClusterProcessor():

    def __init__(self, game):
        self.game = game
        self.game_map = game.game_map
        self.me = game.me
        self.GV = GlobalVariablesSingleton.getInstance()
        self.NR_OF_PLAYERS = self.GV.NR_OF_PLAYERS


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
                     2) if self.NR_OF_PLAYERS in [2, 4] else self.game_map.width
        y_size = self.game_map.height if self.NR_OF_PLAYERS in [2, 1] else int(
            game_map.height / 2)
        diff1, diff2, diff3, diff4 = (0, 0, 0, 0)

        # in 4 player maps limit the scanning area so that we dont classify centers of map
        # or too close to enemies
        if self.NR_OF_PLAYERS == 4:
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
        travel = int(self.game_map.width / self.NR_OF_PLAYERS)
        # get all the centers depending on the amount of players
        if self.NR_OF_PLAYERS == 4:
            cntrs = [Position(travel, travel), Position(travel * 3, travel),
                     Position(travel * 3, travel * 3), Position(travel, travel * 3)]
        elif self.NR_OF_PLAYERS == 2:
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

            if halite < self.GV.MIN_CLUSTER_VALUE:  # if area contains less than 8k then remove it
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
            if diff < GC.CLOSE_TO_SHIPYARD * self.game_map.width\
             or GlobalFunctions(self.game).dist_to_enemy_doff(pos) < GC.CLOSE_TO_SHIPYARD * self.game_map.width\
             or halite < 0.7 * self.GV.MIN_CLUSTER_VALUE:
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
                    if dist <= area and not dist <= 4:
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
        euclid_dist = self.game_map.euclidean_distance(
            Position(p1[0], p1[1]), Position(p2[0], p2[1]))
        return euclid_dist + abs(p1[2] - p2[2])


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
