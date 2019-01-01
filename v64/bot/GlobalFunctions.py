import os
import sys
import inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import bot.GlobalConstants as GC
import time
import logging
from hlt.positionals import Direction, Position
from heapq import heappush, heappop, merge
from bot.GlobalVariablesSingleton import GlobalVariablesSingleton
from collections import deque


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
        self.NR_OF_PLAYERS = GV.NR_OF_PLAYERS

    def halite_priority_q(self, pos, area):
        # h_amount <= 0 to run minheap as maxheap
        h = []  # stores halite amount * -1 with its position in a minheap
        top_left = Position(int(-1 * area / 2),
                            int(-1 * area / 2)) + pos  # top left of scan area
        for y in range(area):
            for x in range(area):
                p = Position((top_left.x + x) % self.game_map.width,
                             (top_left.y + y) % self.game_map.height)  # position of patch
                if p not in self.get_dropoff_positions():  # we dont consider the position of the centre or dropoffs
                    cell = self.game_map[p]
                    # we ignore cells who have 0 halite.
                    # if that cell has small amount of halite, just take a
                    # ratio with 2x distance to lesser the priority
                    if cell.halite_amount > 0:
                        factor = self.game_map.cell_factor(
                            pos, cell, self.me, self.ENABLE_BACKUP)
                        # add negative halite amounts so that would act as
                        # maxheap
                        heappush(h, (factor, p))
        return h

    def dist_to_enemy_doff(self, pos):
        ''' determines how close to an enemy dropoff is the position'''
        if self.NR_OF_PLAYERS < 2:
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
            self.ship_dest[ship_id] = self.game_map[
                self.me.get_ship(ship_id).position].dijkstra_dest

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
