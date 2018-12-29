import os
import sys
import inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import logging
import bot.GlobalConstants as GC
from heapq import heappush, heappop, merge
from bot.GlobalFunctions import GlobalFunctions
from bot.GlobalVariablesSingleton import GlobalVariablesSingleton
from hlt.positionals import Direction, Position
import time

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
        self.NR_OF_PLAYERS = GV.NR_OF_PLAYERS
        self.inv_ship_dest = {}
        self.dropoff_distribution = {}


    def find_new_destination(self, h, ship):
        ''' h: priority queue of halite factors,
                                        halite_pos: dictionary of halite factor -> patch position '''
        removed = set()  # all elements removed from the heap during this function
        ship_id = ship.id
        biggest_halite, position = heappop(h)  # get biggest halite
        removed.add((biggest_halite, position))
        destination = self.game_map.normalize(position)
        self.dropoff_distribution = self.get_ship_distribution_over_dropoffs()
        # create an inverted ship_dest hashmap 
        # with a list of ships per destination
        self.inv_ship_dest = {}
        for ID, pos in self.ship_dest.items():
            if pos in self.inv_ship_dest:
                self.inv_ship_dest[pos].append(ID)
            else:
                self.inv_ship_dest[pos] = [ID]
        # repeat while not a viable destination, or enemies around the position or 
        # too many ships going to that dropoff area
        # or the same destination as before
        while self.bad_destination(ship, destination) or (self.NR_OF_PLAYERS == 4 and self.game_map[destination].enemy_neighbouring > 0):
            # if no more options, return
            if not h:
                GlobalFunctions(self.game).state_switch(ship.id, "returning")
                return
            biggest_halite, position = heappop(h)
            removed.add((biggest_halite, position))
            destination = self.game_map.normalize(position)
        self.ship_dest[ship_id] = destination  # set the destination
        # This cell has a ship so doesn't need to be in heap
        removed.remove((biggest_halite, position))
        # Add add removed ones back to the heap except the cell were going to
        for r in removed:
            heappush(h, r)
        self.reassign_duplicate_dests(destination, ship_id) # deal with duplicate destinations


    def bad_destination(self, ship, destination):
        """ definition of unviable destination for ship """
        if self.game_map[destination].halite_amount == 0:
            return True
        if self.game.turn_number <= GC.SPAWN_TURN:
            return self.game_map[destination].enemy_amount >= GC.UNSAFE_AREA\
                or not self.dest_viable(destination, ship) or self.too_many_near_dropoff(ship, destination)
        else:
            return self.game_map[destination].enemy_amount >= GC.UNSAFE_AREA or not self.dest_viable(destination, ship)


    def reassign_duplicate_dests(self, destination, this_id):
        # if another ship had the same destination
        s = self.get_ship_w_destination(destination, this_id)
        if s:  # find a new destination for the ships with same dest
            for other in s:
                self.process_new_destination(other)


    def process_new_destination(self, ship):
        self.ship_path[ship.id] = []
        if ship.position in GlobalFunctions(self.game).get_dropoff_positions() or ship.id not in self.ship_dest:
            self.find_new_destination(self.game_map.halite_priority, ship)
        else:
            source = ship.position
            ship_h = GlobalFunctions(self.game).halite_priority_q(source, GC.SHIP_SCAN_AREA)
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
            return self.dropoff_distribution[GlobalFunctions(self.game).get_shipyard(destination)] > (1 / len(GlobalFunctions(self.game).get_dropoff_positions()))


    def prcntg_ships_returning_to_doff(self, d_pos):
        amount = 0
        for s in self.me.get_ships():
            eval_pos = s.position if s.id not in self.ship_dest else self.ship_dest[s.id]
            if GlobalFunctions(self.game).get_shipyard(eval_pos) == d_pos:
                amount += 1
        return amount / len(self.me.get_ships())

    def get_ship_distribution_over_dropoffs(self):
        distribution = {}
        for s in self.me.get_ships(): # count ships per dropoff
            d_pos = GlobalFunctions(self.game).get_shipyard(s.position)
            if d_pos in distribution:
                distribution[d_pos] += 1
            else:
                distribution[d_pos] = 1
        for p, amount in distribution.items(): # turn into percentages
            distribution[p] = amount/len(self.me.get_ships())
        return distribution 

    def get_ship_w_destination(self, dest, this_id):
        """ gets ships with dest, s.t. that ship is not this_id """
        other_ships = []
        if dest in self.inv_ship_dest:
            for s_id in self.inv_ship_dest[dest]:
                if s_id != this_id:
                    other_ships.append(self.me.get_ship(s_id))
        return other_ships
