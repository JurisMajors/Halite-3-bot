import os
import sys
import inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import logging


class GlobalVariablesSingleton():
    # Here the instance will be stored
    __instance = None

    @staticmethod
    def getInstance():
        ''' static access method '''
        if GlobalVariablesSingleton.__instance == None:
            GlobalVariablesSingleton()
        return GlobalVariablesSingleton.__instance

    def __init__(self, game):
        ''' virtually private constructor. '''
        if GlobalVariablesSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            GlobalVariablesSingleton.__instance = self
            self.ENABLE_BACKUP = True
            self.ENABLE_COMBAT = True
            self.ship_state = {}  # ship.id -> ship state
            # ship.id -> directional path to ship_dest[ship.id]
            self.ship_path = {}
            self.ship_dest = {}  # ship.id -> destination
            self.previous_position = {}  # ship.id-> previous pos
            self.previous_state = {}  # ship.id -> previous state
            self.fleet_leader = {}
            self.ship_obj = {}  # ship.id to ship obj for processing crashed ship stuff
            self.turn_start = 0  # for timing
            self.NR_OF_PLAYERS = len(game.players.keys())
            self.MIN_CLUSTER_VALUE = 6000 if game.game_map.width >= 64 else 8000
