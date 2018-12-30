import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants
import json
from decimal import Decimal

""" This file contains the global variables used throughout the game 
    Function is called in the main bot file """

def load_global_constants(json_file):

    global PERCENTAGE_SWITCH, SMALL_PERCENTAGE, BIG_PERCENTAGE
    global MEDIUM_HALITE, INITIAL_HALITE_STOP, SPAWN_TURN, A, B, C, D, E, F, CRASH_TURN
    global CRASH_SELECTION_TURN, DETERMINE_CLUSTER_TURN, CLUSTER_TOO_CLOSE, MAX_CLUSTERS
    global FLEET_SIZE, CLOSE_TO_SHIPYARD, ENEMY_SHIPYARD_CLOSE, SHIP_SCAN_AREA, EXTRA_FLEET_MAP_SIZE, BUSY_PERCENTAGE
    global BUSY_RETURN_AMOUNT, UNSAFE_AREA, MAX_SHIP_DROPOFF_RATIO

    adress = r'{}\{}\{}'.format('bot', 'constantsprofiles', json_file)
    with open(adress) as f:
        data = json.load(f)


    CRASH_TURN = constants.MAX_TURNS

    ''' when switch collectable percentage of max halite '''
    PERCENTAGE_SWITCH = int(data["PERCENTAGE_SWITCH"] * constants.MAX_TURNS)

    SMALL_PERCENTAGE = float(data["SMALL_PERCENTAGE"])
    BIG_PERCENTAGE = float(data["BIG_PERCENTAGE"])

    ''' definition of medium patch size for stopping and collecting patch if on the way '''
    MEDIUM_HALITE = int(data["MEDIUM_HALITE"])

    ''' halite left at patch to stop collecting at that patch '''
    INITIAL_HALITE_STOP = int(data["INITIAL_HALITE_STOP"])
    
    ''' until which turn to spawn ships '''
    SPAWN_TURN = int(float(data["SPAWN_TURN"]) * constants.MAX_TURNS)
    
    ''' Coefficients for halite heuristics '''
    A = float(data["A"])
    B = float(data["B"])
    C = float(data["C"])
    D = float(data["D"])
    E = float(data["E"])
    F = float(data["F"])


    CRASH_SELECTION_TURN = int(float(data["CRASH_SELECTION_TURN"]) * constants.MAX_TURNS)

    # Minimum halite needed to join a halite cluster
    DETERMINE_CLUSTER_TURN = int(float(data["DETERMINE_CLUSTER_TURN"]) * constants.MAX_TURNS)
    CLUSTER_TOO_CLOSE = float(data["CLUSTER_TOO_CLOSE"])  # distance two clusters can be within
    MAX_CLUSTERS = int(data["MAX_CLUSTERS"])  # max amount of clusters
    FLEET_SIZE = int(data["FLEET_SIZE"])  # fleet size to send for new dropoff
    CLOSE_TO_SHIPYARD = data["CLOSE_TO_SHIPYARD"] # definition of percentage of map size that we consider close to shipyard
    ENEMY_SHIPYARD_CLOSE = data["ENEMY_SHIPYARD_CLOSE"] # -//- close to enemy shipyard
    SHIP_SCAN_AREA = data["SHIP_SCAN_AREA"] # when ship changing destinations relative to its position, the area used to scan around him for halite
    EXTRA_FLEET_MAP_SIZE = data["EXTRA_FLEET_MAP_SIZE"] # for maps >= we send an extra fleet together with the builder of a dropoff
    # % of patches that have a ship on them for ships to return earlier
    BUSY_PERCENTAGE = data["BUSY_PERCENTAGE"]
    BUSY_RETURN_AMOUNT = data["BUSY_RETURN_AMOUNT"] * constants.MAX_TURNS
    UNSAFE_AREA = data["UNSAFE_AREA"]

    ''' for each 40 ships there should be a dropoff '''
    MAX_SHIP_DROPOFF_RATIO = data["MAX_SHIP_DROPOFF_RATIO"]  