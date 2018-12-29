import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import hlt
from hlt import constants

VARIABLES = ["YEEHAW", 1285, 50, 0.4, 0.9, 0.95, 500, 50, 0.55,
             0, 0.8, 0, 0.01, 0.98, 1.05, 0.9, 0.15, 0.15, 8, 8]
VERSION = VARIABLES[1]
# search area for halite relative to shipyard
SCAN_AREA = int(VARIABLES[2])
# when switch collectable percentage of max halite
PERCENTAGE_SWITCH = int(float(VARIABLES[3]) * constants.MAX_TURNS)
SMALL_PERCENTAGE = float(VARIABLES[4])
BIG_PERCENTAGE = float(VARIABLES[5])
# definition of medium patch size for stopping and collecting patch if on
# the way
MEDIUM_HALITE = int(VARIABLES[6])
# halite left at patch to stop collecting at that patch
INITIAL_HALITE_STOP = int(VARIABLES[7])
# until which turn to spawn ships
SPAWN_TURN = int(float(VARIABLES[8]) * constants.MAX_TURNS)
# Coefficients for halite heuristics
A = float(VARIABLES[9])
B = float(VARIABLES[10])
C = float(VARIABLES[11])
D = float(VARIABLES[12])
E = float(VARIABLES[13])
F = float(VARIABLES[14])
CRASH_TURN = constants.MAX_TURNS
CRASH_SELECTION_TURN = int(float(VARIABLES[15]) * constants.MAX_TURNS)

# Minimum halite needed to join a halite cluster
DETERMINE_CLUSTER_TURN = int(float(VARIABLES[16]) * constants.MAX_TURNS)
CLUSTER_TOO_CLOSE = float(VARIABLES[17])  # distance two clusters can be within
MAX_CLUSTERS = int(VARIABLES[18])  # max amount of clusters
FLEET_SIZE = int(VARIABLES[19])  # fleet size to send for new dropoff
CLOSE_TO_SHIPYARD = 0.18 # definition of percentage of map size that we consider close to shipyard
ENEMY_SHIPYARD_CLOSE = 0.15 # -//- close to enemy shipyard
SHIP_SCAN_AREA = 16 # when ship changing destinations relative to its position, the area used to scan around him for halite
EXTRA_FLEET_MAP_SIZE = 32 # for maps >= we send an extra fleet together with the builder of a dropoff
# % of patches that have a ship on them for ships to return earlier
BUSY_PERCENTAGE = 0.15
BUSY_RETURN_AMOUNT = 0.5 * constants.MAX_TURNS
UNSAFE_AREA = 4

MAX_SHIP_DROPOFF_RATIO = 40  # for each 40 ships there should be a dropoff