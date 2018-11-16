import os
import random
from pathlib import Path
map_sizes = [32, 40, 48, 56, 64] # possible map sizes
BASE_DIR = Path(os.path.dirname(__file__)).parent # path to halite.exe

# VARIABLES = [VERSION, SCAN_AREA, PERCENTAGE_SWITCH, SMALL_PERCENTAGE, BIG_PERCENTAGE, MEDIUM_HALITE, HALITE_STOP, 
# SPAWN_TURN, A, B, C, CRASH_PERCENTAGE_TURN]
# SCAN_AREA - for scanning halite patches
# PERCENTAGE_SWITCH - which turn to switch from big to small percentage collection
# MEDIUM_HALITE - definition of medium sized patch to stop n collect
# HALITE_STOP - how much halite has to be left to stop collecting
# SPAWN TURN - when to stop spawning ships
# A, B, C - constants for patch heuristic polynomial
# CRASH_PERCENTAGE_TURN - at what percent of game to select crash turn
# Our variables - [1, 30, 50, 0.7, 0.95, 300, 10, 220, 0, 2, 1, 0.8]
def run_iteration(variables1, variables2): # run iteration of bots with variables
	map_size= random.choice(map_sizes)
	print("MAP CHOICE: {}x{}".format(map_size, map_size))
	
	bot1_variables = " ".join(map(str, variables1))
	bot2_variables = " ".join(map(str, variables2))

	cmd = '{0}\\halite.exe --replay-directory {0}\\replays/ -vvv --width {1} --height {1}'.format(BASE_DIR, map_size)
	cmd += ' "python {0}\\MyBot1.py {1}" "python {0}\\MyBot2.py {2}" >> data.txt'.format(BASE_DIR, bot1_variables, bot2_variables)

	os.system(cmd)

DEFAULT_VARIABLES = [1, 30, 50, 0.7, 0.95, 300, 10, 220, 0, 2, 1, 0.8]
run_iteration(DEFAULT_VARIABLES, DEFAULT_VARIABLES)
#with open("game.data", "r") as f:
#	contents = f.readlines()



