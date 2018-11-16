import os
import random
import json
from pathlib import Path

map_sizes = [32, 40, 48, 56, 64] # possible map sizes
BASE_DIR = Path(os.path.dirname(__file__)).parent # path to halite.exe
random.seed(2)

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
def run_iteration(variables1, variables2, seed ): # run iteration of bots with variables
	map_size= random.choice(map_sizes)
	print("MAP CHOICE: {}x{}".format(map_size, map_size))
	
	bot1_variables = " ".join(map(str, variables1))
	bot2_variables = " ".join(map(str, variables2))

	cmd = '{0}\\halite.exe --replay-directory {0}\\replays/ --no-replay --results-as-json -vvv -s {2} --width {1} --height {1}'.format(BASE_DIR, map_size, seed)
	cmd += ' "python {0}\\MyBot1.py {1}" "python {0}\\MyBot2.py {2}" > data.json'.format(BASE_DIR, bot1_variables, bot2_variables)
	# save output in data.json, > for rewrite , >> for appending outputs
	os.system(cmd) # run

def get_result():
	# get results from game
	with open("data.json") as f:
		data = json.load(f)
	return data['stats']['0']['score'], data['stats']['1']['score']

POPULATION_SIZE = 5
GENERATIONS = 1
DEFAULT_VARIABLES = [0, 30, 50, 0.7, 0.95, 300, 10, 220, 0, 2, 1, 0.8]
results = {} # VERSION -> RESULT
population = {} # VERSION -> VARIABLES
fitness = {} # VERSION -> FITNESS

def populate():
	# produces the initial population 
	population[0] = DEFAULT_VARIABLES # put our bot in the evolution as well
	for _ in range(1, POPULATION_SIZE):
		# determine random parameters
		VERSION = _
		SCAN_AREA = random.randint(10, 50)
		PERCENTAGE_SWITCH = random.randint(0, 200)
		SMALL_PERCENTAGE = round(random.random(), 2)
		BIG_PERCENTAGE = round(random.random(), 2)
		MEDIUM_HALITE = random.randint(100, 900)
		HALITE_STOP = random.randint(1, 10)
		SPAWN_TURN = random.randint(100, 300)
		A = round(random.random(), 2)
		B = round(random.random(), 2)
		C = round(random.random(), 2) + 0.1 # shouldnt be zero
		CRASH_PERCENTAGE_TURN = random.random()
		population[VERSION] = [VERSION, SCAN_AREA, PERCENTAGE_SWITCH, SMALL_PERCENTAGE, BIG_PERCENTAGE, MEDIUM_HALITE, HALITE_STOP, 
							 SPAWN_TURN, A, B, C, CRASH_PERCENTAGE_TURN]
def get_fitness(version):
	# fitness = version_score / our_score
	if results[0] == 0:
		return 0 # just in case
	return results[version]/results[0]


def determine_scores(seed):
	# offsprings play against our bot
	# loads results stores them in results dictionary
	our_var = population[0]
	run_iteration(our_var, our_var, seed)
	r1, r2 = get_result()
	real_result = max(r1, r2)
	results[0] = real_result
	for version, parameters in population.items():
		if not version == 0:
			run_iteration(our_var, parameters, seed)
			our_result, their_result = get_result()
			print("VERSION 0 : {} ".format(our_result), "VERSION {} : {}".format(version, their_result))
			results[version] = their_result
			fitness[version] = get_fitness(version)


def selection():
	# selection of best performers
	pass

def crossover():
	# crossover of best performers 
	pass

def mutation():
	# mutation step of new offspring
	pass

def determine_best():
	# save best variables somewhere
	pass


populate()
for _ in range(GENERATIONS):
	seed = random.randint(1, 5000)
	print("SEED: {}".format(seed))
	
	determine_scores(seed)

	selection()
	crossover()
	mutation()
	
	determine_best()


