import os
import random
import json
from pathlib import Path
import multiprocessing

map_sizes = [32, 40, 48, 56, 64]  # possible map sizes
BASE_DIR = Path(os.path.dirname(__file__)).parent  # path to halite.exe
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


# run iteration of bots with variables
def run_iteration(variables1, variables2, seed):
    map_size = random.choice(map_sizes)
    #print("MAP CHOICE: {}x{}".format(map_size, map_size))

    bot1_variables = " ".join(map(str, variables1))
    bot2_variables = " ".join(map(str, variables2))

    cmd = '{0}\\halite.exe --replay-directory {0}\\replays/  --results-as-json -vvv -s {2} --width {1} --height {1}'.format(
        BASE_DIR, map_size, seed)
    cmd += ' "python {0}\\MyBot1.py {1}" "python {0}\\MyBot2.py {2}" > data.json'.format(
        BASE_DIR, bot1_variables, bot2_variables)
    # save output in data.json, > for rewrite , >> for appending outputs
    os.system(cmd)  # run


def get_result():
    # get results from game
    with open("data.json", "r") as f:
        data = json.load(f)
    return data['stats']['0']['score'], data['stats']['1']['score']

POPULATION_SIZE = 150
# assuming population size 200, we need about 21 selected invidiuals
PARENT_AMOUNT = 18
GENERATIONS = 100
MUTATION_CHANCE = 0.05

DEFAULT_VARIABLES = [0, 50, 129, 0.87, 0.85, 290, 50,
                     0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 7]
PARAM_AMOUNT = len(DEFAULT_VARIABLES)
results = {}  # VERSION -> RESULT
population = {}  # VERSION -> VARIABLES
fitness = {}  # VERSION -> FITNESS


def populate():
    # produces the initial population
    population[0] = DEFAULT_VARIABLES  # put our bot in the evolution as well
    for v in range(1, POPULATION_SIZE):
        # determine random parameters
        VERSION = v
        SCAN_AREA = random.randint(10, 64)
        PERCENTAGE_SWITCH = round(random.random(), 2)
        SMALL_PERCENTAGE = round(random.random(), 2)
        BIG_PERCENTAGE = round(random.random(), 2)
        MEDIUM_HALITE = random.randint(100, 900)
        HALITE_STOP = random.randint(1, 250)
        SPAWN_TURN = round(random.random() + 0.1, 2)
        A = round(random.random(), 2)
        B = round(random.random(), 2)
        C = round(random.random() + 0.1, 2)  # shouldnt be zero
        D = round(random.random(), 2)
        E = round(random.random(), 2)
        F = round(random.random(), 2)
        CRASH_PERCENTAGE_TURN = round(random.random() + 0.1, 2)
        KILL_ENEMY = random.randint(1, 1000)
        DETERMINE_CLUSTER_TURN = round(random.random() + 0.1, 2)
        CLUSTER_TOO_CLOSE = round(random.random(), 2)
        MAX_CLUSTERS = random.randint(1, 10)
        FLEET_SIZE = random.randint(2, 10)
        population[VERSION] = [VERSION, SCAN_AREA, PERCENTAGE_SWITCH, SMALL_PERCENTAGE, BIG_PERCENTAGE, MEDIUM_HALITE, HALITE_STOP,
                               SPAWN_TURN, A, B, C, D, E, F, CRASH_PERCENTAGE_TURN, KILL_ENEMY, DETERMINE_CLUSTER_TURN, CLUSTER_TOO_CLOSE,
                               MAX_CLUSTERS, FLEET_SIZE]


def get_fitness(version):
    # fitness = version_score / our_score
    if results[0] == 0:
        return 0  # just in case
    return results[version] / results[0]


def determine_scores(seed):
	# offsprings play against our bot
	# loads results stores them in results dictionary
	our_var = population[0]
	for version, parameters in population.items():
		if not version == 0:
			run_iteration(our_var, parameters, seed)
			our_result, their_result = get_result()
			print("VERSION 0 : {} ".format(our_result), "VERSION {} : {}".format(version, their_result))
			results[version] = their_result
			results[0] = our_result
			fitness[version] = get_fitness(version)
			fitness[0] = get_fitness(0)

def selection():
    # selection of best performers
    sorted_by_fitness = sorted(fitness.items(), key=lambda k: k[
                               1], reverse=True)[:PARENT_AMOUNT]
    selected_versions = [v[0] for v in sorted_by_fitness]
    print(selected_versions)
    return selected_versions


def create_off_spring(param1, param2):
    split_index = random.randint(1, PARAM_AMOUNT - 1)
    first_part = param1[1:split_index]  # not including version part
    second_part = param2[split_index:]
    return first_part + second_part  # concatanete into parameter array


def crossover(selected_versions):
    # crossover of previously selected versions
    biggest_version = max(selected_versions)
    # clear population
    for version in list(population.keys()):
        if version not in selected_versions and not version == 0:
            # delete all versions that have not been selected for next gen
            del population[version]
            del results[version]
            del fitness[version]
    # creates list of unordered pairs of versions, PARENT_AMOUNT choose 2
    couples = [(selected_versions[v1], selected_versions[v2]) for v1 in range(
        len(selected_versions)) for v2 in range(v1 + 1, len(selected_versions))]
    print(couples)
    # need to randomly remove len(couples) - (POPULATION_SIZE - PARENT_AMOUNT)
    # couples
    to_remove = len(couples) - (POPULATION_SIZE - PARENT_AMOUNT)
    for _ in range(to_remove):
        choice = random.choice(couples)
        couples.remove(choice)

    for couple in couples:  # add new versions
        biggest_version += 1
        population[biggest_version] = [biggest_version] + \
            create_off_spring(population[couple[0]], population[couple[1]])


def mutate(parameters):
    amount_of_mutations = random.randint(1, PARAM_AMOUNT - 1)
    mutated_parameters = parameters
    for mutation in range(amount_of_mutations):
        variable = random.choice(parameters[1:])
        index = parameters.index(variable)
        if variable < 1:  # if probability, percentage
            m = round(random.uniform(0, 1 - variable), 2)
        else:
            part = int(0.1 * variable)
            m = random.randint(0, part)
        variable += m
        mutated_parameters[index] = variable

    return mutated_parameters


def mutation():
    for version, parameters in population.items():
        probability = random.random()
        if probability < MUTATION_CHANCE:
            mutated_param = mutate(parameters)
            population[version] = mutated_param


def get_best_version():  # returns key with the max value in dictionary
    v = list(results.values())
    k = list(results.keys())
    return k[v.index(max(v))]


def determine_best():  # writes best one so far in text file
    best_version = get_best_version()
    best_score = results[best_version]
    best_param = population[best_version]
    best_fitness = fitness[best_version]
    print("BEST IN THIS GENERATION: {}, {}".format(best_score, best_param))
    output = " FITNESS: {}, SCORE: {}, PARAMETERS: {} \n".format(
        best_fitness, best_score, str(best_param))
    with open("best.txt", "a") as f:
        f.write(output)

if __name__ == '__main__':
	populate()
	for g in range(GENERATIONS):
	    seed = random.randint(1, 5000)
	    print("GENERATION: {}/{}".format(g, GENERATIONS))
	    print("SEED: {}".format(seed))

	    determine_scores(seed)
	    determine_best()

	    selected = selection()
	    crossover(selected)
	    mutation()
