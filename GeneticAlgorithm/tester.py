
import random
from math import log2
from pathlib import Path
import json
import os

OUR_BOT =  [0, 50, 0.3, 0.87, 0.85, 290, 50,
             0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 7]	# 1.0``	``
OTHER_BOTS = [[1682, 50, 0.3, 0.94, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.61, 4, 10], # 1.36
 				[1283, 50, 0.3, 0.94, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 8], # 1.39
 				[525, 50, 0.3, 0.87, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.53, 0.33, 4, 10], # 1.34
 				[341, 50, 0.3, 0.94, 0.85, 300, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 158, 0.2, 0.21, 4, 10] ,	# 1.08
 				[715, 50, 0.3, 0.94, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 7],	# 1.05
 				[694, 54, 0.6, 0.94, 1.0, 302, 50, 0.5, 0, 1, 0, 0.01, 1.0, 1.05, 0.9, 549, 0.18, 0.99, 4, 7], # 1.05
 				[997, 50, 0.3, 0.87, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 10], # 1.18
 				[1285, 50, 0.3, 0.94, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 10], # 1.04
 				[1497, 50, 0.3, 0.89, 0.85, 290, 50, 0.5, 0, 0.58, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.611, 4, 10], # 1.16
 				[1628, 50, 0.3, 0.89, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.61, 4, 10], # 1.0
 				[1900, 50, 0.3, 0.94, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.61, 4, 10], # 1.05
 				[1958, 53, 0.3, 0.94, 0.88, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 501, 0.37, 0.61, 4, 11], # 1.07
 				[867, 50, 0.3, 0.87, 0.85, 290, 50, 0.5, 0, 1, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 8], # 1.08
 				[1242, 50, 0.3, 0.94, 0.85, 300, 50, 0.5, 0, 1, 0, 0.36, 0.98, 1.05, 0.9, 500, 0.15, 0.33, 4, 10], # 1.06
 				[1725, 50, 0.3, 0.89, 0.85, 290, 50, 0.5, 0, 0.58, 0, 0.01, 0.98, 1.05, 0.9, 500, 0.15, 0.61 , 4, 10] # 1.07
 				] 
map_sizes = [32, 40, 48, 56, 64] # possible map sizes
BASE_DIR = Path(os.path.dirname(__file__)).parent # path to halite.exe
random.seed(1)
TOURNAMENT_AMOUNT = 10
DEPTH  = int(log2(len(OTHER_BOTS) + 1)) # tree depth is log of nodes
stats = {} # version - >  list of ranking stats

def run_iteration(variables1, variables2, seed): # run iteration of bots with variables
	map_size= random.choice(map_sizes)
	#print("MAP CHOICE: {}x{}".format(map_size, map_size))
	bot1_variables = " ".join(map(str, variables1))
	bot2_variables = " ".join(map(str, variables2))
	cmd = '{0}\\halite.exe --replay-directory {0}\\replays/ --no-logs --results-as-json -vvv -s {2} --width {1} --height {1}'.format(BASE_DIR, map_size, seed)
	cmd += ' "python {0}\\MyBot1.py {1}" "python {0}\\MyBot2.py {2}" > data.json'.format(BASE_DIR, bot1_variables, bot2_variables)
	# save output in data.json, > for rewrite , >> for appending outputs
	os.system(cmd) # run

def get_result():
	# get results from game
	with open("data.json", "r") as f:
		data = json.load(f)		
	return data['stats']['0']['score'], data['stats']['1']['score']


def init_matchings(all_bots):
	''' assigns initial matchings, aka leaves of binary tree '''
	bots = all_bots[:]
	nr_of_pairs = int(len(bots)/2)
	matchings = []

	for _ in range(nr_of_pairs):
		choice1 = random.choice(bots)
		bots.remove(choice1)
		choice2 = random.choice(bots)
		bots.remove(choice2)
		matchings.append((choice1, choice2))
	return matchings

def interim_matchings(bots):
	nr_of_pairs = int(len(bots)/2)
	matchings = []
	for i in range(nr_of_pairs):
		matchings.append((bots[2*i], bots[2*i + 1]))
	return matchings

def do_tournament(matchings):
	''' runs a tournament with produces matchings and returns the rankings of the tournament'''
	t_matchings = {} # DEPTH - > matchings
	t_matchings[DEPTH] = matchings
	seed = random.randint(0, 5000)
	for lvl in range(DEPTH):
		current_matchings = t_matchings[DEPTH - lvl]
		bots_alive = []
		# run current matching games
		for match in current_matchings:
			print("PLAYING MATCH {} VS {}".format(match[0][0], match[1][0]))
			run_iteration(match[0], match[1], seed)
			first, second = get_result()
			if first >= second:
				bots_alive.append(match[0])
				print("WINNER: {} w SCORE {}".format(match[0][0], first))
			else:
				print("WINNER: {} w SCORE {}".format(match[1][0], second))
				bots_alive.append(match[1])
		if lvl == DEPTH - 1:
			break # bots_alive contains only the winner
		t_matchings[DEPTH - lvl - 1] = interim_matchings(bots_alive) # add next depth matchings
	winner = bots_alive[0] 
	print("TORUNAMENT WINNER: {}".format(winner))
	return produce_rankings(t_matchings, winner[0])

def produce_rankings(tournament, winner):
	# takes dictionary of intermediate results for each level of depth n winner of tournament
	# returns dictionary of rankings for each version rank -> [versions]
	rankings = {}
	rankings[1] = [winner]
	rank = 2
	for lvl in range(DEPTH):
		matchings = tournament[lvl + 1]
		bots_in_rank = []
		for match in matchings:
			if match[0][0] in rankings[rank - 1]: # determine loser
				loser = match[1][0]
			else:
				loser = match[0][0]
			bots_in_rank.append(loser) # add to the bots in this rank
		rankings[rank] = bots_in_rank
		rank += 1 # next rank
	return rankings

def produce_stats(rankings):
	''' given rankings of tournament, produces statistics and adds to the already existing statistics'''
	for rank, versions in rankings.items():
		for version in versions:
			statistics = stats[version] if version in stats.keys() else [0]*(DEPTH+1)
			amount_scored = statistics[rank - 1]
			statistics[rank - 1] = amount_scored + 1
			stats[version] = statistics
	return stats


def write_stats():
	# write stats in txt file
	output = ["  V  1  2  3  4\n"]
	for version, statistic in stats.items():
		percentage = [round(x/TOURNAMENT_AMOUNT, 2) for x in statistic] # percentage of hittin that spot
		line = [str(version)]
		line.append(str(percentage))
		line.append("\n")
		s = "  ".join(line)
		output.append(s)
	final_string = "".join(output)

	with open("results.txt", "w") as f:
		f.write(final_string)


# OTHER_BOTS.append(OUR_BOT)
# for t in range(TOURNAMENT_AMOUNT):
# 	matchings = init_matchings(OTHER_BOTS)	
# 	rankings = do_tournament(matchings)
# 	produce_stats(rankings)
# 	print("TOURNAMENT {} FINISHED".format(t+1))
# write_stats()
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!FINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# FOR DETERMINING SCORES
def scores(results):
	score = 0
	multiplier = len(results)
	for i in results:
		score += i*multiplier
		multiplier -= 1
	print(score)

scores([0.1, 0.1, 0.2, 0.1, 0.7])
scores([0.2, 0.1, 0.2, 0.3, 0.6])
scores([0.0, 0.0, 0.2, 0.4, 0.1])
scores([0.2, 0.0, 0.4, 0.4, 0.4])
scores([0.2, 0.2, 0.3, 0.2, 0.3])
scores([0.1, 0.0, 0.3, 0.2, 0.7])
scores([0.2, 0.0, 0.2, 0.4, 0.5])
scores([0.1, 0.2, 0.0, 0.3, 0.4])


