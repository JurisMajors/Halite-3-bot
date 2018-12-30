import os
from random import randint, choice, seed
import json

seed(6)
map_sizes = [32, 40, 48, 56, 64]  # possible map sizes
# run iteration of bots with variables
def run_iteration(bot_name, map_size, seed):
    cmd = f'halite.exe --replay-directory replays/ --results-as-json\
     --height {map_size} --width {map_size} -s {seed} -vvv "python {bot_name}.py" > benchmark.json'
    print(cmd)
    os.system(cmd)

def get_result():
    # get results from game
    with open("benchmark.json", "r") as f:
        data = json.load(f)
    return data['stats']['0']['score']

def benchmark(bot1, bot2, iterations):
    for map_choice in map_sizes:
        for _ in range(iterations):
                s = randint(1, 5000)
                run_iteration(bot1, map_choice, s)
                result1 = get_result()
                run_iteration(bot2, map_choice, s)
                result2 = get_result()

                print(f"{bot1} scored {result1}")
                print(f"{bot2} scored {result2}")

benchmark("MyBot", "v58/58", 3)

