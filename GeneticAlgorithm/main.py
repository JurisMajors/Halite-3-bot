import os
import random
from pathlib import Path

map_sizes = [32, 40, 48, 56, 64]

map_size= random.choice(map_sizes)
print("MAP CHOICE: {}x{}".format(map_size, map_size))
BASE_DIR = os.path.join(os.path.dirname(__file__), os.path.pardir)
BASE_DIR = Path(os.path.dirname(__file__)).parent
#os.system("..")
print(BASE_DIR)
os.system('{0}\\halite.exe --replay-directory {0}\\replays/ -vvv --width {1} --height {1}  "python {0}\\MyBot1.py" "python {0}\\MyBot2.py"'.format(BASE_DIR, map_size))