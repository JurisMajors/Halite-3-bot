import copy
import json
import os
import os.path
import zstd
import sys
import random
import numpy as np
sys.path.append('C:/Users/juris/OneDrive/Documents/GitHub/halite-bot')
from hlt import *


class Parser:

    def parse_replay_file(self, file_name, player_name):
        ARBITRARY_ID = -1
        with open(file_name, 'rb') as f:
            data = json.loads(zstd.loads(f.read()))


        player = [p for p in data['players'] if p[
            'name'].split(" ")[0] == player_name][0]
        player_id = int(player['player_id'])
        my_shipyard = entity.Shipyard(player_id, ARBITRARY_ID,
                                      positionals.Position(player['factory_location']['x'], player['factory_location']['y']))
        other_shipyards = [
            entity.Shipyard(p['player_id'], ARBITRARY_ID, positionals.Position(
                p['factory_location']['x'], p['factory_location']['y']))
            for p in data['players'] if int(p['player_id']) != player_id]
        width = data['production_map']['width']
        height = data['production_map']['height']

        first_cells = []
        for x in range(len(data['production_map']['grid'])):
            row = []
            for y in range(len(data['production_map']['grid'][x])):
                row += [game_map.MapCell(positionals.Position(x, y),
                                         data['production_map']['grid'][x][y]['energy'])]
            first_cells.append(row)
        frames = []
        frames.append(game_map.GameMap(first_cells, width, height))

        first_my_dropoffs = [my_shipyard]
        first_them_dropoffs = other_shipyards
        my_dropoffs = []
        them_dropoffs = []
        for f in data['full_frames']:
            new_my_dropoffs = copy.deepcopy(first_my_dropoffs if len(
                my_dropoffs) == 0 else my_dropoffs[-1])
            new_them_dropoffs = copy.deepcopy(first_them_dropoffs if len(
                them_dropoffs) == 0 else them_dropoffs[-1])
            for e in f['events']:
                if e['type'] == 'construct':
                    if int(e['owner_id']) == player_id:
                        new_my_dropoffs.append(
                            entity.Dropoff(player_id, ARBITRARY_ID, positionals.Position(e['location']['x'], e['location']['y'])))
                    else:
                        new_them_dropoffs.append(
                            entity.Dropoff(e['owner_id'], ARBITRARY_ID, positionals.Position(e['location']['x'], e['location']['y'])))
            my_dropoffs.append(new_my_dropoffs)
            them_dropoffs.append(new_them_dropoffs)

        return frames[0], my_dropoffs[-1], them_dropoffs[-1]

    def do_data(self, file_name, player_name):
        game_map, my, they = self.parse_replay_file(file_name, player_name)
        w = game_map.width
        h = game_map.height
        all_shipyards = my + they
        top_left = positionals.Position(0, 0)
        data = [w]  # map size and our shipyard
        map_data = []
        for y in range(h):
            row = []
            for x in range(w):
                a_dropoff = 0
                pos = top_left + positionals.Position(x, y)
                if self.is_dropoff(all_shipyards, pos):
                    a_dropoff = 1
                    distance = self.dropoff_distance_to_shipyard(
                        all_shipyards, game_map, pos)
                else:
                    distance = self.cell_distance_to_shipyard(
                        game_map, pos, all_shipyards)
                # halite, distance to shipyard, 0 if not dropoff else 1
                row.append([x, y, round(game_map[pos].halite_amount /
                                        1000, 2), round(distance / w, 2), a_dropoff])
            map_data.append(row)

        # to access cell x, y, data[1][y][x]
        data.append(map_data)
        data = self.transform_data(data)
        return data

    def get_coord(self, x, y, diff_x, diff_y, size):
        new_x = (x - diff_x) % size
        new_y = (y - diff_y) % size
        return int(new_x), int(new_y)

    def transform_data(self, data, pooling_size=4):
        new_data = []
        map_data = []
        size = data[0]  # map size
        new_data.append(size)
        for row in data[1]:
            new_row = []
            for cell in row:
                cell_x, cell_y = cell[0], cell[1]
                # halite and distance of pooling cntr
                neighbours = [*cell[2:-1]]
                a_dropoff = 0
                # pooling_size x pooling_size square
                for x_diff in range(-int(pooling_size / 2), int(pooling_size / 2) + 1):
                    for y_diff in range(-int(pooling_size / 2), int(pooling_size / 2) + 1):
                        neighbour_x, neighbour_y = self.get_coord(
                            cell_x, cell_y, x_diff, y_diff, size)
                        if data[1][neighbour_y][neighbour_x][-1] == 1:
                            a_dropoff = 1
                        for info in data[1][neighbour_y][neighbour_x][2:-1]:
                            neighbours.append(info)
                # add whether that pooling sqr had a dropoff
                neighbours.insert(0, a_dropoff)
                new_row.append(neighbours)
            map_data.append(new_row)

        new_data.append(map_data)
        grouped_data = self.group_data(new_data)
        return grouped_data

    def group_data(self, transformed_data):
        g_data = {}
        g_data[0], g_data[1] = [], []
        for row in transformed_data[1]:
            for cell in row:
                cell_data = cell[1:]
                cell_data.insert(0, round(transformed_data[0] / 64, 2))
                g_data[cell[0]].append(cell_data)
        return g_data

    def dropoff_distance_to_shipyard(self, shipyards, game_map, dropoff_pos):
        shipyard_pos = self.find_shipyard_pos(
            shipyards, self.get_id(shipyards, dropoff_pos))
        return game_map.calculate_distance(shipyard_pos, dropoff_pos)

    def cell_distance_to_shipyard(self, game_map, pos, shipyards):
        dist = 10000
        for s in shipyards:
            if isinstance(s, entity.Shipyard):
                d = game_map.calculate_distance(s.position, pos)
                dist = min(dist, d)
        return dist

    def get_id(self, shipyards, dropoff_pos):
        for e in shipyards:
            if e.position == dropoff_pos:
                return e.owner

    def find_shipyard_pos(self, shipyards, owner_id):
        for e in shipyards:
            if isinstance(e, entity.Shipyard) and e.owner == owner_id:
                return e.position

    def is_dropoff(self, shipyards, pos):
        for e in shipyards:
            if e.position == pos and isinstance(e, entity.Dropoff):
                return True
        return False

    def split_data(self, file_name, player_name):
        d = self.do_data(file_name, player_name)
        size = d[0]

    def parse_replay_folder(self, folder_name, player_name):
        replay_buffer = {0: [], 1: []}
        print("Parsing {}".format(folder_name))
        for file_name in sorted(os.listdir(folder_name)):
            if not file_name.endswith(".hlt"):
                continue
            else:
                data_dic = self.do_data(os.path.join(
                    folder_name, file_name), player_name)
                replay_buffer[0] += data_dic[0]
                replay_buffer[1] += data_dic[1]
        print("FOLDER {} PARSED".format(folder_name))
        return replay_buffer


class Balance:

    def __init__(self, data):
        self.data = data

    def get_balanced_data(self):
        self.do_balance()
        return self.data

    def do_balance(self):
        to_remove_indices = []
        while len(self.data[0]) / len(self.data[1]) > 1.1:
            not_dropoffs = len(self.data[0])
            choice = random.randint(0, not_dropoffs - 1)
            del self.data[0][choice]
        np.random.shuffle(self.data[0])
        np.random.shuffle(self.data[1])

def combine_data(data):
    print("Combining data")
    c_data = {0: [], 1: []}
    for d in data:
        c_data[0] += d[0]
        c_data[1] += d[1]
    return c_data


p = Parser()
data = p.parse_replay_folder("training_replays/tecclas", "teccles")
data_2 = p.parse_replay_folder("training_replays/Duck", "TheDuck314")
data_3 = p.parse_replay_folder("training_replays/Tony", "TonyK")
combined_data = combine_data([data, data_2, data_3])
b = Balance(combined_data)
c_data = b.get_balanced_data()
with open('all_data.json', 'w') as fp:
    json.dump(c_data, fp)
