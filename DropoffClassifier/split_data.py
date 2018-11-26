import json
import os
import os.path
from random import choice
import numpy as np
from sklearn.utils import shuffle


def split_data(dictionary, percentage=1):
    data_in, data_out, d = [], [], {}

    no_dropoffs = dictionary['0'][:int(len(dictionary['0']) * percentage)]
    dropoffs = dictionary['1'][:int(len(dictionary['1']) * percentage)]
    # remove training
    d['0'] = dictionary['0'][int(len(dictionary['0']) * percentage):]
    d['1'] = dictionary['1'][int(len(dictionary['1']) * percentage):]

    data_in = np.array(no_dropoffs + dropoffs)
    data_out = np.array([0] * len(no_dropoffs) + [1] * len(dropoffs))

    data_in, data_out = shuffle(data_in, data_out, random_state=0)

    return data_in, data_out, d


def save_data(folder_name, file_name, data):
    np.save(os.path.join(folder_name, file_name), data)

with open('all_data.json', 'r') as fp:
    data = json.load(fp)

training_percentage = 0.75  # use % for training
train_in, train_out, d = split_data(data, training_percentage)
test_in, test_out, _ = split_data(d)
print(test_in)

# save as np arrays in data folder
folder = 'data'

save_data(folder, 'test_in', test_in)
save_data(folder, 'test_out', test_out)
save_data(folder, 'train_in', train_in)
save_data(folder, 'train_out', train_out)
