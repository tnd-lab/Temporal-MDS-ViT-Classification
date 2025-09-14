import os

import numpy as np

from src.utils.utils import fetch_data


def load_data(data_sets, if_shuffle=False):
    """
    This function is for loading data with directory and label
    """
    root_dir = data_sets["root_dir"]
    capture_date = data_sets["dates"]
    seqs = data_sets["seqs"]
    training_set = []
    for date_counter, date in enumerate(capture_date):
        directory = root_dir + capture_date[date_counter] + "/"
        if seqs[date_counter] is None:
            # find all in current folder
            train_file = os.listdir(directory)
        else:
            train_file = seqs[date_counter]
        training_set = training_set + fetch_data(directory, train_file)

    if if_shuffle:
        np.random.shuffle(training_set)
    train_set_data = []
    train_set_labels = []
    for i in range(len(training_set)):
        train_set_data.append(training_set[i][0])
        train_set_labels.append(training_set[i][1])

    return train_set_data, train_set_labels
