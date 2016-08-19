#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

# Imports
import os
import pandas
from sklearn.cross_validation import train_test_split
import time
import numpy as np
import shutil

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '19-Aug-2016 16:00:09 AEST'
__license__ = 'MPL v2.0'


RESOURCE_DIR = os.path.dirname(__file__)
DEFAULT_TRAIN_SET = os.path.join(RESOURCE_DIR, 'training.csv')

def import_training_csv(filename):
    copied_file = shutil.copy(filename, DEFAULT_TRAIN_SET)
    assert(copied_file == DEFAULT_TRAIN_SET)


def load_training_data(filename=DEFAULT_TRAIN_SET):
    """Load the training set

    Parameters
    ------------

    filename : the filename of the data set to load

    Returns
    ------------

    a pandas DataFrame containing the data
    """

    data = pandas.read_csv(filename)
    return data


def remove_incomplete_data(data):
    return data.dropna()


def extract_image_landmarks(data_in):

    # Convert the images
    data_in['Image'] = \
            data_in['Image'].apply(
            lambda im: np.fromstring(im, sep=' '))

    # Extract the images
    # Scale the images
    x = np.vstack(data_in['Image'].values) / 255.

    # Centre to the mean
    x -= np.mean(x, axis=1).reshape((x.shape[0], -1))
    x -= np.mean(x, axis=0)

    x = x.astype(np.float32)

    # Extract the labels
    labels = data_in.columns.tolist()
    labels.pop(labels.index('Image'))
    y = data_in[labels].values

    y = (y - 48) / 48  # Scale between -1 and 1

    y = y.astype(np.float32)

    return x, y


def split_training_data(x, y, split_ratio=0.7):

    x_train, x_valid, y_train, y_valid = \
        train_test_split(x, y,
                         train_size=split_ratio,
                         random_state=int(time.time()))

    return x_train, y_train, x_valid, y_valid


def load_data(filename=DEFAULT_TRAIN_SET, dropna=True, split_ratio=0.7):
    """Load the training set, extract features and split into training and validation groups

    Parameters
    ------------

    filename : (optional) the filename of the data set to load, default set to nnet.resources.DEFAULT_TRAIN_SET
    dropna   : (optional) remove samples with incomplete data from the original data set.  Set to True by default, det
               to False to keep all data
    split_ratio : (optional) the train / validation ratio for the data set.  0.7 by default (70% of the data is used for
                  the training set)

    Returns
    ------------

    a list of np.arrays containing the training and validation sets
    [train_in, train_targets, valid_in, valid_targets]
    """
    data = load_training_data(filename)
    if dropna:
        data = remove_incomplete_data(data)
    x, y = extract_image_landmarks(data)
    return split_training_data(x, y, split_ratio=split_ratio)
