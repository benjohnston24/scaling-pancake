#! /usr/bin/env python
"""!
-----------------------------------------------------------------------------
File Name: __init__.py

Purpose:

Created: 04-Aug-2016 16:28:02 AEST
-----------------------------------------------------------------------------
Revision History



-----------------------------------------------------------------------------
S.D.G
"""
__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 16:28:02 AEST'
__license__ = 'MPL v2.0'

# LICENSE DETAILS############################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# IMPORTS#####################################################################
import os
import pandas
from sklearn.cross_validation import train_test_split
import time
import numpy as np
##############################################################################
RESOURCE_DIR =  os.path.dirname(__file__)
DEFAULT_TRAIN_SET = os.path.join(RESOURCE_DIR,'training.csv') 

__all__ = [
        "RESOURCE_DIR", 
        "load_training_data"
        ]

def load_training_data(filename=DEFAULT_TRAIN_SET):
    """Load the training set

    Parameters
    ------------

    filename : the filename of the data set to load

    Returns
    ------------

    an array containing the data 
    """

    data = pandas.read_csv(filename)
    return data

def remove_incomplete_data(data):
    return data.dropna() 

def extract_image_landmarks(data_in):

    complete_data = pandas.DataFrame.copy(data_in)

    # Convert the images
    complete_data['Image'] = \
            complete_data['Image'].apply(
            lambda im: np.fromstring(im, sep=' '))

    # Extract the images
    # Scale the images
    x = np.vstack(complete_data['Image'].values) / 255.

    # Centre to the mean
    x -= np.mean(x, axis=1).reshape((x.shape[0], -1))
    x -= np.mean(x, axis=0)

    x = x.astype(np.float32)
    # Extract the labels
    y = complete_data[complete_data.columns[:-1]].values

    y = (y - 48) / 48 # Scale between -1 and 1

    y = y.astype(np.float32)

    return x, y



def split_training_data(x, y, split_ratio=0.7):


    x_train, x_valid, y_train, y_valid = \
        train_test_split(x, y, 
                         train_size=split_ratio, 
                         random_state=int(time.time()))

    return x_train, y_train, x_valid, y_valid 

if __name__ == "__main__":
    dat = remove_incomplete_data(load_training_data())
    split_training_data([1, 2, 3],[4,5,6])

