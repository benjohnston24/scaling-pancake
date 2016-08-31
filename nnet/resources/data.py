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
import gzip

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '19-Aug-2016 16:00:09 AEST'
__license__ = 'MPL v2.0'


RESOURCE_DIR = os.path.dirname(__file__)
DEFAULT_TRAIN_SET = os.path.join(RESOURCE_DIR, 'training.csv')
MNIST_TRAIN_IMAGES = os.path.join(RESOURCE_DIR, 'train-images-idx3-ubyte.gz')
MNIST_TRAIN_LABELS = os.path.join(RESOURCE_DIR, 'train-labels-idx1-ubyte.gz')
MNIST_TEST_IMAGES = os.path.join(RESOURCE_DIR, 't10k-images-idx3-ubyte.gz')
MNIST_TEST_LABELS = os.path.join(RESOURCE_DIR, 't10k-labels-idx1-ubyte.gz')
MNIST_NUMBER_LABELS = 10

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

def _read(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def load_mnist_train_images():
    """Load the MNIST training set images

    Parameters
    ------------
    
    None 

    Returns
    ------------
    The MNIST training set images as a numpy array with shape (number of images, rows, cols)

    """
    return load_mnist_images(MNIST_TRAIN_IMAGES)

def load_mnist_train_labels():
    """Load the MNIST training set labels 

    Parameters
    ------------
    
    None 

    Returns
    ------------
    The MNIST training set labels using one hot encoding  as a numpy array with shape (number of images, 10) 

    """
    return load_mnist_labels(MNIST_TRAIN_LABELS)



def load_mnist_test_images():
    """Load the MNIST test set images

    Parameters
    ------------
    
    None 

    Returns
    ------------
    The MNIST test set images as a numpy array with shape (number of images, rows, cols)

    """
    return load_mnist_images(MNIST_TEST_IMAGES)

def load_mnist_test_labels():
    """Load the MNIST test set labels 

    Parameters
    ------------
    
    None 

    Returns
    ------------
    The MNIST test set labels using one hot encoding  as a numpy array with shape (number of images, 10) 

    """
    return load_mnist_labels(MNIST_TEST_LABELS)



def load_mnist_images(filename=MNIST_TRAIN_IMAGES):
    """Load a set of MNIST images
    This function extracts the MNIST images contained within a gzip file.


    Parameters
    ------------
    
    filename : (optional) the filename of the data set to load, default set to nnet.resources.MNIST_TRAIN_IMAGES

    Returns
    ------------
    The images as a numpy array with shape (number of images, rows, cols)
    
    """

    with gzip.open(filename, 'rb') as bytestream:
        magic = _read(bytestream)

        # Check correct magic number
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in %s' %
                (magic, filename))

        num_images = _read(bytestream)
        rows = _read(bytestream)
        cols = _read(bytestream)
        buf = bytestream.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols)
        return data

def load_mnist_labels(filename=MNIST_TRAIN_LABELS):
    """Load a set of MNIST labels 
    This function extracts the MNIST labels contained within a gzip file.

    Parameters
    ------------
    
    filename : (optional) the filename of the data set to load, default set to nnet.resources.MNIST_TRAIN_LABELS

    Returns
    ------------
    The labels using one hot encoding  as a numpy array with shape (number of samples, 10) 
    
    """
    with gzip.open(filename, 'rb') as bytestream:
        magic = _read(bytestream)

        # Check correct magic number
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in %s' %
                (magic, filename))

        num_labels = _read(bytestream)
        buf = bytestream.read(num_labels)
        data = np.frombuffer(buf, dtype=np.uint8)
        # Return with one hot encoding
        encoding = np.zeros((num_labels, MNIST_NUMBER_LABELS)) 
        for idx, label in enumerate(data):
            encoding[idx, label] = 1
        return encoding 

