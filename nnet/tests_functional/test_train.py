#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Functional testing of the train module"""

# Imports
import unittest
from unittest.mock import MagicMock, patch, mock_open
from nnet.train import trainBase as train
from nnet.resources import load_mnist_test_images, load_mnist_test_labels, \
        MNIST_IMAGE_SIZE, MNIST_NUMBER_LABELS
import theano
import lasagne
from lasagne.layers import InputLayer, DenseLayer 
from lasagne.nonlinearities import rectify, softmax 
from lasagne.init import Normal
import numpy as np
import random
import time

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__  = 'Thursday 11 August 12:40:37 AEST 2016'
__license__ = 'MPL v2.0'


class trainMNIST(train.trainBase):

    def model(self):
        # Initialise input / output tensors
        self.input_var = theano.tensor.matrix('x')
        self.output_var = theano.tensor.matrix('y')

        input_layer = InputLayer(
                input_var=self.input_var,
                shape=(None,MNIST_IMAGE_SIZE),
                W=Normal(0.01),
                b=Normal(0.01),
                name='input',
                )

        hidden_layer = DenseLayer(
                input_layer,
                nonlinearity=rectify,
                W=Normal(0.01),
                b=Normal(0.01),
                num_units=100,
                name='hidden',
                )
        output_layer = DenseLayer(
                hidden_layer,
                nonlinearity=softmax,
                num_units=MNIST_NUMBER_LABELS,
                name='output'
                )

        self.network = output_layer


class TestTrainingFunctional(unittest.TestCase):

    def setUp(self):
        pass

    def test_training_error_to_zero(self):
        """Test the training error approaches zero with only a single training sample"""

        train_object = train.trainBase(verbose=False, max_epochs=200) 
        train_object.test_model()
        np.testing.assert_almost_equal(train_object.best_valid_err,0)

    def test_early_stopping(self):
        """Test early stopping can be triggered before max_epoch"""
        max_epochs = 200
        train_object = train.trainBase(verbose=False,
                                       max_epochs=max_epochs,
                                       patience=10
                                       ) 
        train_object.test_model()

        np.testing.assert_almost_equal(train_object.best_valid_err,0)
        self.assertTrue(train_object.best_epoch < max_epochs) 
        self.assertTrue(train_object.current_epoch < max_epochs)

    def test_classification_error_to_zero(self):
        """Test the base class can train a classification task to zero"""

        train_object = trainMNIST(verbose=False,
                                  objective=lasagne.objectives.categorical_crossentropy,
                                  updates=lasagne.updates.sgd,
                                  max_epochs=200,
                                  ) 

        train_object.x_train = load_mnist_test_images()
        train_object.x_train = train_object.x_train.reshape((-1, MNIST_IMAGE_SIZE)) 

        train_object.y_train = load_mnist_test_labels()

        train_object.test_model(load_training_data=False)
        np.testing.assert_almost_equal(train_object.best_valid_err,0)
