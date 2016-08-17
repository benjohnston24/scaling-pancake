#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Functional testing of the train module"""

# Imports
import unittest
from unittest.mock import MagicMock, patch, mock_open
import nnet.train as train
import theano
import lasagne
import numpy as np
import random
import time

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__  = 'Thursday 11 August  12:40:37 AEST 2016'
__license__ = 'MPL v2.0'


class TestTrainingFunctional(unittest.TestCase):

    def setUp(self):
        pass

    def test_training_error_to_zero(self):
        """Test the training error approaches zero with only a single training sample"""

        train_object = train.trainBase(verbose=False, max_epochs=200) 
        train_object.test_model()
        np.testing.assert_almost_equal(train_object.best_valid_err,0)

    def test_training_error_to_zero_no_build(self):
        """Test the training error approaches zero without a built model"""
        train_object = train.trainBase(verbose=False, max_epochs=200) 
        train_object.load_data()
        # Make the training data set only a single sample
        train_object.x_train = train_object.x_train[:1,:] 
        train_object.y_train = train_object.y_train[:1,:] 

        # Test the network is capable of memorising the training sample
        train_object.x_valid = train_object.x_train[:1,:] 
        train_object.y_valid = train_object.y_train[:1,:] 

        train_object.train()
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
