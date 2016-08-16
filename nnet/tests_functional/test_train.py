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
        train_object = train.trainBase(verbose=True)
        train_object.build_model()
        train_object.load_data()

        # Make the training data set only a single sample
        train_object.x_train = train_object.x_train[:1,:] 
        train_object.y_train = train_object.y_train[:1,:] 

        self.assertEqual

        # Train the object
        train_object.train()

        np.testing.assert_approx_equal(train_object.y_train_err_history[-1],0)
