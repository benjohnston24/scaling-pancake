#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Test the training module of the package
"""

# Imports
import unittest
import nnet.train as train
import theano
import lasagne

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 23:25:33 AEST'
__license__ = 'MPL v2.0'


class TestRates(unittest.TestCase):
    """Test the adaptive rates"""

    def test_fixed_rate(self):
        """Test fixed rate learning rate"""
        rate = train.adaptiverates.fixedRate(0.2)
        self.assertEqual(rate(), 0.2)

    def test_linear_rate(self):
        """Test linear rate learning rate"""
        rate = train.adaptiverates.linearRate(start=10, end=1, epochs=10)
        rate_check = 10

        for step in rate:
            with self.subTest(rate_check=rate_check):
                # Assert the same linear rate
                self.assertEqual(step, rate_check, "linear rate error {} != {}".format(step, rate_check))


class TestTrainClass(unittest.TestCase):
    """Test the training base class"""

    def test_attributes(self):
        """Test presence of attributes"""
        train_object = train.trainBase()
        attributes = [
                'model',
                'build_model',
                'iterate_minibatches',
                'train',
                'predict',
                'save_params',
                'load_params',
                ]
        for attr in attributes:
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(train_object, attr), "trainBase missing {} attribute".format(attr))

    def test_model_input_output(self):
        """Test the default model input tensor is of the correct type"""
        train_object = train.trainBase()
        train_object.model()
        self.assertIsInstance(train_object.input_var, type(theano.tensor.matrix('x')))
        self.assertIsInstance(train_object.output_var, type(theano.tensor.vector('y')))

    def test_model_default_architecture(self):
        """Test the default model architecture is a simple single layer neural network"""
        train_object = train.trainBase()
        train_object.model()
        # Check number of layers should be 3 in total, input, hidden and output
        current_layer = train_object.network
        layer_count = 0
        while hasattr(current_layer, 'input_layer', False):
            layer_count += 1
        self.assertEqual(layer_count, 3, "Incorrect number of layers in default nnet")
