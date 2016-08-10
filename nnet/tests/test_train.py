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
import numpy as np
import random
import time

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
        """Test the default model architecture is a simple single layer neural network - input-hidden-output layers"""
        train_object = train.trainBase()
        train_object.model()
        # Check number of layers should be 3 in total, input, hidden and output
        current_layer = train_object.network
        layer_count = 1
        while hasattr(current_layer, 'input_layer'):
            layer_count += 1
            current_layer = current_layer.input_layer
        self.assertEqual(layer_count, 3, "Incorrect number of layers in default nnet")

    def test_model_layer_shape(self):
        """Test the shape of the layers in the network"""
        train_object = train.trainBase()
        train_object.model()

        # Check shapes
        # Output layer
        self.assertEqual(train_object.network.num_units, 30,
                         "Incorrect number of output layer units")
        # Hidden layer
        self.assertEqual(train_object.network.input_layer.num_units, 500,
                         "Incorrect number of hidden layer units")

        # Input layer
        self.assertEqual(train_object.network.input_layer.input_layer.shape,
                         (None, train.DEFAULT_IMAGE_SIZE),
                         "Incorrect input layer shape")

    def test_generate_layer_list_w_network(self):
        """Test the construction of the layer list if the network is defined"""
        train_object = train.trainBase()
        train_object.model()
        train_object._generate_layer_list()

        # Check the layers are correctly mapped
        current_layer = train_object.network
        layer_count = len(train_object.layers)
        while hasattr(current_layer, 'input_layer'):
            with self.subTest(layer_count=layer_count):
                self.assertEqual(train_object.layers[layer_count - 1],
                                 current_layer,
                                 "{} layer != {} layer".format(train_object.layers[layer_count - 1].name,
                                                               current_layer.name)
                                 )
            current_layer = current_layer.input_layer
            layer_count -= 1

    def test_build_model_training_loss_function(self):
        """Test the correct construction of the training loss function - type(theano.function)"""
        train_object = train.trainBase()
        train_object.build_model()

        # Check is instance of theano.function
        self.assertIsInstance(train_object.train_loss, type(theano.function(inputs=[])))

    def test_build_model_validation_loss_function(self):
        """Test the correct construction of the validation loss function - type(theano.function)"""
        train_object = train.trainBase()
        train_object.build_model()

        # Check is instance of theano.function
        self.assertIsInstance(train_object.valid_loss, type(theano.function(inputs=[])))

    def test_build_model_prediction_function(self):
        """Test the correct construction of the prediction function - type(theano.function)"""
        train_object = train.trainBase()
        train_object.build_model()

        # Check is instance of theano.function
        self.assertIsInstance(train_object.predict, type(theano.function(inputs=[])))

    def test_iterate_minibatches_size(self):
        """Test the correct size of data returned by minibatches - no shuffle"""
        train_object = train.trainBase()

        random.seed(int(time.time()))

        # Produce a random number of samples between 1000 and 2000
        random_samples = random.randint(1000, 2000)
        inputs = np.ones((random_samples, train.DEFAULT_IMAGE_SIZE))
        targets = np.ones((random_samples, 30))

        # Produce a random batch size
        random_batch = random.randint(100, 200)

        batch_counter = 0

        for batch in train_object.iterate_minibatches(inputs, targets, random_batch, shuffle=False):
            samp, tar = batch
            batch_counter += len(samp)

        self.assertEqual(batch_counter, random_samples, "batch counter != size of target set")

    def test_iterate_minibatches_shuffle(self):
        """Test the correct shuffling of data in minibatches"""
        train_object = train.trainBase()

        random.seed(int(time.time()))

        # Produce a random number of samples between 1000 and 2000
        random_samples = random.randint(100, 200)
        inputs = np.ones((random_samples, train.DEFAULT_IMAGE_SIZE))
        inputs[:, 0] = np.int_(np.linspace(0, len(inputs), len(inputs)))
        targets = np.ones((random_samples, 30))

        # Produce a random batch size
        random_batch = random.randint(10, 20)

        batch_counter = 0

        for batch in train_object.iterate_minibatches(inputs, targets, random_batch, shuffle=True):
            samp, tar = batch
            non_shuffled_idx = np.int_(np.linspace(batch_counter, batch_counter + len(samp), len(samp)))
            with self.subTest(batch_counter=batch_counter):
                self.assertTrue(
                        np.any(np.not_equal(samp[:, 0], non_shuffled_idx)),
                        "no minibatch shuffling")
            batch_counter += len(samp)

    @unittest.skip("Test training in integration tests")
    def test_train(self):
        pass
