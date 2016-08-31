#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Test the training module of the package
"""

# Imports
import unittest
from unittest.mock import MagicMock, patch, mock_open
from nnet.train import trainBase as train
from nnet.train import adaptiverates as adaptiverates
from nnet.tests_unit.test_resources import assert_data_division 
import theano
import lasagne
import numpy as np
import random
import time
import os

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '18-Aug-2016 11:18:07 AEST'
__license__ = 'MPL v2.0'


class TestRates(unittest.TestCase):
    """Test the adaptive rates"""

    def test_fixed_rate(self):
        """Test fixed rate learning rate"""
        rate = adaptiverates.fixedRate(0.2)
        self.assertEqual(rate(), 0.2)

    def test_linear_rate(self):
        """Test linear rate learning rate"""
        rate = adaptiverates.linearRate(start=10, end=1, epochs=10)
        rate_check = 10

        for step in rate:
            with self.subTest(rate_check=rate_check):
                # Assert the same linear rate
                self.assertEqual(int(step), rate_check, "linear rate error {} != {}".format(step, rate_check))
            rate_check -= 1

log_file_mock_one_file = MagicMock(side_effect=[True, False])
log_file_mock_two_files = MagicMock(side_effect=[True, True, False])
mock_pickle = MagicMock()
mock_file_open = mock_open()

class TestTrainClass(unittest.TestCase):
    """Test the training base class"""

    def setUp(self):
        self.model_str_template = "input (None, %i)\n" % train.DEFAULT_IMAGE_SIZE 
        self.model_str_template += "hidden (None, 500)\n" 
        self.model_str_template += "output (None, 30)" 

    def test_attributes(self):
        """Test presence of attributes"""
        train_object = train.trainBase()
        attributes = [
                'model',
                'build_model',
                'iterate_minibatches',
                'train',
                'save_progress',
                'load_progress',
                ]
        for attr in attributes:
            with self.subTest(attr=attr):
                self.assertTrue(hasattr(train_object, attr), "trainBase missing {} attribute".format(attr))

    def test_load_data(self):
        """Test loading the default data"""
        train_object = train.trainBase()
        train_object.load_data()

        split_ratio_calculated = np.round(len(train_object.x_train) / 
                                          (len(train_object.x_train) + len(train_object.x_valid)), 1)
        assert_data_division(self, train_object.x_train, train_object.y_train, 
                             train_object.x_valid, train_object.y_valid, 
                             0.7, split_ratio_calculated)

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

    def test_model_different_num_nodes(self):
        """Test the default model architecture with a different number of nodes is a simple single layer neural network - input-hidden-output layers"""
        train_object = train.trainBase()
        train_object.model(num_units=10)

        # Check the hidden layer has the correct number of units
        self.assertEqual(train_object.network.input_layer.num_units, 10,
                         "Incorrect number of hidden layer units")

    def test_model_different_num_nodes_build(self):
        """Test the default model architecture with a different number of nodes with model building"""
        train_object = train.trainBase()
        train_object.model(num_units=10)
        train_object.build_model()

        # Check the hidden layer has the correct number of units
        self.assertEqual(train_object.network.input_layer.num_units, 10,
                         "Incorrect number of hidden layer units")


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

    def test_model_to_str(self):
        """Test the description of the model as a string"""
        train_object = train.trainBase()
        train_object.model()
        train_object._generate_layer_list()

        description = train_object.model_str()
        self.assertEqual(description, self.model_str_template, "model string doesn't match template")

    def test_model_to_str_no_layers(self):
        """Test the description of the model as a string - autogenerate layer list"""
        train_object = train.trainBase()
        train_object.model()

        description = train_object.model_str()
        self.assertEqual(description, self.model_str_template, "model string doesn't match template")

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

    def test_use_of_non_momentum_update(self):
        """Test building a model using an update method that does not have a momentum property"""

        train_object = train.trainBase(verbose=False, 
                                       updates=lasagne.updates.sgd,
                                       max_epochs=200,
                                       ) 
        train_object.build_model()

        # If an error is not thrown the model has built correctly
        self.assertIsInstance(train_object.valid_loss, type(theano.function(inputs=[])))


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

    def test_iterate_minibatches_one_sample(self):
        """Test minibatches is able to handle a single sample only"""
        train_object = train.trainBase()
        inputs = np.ones((1, train.DEFAULT_IMAGE_SIZE))
        targets = np.ones((1, 30))

        random_batch = 128
        
        batch_counter = 0

        for batch in train_object.iterate_minibatches(inputs, targets, random_batch, shuffle=False):
            samp, tar = batch
            batch_counter += len(samp)

        self.assertEqual(batch_counter, 1, "batch counter != size of target set")

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

    def test_iterate_minibatches_assertion(self):
        """Test ValueError is raised in minibatches when inputs and targets differ in length"""
        train_object = train.trainBase()

        inputs = np.ones((10, train.DEFAULT_IMAGE_SIZE)) 
        targets = np.ones((8, 30)) 

        with self.assertRaises(ValueError):
            for batch in train_object.iterate_minibatches(inputs, targets, 1, shuffle=True):
                pass

    @patch('pickle.dump', mock_pickle)
    @patch('builtins.open', mock_file_open)
    def test_save_progress(self):
        """Test the correct data is pickled"""
        train_object = train.trainBase()

        # Assign some data to pickle

        expected_data = {
                'weights': [1, 2],
                'curr_epoch': 100,
                'best_valid_err': 33,
                'best_train_err': 15,
                'y_train_err_history': [78, 98],
                'y_valid_err_history': [102, 233],
                }

        train_object.best_weights = expected_data['weights'] 
        train_object.current_epoch = expected_data['curr_epoch'] 
        train_object.best_valid_err = expected_data['best_valid_err'] 
        train_object.best_train_err = expected_data['best_train_err'] 
        train_object.y_train_err_history = expected_data['y_train_err_history'] 
        train_object.y_valid_err_history = expected_data['y_valid_err_history'] 

        train_object.save_progress()

        mock_file_open.assert_called_with('trainBase.pkl', 'wb')

        mock_pickle.assert_called_once_with(expected_data, mock_file_open())

    def test_load_progress(self):
        """Test the correct data is being loaded from the pickle"""
        train_object = train.trainBase()

        expected_data = {
                'weights': [1, 2],
                'curr_epoch': 10,
                'best_valid_err': 12,
                'best_train_err': 13,
                'y_train_err_history': [7, 8, 9],
                'y_valid_err_history': [10, 11, 12],
                }

        train_object.load_progress(os.path.join(os.path.dirname(__file__),'trainBase.pkl'))
        self.assertEqual(train_object.best_weights, expected_data['weights'])
        self.assertEqual(train_object.current_epoch, expected_data['curr_epoch'])
        self.assertEqual(train_object.best_valid_err, expected_data['best_valid_err'])
        self.assertEqual(train_object.best_train_err, expected_data['best_train_err'])
        self.assertEqual(train_object.y_train_err_history, expected_data['y_train_err_history'])
        self.assertEqual(train_object.y_valid_err_history, expected_data['y_valid_err_history'])

    def test_prepare_log(self):
        """Check the log filenames are correctly established - no existing log exists"""
        train_object = train.trainBase(name="test_prepare")

        self.assertEqual(train_object.log_filename, "test_prepare.log", 
                         "incorrect log filename")
        self.assertEqual(train_object.save_params_filename, "test_prepare.pkl", 
                         "incorrect pickle filename")

    @patch('os.path.exists', log_file_mock_one_file)
    def test_prepare_log_exists(self):
        """Check the log filenames are correctly established - a log of the same name already exists"""
        train_object = train.trainBase(name="test_prepare")

        self.assertEqual(train_object.log_filename, "test_prepare.log.0", 
                         "incorrect log filename")
        self.assertEqual(train_object.save_params_filename, "test_prepare.pkl.0", 
                         "incorrect pickle filename")

    @patch('os.path.exists', log_file_mock_two_files)
    def test_prepare_log_exists(self):
        """Check the log filenames are correctly established - two logs of the same name already exist"""
        train_object = train.trainBase(name="test_prepare")

        self.assertEqual(train_object.log_filename, "test_prepare.log.1", 
                         "incorrect log filename")
        self.assertEqual(train_object.save_params_filename, "test_prepare.pkl.1", 
                         "incorrect pickle filename")

    def test_info_logging(self):
        """Test the logging method correctly logs"""

        train_object = train.trainBase(name="test_log", verbose=True)

        file_patch = mock_open()

        with patch('builtins.open',file_patch):
            train_object.log_msg('test')

        file_patch.assert_called_with('test_log.log', 'a')
        handle = file_patch()
        handle.write.assert_called_once_with('test\n')
