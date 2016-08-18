#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Base training class for nnet"""

# Imports
from . import adaptiverates
import nnet.resources as resources
import theano
from lasagne.nonlinearities import rectify, linear
from lasagne.init import Normal
from lasagne.layers import InputLayer, DenseLayer, get_output, \
        get_all_params, get_all_param_values
from lasagne.objectives import squared_error, aggregate
from lasagne.updates import nesterov_momentum
import numpy as np
import os
from six.moves import cPickle as pickle
import time


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = "Thursday 18 August 10:38:47 AEST 2016"
__license__ = 'MPL v2.0'

__all__ = [
        "DEFAULT_IMAGE_SIZE",
        "DEFAULT_LOG_EXTENSION",
        "DEFAULT_PKL_EXTENSION",
        "trainBase",
        ]


DEFAULT_IMAGE_SIZE = 96 ** 2
DEFAULT_LOG_EXTENSION = '.log'
DEFAULT_PKL_EXTENSION = '.pkl'
LINE = "-" * 156


class trainBase(object):

    def __init__(self,
                 input_var=None,
                 output_var=None,
                 learning_rate=adaptiverates.fixedRate(0.01),
                 objective=squared_error,
                 updates=nesterov_momentum,
                 momentum=adaptiverates.fixedRate(0.9),
                 patience=500,
                 batch_size=128,
                 verbose=False,
                 name='trainBase',
                 max_epochs = 1000,
                 ):
        self.input_var = input_var
        self.output_var = output_var
        self.objective = objective
        self.updates = updates
        self.learning_rate = learning_rate
        self.learning_rate_tensor = theano.shared(np.cast['float32'](self.learning_rate.rate))
        self.momentum = momentum
        self.momentum_tensor = theano.shared(np.cast['float32'](self.momentum.rate))
        self.patience = patience
        self.name = name
        self.verbose = verbose
        self.image_size = 96
        self.max_epochs = max_epochs

        # Performance parameters
        self.best_weights = None
        self.current_epoch = 0
        self.best_train_err = np.inf
        self.best_valid_err = np.inf
        self.y_train_err_history = []
        self.y_valid_err_history = []
        self.batch_size = batch_size

        # Prepare the log file
        self._prepare_log()

    def load_data(self, filename=resources.DEFAULT_TRAIN_SET, split_ratio=0.7):
        # Load the training data
        self.x_train, self.y_train, self.x_valid, self.y_valid = \
                resources.load_data(filename=filename, split_ratio=split_ratio)

    def model(self, num_units=500):
        """
        This method defines the model to be used during the training algorithm.  When the class
        is being inherited override this method to change the architecture of the model to
        be trained.

        When overriding this method self.input_var and self.output_var must be defined as
        theano Tensor objects

        Parameters
        ------------

        self    :  the reference for the object


        Returns
        ------------
        None

        """

        # Initialise input / output tensors
        self.input_var = theano.tensor.matrix('x')
        self.output_var = theano.tensor.matrix('y')

        input_layer = InputLayer(
                input_var=self.input_var,
                shape=(None, DEFAULT_IMAGE_SIZE),
                W=Normal(0.01),
                b=Normal(0.01),
                name='input',
                )

        hidden_layer = DenseLayer(
                input_layer,
                nonlinearity=rectify,
                W=Normal(0.01),
                b=Normal(0.01),
                num_units=num_units,
                name='hidden',
                )
        output_layer = DenseLayer(
                hidden_layer,
                nonlinearity=linear,
                num_units=30,
                name='output'
                )

        self.network = output_layer

    def _generate_layer_list(self):

        # Only execute if self.network is defined
        self.layers = []
        current_layer = self.network
        while hasattr(current_layer, 'input_layer'):
            self.layers.insert(0, current_layer)
            current_layer = current_layer.input_layer

        # Add the first input layer
        self.layers.insert(0, current_layer)

    def model_str(self):
        # Generate a string describing the model architecture

        if not hasattr(self, 'layers'):
            self._generate_layer_list()

        model_str = ""
        for layer in self.layers:
            model_str += "{} {}\n".format(layer.name, layer.output_shape)

        return model_str[:-1]

    def build_model(self):
        # Define the model prior to building it
        self.model()
        self._generate_layer_list()

        # Print the model architecture
        self.log_msg(self.model_str())

        # Training loss
        train_prediction = get_output(self.network)
        train_loss = aggregate(self.objective(train_prediction, self.output_var))
        self.log_msg("Objective: {}".format(self.objective.__name__))

        # Validation loss
        validation_prediction = get_output(self.network, deterministic=True)
        validation_loss = aggregate(self.objective(validation_prediction, self.output_var))

        # Update the parameters
        params = get_all_params(self.network, trainable=True)
        updates = self.updates(
                loss_or_grads=train_loss,
                params=params,
                learning_rate=self.learning_rate_tensor,
                momentum=self.momentum_tensor,
                )

        # Print the learning rate type
        self.log_msg('Update: %s' % self.updates.__name__)
        self.log_msg("Learning Rate: %s" % self.learning_rate.__name__)
        self.log_msg("Momentum: %s" % self.momentum.__name__)

        # Define training loss function
        self.train_loss = theano.function(
                inputs=[self.input_var, self.output_var],
                outputs=train_loss,
                updates=updates,
                allow_input_downcast=True,
                )

        # Define validation loss function
        self.valid_loss = theano.function(
                inputs=[self.input_var, self.output_var],
                outputs=validation_loss)

        # Define predict
        self.predict = theano.function(
                inputs=[self.input_var],
                outputs=validation_prediction)

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        """Iterate minibatches"""

        if len(inputs) != len(targets):
            raise ValueError('input and target lengths differ')

        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        for lot in range((len(inputs) + batchsize - 1) // batchsize):
            if shuffle:
                excerpt = indices[lot * batchsize: (lot + 1) * batchsize]
            else:
                excerpt = slice(lot * batchsize, (lot + 1) * batchsize)
            yield inputs[excerpt], targets[excerpt]

    def train(self):

        # If not already done, build the model
        if not hasattr(self, 'predict'):
            self.build_model()

        self.log_msg("Batch Size: %s" % self.batch_size)
        self.log_msg("Patience: %s" % self.patience)

        # Print the header
        self.log_msg(LINE)
        self.log_msg(
                     "|{:^20}|{:^20}|{:^20}|{:^30}|{:^20}|{:^20}|{:^20}|".
                     format("Epoch",
                            "Train Error",
                            "Valid Error",
                            "Valid / Train Error",
                            "Time",
                            "Best Error",
                            "Learning Rate")
                     )
        self.log_msg(LINE)

        prev_weights = None
        prev_train_err = np.inf
        self.best_valid_err = np.inf


        for i in range(self.current_epoch, self.max_epochs):
            start_time = time.time()

            train_err = 0
            train_batches = 0
            for batch in self.iterate_minibatches(self.x_train, self.y_train, self.batch_size, shuffle=True):
                inputs, targets = batch
                train_err += self.train_loss(inputs, targets)
                train_batches += 1
            train_err /= train_batches

            self.y_train_err_history.append(train_err)

            valid_err = self.valid_loss(self.x_valid, self.y_valid)

            self.y_valid_err_history.append(valid_err)
            curr_learning_rate = self.learning_rate_tensor.get_value()

            improvement = ' '
            # Check the validation error to see if it is time to exit
            if valid_err < self.best_valid_err:
                self.best_epoch = i
                self.best_valid_err = valid_err
                self.best_weights = get_all_param_values(self.network)
                #curr_learning_rate *= 1 + (1 - self.learning_adjustment)
                improvement = '*'
            else:
                #curr_learning_rate *= self.learning_adjustment
                #if curr_learning_rate < self.base_learning_rate:
                #    curr_learning_rate = self.base_learning_rate
                pass

            curr_learning_rate = np.cast['float32'](curr_learning_rate)
            self.learning_rate_tensor.set_value(np.float32(curr_learning_rate))

            if (i - self.best_epoch) > self.patience:
                self.log_msg("Early Stopping")
                self.log_msg("Best validation error: %0.6f at epoch %d" %
                             (self.best_valid_err, self.best_epoch))
                break

            self.log_msg(
                         "|{:^20}|{:^20}|{:^20}|{:^30}|{:^20}|{:^20}|{:^20}|".
                         format(i,
                                "%0.6f" % np.cast['float32'](train_err),
                                "%0.6f" % np.cast['float32'](valid_err),
                                "%0.6f" % (np.cast['float32'](valid_err) / np.cast['float32'](train_err)),
                                "%0.6f" % (time.time() - start_time),
                                improvement,
                                "%0.6E" % self.learning_rate_tensor.get_value())
                         )
            i += 1

    def test_model(self, max_epochs=200):
        """This method is used to execute a basic test of the model.  When trained with a single example
        and validated against this example the validation error should be approximately zero"""
        self.build_model()
        self.load_data()

        self.max_epochs = max_epochs

        # Make the training data set only a single sample
        self.x_train = self.x_train[:1,:] 
        self.y_train = self.y_train[:1,:] 

        # Test the network is capable of memorising the training sample
        self.x_valid = self.x_train[:1,:] 
        self.y_valid = self.y_train[:1,:] 

        self.train()

        np.testing.assert_almost_equal(self.best_valid_err,0,
                                       err_msg="Single sample did not memorize")


    def save_progress(self):
        save_data = {
                'weights': self.best_weights,
                'curr_epoch': self.current_epoch,
                'best_valid_err': self.best_valid_err,
                'best_train_err': self.best_train_err,
                'y_train_err_history': self.y_train_err_history,
                'y_valid_err_history': self.y_valid_err_history,
            }
        with open(self.save_params_filename, "wb") as f:
            pickle.dump(save_data, f)

    def load_progress(self, filename):
        with open(filename, "rb") as f:
            load_data = pickle.load(f)

        self.best_weights = load_data['weights']
        self.current_epoch = load_data['curr_epoch']
        self.best_valid_err = load_data['best_valid_err']
        self.best_train_err = load_data['best_train_err']
        self.y_train_err_history = load_data['y_train_err_history']
        self.y_valid_err_history = load_data['y_valid_err_history']

    # Logging functionality
    def _prepare_log(self):
        # Data logging
        # If the log already exists append a .x to the end of the file
        self.log_extension = DEFAULT_LOG_EXTENSION
        self.save_params_extension = DEFAULT_PKL_EXTENSION
        log_basename = self.name
        # log_basename = "%s-%d" % (log_name, hidden_units)
        if os.path.exists("{}{}".format(log_basename, self.log_extension)):
            # See if any other logs exist of the .x format
            log_iter = 0
            while os.path.exists("{}{}.{}".format(log_basename, self.log_extension, log_iter)):
                log_iter += 1

            self.log_filename = "{}{}.{}".format(log_basename, self.log_extension, log_iter)
            self.save_params_filename = "{}{}.{}".format(log_basename, self.save_params_extension, log_iter)
        else:
            self.log_filename = log_basename + self.log_extension
            self.save_params_filename = log_basename + self.save_params_extension

    def log_msg(self, msg):
        if self.verbose:
            with open(self.log_filename, "a") as f:
                f.write("%s\n" % msg)
            print(msg)
