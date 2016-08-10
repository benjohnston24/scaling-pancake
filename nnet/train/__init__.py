#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Network training functionality
"""

# Imports
from . import adaptiverates
import theano
from lasagne.layers import InputLayer, DenseLayer, get_output, \
        get_all_params
from lasagne.objectives import squared_error, aggregate
from lasagne.updates import nesterov_momentum
import numpy as np

# Details of the script

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 23:30:01 AEST'
__license__ = 'MPL v2.0'


__all__ = [
        "adaptiverates",
        "trainBase",
        ]

DEFAULT_IMAGE_SIZE = 96 ** 2


class trainBase(object):

    def __init__(self,
                 input_var=None,
                 output_var=None,
                 learning_rate=adaptiverates.fixedRate(0.01),
                 objective=squared_error,
                 updates=nesterov_momentum,
                 # momentum=0.9,  # TODO: Update later to be variable
                 patience=500,
                 ):
        self.input_var = input_var
        self.output_var = output_var
        self.objective = objective
        self.updates = updates
        self.learning_rate = learning_rate
        self.learning_rate_tensor = theano.tensor.scalar(dtype='float32')
        self.patience = patience

    def model(self):
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
        self.output_var = theano.tensor.vector('y')

        input_layer = InputLayer(
                input_var=self.input_var,
                shape=(None, DEFAULT_IMAGE_SIZE),
                name='input',
                )

        hidden_layer = DenseLayer(
                input_layer,
                num_units=500,
                name='hidden',
                )

        output_layer = DenseLayer(
                hidden_layer,
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

    def build_model(self):
        # Define the model prior to building it
        self.model()
        self._generate_layer_list()

        # Training loss
        train_prediction = get_output(self.network)
        train_loss = aggregate(self.objective(train_prediction, self.output_var))

        # Validation loss
        validation_prediction = get_output(self.network, deterministic=True)
        validation_loss = aggregate(self.objective(validation_prediction, self.output_var))

        # Update the parameters
        params = get_all_params(self.network, trainable=True)
        updates = self.updates(
                loss_or_grads=train_loss,
                params=params,
                learning_rate=self.learning_rate_tensor,
                # momentum=self.momentum  TODO update to be a tensor
                )

        # Define training loss function
        self.train_loss = theano.function(
                inputs=[self.input_var, self.output_var, self.learning_rate_tensor],
                outputs=[train_loss, self.learning_rate_tensor],
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
        pass

    def save_params(self):
        pass

    def load_params(self):
        pass
