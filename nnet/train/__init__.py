#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Network training functionality
"""

# Imports
from . import adaptiverates
import theano
from lasagne.layers import InputLayer, DenseLayer

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

    def __init__(self):
        self.input_var = None
        self.output_var = None

    def model(self):
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

    def build_model(self):
        pass

    def iterate_minibatches(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save_params(self):
        pass

    def load_params(self):
        pass
