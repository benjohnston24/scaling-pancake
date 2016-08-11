#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Provide adaptive learning and momentum rates for the training module
"""

# Imports
import numpy as np

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 23:34:32 AEST'
__license__ = 'MPL v2.0'


class fixedRate(object):
    """Fixed Rate class

    Parameters
    ------------

    rate: The fixed rate to be used

    Returns
    ------------

    the rate when the object is called
    """

    def __init__(self, rate):
        self.rate = rate
        self.__name__ = "Fixed: {}".format(self.rate)

    def __call__(self):
        return self.rate


class linearRate(object):
    """Linear rate class"""

    def __init__(self, start, end, epochs):

        self.start = start
        self.end = end
        self.epochs = epochs
        self.rates = list(np.linspace(start, end, epochs))
        self.rate = self.rates[0]
        self.__name__ = "Linear: {}, {}, {}".format(start, end, epochs)

    def __iter__(self):
        for rate in self.rates:
            self.rate = self.rates.pop(0) 
            yield self.rate
