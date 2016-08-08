#! /usr/bin/env python
"""!
-----------------------------------------------------------------------------
File Name: adaptiverates.py

Purpose: A number of different adaptive rate options

Created: 04-Aug-2016 23:34:32 AEST
-----------------------------------------------------------------------------
Revision History



-----------------------------------------------------------------------------
S.D.G
"""
__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 23:34:32 AEST'
__license__ = 'MPL v2.0'

# LICENSE DETAILS############################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# IMPORTS#####################################################################
import numpy as np
##############################################################################

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

    def __call__(self):
        return self.rate

class linearRate(object):
    """Linear rate class"""

    def __init__(self, start, end, epochs):

        self.start = start
        self.end = end
        self.epochs = epochs
        self.rates = list(np.linspace(start, end, epochs))

    def __iter__(self):
        if len(self.rates) <= 0:
            raise StopIteration
        yield self.rates.pop(0)
