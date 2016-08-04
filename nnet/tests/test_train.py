#! /usr/bin/env python
"""!
-----------------------------------------------------------------------------
File Name : test_train.py

Purpose:

Created: 04-Aug-2016 23:25:33 AEST
-----------------------------------------------------------------------------
Revision History



-----------------------------------------------------------------------------
S.D.G
"""
__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 23:25:33 AEST'
__license__ = 'MPL v2.0'

# LICENSE DETAILS############################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# IMPORTS#####################################################################
import unittest
import nnet
##############################################################################

class TestRates(unittest.TestCase):
    """Test the adaptive rates"""

    def  test_fixed_rate(self):
        print(dir(nnet))
        pass
        #rate = train.adaptiverates.fixedRate(0.2)
        #self.assertEqual(rate(), 0.2)

