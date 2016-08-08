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
import nnet.train as train
##############################################################################

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
            with self.subTest(rate_check = rate_check):
                # Assert the same linear rate
                self.assertEqual(step, rate_check, "linear rate error {} != {}".format(step, rate_check))

class TestTrainClass(unittest.TestCase):
    """Test the training base class"""

    def test_attributes(self):
        """Test presence of attributes"""
        train_object = train.trainBase()
        self.assertTrue(hasattr(train_object, 'model'))
        self.assertTrue(hasattr(train_object, 'build_model'))
        self.assertTrue(hasattr(train_object, 'iterate_minibatches'))
        self.assertTrue(hasattr(train_object, 'train'))
        self.assertTrue(hasattr(train_object, 'predict'))
        self.assertTrue(hasattr(train_object, 'save_params'))
        self.assertTrue(hasattr(train_object, 'load_params'))

