#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""Functional testing of resources module"""

# Imports
from nnet.resources import load_mnist_test_images, load_mnist_test_labels, \
        load_mnist_train_images, load_mnist_train_labels
import unittest
import numpy as np


__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = 'Wednesday 31 August  14:44:48 AEST 2016'
__license__ = 'MPL v2.0'

class TestMNISTData(unittest.TestCase):

    def test_load_mnist_test_images(self):
        """Test the MNIST test set images load correctly"""

        images = load_mnist_test_images()

        self.assertEqual(images.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(images.shape[0], 10000,
                         "The test set should contain 10k images")
        self.assertEqual(images.shape[1], 28,
                         "Each test set image should be 28 x 28 pixels")
        self.assertEqual(images.shape[2], 28,
                         "Each test set image should be 28 x 28 pixels")

    def test_load_mnist_test_labels(self):
        """Test the MNIST test set labels load correctly"""

        first_label = np.zeros((10))
        first_label[7] = 1

        last_label = np.zeros((10))
        last_label[6] = 1

        labels = load_mnist_test_labels()

        self.assertEqual(labels.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(labels.shape[0], 10000,
                         "The test set should contain 10k images")
        self.assertEqual(labels.shape[1], 10,
                         "The labels should be present as a vector of length 10")
        np.testing.assert_equal(labels[0], first_label,
                                "First sample incorrectly labelled")
        np.testing.assert_equal(labels[-1], last_label,
                                "First sample incorrectly labelled")

    def test_load_mnist_train_images(self):
        """Test the MNIST training set images load correctly"""
        images = load_mnist_train_images()

        self.assertEqual(images.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(images.shape[0], 60000,
                         "The test set should contain 10k images")
        self.assertEqual(images.shape[1], 28,
                         "Each test set image should be 28 x 28 pixels")
        self.assertEqual(images.shape[2], 28,
                         "Each test set image should be 28 x 28 pixels")

    def test_load_mnist_training_labels(self):
        """Test the MNIST test set labels load correctly"""

        first_label = np.zeros((10))
        first_label[5] = 1

        last_label = np.zeros((10))
        last_label[8] = 1

        labels = load_mnist_train_labels()

        self.assertEqual(labels.dtype, np.float32,
                         'images should be type np.float32')

        self.assertEqual(labels.shape[0], 60000,
                         "The test set should contain 10k images")
        self.assertEqual(labels.shape[1], 10,
                         "The labels should be present as a vector of length 10")
        np.testing.assert_equal(labels[0], first_label,
                                "First sample incorrectly labelled")
        np.testing.assert_equal(labels[-1], last_label,
                                "First sample incorrectly labelled")


