#! /usr/bin/env python
# -*- coding: utf-8 -*-
# S.D.G

"""
Test the resources module of the package
"""

# Imports
import unittest
from unittest.mock import mock_open, patch
import nnet.resources as resources
import os
import pandas
import numpy as np
from collections import OrderedDict

__author__ = 'Ben Johnston'
__revision__ = '0.1'
__date__ = '04-Aug-2016 14:34:55 AEST'
__license__ = 'MPL v2.0'

mock_file_open = mock_open()

def assert_data_division(utest_obj, x_train, y_train, x_valid, y_valid, split_ratio, split_ratio_calculated):
    # Check equal lengths
    utest_obj.assertEqual(len(x_train), len(y_train), 'x and y train dataset lengths not equal: %d != %d' %
                          (len(x_train), len(y_train)))
    utest_obj.assertEqual(len(x_valid), len(y_valid), 'x and y valid dataset lengths not equal: %d != %d' %
                          (len(x_valid), len(y_valid)))
    # Check the correct ratios
    utest_obj.assertEqual(split_ratio_calculated, split_ratio,
                         'incorrect split ratio: %0.2f' % split_ratio_calculated)


class TestResources(unittest.TestCase):
    """Test the resources"""

    def setUp(self):
        self.train_data_extract_landmarks = pandas.DataFrame({
            'left_eye_center_x': pandas.Series([1, 1]),
            'left_eye_center_y': pandas.Series([2, 2]),
            'left_eye_inner_corner_x': pandas.Series([3, 3]),
            'right_eye_center_x': pandas.Series([4, 4]),
            'right_eye_center_y': pandas.Series([5, 5]),
            'Image': pandas.Series(["255 255 255 255", "255 255 255 255"]),
        })

    def test_resources_path(self):
        """Test the correct resources path """
        # Check the path is correct
        self.assertEqual(os.path.relpath(resources.RESOURCE_DIR, __file__),
                         '../../resources')

    def test_training_set_filename(self):
        """Test the training set filename"""
        # Check the default training set name
        self.assertEqual(os.path.basename(resources.DEFAULT_TRAIN_SET), 'training.csv')

    def test_load_training_data(self):
        """Load the training set"""
        train_data = resources.load_training_data()
        # Check the default number of training samples
        self.assertEqual(train_data.shape[0], 7049, 'incorrect number of training samples %d != %d' %
                         (train_data.shape[0], 7049))
        self.assertEqual(train_data.shape[1], 31, 'incorrect number of training features %d != %d' %
                         (train_data.shape[1], 31))

    def test_load_data(self):
        """Load the data set with landmarks extracted and training / validation sets split"""
        train_data = resources.load_data()
        self.assertEqual(len(train_data), 4)
        self.assertEqual(train_data[0].shape[0], train_data[1].shape[0])
        self.assertEqual(train_data[2].shape[0], train_data[3].shape[0])
        self.assertEqual(train_data[0].shape[1], train_data[2].shape[1])
        self.assertEqual(train_data[1].shape[1], train_data[3].shape[1])

    def test_load_data_from_different_file(self):
        """Test load_data tries to load from a different file, when not present and exception is raised"""

        with self.assertRaises(OSError):
            train_data = resources.load_data("new_training_set.csv")

    def test_remove_incomplete(self):
        """Remove incomplete data"""
        train_data = pandas.DataFrame(np.array([
                [1, 2],
                [3, 4],
                [5, np.NaN]]))
        selected_data = resources.remove_incomplete_data(train_data)
        self.assertLess(selected_data.shape[0], train_data.shape[0])
        self.assertEqual(selected_data.shape[1], 2)

    def test_image_landmark_extraction_shape(self):
        """Extract landmarks and images"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        self.assertEqual(len(x), len(y))
        self.assertEqual(x.shape[1], 4)
        self.assertEqual(y.shape[1], 5)

    def test_image_landmark_extraction_x(self):
        """Test image extraction of extract_image_landmarks"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        np.testing.assert_allclose(x[0], [0, 0, 0, 0])

    def test_image_landmark_extraction_y_0(self):
        """Test landmark extraction of extract_image_landmarks 0"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 0], np.float32((1 - 48) / 48))

    def test_image_landmark_extraction_y_1(self):
        """Test landmark extraction of extract_image_landmarks 1"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 1], np.float32((2 - 48) / 48))

    def test_image_landmark_extraction_y_2(self):
        """Test landmark extraction of extract_image_landmarks 2"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 2], np.float32((3 - 48) / 48))

    def test_image_landmark_extraction_y_3(self):
        """Test landmark extraction of extract_image_landmarks 3"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 3], np.float32((4 - 48) / 48))

    def test_image_landmark_extraction_y_4(self):
        """Test landmark extraction of extract_image_landmarks 4"""
        train_data = self.train_data_extract_landmarks
        x, y = resources.extract_image_landmarks(train_data)
        np.testing.assert_approx_equal(y[0, 4], np.float32((5 - 48) / 48))


    def test_splitting_training_data(self):
        """Test default train / valid set split"""
        train_data = pandas.DataFrame({
            'left_eye_center_x': pandas.Series([1] * 10),
            'left_eye_center_y': pandas.Series([2] * 10),
            'left_eye_inner_corner_x': pandas.Series([3] * 10),
            'right_eye_center_x': pandas.Series([4] * 10),
            'right_eye_center_y': pandas.Series([5] * 10),
            'Image': pandas.Series(["255 255 255 255"] * 10),
        })

        x, y = resources.extract_image_landmarks(train_data)

        for split_ratio in [0.5, 0.7]:
            x_train, y_train, x_valid, y_valid = \
                    resources.split_training_data(x, y, split_ratio=split_ratio)
            split_ratio_calculated = np.round(len(x_train) / (len(x_train) + len(x_valid)), 1)
            with self.subTest(split_ratio=split_ratio):
                assert_data_division(self, x_train, y_train, x_valid, y_valid, split_ratio, split_ratio_calculated)
                # Check the shape of the features
                self.assertEqual(x_train.shape[1], 4)
                self.assertEqual(y_train.shape[1], 5)
