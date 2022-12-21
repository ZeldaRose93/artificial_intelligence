#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:59:02 2022

@author: zelda
"""

import os
import sys

import numpy as np
import numpy.testing as np_test
import pytest

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))
from src.preprocessing.preprocessing_base import binarizer
from src.preprocessing.preprocessing_base import mean_removal


class TestBinarizer:
    def test_book_example(self):
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])

        actual = binarizer(input_data, threshold=2.1)
        expected = np.array([[1., 0., 1.],
                            [0., 1., 0.],
                            [1., 0., 0.],
                            [1., 0., 0.]])
        assert actual.shape == expected.shape, \
            "Input and Output are different sizes."
        np_test.assert_array_equal(actual,
                                   expected,
                                   "Actual does not match expected")


class TestMeanRemoval:
    def test_mean_removal_shape(self):
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])
        actual = mean_removal(input_data)
        expected = np.array([[0.42462551, -0.2748757,  1.13244172],
                            [-1.59434861,  1.40579288, -1.18167831],
                            [0.04005901,  0.24346134,  0.83702214],
                            [1.12966409, -1.37437851, -0.78778554]])
        assert actual.shape == expected.shape, \
            "Input and Output are different sizes."

    def test_mean_removal_std(self):
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])
        actual = mean_removal(input_data)
        act_std = actual.std(axis=0)
        expected_std = np.array([1., 1., 1.])
        np_test.assert_allclose(act_std,
                                expected_std,
                                rtol=1e-04,
                                err_msg="Std not normalized.")

    def test_mean_removal_array(self):
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])
        actual = mean_removal(input_data)
        expected = np.array([[0.42462551, -0.2748757,  1.13244172],
                            [-1.59434861,  1.40579288, -1.18167831],
                            [0.04005901,  0.24346134,  0.83702214],
                            [1.12966409, -1.37437851, -0.78778554]])

        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-07,
                                err_msg="Actual and expected values vary.")

    def test_mean_removal_std_rowwise(self):
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])
        actual = mean_removal(input_data, axis=0)
        act_std = actual.std(axis=1)
        expected_std = np.array([1., 1., 1., 1.])
        np_test.assert_allclose(act_std,
                                expected_std,
                                rtol=1e-04,
                                err_msg="Std not normalized.")

    def test_mean_removal_array_rowwise(self):
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])
        actual = mean_removal(input_data, axis=0)
        expected = np.array([[0.95330018, -1.3813125, 0.42801232],
                            [-0.237419, 1.32607198, -1.08865298],
                            [1.23624092, -1.21291562, -0.0233253],
                            [1.34594316, -1.04890743, -0.29703573]])

        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-07,
                                err_msg="Actual and expected values vary.")
