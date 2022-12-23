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
from src.preprocessing.preprocessing_base import min_max_scaler
from src.preprocessing.preprocessing_base import \
    least_absolute_deviation_norm
from src.preprocessing.preprocessing_base import \
    least_squares_deviation_norm


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

    def test_mean_removal_axis_value_error(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        with pytest.raises(ValueError):
            mean_removal(input_data, axis=2)

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


class TestMinMaxScaler:
    def test_min_max_scaler_shape(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = min_max_scaler(input_data, axis=0)
        act_shape = actual.shape
        exp_shape = input_data.shape
        assert act_shape == exp_shape, "Input and Output arrays don't match"

    def test_min_max_axis(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        with pytest.raises(ValueError):
            min_max_scaler(input_data, axis=2)

    def test_min_max_scaler_ax1(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = min_max_scaler(input_data, axis=1)
        expected = np.array([[0.74117647, 0.39548023, 1.],
                             [0., 1., 0.],
                             [0.6, 0.5819209, 0.87234043],
                             [1., 0., 0.17021277]])
        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-05,
                                err_msg="Output does not \
                                         match expectations.")

    def test_min_max_scaler_ax0(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = min_max_scaler(input_data, axis=0)
        expected = np.array([[1.0, 0., 0.775],
                             [0.35251799, 1., 0.],
                             [1.0, 0., 0.48571429],
                             [1., 0., 0.31395349]])
        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-05,
                                err_msg="Output does not \
                                         match expectations.")


class TestLeastAbsoluteDeviationNorm:
    def test_least_absolute_dev_shape_ax0(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_absolute_deviation_norm(input_data, axis=0)
        act_shape = actual.shape
        expected_shape = (4, 3)
        assert act_shape == expected_shape, \
            "Input and output shape don't match"

    def test_least_absolute_dev_shape_ax1(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_absolute_deviation_norm(input_data, axis=1)
        act_shape = actual.shape
        expected_shape = (4, 3)
        assert act_shape == expected_shape, \
            "Input and output shape don't match"

    def test_least_absolute_dev_axis_valerr(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        with pytest.raises(ValueError):
            least_absolute_deviation_norm(input_data, axis=3)

    def test_least_absolute_dev_values_ax0(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_absolute_deviation_norm(input_data, axis=0)
        expected = np.array([[0.45132743, -0.25663717, 0.2920354],
                            [-0.0794702, 0.51655629, -0.40397351],
                            [0.609375, 0.0625, 0.328125],
                            [0.33640553, -0.4562212, -0.20737327]])
        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-05,
                                err_msg="Actual and expected do not match.")

    def test_least_absolute_dev_values_ax1(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_absolute_deviation_norm(input_data, axis=1)
        expected = np.array([[0.29142857, -0.13809524, 0.20625],
                            [-0.06857143, 0.37142857, -0.38125],
                            [0.22285714, 0.01904762, 0.13125],
                            [0.41714286, -0.47142857, -0.28125]])
        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-05,
                                err_msg="Actual and expected do not match.")


class TestLeastSquaresDeviationNorm:
    def test_least_squares_dev_shape_ax0(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_squares_deviation_norm(input_data, axis=0)
        act_shape = actual.shape
        expected_shape = (4, 3)
        assert act_shape == expected_shape, \
            "Input and output shape don't match"

    def test_least_squares_dev_shape_ax1(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_squares_deviation_norm(input_data, axis=1)
        act_shape = actual.shape
        expected_shape = (4, 3)
        assert act_shape == expected_shape, \
            "Input and output shape don't match"

    def test_least_squares_dev_axis_valerr(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        with pytest.raises(ValueError):
            least_squares_deviation_norm(input_data, axis=3)

    def test_least_squares_dev_values_ax0(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_squares_deviation_norm(input_data, axis=0)
        expected = np.array([[0.75765788, -0.43082507, 0.49024922],
                            [-0.12030718, 0.78199664, -0.61156148],
                            [0.87690281, 0.08993875, 0.47217844],
                            [0.55734935, -0.75585734, -0.34357152]])
        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-05,
                                err_msg="Actual and expected do not match.")

    def test_least_squares_dev_values_ax1(self):
        input_data = input_data = np.array([[5.1, -2.9, 3.3],
                                           [-1.2, 7.8, -6.1],
                                           [3.9, 0.4, 2.1],
                                           [7.3, -9.9, -4.5]])
        actual = least_squares_deviation_norm(input_data, axis=1)
        expected = np.array([[0.52065217, -0.22412708, 0.38687226],
                            [-0.12250639, 0.60282455, -0.71512752],
                            [0.39814578, 0.03091408, 0.24619144],
                            [0.74524723, -0.76512347, -0.52755309]])
        np_test.assert_allclose(actual,
                                expected,
                                rtol=1e-05,
                                err_msg="Actual and expected do not match.")
