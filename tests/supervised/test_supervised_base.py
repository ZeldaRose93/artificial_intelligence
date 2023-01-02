#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:51:00 2022

@author: zelda
"""


import os
import sys

import numpy as np
import numpy.testing as np_test
import pytest

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.supervised.supervised_base import LogisticRegression


class TestLogisticRegressionLinearTransform():
    @pytest.fixture(autouse=True)
    def input_data(self):
        return (np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8]]))

    def test_linear_transform_1(self, input_data):
        test_class = LogisticRegression()
        test_class.weights = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        actual = test_class.linear_transform(input_data)
        expected = [0, 0, 0, 0, 0, 0, 0, 0]
        np_test.assert_allclose(actual, expected, atol=1e-6)

    def test_linear_transform_2(self, input_data):
        test_class = LogisticRegression()
        test_class.weights = np.array([1, 0.5, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8])
        actual = test_class.linear_transform(input_data)
        expected = np.array([8, 8, 8, 8, 8, 8, 8, 8])
        np_test.assert_allclose(actual, expected, atol=1e-6)


class TestLogisticRegressionLogisticEquation:
    @pytest.fixture(autouse=True)
    def input_data(self):
        return (np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8],
                         [1, 2, 3, 4, 5, 6, 7, 8]]))

    def test_logistic_equation_1(self, input_data):
        test_class = LogisticRegression()
        test_class.weights = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        actual = test_class.logistic_equation(input_data)
        expected = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        print(self.input_data)
        np_test.assert_allclose(actual,
                                expected,
                                atol=1e-6,
                                err_msg=f"Actual: \n{actual}\n\n"
                                + f"Expected: \n{expected}")

    def test_logistic_equation_2(self, input_data):
        test_class = LogisticRegression()
        test_class.weights = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        test_class.weights = test_class.weights / 36
        actual = test_class.logistic_equation(input_data)
        expected = np.array([0.73105858,
                             0.73105858,
                             0.73105858,
                             0.73105858,
                             0.73105858,
                             0.73105858,
                             0.73105858,
                             0.73105858])
        print(self.input_data)
        np_test.assert_allclose(actual,
                                expected,
                                atol=1e-6,
                                err_msg=f"Actual: \n{actual}\n\n"
                                + f"Expected: \n{expected}")


class TestGradientDescent:
    def test_gradient_descent_base(self):
        test_class = LogisticRegression()
        input_data = np.array([[0.74117647, 0.39548023, 1.],
                               [0., 1., 0.],
                               [0.6, 0.5819209, 0.87234043],
                               [1., 0., 0.17021277]])
        labels = [1, 1, 0, 0]
        test_class.weights = [0, 0, 0]
        actual = test_class._gradient_descent(input_data,
                                              labels,
                                              tuning_parameter=0.01)
        expected = np.array([-0.00429411765,
                             0.00406779665,
                             -0.00021276599999999965])
        np_test.assert_allclose(actual,
                                expected,
                                atol=1e-8,
                                err_msg="Actual weights don't match" \
                                        + "expected weights")
