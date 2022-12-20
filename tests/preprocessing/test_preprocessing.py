#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 14:59:02 2022

@author: zelda
"""

import os
import sys

import numpy as np
from numpy.testing import assert_array_equal
import pytest

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))
from src.preprocessing.preprocessing_base import binarizer



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
        assert_array_equal(actual,
                           expected,
                           "Output array does not match expected value")
