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

class TestLogisticRegressionClassifier:
    