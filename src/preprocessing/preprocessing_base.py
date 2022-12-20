#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:17:18 2022

@author: zelda
"""

import numpy as np


def binarizer(input_data, threshold):
    output_data = input_data
    output_data[output_data > threshold] = 1 + threshold
    output_data[output_data <= threshold] = 0 + threshold
    output_data = output_data - threshold
    return output_data
