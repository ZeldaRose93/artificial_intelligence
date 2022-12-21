#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:17:18 2022

@author: zelda
"""

import numpy as np


def binarizer(input_data, threshold):
    """
    Changes values in a np array to either 1 or 0 based on the threshold value.

    Args:
        input_data (np.array): The array of values you want to binarize
        threshold (float): the value where numbers will switch from 0 to 1.

    Returns:
        a binarized numpy array.
    """
    output_data = input_data
    output_data[output_data > threshold] = 1 + threshold
    output_data[output_data <= threshold] = 0 + threshold
    output_data = output_data - threshold
    return output_data


def mean_removal(input_data, axis=1):
    """
    Returns the input dataframe after normalizing the values by column.
    Uses the formula z = (x - mu) / sigma

    Args:
        input_data: Numpy array of the values to separate
        axis=1: Sets default normalization axis to columns

    Returns: Numpy array with same dimensions as the input and values
             standardized to mean and std.
    """
    if axis == 1:
        input_columns = np.split(input_data, input_data.shape[1], axis=axis)
        for index in range(len(input_columns)):
            inp_mean = input_columns[index].mean()
            inp_std = input_columns[index].std()
            input_columns[index] = (input_columns[index] - inp_mean) / inp_std
        output = np.concatenate(input_columns, axis=1)
        return output
    elif axis == 0:
        input_rows = np.split(input_data, input_data.shape[0], axis=axis)
        for index in range(len(input_rows)):
            inp_mean = input_rows[index].mean()
            inp_std = input_rows[index].std()
            input_rows[index] = (input_rows[index] - inp_mean) / inp_std
        output = np.concatenate(input_rows, axis=0)
        return output
