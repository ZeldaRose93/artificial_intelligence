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
    else:
        raise ValueError("Axis must be 1 or 0")


def min_max_scaler(input_data, axis=1):
    """
    Scales a numpy array to the minimum and maximum values.

    Args:
        input_data (np.array): The array to normalize
        axis=1 (int): 0 for row-wise normalization, 1 for column-wise norm

    Returns:
        (np.array): Normalized matrix
    """
    if axis == 1:
        input_columns = np.split(input_data, input_data.shape[1], axis=axis)
        for index in range(len(input_columns)):
            col_max = input_columns[index].max()
            col_min = input_columns[index].min()
            input_columns[index] = (input_columns[index] - col_min) \
                / (col_max - col_min)
        output = np.concatenate(input_columns, axis=1)
        return output
    elif axis == 0:
        input_rows = np.split(input_data, input_data.shape[0], axis=axis)
        for index in range(len(input_rows)):
            row_max = input_rows[index].max()
            row_min = input_rows[index].min()
            input_rows[index] = (input_rows[index] - row_min) \
                / (row_max - row_min)
        output = np.concatenate(input_rows, axis=0)
        return output
    else:
        raise ValueError("Axis must be 1 or 0")


def least_absolute_deviation_norm(input_data, axis=1):
    """
    Normalizes a numpy matrix so that the sum of the absolute value of
    all rows or columns is equal to 1.

    Args:
        input_data: Numpy matrix to normalize
        axis: 0 to normalize by rows or 1 to normalized by columns

    Returns:
        Numpy matrix with normalized values.
    """
    if axis == 1:
        input_columns = np.split(input_data, input_data.shape[1], axis=axis)
        for index in range(len(input_columns)):
            col_sum = sum(abs(input_columns[index]))
            input_columns[index] = input_columns[index] / col_sum
        output = np.concatenate(input_columns, axis=1)
        return output
    elif axis == 0:
        input_rows = np.split(input_data, input_data.shape[0], axis=axis)
        for index in range(len(input_rows)):
            row_sum = abs(input_rows[index]).sum()
            input_rows[index] = input_rows[index] / row_sum
        output = np.concatenate(input_rows, axis=0)
        return output
    else:
        raise ValueError("Axis must be 1 or 0")


def least_squares_deviation_norm(input_data, axis=1):
    """
    Normalizes a numpy matrix so that the sum of the squares of
    all rows or columns is equal to 1.

    Args:
        input_data: Numpy matrix to normalize
        axis: 0 to normalize by rows or 1 to normalized by columns

    Returns:
        Numpy matrix with normalized values.
    """
    if axis == 1:
        input_columns = np.split(input_data, input_data.shape[1], axis=axis)
        for index in range(len(input_columns)):
            col_sum = sum((input_columns[index]) ** 2) ** 0.5
            input_columns[index] = input_columns[index] / col_sum
        output = np.concatenate(input_columns, axis=1)
        return output
    elif axis == 0:
        input_rows = np.split(input_data, input_data.shape[0], axis=axis)
        for index in range(len(input_rows)):
            row_sum = (input_rows[index] ** 2).sum() ** 0.5
            input_rows[index] = input_rows[index] / row_sum
        output = np.concatenate(input_rows, axis=0)
        return output
    else:
        raise ValueError("Axis must be 1 or 0")


class Encoder:
    def fit(self, target: np.array):
        """
        Builds dictionaries for transform and inverse_transform.

        Args:
            target: list or np.array with labels to encode

        Returns:
            copy of the encoder.
        """
        self.classes_ = np.unique(target)
        self._encoder_dict = {label: index for index, label in
                              enumerate(self.classes_)}
        self._rev_encoder_dict = {index: label for index, label in
                                  enumerate(self.classes_)}
        return self

    def transform(self, target: np.array):
        """
        Convert a vector from labels to encoded values.

        Args:
            target: the list of labels to convert

        Returns:
            Encoded np.array of encoded values.
        """
        coded_vector = [self._encoder_dict[x] for x in target]
        return coded_vector

    def inverse_transform(self, target):
        """
        Transforms encoded values back into their labels

        Args:
            target: The list of encoded labels to convert

        Returns:
            list of labels
        """
        coded_vector = [self._rev_encoder_dict[x] for x in target]
        return coded_vector
