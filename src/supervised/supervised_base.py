#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 16:51:00 2022

@author: zelda
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# This is used only to help check how the algorith performs.
from src.preprocessing.preprocessing_base import min_max_scaler


class LogisticRegression:
    def __init__(self):
        self.weights = np.zeros([0])
        self.bias = 0

    def linear_transform(self, input_parameters: np.array):
        return (input_parameters @ self.weights) + self.bias

    def logistic_equation(self,
                          input_parameters: np.array):
        return (1 / (1 + np.exp(-(self.linear_transform(input_parameters)))))

    def _gradient_descent(self,
                          input_parameters: np.array,
                          labels: np.array,
                          tuning_parameter: float = 0.01):
        h_x = self.logistic_equation(input_parameters)
        h_x_minus_y = h_x - labels

        final_matrix = input_parameters
        for index in range(final_matrix.shape[0]):
            final_matrix[index, :] = h_x_minus_y[index] \
                * final_matrix[index, :]

        final_matrix *= tuning_parameter
        alpha_sums = final_matrix.sum(axis=0)

        for index in range(len(self.weights)):
            self.weights[index] -= alpha_sums[index]
        return self.weights

    # TODO: rebuild using the _gradient_descent function.
    def fit(self,
            input_parameters: np.array,
            labels: np.array,
            iteration_cap: int = 100000,  # Not yet implemented
            alpha_init: float = 0.05):

        # Verify labels are either 1 or 0
        if set(labels) | set([0, 1]) == set([0, 1]):
            pass
        else:
            raise ValueError("The set of labels must only contain 0 and 1.")

        # Coerce labels into a numpy array in the event they are passed in as a
        # list.
        labels = np.array(labels)
        self.input_parameters = input_parameters
        # initialize weights
        self.weights = np.zeros(input_parameters.shape[1])

        # Initialize overall cost list.
        cost_list = []
        cost_delta = 10  # dummy value

        # Begin loop to train our models weights.
        iteration = 0
        # while (cost_delta > 1e-10) or not (cost_delta < 0):
        while iteration < iteration_cap:
            predicted_values = self.logistic_equation(self.input_parameters)
            self.predictions = np.array([1 if x >= 0.5 else 0
                                         for x in predicted_values])
            cost_vector = -(labels) * (np.log10(predicted_values)) \
                - (1 - labels) * np.log10(1 - predicted_values)
            overall_cost = cost_vector.mean()
            # Track costs and convergence optimization
            cost_list.append(overall_cost)
            if len(cost_list) >= 2:
                cost_delta = cost_list[-2] - cost_list[-1]
                print(cost_delta, '     ', iteration)
            # Update weights
            grad_descent = []
            for j in range(self.weights.shape[0]):
                for i in range(self.input_parameters.shape[0]):
                    grad_descent.append((self.logistic_equation(
                        input_parameters[i,]) - labels[i])
                            * input_parameters[i, j])
                self.weights[j] = self.weights[j] \
                    - alpha_init * sum(grad_descent)
                grad_descent = []
            iteration += 1
        self.cost_list = cost_list


# iris = datasets.load_iris()
# iris
# x = iris['data'][:, 2:]
# y = iris['target']
# y[y == 2] = 1
# y
# x = min_max_scaler(x, axis=1)


# iris_class = LogisticRegression()
# iris_class.fit(x, y, alpha_init=0.3, iteration_cap=10000)
# len(iris_class.cost_list)

# plt.scatter(range(len(iris_class.cost_list)), iris_class.cost_list)
# plt.show()

# pred_match = []
# for i in range(len(y)):
#     if y[i] == iris_class.predictions[i]:
#         pred_match.append(1)
#     else:
#         pred_match.append(0)
# color = ['red' if x == 1 else 'blue' for x in y]
# color_match = ['red' if x == 1 else 'blue' for x in pred_match]

# plt.scatter(x[:, 0], x[:, 1], color=color)
# plt.scatter(x[:, 0], x[:, 1], color=color_match)

if __name__ == '__main__':
    pass
