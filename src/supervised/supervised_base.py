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
        self.weights = np.ones([0])

    def logistic_equation(self,
                          input_parameters: np.array
                          ):
        return 1 / (1 + np.exp(-(input_parameters @ self.weights)))

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
        # Set initial weights
        self.input_parameters = input_parameters
        self.weights = np.ones(input_parameters.shape[1])  # initialize weights

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






# This is to help train the dataset.
input_data = np.array([[5.1, -2.9, 3.3],
                       [-1.2, 7.8, -6.1],
                       [3.9, 0.4, 2.1],
                       [7.3, -9.9, -4.5]])


## Test the class
test = LogisticRegression()
test.fit(input_parameters=input_data, labels=[1, 1, 1, 0], alpha_init=0.05)
test.weights
len(test.cost_list)

equation_results = test.logistic_equation(input_data)
for value in equation_results:
    print(f"{value:.7f}")


iris = datasets.load_iris()
iris
x = iris['data'][:, 2:]
y = iris['target']
y[y == 2] = 1
y
x = min_max_scaler(x, axis=1)


iris_class = LogisticRegression()
iris_class.fit(x, y, alpha_init=0.3, iteration_cap=10000)
len(iris_class.cost_list)

plt.scatter(range(len(iris_class.cost_list)), iris_class.cost_list)
plt.show()

pred_match = []
for i in range(len(y)):
    if y[i] == iris_class.predictions[i]:
        pred_match.append(1)
    else:
        pred_match.append(0)
color = ['red' if x == 1 else 'blue' for x in y]
color_match = ['red' if x == 1 else 'blue' for x in pred_match]

plt.scatter(x[:, 0], x[:, 1], color=color)
plt.scatter(x[:, 0], x[:, 1], color=color_match)
