# -*- coding: utf-8 -*-
"""
@author: dileepshukla901
"""

import numpy as np


class LogisticRegressionModel:
    def __init__(self):
        pass

    def fit(self, X, y, lr = 0.01, n_iters = 1000, Verbose=False, get_para=False, algo='gd'):
        """
        fits an logisitc regression model with gradient descent
        """

        self.intercept = np.ones((X.shape[0], 1))
        self.X = np.concatenate((self.intercept, X), axis=1)
        self.W = np.zeros(self.X.shape[1])
        self.weights = []
        self.y = y
        self.algo = algo
        self.lr = lr
        self.costs = []

        for i in range(n_iters):
            """"
            1. Compute y hat
            """
            y_hat = self.sigmoid(self.X, self.W)
            """
            2. find Cost error
            """
            cost = self.loss(y_hat, self.y)
            self.costs.append(cost)

            """
            3. Verbose: Description of cost at each iteration
            """

            if Verbose == True:
                print("Cost at iteration {0}: {1}".format(i, cost))

            """
            4. Updating the derivative
            """
            dW = self.gradient_descent(self.X, y_hat, self.y)

            """"
            5. Updating weights
            """

            self.W -= lr * dW

            """
            6. Save the weights for visualisation
            """
            self.weights.append(self.W)

        print('Training Completed.....')
        if get_para == True:
            return self.weights, self.costs

    # Sigmoid method
    def sigmoid(self, x, weight):
        z = np.dot(x, weight)
        return 1 / (1 + np.exp(-z))

    # method to calculate the Loss
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # Method for calculating the gradients
    def gradient_descent(self, X, h, y):
        return np.dot(X.T, (h - y)) / y.shape[0]


    # Method to predict the class label.
    def predict(self, X_test, treshold):
        X_test = np.concatenate((self.intercept, X_test), axis=1)
        result = self.sigmoid(X_test, self.W)
        result = result >= treshold
        y_pred = np.zeros(result.shape[0])
        for i in range(len(y_pred)):
            if result[i] == True:
                y_pred[i] = 1
            else:
                continue
        return y_pred

    def plot_cost(self):
        """
        1. Plotting cost with each iteration
        """
        plt.title('Cost Function J')
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.plot(self.costs)
        plt.show()