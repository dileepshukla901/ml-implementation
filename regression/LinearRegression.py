# -*- coding: utf-8 -*-
"""
@author: dileepshukla901
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionModel:
    """
    Linear Regression Model Class
    
    """
    def __init__(self):
        pass

    def fit(self, X, y, lr = 0.01, n_iters = 1000, Verbose=False, get_para=False, algo='gd'):
        self.algo = algo
        if (algo == 'gd'):
            """

            Trains a linear regression model using gradient descent

            """

            (n_samples, self.n_features) = X.shape
            self.num_sample = n_samples
            self.W = np.zeros(shape=(self.n_features,1))
            self.b = 0
            self.weights = []
            self.bias = []
            self.X = X
            self.y = y
            self.costs = []

            for i in range(n_iters):
                """"
                1. Compute y hat
                """
                y_hat = (np.dot(X, self.W) + self.b)[:,0]

                """
                2. find Cost error
                """
                cost = (1 / n_samples) * np.sum((y_hat - y)**2)
                self.costs.append(cost)

                """
                3. Verbose: Description of cost at each iteration
                """
                if Verbose == True:
                    print("Cost at iteration {0}: {1}".format(i,cost))

                """
                4. Updating the derivative
                """
                dw = (2 / n_samples) * np.dot(X.T, (y_hat - y).reshape(-1,1))
                db = (2 / n_samples) * np.sum((y_hat - y)) 

                """"
                5. Updating weights and bias
                """
                self.W = self.W - lr * dw
                self.b = self.b - lr * db

                """
                6. Save the weights for visualisation
                """
                self.weights.append(self.W)
                self.bias.append(self.b)
            print('Training Completed.....')
            if get_para == True:
                return self.W, self.b, self.costs
            
        elif(algo == 'ols'):
            
            (n_samples, self.n_features) = X.shape
            self.num_sample = n_samples
            self.y = y
            
            X = np.c_[np.ones((n_samples)), X]
            self.X = X
            self.W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
            self.b = 0
            
            print('Training Completed.....')
            if get_para == True:
                return self.W, self.b

    def predict(self, X_test):
        """
        1. Predicting the values by using Linear Regression Model
        """
        if(self.algo == 'ols'):
            (n_samples_test, n_feature_test) = X_test.shape
            X_test = np.c_[np.ones((n_samples_test)), X_test]
            return (np.dot(X_test, self.W))
        elif(self.algo == 'gd'):  
            return (np.dot(X_test, self.W)+self.b)[:,0]
    def plot_cost(self):
        """
        1. Plotting cost with each iteration 
        """
        if(self.algo == 'gd'):
            #Plot the cost function...
            plt.title('Cost Function J')
            plt.xlabel('No. of iterations')
            plt.ylabel('Cost')
            plt.plot(self.costs)
            plt.show()
        elif(self.algo == 'ols'):
            print('Can not print cost while algo = "ols"')

    def plotLine(self):
        if(self.algo == 'gd'):
            max_x = np.max(self.X[:,0])
            min_x = np.min(self.X[:,0])

            xplot = np.linspace(min_x, max_x, self.num_sample)
            yplot = self.b + self.W[0] * xplot

            plt.plot(xplot, yplot, color='#58b970', label='Regression Line')

            plt.title('Visualise line fit')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.scatter(self.X[:,0],self.y)
            plt.show()
        elif(self.algo == 'ols'):
            max_x = np.max(self.X[:,1])
            min_x = np.min(self.X[:,1])

            xplot = np.linspace(min_x, max_x, self.num_sample)
            yplot = self.W[0] + self.W[1] * xplot

            plt.plot(xplot, yplot, color='#58b970', label='Regression Line')

            plt.title('Visualise line fit')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.scatter(self.X[:,1],self.y)
            plt.show()