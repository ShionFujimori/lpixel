#!/usr/bin/env python
"""kadai01.py: Implementing a perceptron learning algorithm in Python

__Author__ = "Shion Fujimori"
__Date__ = "April 4th, 2019"
__Affiliation__ = ["University of Toronto, CS Specialist", "LPixel Inc."]
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Perceptron:
    """Perceptron classifier

    === Public Attributes ===
    learning_rate:
        a hyper-parameter that controls how much we are adjusting the weights
        of the network with respect the loss gradient.
    num_iterations:
        number of iterations of the optimization loop.
    weight:
        This determines the strength of the connection of the neurons.
    bias:
        Bias neurons allow the output of an activation funcrtion to be shifted.
    """
    learning_rate: float
    num_iterations: int
    weight: np.array
    bias: np.array

    def __init__(self, learning_rate, num_iterations) -> None:
        """Initialize a new Perceptron with the provided
        <learning_rate> and <num_iterations>
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weight = np.array([0])
        self.bias = np.array([0])

    def net_input(self, x: np.array) -> np.array:
        """Calculate the input of the activation function

        :param x: input data, of shape (n_x, n_samples)
        :return: the input of the activation function
        """
        return np.dot(self.weight.T, x) + self.bias

    def predict(self, x: np.array) -> np.array:
        """Activation function.  Calculate the output prediction

        :param x: input data, of shape (n_x, n_samples)
        :return: the output prediction
        """
        return np.where(self.net_input(x) > 0, 1, -1)

    def fit(self, x: np.array, y: np.array) -> np.array:
        """Fit training data

        :param x: input data, of shape (n_x, n_samples)
        :param y: true "label" vector, of shape (1, n_samples)
        :return: the output prediction after training data
        """
        m = x.shape[1]
        y_pred = np.array
        self.weight = np.random.randn(x.shape[0], 1)*0.01
        self.bias = np.zeros((1, 1))

        for i in range(self.num_iterations):
            y_pred = self.predict(x)

            dw = (1/m)*np.dot(x, (y_pred-y).T)
            db = (1/m)*np.sum(y_pred-y)

            self.weight -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

        return y_pred


if __name__ == '__main__':
    # acquire Data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # select only "setosa" and "versicolor"
    # extract only "sepal length" and "petal length"
    X = np.delete(X, [1, 3], axis=1)
    delete_target = np.where(y == 2)
    y = np.delete(y, delete_target)
    X = np.delete(X, delete_target, axis=0)
    y = np.where(y == 0, -1, 1)

    # plot data
    setosa = np.where(y == -1)
    versicolor = np.where(y == 1)
    plt.scatter(X[setosa, 0], X[setosa, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[versicolor, 0], X[versicolor, 1],
                color='blue', marker='x', label='versicolor')
    plt.title('Training a perceptron model on the Iris dataset')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend(loc='upper left')
    plt.show()

    # training the perceptron model
    ppn = Perceptron(learning_rate=0.1, num_iterations=10)
    y_pred = ppn.fit(X.T, y)

    # plot the decision boundary and data
    x1 = np.arange(X[:, 0].min()-1, X[:, 0].max()+1, 0.01)
    x2 = np.arange(X[:, 1].min()-1, X[:, 1].max()+1, 0.01)
    xx1, xx2 = np.meshgrid(x1, x2)
    z = ppn.predict(np.array([xx1.ravel(), xx2.ravel()]))
    Z = z.reshape(xx1.shape)
    plt.contour(xx1, xx2, Z)

    setosa = np.where(y == -1)
    versicolor = np.where(y == 1)
    plt.scatter(X[setosa, 0], X[setosa, 1],
                color='red', marker='o', label='setosa')
    plt.scatter(X[versicolor, 0], X[versicolor, 1],
                color='blue', marker='x', label='versicolor')
    plt.title('Training a perceptron model on the Iris dataset')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend(loc='upper left')
    plt.show()

    # train accuracy
    print('Accuracy: ' + str(np.mean(y_pred == y) * 100) + '%')



