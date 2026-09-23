import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

# STEP 1: Define Perceptron with Sigmoid

class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.b_ = 0.0

    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # STEP 2: Training
    def fit(self, X, y):

        # Initialize weights with zeros
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0

        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                # Calculate weighted sum
                z = np.dot(xi, self.w_) + self.b_

                # Apply sigmoid
                output = self.sigmoid(z)

                # Calculate error
                error = target - output

                # Update weights
                self.w_ += self.eta * error * xi

                # Update bias
                self.b_ += self.eta * error

        return self

    # STEP 3: Prediction
    def predict_probability(self, X):
        z = np.dot(X, self.w_) + self.b_
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_probability(X)
        # Convert probability into 0 or 1
        return np.where(probabilities >= 0.5, 1, 0)
