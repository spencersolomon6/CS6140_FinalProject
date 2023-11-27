import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

class LogRegression:

    def __init__(self, random_state):
        self.model = LogisticRegression(random_state=random_state, multi_class='ovr')

    def fit(self, X, y):
        # Remove the time dimension from our data since this model is not time sensitive
        inputs = X.reshape(-1, X.shape[-1])
        targets = y.reshape(-1)

        self.model = self.model.fit(inputs, targets)

    def predict(self, X):
        inputs = X.reshape(-1, X.shape[-1])

        return self.model.predict(inputs)


