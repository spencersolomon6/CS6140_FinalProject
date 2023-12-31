from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class GaussianProcess:

    def __init__(self, random_state, kernel):
        if kernel is None:
            self.kernel = RBF(1.0)
        else:
            self.kernel = kernel
        self.model = GaussianProcessClassifier(kernel=self.kernel, random_state=random_state)

    def fit(self, X, y):
        inputs = X.reshape(-1, X.shape[-1])
        targets = y.reshape(-1)

        self.model = self.model.fit(inputs, targets)

    def predict(self, X):
        inputs = X.reshape(-1, X.shape[-1])

        return self.model.predict(inputs)

