import numpy as np

class RandomBaseline:
    def fit(self, X, y):
        self.labels = np.unique(y)

    def predict(self, X):
        n_samples = X.shape[0]
        return np.random.choice(self.labels, size=n_samples)

