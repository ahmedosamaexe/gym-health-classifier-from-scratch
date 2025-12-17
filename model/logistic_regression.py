import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, lr=0.01, epochs=1500):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y, y_hat):
        eps = 1e-9
        return -np.mean(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))

    def fit(self, X, y):
        n, m = X.shape
        self.w = np.zeros(m)

        for i in range(self.epochs):
            z = X @ self.w + self.b
            y_hat = self.sigmoid(z)

            dw = (1/n) * (X.T @ (y_hat - y))
            db = (1/n) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            l = self.loss(y, y_hat)
            self.losses.append(l)

            if i % 100 == 0:
                print(f"Epoch {i} | Loss: {l:.4f}")

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
