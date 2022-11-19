import numpy as np 

def cross_entropy(preds, targets):
    # binary cross entropy
    y_zero_loss = targets * np.log(preds + 1e-9)
    y_one_loss = (1 - targets) * np.log(1 - preds + 1e-9)
    return -np.mean(y_zero_loss + y_one_loss)


class LogisticRegression:
    def __init__(self, lr=0.001, max_iter=200, lamb=0):
        self.lr = lr 
        self.max_iter = max_iter
        self.lamb = lamb # regularization
        self.weights = None
        self.bias = None 

    def train(self, X, y):
        N = X.shape[0]
        feats = X.shape[1]

        self.weights = np.zeros(feats)
        self.bias = 0 

        for i in range(self.max_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)
            
            dw, db = self.gradient(X, y, y_pred)
            dw /= N 
            db /= N

            if self.lamb > 0:
                dw += 2 * self.lamb * self.weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 

            loss = self.compute_loss(y, y_pred)
            print(f"loss at iteration {i} = {loss}")
      
    def gradient(self, X, y, y_pred):
        error = y_pred - y
        dw = np.dot(X.T, error)
        db = np.sum(error)
        return dw, db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(z)
        return [1 if p > 0.5 else 0 for p in y_pred]

    def compute_loss(self, y, y_pred):
        loss = cross_entropy(y_pred, y)
        if self.lamb:
            loss += self.lamb * np.sum(self.weights**2)
        return loss

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
