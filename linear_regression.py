import numpy as np 
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
# import seaborn as sns


def derivative(X, y, theta):
    K = len(theta)
    N = len(X)
    dtheta = [0.0] * K # Initialize the derivative vector to be a vector of zeros
    MSE = 0 # Compute the MSE as we go, just to print it for debugging

    for i in range(N):
        error = np.dot(X[i], theta) - y[i]
        for k in range(K):
            dtheta[k] += 2*X[i][k]*error/N # See the lectures to understand how this expression was derived
        MSE += error*error/N
    return dtheta, MSE


def mean_squared_error(preds, targets):
    mse = np.sum((preds - targets)**2) / len(targets)
    return mse 


class Linear_Regression:
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
            y_pred = np.dot(X, self.weights) + self.bias
            dw, db = self.gradient(X, y, y_pred)
            dw /= N 
            db /= N

            # apply a regularizer to the weights gradient
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
        y_pred = np.dot(X, self.weights) + self.bias 
        return y_pred
    
    def compute_loss(self, y, y_pred):
        loss = mean_squared_error(y_pred, y)
        if self.lamb > 0:
            loss += self.lamb * np.sum(self.weights**2)
        return loss



def main():
    X, y = make_regression(n_features=2, n_samples=200, noise=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    linear_reg = Linear_Regression(lr=0.02)
    linear_reg.train(X_train, y_train)

    y_pred = linear_reg.predict(X_test)

    mse = mean_squared_error(y_pred, y_test)
    print(f"MSE = {mse}")


if __name__ == "__main__":
    np.random.seed(15)
    main()