import numpy as np 


def hinge_loss(preds, targets):
    z = 1 - preds * targets 
    result = np.where(z < 0, 0, z)
    return result 

class Support_Vector_Machine:
    def __init__(self, lr=0.001, max_iter=200, lamb=1):
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
        
        y_ = np.where(y <= 0, -1, 1)

        for i in range(self.max_iter):
            y_pred = (np.dot(X, self.weight) + self.bias
            dw, db = self.gradient(X, y_, y_pred)
            dw /= N 
            db /= N 

            self.weights -= self.lr * dw
            self.bias -= self.lr * db 
            
            loss = self.compute_loss(y_, y_pred)
            print(f"loss at iteration {i} = {loss}")

    def gradient(self, X, y, y_pred):
        dw = self.weights * 0 
        db = self.bias * 0

        for i in range(len(X)):
            if y[i] * y_pred[i] >= 1:
                dw[i] = self.lamb * self.weights[i]**2
                db[i] = 0 
            else:
                dw[i] = self.lamb * self.weights[i]**2 + y[i] * X[i]
                db[i] = y[i]
        return dw, db

    def predict(self, X):
        y_pred = np.dot(X, self.weight) + self.bias
        return np.sign(y_pred)

    def compute_loss(self, y, y_pred):
        loss = hinge_loss(y_pred, y)
        loss += 0.5 * self.lamb * np.sum(self.weights**2) 