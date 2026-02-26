import numpy as np

from ..base import BaseEstimator, RegressorMixin
from ..utils.validation import _validate_X_y, _validate_X

# from ..metrics.regression import mean_squared_error


class GDLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lr=0.01, n_iters=1000, tol=None):
        self.lr = lr
        self.n_iters = n_iters
        self.tol = None if tol is None else float(tol)

        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X, y = _validate_X_y(X, y)

        n, d = X.shape

        # START WEIGHT EQUALS 0
        w = np.zeros(d)
        b = 0  # Beta_0

        step = 0
        while step < self.n_iters:
            w_old = w.copy()
            b_old = b

            y_pred = X @ w + b  # GETTING y_pred THAT EQUALS X @ current weight
            error = y_pred - y  # mse

            # CALCULATE GRADIENTS
            w_grad = (2 / n) * X.T @ error
            b_grad = (2 / n) * error.sum()

            # DESCENT STEP
            w = w - self.lr * w_grad
            b = b - self.lr * b_grad

            if self.tol is not None:
                if np.linalg.norm(w - w_old) < self.tol and abs(b - b_old) < self.tol:
                    break

            step += 1

        self.coef_ = w
        self.intercept_ = b

        return self

    def predict(self, X):
        X = _validate_X(X)
        if self.coef_ is None:
            raise RuntimeError("call fit(X, y) first")
        return X @ self.coef_ + self.intercept_
