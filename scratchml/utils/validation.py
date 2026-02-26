import numpy as np


def _validate_X(X):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be 2d matrix")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be non-empty")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or inf")

    return X


def _validate_X_y(X, y):
    X = _validate_X(X)
    y = np.asarray(y, dtype=float)

    # Y TO 1D VECTOR
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()

    if y.ndim != 1:
        raise ValueError("y must have shape (n_samples,)")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")

    if y.shape[0] == 0:
        raise ValueError("y must be non-empty")

    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or inf")

    return X, y
