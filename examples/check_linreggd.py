import numpy as np
from scratchml.linear_model.gd_lin_regression import GDLinearRegression

# создаём синтетические данные
np.random.seed(42)

X = np.random.randn(200, 1)
true_w = 3.0
true_b = 5.0

y = true_w * X[:, 0] + true_b + np.random.randn(200) * 0.5

model = GDLinearRegression(lr=0.1, n_iters=2000, tol=1e-8)
model.fit(X, y)

print("coef:", model.coef_)
print("intercept:", model.intercept_)
