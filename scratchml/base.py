class BaseEstimator:

    pass


class RegressorMixin:

    # Adding score()
    def score(self, X, y):
        from .metrics.regression import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred)
