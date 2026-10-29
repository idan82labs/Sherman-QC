"""zs_lane_v4 selector model wrapper (pickled with the frozen model; runtime side).

sklearn HistGradientBoostingClassifier(max_depth 3, learning_rate 0.05, max_iter 300, l2_regularization 1.0,
random_state 0, early_stopping False): the zs_lane_v1 recipe with early stopping pinned off, so the fit does not
depend on the training-set size crossing sklearn's 10 000-sample auto threshold.
"""
from __future__ import annotations

import numpy as np


class GBM:
    def __init__(self, depth: int = 3, lr: float = 0.05, iters: int = 300, l2: float = 1.0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.params = dict(max_depth=depth, learning_rate=lr, max_iter=iters, l2_regularization=l2)
        self.m = HistGradientBoostingClassifier(random_state=0, early_stopping=False, **self.params)

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def predict(self, X):
        return self.m.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
