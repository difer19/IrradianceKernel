"""Clasificadores kernel compatibles con la API de scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .constants import RANDOM_STATE
from .kernels import kernel_matrix, resolve_kernel
from .original_estimators import KANNC, KSVC


class KRidgeClassifier(RidgeClassifier):
    """RidgeClassifier dual multiclase con una matriz Gram configurable."""

    def __init__(
        self,
        alpha: float = 1.0,
        kernel: str = "rbf",
        degree: int = 2,
        gamma: float | None = None,
        coef0: float | None = None,
        random_state: int = RANDOM_STATE,
    ):
        super().__init__(alpha=alpha)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X_checked, y_checked = check_X_y(X, y)
        self.classes_, encoded = np.unique(y_checked, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("KRidgeClassifier necesita al menos dos clases")
        self.X_fit_ = X_checked
        self.n_features_in_ = X_checked.shape[1]
        self.resolved_kernel_ = resolve_kernel(
            X_checked, self.kernel, self.degree, self.gamma, self.coef0
        )
        gram = kernel_matrix(X_checked, resolved=self.resolved_kernel_)
        targets = -np.ones((len(y_checked), len(self.classes_)), dtype=float)
        targets[np.arange(len(y_checked)), encoded] = 1.0
        if sample_weight is not None:
            weights = np.sqrt(np.asarray(sample_weight, dtype=float)).reshape(-1, 1)
            gram = weights * gram * weights.T
            targets = weights * targets
        regularized = gram + float(self.alpha) * np.eye(len(gram))
        try:
            self.dual_coef_ = np.linalg.solve(regularized, targets)
        except np.linalg.LinAlgError:
            self.dual_coef_ = np.linalg.pinv(regularized) @ targets
        return self

    def decision_function(self, X):
        check_is_fitted(self, "dual_coef_")
        cross_kernel = kernel_matrix(check_array(X), self.X_fit_, resolved=self.resolved_kernel_)
        return cross_kernel @ self.dual_coef_

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exponentials = np.exp(scores)
        return exponentials / exponentials.sum(axis=1, keepdims=True)
