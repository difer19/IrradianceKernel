"""Clasificadores kernel compatibles con la API de scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .constants import RANDOM_STATE
from .kernels import PairwiseKernel, ScalarKernel, kernel_matrix, resolve_kernel


class KSVC(ClassifierMixin, BaseEstimator):
    """Adaptación de KSVC que conserva el nombre del proyecto de referencia."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        degree: int = 2,
        gamma: float | None = None,
        coef0: float | None = None,
        class_weight=None,
        random_state: int = RANDOM_STATE,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X_checked, y_checked = check_X_y(X, y)
        self.resolved_kernel_ = resolve_kernel(
            X_checked, self.kernel, self.degree, self.gamma, self.coef0
        )
        self.model_ = SVC(
            C=self.C,
            kernel=PairwiseKernel(self.resolved_kernel_),
            class_weight=self.class_weight,
            decision_function_shape="ovr",
            random_state=self.random_state,
        )
        self.model_.fit(X_checked, y_checked, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X_checked.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(check_array(X))

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        scores = self.model_.decision_function(check_array(X))
        if np.ndim(scores) == 1:
            scores = np.column_stack([-scores, scores])
        return scores


class KANNC(ClassifierMixin, BaseEstimator):
    """Red neuronal sobre una aproximación Nystroem del kernel."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "identity",
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 1000,
        early_stopping: bool = True,
        n_components: int = 100,
        kernel: str = "rbf",
        degree: int = 2,
        gamma: float | None = None,
        coef0: float | None = None,
        random_state: int = RANDOM_STATE,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_components = n_components
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, y):
        X_checked, y_checked = check_X_y(X, y)
        self.resolved_kernel_ = resolve_kernel(
            X_checked, self.kernel, self.degree, self.gamma, self.coef0
        )
        components = min(self.n_components, X_checked.shape[0])
        self.feature_map_ = Nystroem(
            kernel=ScalarKernel(self.resolved_kernel_),
            n_components=components,
            random_state=self.random_state,
        )
        transformed = self.feature_map_.fit_transform(X_checked)
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        self.model_.fit(transformed, y_checked)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X_checked.shape[1]
        return self

    def _transform(self, X):
        check_is_fitted(self, "feature_map_")
        return self.feature_map_.transform(check_array(X))

    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))

    def decision_function(self, X):
        probabilities = np.clip(self.predict_proba(X), 1e-12, 1.0)
        return np.log(probabilities)


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
