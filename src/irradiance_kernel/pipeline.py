"""Pipeline integral: discretiza y y transforma X sin fuga entre folds."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .constants import RANDOM_STATE
from .discretizers import make_discretizer
from .estimators import KANNC, KSVC, KRidgeClassifier


class IrradiancePipeline(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        scaler: str = "standard",
        discretizer: str = "uniform",
        reducer: str = "pca",
        model: str = "ksvc",
        kernel: str = "rbf",
        random_state: int = RANDOM_STATE,
    ):
        self.scaler = scaler
        self.discretizer = discretizer
        self.reducer = reducer
        self.model = model
        self.kernel = kernel
        self.random_state = random_state

    def _make_scaler(self):
        options = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "normalizer": Normalizer,
        }
        try:
            return options[self.scaler]()
        except KeyError as exc:
            raise ValueError(f"Escalador desconocido: {self.scaler!r}") from exc

    def _make_model(self):
        options = {
            "ksvc": lambda: KSVC(C=1.0, kernel=self.kernel, random_state=self.random_state),
            "kannc": lambda: KANNC(kernel=self.kernel, random_state=self.random_state),
            "kridge": lambda: KRidgeClassifier(
                alpha=1.0, kernel=self.kernel, random_state=self.random_state
            ),
        }
        try:
            return options[self.model]()
        except KeyError as exc:
            raise ValueError(f"Modelo desconocido: {self.model!r}") from exc

    def fit(self, X, y):
        X_checked, y_continuous = check_X_y(X, y, y_numeric=True)
        self.discretizer_ = make_discretizer(self.discretizer, self.random_state)
        y_labels = self.discretizer_.fit_transform(y_continuous)
        self.scaler_ = self._make_scaler()
        scaled = self.scaler_.fit_transform(X_checked)
        if self.reducer == "pca":
            self.reducer_ = PCA(n_components=0.95, svd_solver="full", random_state=self.random_state)
            reduced = self.reducer_.fit_transform(scaled)
        elif self.reducer == "lda":
            max_components = min(scaled.shape[1], len(np.unique(y_labels)) - 1)
            if max_components < 1:
                raise ValueError("LDA necesita al menos dos clases")
            self.reducer_ = LinearDiscriminantAnalysis(n_components=max_components)
            reduced = self.reducer_.fit_transform(scaled, y_labels)
        else:
            raise ValueError(f"Reductor desconocido: {self.reducer!r}")
        self.model_ = self._make_model()
        self.model_.fit(reduced, y_labels)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X_checked.shape[1]
        self.class_descriptions_ = self.discretizer_.class_descriptions()
        return self

    def _transform_features(self, X):
        check_is_fitted(self, "model_")
        checked = check_array(X)
        return self.reducer_.transform(self.scaler_.transform(checked))

    def predict(self, X):
        return self.model_.predict(self._transform_features(X))

    def predict_proba(self, X):
        transformed = self._transform_features(X)
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(transformed)
        scores = self.model_.decision_function(transformed)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def decision_function(self, X):
        transformed = self._transform_features(X)
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(transformed)
        return np.log(np.clip(self.model_.predict_proba(transformed), 1e-12, 1.0))

    def transform_target(self, y):
        check_is_fitted(self, "discretizer_")
        return self.discretizer_.transform(y)

    def resolved_kernel(self) -> dict:
        check_is_fitted(self, "model_")
        return self.model_.resolved_kernel_.as_dict()
