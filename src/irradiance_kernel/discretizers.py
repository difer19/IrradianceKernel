"""Discretizadores de irradiancia que pueden ajustarse dentro de cada fold."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from .constants import RANDOM_STATE


def _as_1d(y) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("La irradiancia debe contener valores finitos")
    return values


class _CentroidDiscretizer(BaseEstimator):
    """Base interna: asigna valores nuevos al centro de irradiancia más próximo."""

    def _set_centers(self, centers) -> None:
        ordered = np.sort(np.unique(np.asarray(centers, dtype=float)))
        if ordered.size < 2:
            raise ValueError("Se necesitan al menos dos clases de irradiancia")
        self.centers_ = ordered
        self.n_classes_ = int(ordered.size)
        midpoints = (ordered[:-1] + ordered[1:]) / 2.0
        self.edges_ = np.r_[-np.inf, midpoints, np.inf]

    def transform(self, y):
        check_is_fitted(self, "centers_")
        values = _as_1d(y)
        return np.argmin(np.abs(values[:, None] - self.centers_[None, :]), axis=1)

    def fit_transform(self, y, *_):
        return self.fit(y).transform(y)

    def class_descriptions(self) -> list[dict]:
        check_is_fitted(self, "centers_")
        descriptions = []
        for label, center in enumerate(self.centers_):
            lower = None if not np.isfinite(self.edges_[label]) else float(self.edges_[label])
            upper = None if not np.isfinite(self.edges_[label + 1]) else float(self.edges_[label + 1])
            descriptions.append(
                {"label": int(label), "center": float(center), "lower": lower, "upper": upper}
            )
        return descriptions


class UniformTargetDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def fit(self, y, *_):
        values = _as_1d(y)
        if self.n_bins < 2:
            raise ValueError("n_bins debe ser al menos 2")
        if np.isclose(values.min(), values.max()):
            raise ValueError("No se puede discretizar un objetivo constante")
        self.edges_ = np.linspace(values.min(), values.max(), self.n_bins + 1)
        self.centers_ = (self.edges_[:-1] + self.edges_[1:]) / 2.0
        self.n_classes_ = self.n_bins
        return self

    def transform(self, y):
        check_is_fitted(self, "edges_")
        values = _as_1d(y)
        return np.clip(np.digitize(values, self.edges_[1:-1]), 0, self.n_bins - 1).astype(int)

    def fit_transform(self, y, *_):
        return self.fit(y).transform(y)

    def class_descriptions(self) -> list[dict]:
        check_is_fitted(self, "edges_")
        return [
            {
                "label": int(index),
                "center": float(self.centers_[index]),
                "lower": float(self.edges_[index]),
                "upper": float(self.edges_[index + 1]),
            }
            for index in range(self.n_bins)
        ]


class KMeansTargetDiscretizer(_CentroidDiscretizer, TransformerMixin):
    def __init__(self, n_clusters: int = 5, random_state: int = RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, y, *_):
        values = _as_1d(y)
        model = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=self.random_state)
        model.fit(values[:, None])
        self._set_centers(model.cluster_centers_.reshape(-1))
        return self


class AgglomerativeTargetDiscretizer(_CentroidDiscretizer, TransformerMixin):
    def __init__(self, n_clusters: int = 5, linkage: str = "ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, y, *_):
        values = _as_1d(y)
        labels = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage).fit_predict(
            values[:, None]
        )
        centers = [values[labels == label].mean() for label in np.unique(labels)]
        self._set_centers(centers)
        return self


class DBSCANTargetDiscretizer(_CentroidDiscretizer, TransformerMixin):
    def __init__(self, min_samples: int = 5, eps_quantile: float = 0.90):
        self.min_samples = min_samples
        self.eps_quantile = eps_quantile

    def fit(self, y, *_):
        values = _as_1d(y)
        self.mean_ = float(values.mean())
        self.scale_ = float(values.std()) or 1.0
        standardized = ((values - self.mean_) / self.scale_)[:, None]
        neighbors = min(self.min_samples, values.size)
        distances, _ = NearestNeighbors(n_neighbors=neighbors).fit(standardized).kneighbors(standardized)
        base_eps = float(np.quantile(distances[:, -1], self.eps_quantile))
        candidates = [base_eps]
        if base_eps <= 0:
            candidates = []
        candidates.extend(float(np.quantile(distances[:, -1], q)) for q in np.linspace(0.5, 0.95, 10))

        best = None
        for eps in sorted(set(value for value in candidates if value > 0)):
            labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(standardized)
            valid = sorted(set(labels) - {-1})
            if len(valid) < 2:
                continue
            noise = int(np.sum(labels == -1))
            score = (abs(len(valid) - 5), noise, abs(eps - base_eps))
            if best is None or score < best[0]:
                best = (score, eps, labels, valid)
        if best is None:
            raise ValueError("DBSCAN no encontró al menos dos grupos de irradiancia")

        _, self.eps_, labels, valid = best
        centers = [values[labels == label].mean() for label in valid]
        self.noise_count_ = int(np.sum(labels == -1))
        self._set_centers(centers)
        return self


def make_discretizer(name: str, random_state: int = RANDOM_STATE):
    factories = {
        "uniform": lambda: UniformTargetDiscretizer(5),
        "kmeans": lambda: KMeansTargetDiscretizer(5, random_state=random_state),
        "dbscan": lambda: DBSCANTargetDiscretizer(5, 0.90),
        "agglomerative": lambda: AgglomerativeTargetDiscretizer(5),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"Discretizador desconocido: {name!r}") from exc
