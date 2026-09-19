"""Funciones kernel vectorizadas basadas en KernelUtilities.py de magohector/fkernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances


KERNEL_ALIASES = {"can": "canberra", "tru": "truncated", "rq": "rquadratic"}


@dataclass(frozen=True)
class ResolvedKernel:
    name: str
    degree: int
    gamma: float
    coef0: float

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "degree": int(self.degree),
            "gamma": float(self.gamma),
            "coef0": float(self.coef0),
        }


def _canonical(name: str) -> str:
    return KERNEL_ALIASES.get(name, name)


def _sample_rows(X: np.ndarray, limit: int = 300) -> np.ndarray:
    if len(X) <= limit:
        return X
    indices = np.linspace(0, len(X) - 1, limit, dtype=int)
    return X[indices]


def resolve_kernel(
    X,
    name: str,
    degree: int = 2,
    gamma: float | None = None,
    coef0: float | None = None,
) -> ResolvedKernel:
    """Resuelve parámetros fijos mediante heurísticas del fold de entrenamiento."""
    values = np.asarray(X, dtype=float)
    sampled = _sample_rows(values)
    sq_distances = pairwise_distances(sampled, metric="sqeuclidean")
    positive_sq = sq_distances[sq_distances > 0]
    median_sq = float(np.median(positive_sq)) if positive_sq.size else 1.0
    median_distance = float(np.sqrt(median_sq))
    canonical = _canonical(name)

    if gamma is None:
        if canonical in {"rbf", "radial_basic"}:
            gamma = 1.0 / max(2.0 * median_sq, np.finfo(float).eps)
        elif canonical == "triangle":
            gamma = max(median_distance, np.finfo(float).eps)
        elif canonical == "truncated":
            differences = np.abs(sampled[:, None, :] - sampled[None, :, :]).reshape(-1)
            positive = differences[differences > 0]
            gamma = float(np.median(positive)) if positive.size else 1.0
        elif canonical == "canberra":
            gamma = 0.5
        else:
            gamma = 1.0 / max(values.shape[1], 1)
    if coef0 is None:
        if canonical == "rquadratic":
            coef0 = max(median_sq, np.finfo(float).eps)
        elif canonical == "hyperbolic":
            coef0 = -1.0
        elif canonical == "poly":
            coef0 = 1.0
        else:
            coef0 = 0.0
    return ResolvedKernel(canonical, degree, float(gamma), float(coef0))


def kernel_matrix(X, Y=None, *, resolved: ResolvedKernel) -> np.ndarray:
    """Calcula K(X,Y) sin bucles Python por pareja de muestras."""
    left = np.asarray(X, dtype=float)
    right = left if Y is None else np.asarray(Y, dtype=float)
    name = resolved.name
    gamma = resolved.gamma
    degree = resolved.degree
    coef0 = resolved.coef0

    if name == "linear":
        matrix = left @ right.T
    elif name == "poly":
        matrix = (gamma * (left @ right.T) + coef0) ** degree
    elif name == "rbf":
        matrix = np.exp(-gamma * pairwise_distances(left, right, metric="sqeuclidean"))
    elif name == "hyperbolic":
        matrix = np.tanh(gamma * (left @ right.T) + coef0)
    elif name == "triangle":
        distances = pairwise_distances(left, right, metric="euclidean")
        matrix = np.maximum(0.0, 1.0 - distances / gamma)
    elif name == "radial_basic":
        coordinate_distances = left[:, None, :] - right[None, :, :]
        matrix = np.sum(np.exp(-gamma * coordinate_distances**2), axis=2) ** degree
    elif name == "rquadratic":
        squared = pairwise_distances(left, right, metric="sqeuclidean")
        matrix = 1.0 - squared / (squared + coef0)
    elif name == "canberra":
        numerator = np.abs(left[:, None, :] - right[None, :, :])
        denominator = np.abs(left[:, None, :]) + np.abs(right[None, :, :])
        ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        matrix = 1.0 - gamma * np.mean(ratio, axis=2)
    elif name == "truncated":
        differences = np.abs(left[:, None, :] - right[None, :, :])
        matrix = np.mean(np.maximum(0.0, 1.0 - differences / gamma), axis=2)
    else:
        raise ValueError(f"Kernel desconocido: {name!r}")
    return np.nan_to_num(matrix, nan=0.0, posinf=1e12, neginf=-1e12)


@dataclass(frozen=True)
class PairwiseKernel:
    """Callable serializable para estimadores que esperan matrices X e Y."""

    resolved: ResolvedKernel

    def __call__(self, X, Y):
        return kernel_matrix(X, Y, resolved=self.resolved)


@dataclass(frozen=True)
class ScalarKernel:
    """Callable serializable para Nystroem, que entrega dos observaciones."""

    resolved: ResolvedKernel

    def __call__(self, x, y):
        return float(
            kernel_matrix(
                np.asarray(x)[None, :], np.asarray(y)[None, :], resolved=self.resolved
            )[0, 0]
        )
