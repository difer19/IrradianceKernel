import numpy as np
import pytest

from irradiance_kernel.constants import KERNELS
from irradiance_kernel.kernels import kernel_matrix, resolve_kernel


@pytest.mark.parametrize("name", KERNELS)
def test_kernel_matrix_is_symmetric_finite(name):
    rng = np.random.default_rng(2021)
    X = rng.normal(size=(12, 4))
    resolved = resolve_kernel(X, name)
    gram = kernel_matrix(X, resolved=resolved)
    assert gram.shape == (12, 12)
    assert np.isfinite(gram).all()
    np.testing.assert_allclose(gram, gram.T, atol=1e-10)


def test_cross_kernel_shape():
    X = np.arange(20, dtype=float).reshape(5, 4)
    Y = np.arange(12, dtype=float).reshape(3, 4)
    resolved = resolve_kernel(X, "rbf")
    assert kernel_matrix(X, Y, resolved=resolved).shape == (5, 3)
