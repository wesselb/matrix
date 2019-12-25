import warnings

import lab as B
import lab.util
import numpy as np

from ..matrix import AbstractMatrix
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(object)
@lab.util.abstract()
def sample(a, num=1):  # pragma: no cover
    """Sample from covariance matrices.

    Args:
        a (tensor): Covariance matrix to sample from.
        num (int): Number of samples.

    Returns:
        tensor: Samples as rank 2 column vectors.
    """
    pass


B.sample = sample


@B.dispatch(AbstractMatrix)
def sample(a, num=1):
    # Convert integer data types to floats.
    dtype = B.dtype(a)
    if B.issubdtype(B.dtype(a), np.integer):
        warnings.warn('Data type of covariance matrix is integer: '
                      'sampling floats anyway.')
        dtype = float

    # Perform sampling operation.
    chol = B.cholesky(a)
    return B.matmul(chol, B.randn(dtype, B.shape(chol)[1], num))


@B.dispatch(Woodbury)
def sample(a, num=1):
    return B.sample(a.diag, num=num) + B.sample(a.lr, num=num)
