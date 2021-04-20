import lab as B
import numpy as np
from wbml.warning import warn_upmodule

from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def sample(a, num=1):  # pragma: no cover
    """Sample from covariance matrices.

    Args:
        a (tensor): Covariance matrix to sample from.
        num (int): Number of samples.

    Returns:
        tensor: Samples as rank 2 column vectors.
    """
    # Convert integer data types to floats.
    dtype = B.dtype(a)
    if B.issubdtype(B.dtype(a), np.integer):
        warn_upmodule(
            "Data type of covariance matrix is integer: sampling floats anyway."
        )
        dtype = float

    # Perform sampling operation.
    chol = B.cholesky(a)
    return B.matmul(chol, B.randn(dtype, B.shape(chol)[1], num))


B.sample = sample


@B.dispatch
def sample(a: Woodbury, num=1):
    return B.sample(a.diag, num=num) + B.sample(a.lr, num=num)