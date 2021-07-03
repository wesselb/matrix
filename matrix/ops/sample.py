import lab as B

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
    chol = B.cholesky(a)
    return B.matmul(chol, B.randn(B.dtype_float(a), B.shape(chol)[1], num))


B.sample = sample


@B.dispatch
def sample(a: Woodbury, num=1):
    return B.sample(a.diag, num=num) + B.sample(a.lr, num=num)
