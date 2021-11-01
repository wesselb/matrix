import lab as B

from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def ratio(a, b):
    """Compute the ratio between two positive-definite matrices.

    Args:
        a (matrix): Numerator.
        b (matrix): Denominator.

    Returns:
        scalar: Ratio.
    """
    chol = B.cholesky(a)
    return B.sum(B.iqf_diag(b, chol), axis=-1)


@B.dispatch
def ratio(a, b: Woodbury):
    # Split up computation to avoid assembling a low-rank matrix.
    b_inv = B.pd_inv(b)
    return B.trace(B.matmul(a, b_inv.diag)) + B.trace(B.matmul(a, b_inv.lr))


@B.dispatch
def ratio(a: Woodbury, b: Woodbury):
    # Split up computation to avoid assembling a low-rank matrix.
    b_inv = B.pd_inv(b)
    return (
        B.trace(B.matmul(a.diag, b_inv))
        + B.trace(B.matmul(a.lr, b_inv.diag))
        + B.trace(B.matmul(a.lr, b_inv.lr))
    )


B.ratio = ratio
