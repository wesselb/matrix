import lab as B

from ..matrix import AbstractMatrix
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(AbstractMatrix, AbstractMatrix, AbstractMatrix)
def iqf_diag(a, b, c):
    """Compute the diagonal of `transpose(b) inv(a) c` where `a` is assumed
    to be positive definite.

    Args:
        a (matrix): Matrix `a`.
        b (matrix): Matrix `b`.
        c (matrix): Matrix `c`.

    Returns:
        vector: Diagonal of resulting quadratic form.
    """
    chol = B.cholesky(a)
    chol_b = B.solve(chol, b)
    if c is b:
        chol_c = chol_b
    else:
        chol_c = B.solve(chol, c)
    return B.sum(chol_b * chol_c, axis=0)


B.iqf_diag = iqf_diag


@B.dispatch(Woodbury, AbstractMatrix, AbstractMatrix)
def iqf_diag(a, b, c):
    a_inv_c = B.solve(a, c)  # Use the matrix inversion lemma.
    return B.sum(b * a_inv_c, axis=0)
