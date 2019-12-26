import lab as B

from ..matrix import AbstractMatrix
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(AbstractMatrix, AbstractMatrix, AbstractMatrix)
def iqf(a, b, c):
    """Compute `transpose(b) inv(a) c` where `a` is assumed to be positive
    definite.

    Args:
        a (matrix): Matrix `a`.
        b (matrix): Matrix `b`.
        c (matrix): Matrix `c`.

    Returns:
        matrix: Resulting quadratic form.
    """
    chol = B.cholesky(a)
    chol_b = B.solve(chol, b)
    if c is b:
        chol_c = chol_b
    else:
        chol_c = B.solve(chol, c)
    return B.mm(chol_b, chol_c, tr_a=True)


B.iqf = iqf


@B.dispatch(Woodbury, AbstractMatrix, AbstractMatrix)
def iqf(a, b, c):
    a_inv = B.inv(a)  # Use the matrix inversion lemma to compute the inverse.
    return B.mm(B.mm(b, a_inv, tr_a=True), c)
