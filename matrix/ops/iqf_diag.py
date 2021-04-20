import lab as B

from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def iqf_diag(a, b, c):
    """Compute the diagonal of `transpose(b) inv(a) c` where `a` is assumed
    to be positive definite.

    Args:
        a (matrix): Matrix `a`.
        b (matrix): Matrix `b`.
        c (matrix, optional): Matrix `c`. Defaults to `b`.

    Returns:
        vector: Diagonal of resulting quadratic form.
    """
    chol = B.cholesky(a)
    chol_b = B.solve(chol, b)
    if c is b:
        chol_c = chol_b
    else:
        chol_c = B.solve(chol, c)
    return B.matmul_diag(chol_b, chol_c, tr_a=True)


B.iqf_diag = iqf_diag


@B.dispatch
def iqf_diag(a, b):
    return iqf_diag(a, b, b)


@B.dispatch
def iqf_diag(a: Woodbury, b, c):
    return B.matmul_diag(b, B.solve(a, c), tr_a=True)