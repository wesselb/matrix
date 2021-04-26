import lab as B

from ..lowrank import LowRank
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def iqf(a, b, c):
    """Compute `transpose(b) inv(a) c` where `a` is assumed to be positive
    definite.

    Args:
        a (matrix): Matrix `a`.
        b (matrix): Matrix `b`.
        c (matrix, optional): Matrix `c`. Defaults to `b`.

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


@B.dispatch
def iqf(a, b):
    return iqf(a, b, b)


@B.dispatch
def iqf(a: Woodbury, b, c):
    return B.mm(b, B.pd_inv(a), c, tr_a=True)


# @B.dispatch
# def iqf(a: Woodbury, b: LowRank, c: LowRank):
#     middle_left = B.mm(b.right, b.middle, tr_b=True)
#     middle_right = B.mm(c.right, c.middle, tr_b=True)
#     return LowRank(b.right, c.right, B.iqf(a, middle_left, middle_right))
