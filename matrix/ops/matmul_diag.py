import lab as B

from .matmul import _tr
from ..lowrank import LowRank

__all__ = []


@B.dispatch
def matmul_diag(a, b, tr_a=False, tr_b=False):
    """Compute the diagonal of the matrix product of `a` and `b`.

    Args:
        a (matrix): First matrix.
        b (matrix): Second matrix.
        tr_a (bool, optional): Transpose first matrix. Defaults to `False`.
        tr_b (bool, optional): Transpose second matrix. Defaults to `False`.

    Returns:
        vector: Diagonal of matrix product of `a` and `b`.
    """
    a = _tr(a, not tr_a)
    b = _tr(b, tr_b)
    return B.sum(B.multiply(a, b), axis=0)


B.matmul_diag = matmul_diag


@B.dispatch
def matmul_diag(a: LowRank, b: LowRank, tr_a=False, tr_b=False):
    return B.diag(B.matmul(a, b, tr_a=tr_a, tr_b=tr_b))