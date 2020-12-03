import lab as B

from plum import convert
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix
from ..lowrank import LowRank
from ..woodbury import Woodbury

__all__ = []


@B.dispatch({B.Numeric, AbstractMatrix})
def pd_inv(a):
    """Invert a positive-definite matrix.

    Args:
        a (matrix): Positive-definite matrix to invert.

    Returns:
        matrix: Inverse of `a`, which is also positive definite.
    """
    a = convert(a, AbstractMatrix)
    # The call to `cholesky_solve` will convert the identity matrix to dense, because
    # `cholesky(a)` will not have any exploitable structure. We suppress the expected
    # warning by converting `B.eye(a)` to dense here already.
    return B.cholesky_solve(B.cholesky(a), B.dense(B.eye(a)))


@B.dispatch(Diagonal)
def pd_inv(a):
    return B.inv(a)


@B.dispatch(Woodbury)
def pd_inv(a):
    diag_inv = B.inv(a.diag)
    # See comment in `inv`.
    return B.subtract(
        diag_inv,
        LowRank(
            B.matmul(diag_inv, a.lr.left),
            B.matmul(diag_inv, a.lr.right),
            B.pd_inv(B.pd_schur(a)),
        ),
    )


@B.dispatch(Kronecker)
def pd_inv(a):
    return Kronecker(B.pd_inv(a.left), B.pd_inv(a.right))


B.pd_inv = pd_inv
