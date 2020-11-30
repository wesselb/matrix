import lab as B

from ..matrix import AbstractMatrix
from ..diagonal import Diagonal
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury
from ..kronecker import Kronecker
from ..shape import assert_square

__all__ = []


@B.dispatch(AbstractMatrix)
def logdet(a):
    return 2 * B.logdet(B.cholesky(a))


@B.dispatch({Diagonal, LowerTriangular, UpperTriangular})
def logdet(a):
    return B.sum(B.log(B.diag(a)))


@B.dispatch(Woodbury)
def logdet(a):
    return B.logdet(a.diag) + B.logdet(a.lr.middle) + B.logdet(B.schur(a))


@B.dispatch(Kronecker)
def logdet(a):
    assert_square(
        a.left, f"Left factor of {a} must be square to compute the log-determinant."
    )
    assert_square(
        a.right,
        f"Right factor of {a} must be square to compute the log-determinant.",
    )
    n = B.shape(a.left)[0]
    m = B.shape(a.right)[0]
    return m * B.logdet(a.left) + n * B.logdet(a.right)
