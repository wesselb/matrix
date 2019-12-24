import lab as B

from ..matrix import AbstractMatrix
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury
from ..kronecker import Kronecker
from ..shape import assert_square

__all__ = []


@B.dispatch(AbstractMatrix)
def logdet(a):
    return 2 * B.logdet(B.cholesky(a))


@B.dispatch({LowerTriangular, UpperTriangular})
def logdet(a):
    return B.sum(B.log(B.diag(a)))


@B.dispatch(Woodbury)
def logdet(a):
    # Use the matrix determinant lemma.
    pass


@B.dispatch(Kronecker)
def logdet(a):
    assert_square(a.left, f'Left factor of {a} must be square to compute '
                          f'the log-determinanta.')
    assert_square(a.right, f'Right factor of {b} must be square to compute '
                           f'the log-determinant.')
    n = B.shape(a.left)[0]
    m = B.shape(a.right)[0]
    return m * B.logdet(a.left) + n * B.logdet(a.right)
