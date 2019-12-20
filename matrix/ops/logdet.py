import lab as B

from ..matrix import AbstractMatrix
from ..woodbury import Woodbury

__all__ = []


@B.dispatch(AbstractMatrix)
def logdet(a):
    return 2 * B.sum(B.log(B.diag(B.cholesky(a))))


@B.dispatch(Woodbury)
def logdet(a):
    # Use the matrix determinant lemma.
    pass
