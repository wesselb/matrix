import lab as B

from ..matrix import AbstractMatrix

__all__ = []


@B.dispatch(AbstractMatrix)
def logdet(a):
    return 2 * B.sum(B.log(B.diag(B.cholesky(a))))
