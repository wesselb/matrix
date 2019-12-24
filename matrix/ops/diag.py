import warnings

import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _diag_len(a):
    return B.minimum(*B.shape(a))


@B.dispatch(Zero)
def diag(a):
    return B.zeros(B.dtype(a), _diag_len(a))


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def diag(a):
    return B.diag(a.mat)


@B.dispatch(Diagonal)
def diag(a):
    return a.diag


@B.dispatch(Constant)
def diag(a):
    return a.const * B.ones(B.dtype(a), _diag_len(a))


@B.dispatch(LowRank)
def diag(a):
    if structured(a.left, a.right):
        warnings.warn(f'Getting the diagonal of {a}: '
                      f'converting the factors to dense.',
                      category=ToDenseWarning)
    diag_len = _diag_len(a)
    return B.sum(B.multiply(B.dense(a.left)[:diag_len, :],
                            B.dense(a.right)[:diag_len, :]), axis=1)


@B.dispatch(Woodbury)
def diag(a):
    return B.diag(a.diag) + B.diag(a.lr)


@B.dispatch(Kronecker)
def diag(a):
    return B.kron(B.diag(a.left), B.diag(a.right))
