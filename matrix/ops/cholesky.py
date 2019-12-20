import lab as B

from ..constant import Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..shape import assert_square

__all__ = []


def _assert_square_cholesky(a):
    assert_square(a, 'Can only take the Cholesky decomposition of square '
                     'matrices.')


@B.dispatch(Dense)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = Dense(B.cholesky(B.reg(a.mat)))
    return a.cholesky


@B.dispatch(Diagonal)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = Diagonal(B.sqrt(a.diag))
    return a.cholesky


@B.dispatch(Constant)
def cholesky(a):
    _assert_square_cholesky(a)
    if a.cholesky is None:
        a.cholesky = Constant(B.sqrt(a.const), a.rows, a.cols)
    return B.sqrt(a.const) * B.ones(B.dtype(a), a.rows, 1)


@B.dispatch(LowRank)
def cholesky(a):
    _assert_square_cholesky(a)
    assert a.symmetric, 'Can only take the Cholesky decomposition of ' \
                        'symmetric matrices.'
    return a.left


@B.dispatch(Kronecker)
def cholesky(a):
    return Kronecker(B.cholesky(a.left), B.cholesky(a.right))
