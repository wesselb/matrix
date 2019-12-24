import lab as B
import warnings
from algebra import proven

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import assert_compatible, broadcast
from ..woodbury import Woodbury
from ..util import ToDenseWarning
from ..triangular import LowerTriangular, UpperTriangular

__all__ = []


def _reverse_call(*types):
    @B.add.extend(*reversed(types))
    def add(a, b):
        return add(b, a)


# Zero

@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def add(a, b):
    assert_compatible(a, b)
    return a


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def add(a, b):
    assert_compatible(a, b)
    return b


# Dense

@B.dispatch(AbstractMatrix, AbstractMatrix)
def add(a, b):
    if structured(a) and structured(b):
        warnings.warn(f'Adding {a} and {b}: converting to dense.',
                      category=ToDenseWarning)
    return Dense(B.add(B.dense(a), B.dense(b)))


@B.dispatch(Dense, Dense)
def add(a, b):
    return Dense(B.add(a.mat, b.mat))


# Diagonal

@B.dispatch(Diagonal, Diagonal)
def add(a, b):
    return Diagonal(B.add(a.diag, b.diag))


# Constant

@B.dispatch(Constant, Constant)
def add(a, b):
    assert_compatible(a, b)
    return Constant(a.const + b.const, *broadcast(a, b).as_tuple())


@B.dispatch(Constant, AbstractMatrix)
def add(a, b):
    if structured(b):
        warnings.warn(f'Adding {a} and {b}: converting to dense.',
                      category=ToDenseWarning)
    return Dense(a.const + B.dense(b))


_reverse_call(Constant, AbstractMatrix)


# LowerTriangular

@B.dispatch(LowerTriangular, LowerTriangular)
def add(a, b):
    return LowerTriangular(a.mat + b.mat)


@B.dispatch(LowerTriangular, Diagonal)
def add(a, b):
    # TODO: Optimise away `B.dense` call.
    return LowerTriangular(a.mat + B.dense(b))


_reverse_call(LowerTriangular, Diagonal)


# UpperTriangular

@B.dispatch(UpperTriangular, UpperTriangular)
def add(a, b):
    return UpperTriangular(a.mat + b.mat)


@B.dispatch(UpperTriangular, LowerTriangular)
def add(a, b):
    return Dense(a.mat + b.mat)


@B.dispatch(UpperTriangular, Diagonal)
def add(a, b):
    return UpperTriangular(a.mat + B.dense(b))


_reverse_call(UpperTriangular, Diagonal)
_reverse_call(UpperTriangular, LowerTriangular)


# LowRank

@B.dispatch(LowRank, LowRank)
def add(a, b):
    assert_compatible(a, b)
    if structured(a.left, a.right, b.left, b.right):
        warnings.warn(f'Adding {a} and {b}: converting factors to dense.',
                      category=ToDenseWarning)
    return LowRank(B.concat(B.dense(a.left), B.dense(b.left), axis=1),
                   B.concat(B.dense(a.right), B.dense(b.right), axis=1))


@B.dispatch(LowRank, Constant)
def add(a, b):
    assert_compatible(a, b)
    if structured(a.left, a.right):
        warnings.warn(f'Adding {b} and {a}: converting the factors to dense.',
                      category=ToDenseWarning)
    dtype = B.dtype(a)
    rows, cols = B.shape(a)
    ones_left = B.ones(dtype, rows, 1)
    ones_right = B.ones(dtype, cols, 1)
    return LowRank(B.concat(ones_left, B.dense(a.left), axis=1),
                   B.concat(b.const * ones_right, B.dense(a.right), axis=1))


@B.dispatch(Constant, LowRank)
def add(a, b):
    return add(b, a)


@B.dispatch(LowRank, Diagonal)
def add(a, b):
    return Woodbury(b, a)


@B.dispatch(Diagonal, LowRank)
def add(a, b):
    return Woodbury(a, b)


# Woodbury

@B.dispatch(Woodbury, Woodbury)
def add(a, b):
    return Woodbury(a.diag + b.diag, a.lr + b.lr)


@B.dispatch(Woodbury, Diagonal)
def add(a, b):
    return Woodbury(a.diag + b, a.lr)


@B.dispatch(Diagonal, Woodbury)
def add(a, b):
    return add(b, a)


@B.dispatch.multi((Woodbury, Constant),
                  (Woodbury, LowRank))
def add(a, b):
    return Woodbury(a.diag, a.lr + b)


@B.dispatch.multi((Constant, Woodbury),
                  (LowRank, Woodbury))
def add(a, b):
    return add(b, a)
