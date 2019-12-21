import lab as B
from algebra import proven

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense
from ..shape import assert_compatible, broadcast
from ..woodbury import Woodbury

__all__ = []


def _redirect(types_from, types_to):
    target_method = B.multiply.invoke(*types_to)
    B.multiply.extend(*types_from)(target_method)

    target_method = B.multiply.invoke(*reversed(types_to))
    B.multiply.extend(*reversed(types_from))(target_method)


# Zero

@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def multiply(a, b):
    assert_compatible(a, b)
    return b


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def multiply(a, b):
    assert_compatible(a, b)
    return a


# Dense

@B.dispatch(Dense, Dense)
def multiply(a, b):
    return Dense(B.multiply(a.mat, b.mat))


# Diagonal

@B.dispatch(Diagonal, Diagonal)
def multiply(a, b):
    return Diagonal(B.multiply(a.diag, b.diag))


@B.dispatch(Diagonal, AbstractMatrix)
def multiply(a, b):
    assert_compatible(a, b)
    return Diagonal(a.diag * B.diag(b))


@B.dispatch(AbstractMatrix, Diagonal)
def multiply(a, b):
    return multiply(b, a)


# Constant

@B.dispatch(Constant, Constant)
def multiply(a, b):
    assert_compatible(a, b)
    return Constant(a.const * b.const, *broadcast(a, b).as_tuple())


@B.dispatch(Constant, AbstractMatrix)
def multiply(a, b):
    assert_compatible(a, b)
    return Dense(a.const * B.dense(b))


@B.dispatch(AbstractMatrix, Constant)
def multiply(a, b):
    return multiply(b, a)


@B.dispatch(Constant, Diagonal)
def multiply(a, b):
    assert_compatible(a, b)
    return Diagonal(a.const * b.diag)


@B.dispatch(Diagonal, Constant)
def multiply(a, b):
    return multiply(b, a)


# LowRank

@B.dispatch(LowRank, LowRank)
def multiply(a, b):
    assert_compatible(a, b)

    # Pick apart the matrices.
    al, ar = B.unstack(a.left, axis=1), B.unstack(a.right, axis=1)
    bl, br = B.unstack(b.left, axis=1), B.unstack(b.right, axis=1)

    # Construct the factors.
    left = B.stack(*[B.multiply(ali, blk)
                     for ali in al
                     for blk in bl], axis=1)
    right = B.stack(*[B.multiply(arj, brl)
                      for arj in ar
                      for brl in br], axis=1)

    return LowRank(left, right)


@B.dispatch(Constant, LowRank)
def multiply(a, b):
    assert_compatible(a, b)
    return LowRank(B.multiply(a.const, b.left), b.right)


@B.dispatch(LowRank, Constant)
def multiply(a, b):
    return multiply(b, a)


# Woodbury

@B.dispatch.multi((Woodbury, Woodbury),
                  (Woodbury, AbstractMatrix))
def multiply(a, b):
    # Expand out Woodbury matrices.
    return B.add(B.multiply(a.diag, b), B.multiply(a.lr, b))


@B.dispatch(AbstractMatrix, Woodbury)
def multiply(a, b):
    return multiply(b, a)


_redirect((Woodbury, Diagonal), (AbstractMatrix, Diagonal))
_redirect((Woodbury, Constant), (Woodbury, AbstractMatrix))


# Kronecker

@B.dispatch(Kronecker, Kronecker)
def multiply(a, b):
    assert (B.shape(a.left) == B.shape(b.left) and
            B.shape(a.right) == B.shape(b.right)), \
        f'Kronecker products {a} and {b} must be compatible, ' \
        f'but they are not.'
    assert_compatible(a.left, b.left)
    assert_compatible(a.right, b.right)
    return Kronecker(B.multiply(a.left, b.left),
                     B.multiply(a.right, b.right))


@B.dispatch(Constant, Kronecker)
def multiply(a, b):
    assert_compatible(a, b)
    return Kronecker(a.const * b.left, b.right)


@B.dispatch(Kronecker, Constant)
def multiply(a, b):
    return multiply(b, a)
