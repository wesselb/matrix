import lab as B
from algebra import proven
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import assert_compatible, broadcast
from ..triangular import LowerTriangular, UpperTriangular
from ..util import redirect, ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _reverse_call(*types):
    @B.multiply.extend(*reversed(types))
    def multiply(a, b):
        return multiply(b, a)


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


@B.dispatch(AbstractMatrix, AbstractMatrix)
def multiply(a, b):
    if structured(a, b):
        warn_upmodule(
            f"Multiplying {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.multiply(B.dense(a), B.dense(b)))


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
    # In the case of broadcasting, `B.diag(b)` will not get the diagonal of the
    # broadcasted version of `b`, so we exercise extra caution in that case.
    rows, cols = B.shape(b)
    if rows == 1 or cols == 1:
        b_diag = B.squeeze(B.dense(b))
    else:
        b_diag = B.diag(b)
    return Diagonal(a.diag * b_diag)


_reverse_call(Diagonal, AbstractMatrix)


# Constant


@B.dispatch(Constant, Constant)
def multiply(a, b):
    assert_compatible(a, b)
    return Constant(a.const * b.const, *broadcast(a, b).as_tuple())


@B.dispatch(Constant, AbstractMatrix)
def multiply(a, b):
    assert_compatible(a, b)
    if structured(b):
        warn_upmodule(
            f"Multiplying {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(a.const * B.dense(b))


@B.dispatch(Constant, Diagonal)
def multiply(a, b):
    assert_compatible(a, b)
    return Diagonal(a.const * b.diag)


_reverse_call(Constant, AbstractMatrix)
_reverse_call(Constant, Diagonal)


# LowerTriangular


@B.dispatch(LowerTriangular, LowerTriangular)
def multiply(a, b):
    return LowerTriangular(B.multiply(a.mat, b.mat))


@B.dispatch(LowerTriangular, AbstractMatrix)
def multiply(a, b):
    # TODO: Optimise away `B.dense` call.
    return LowerTriangular(a.mat * B.dense(b))


@B.dispatch(LowerTriangular, Constant)
def multiply(a, b):
    return LowerTriangular(B.multiply(a.mat, b.const))


_reverse_call(LowerTriangular, AbstractMatrix)
_reverse_call(LowerTriangular, Constant)

redirect(B.multiply, (LowerTriangular, Diagonal), (AbstractMatrix, Diagonal))


# UpperTriangular


@B.dispatch(UpperTriangular, UpperTriangular)
def multiply(a, b):
    return UpperTriangular(B.multiply(a.mat, b.mat))


@B.dispatch(UpperTriangular, LowerTriangular)
def multiply(a, b):
    return Diagonal(B.multiply(B.diag(a), B.diag(b)))


@B.dispatch(UpperTriangular, AbstractMatrix)
def multiply(a, b):
    # TODO: Optimise away `B.dense` call.
    return UpperTriangular(a.mat * B.dense(b))


@B.dispatch(UpperTriangular, Constant)
def multiply(a, b):
    return UpperTriangular(B.multiply(a.mat, b.const))


_reverse_call(UpperTriangular, LowerTriangular)
_reverse_call(UpperTriangular, AbstractMatrix)
_reverse_call(UpperTriangular, Constant)

redirect(B.multiply, (UpperTriangular, Diagonal), (AbstractMatrix, Diagonal))


# LowRank


@B.dispatch(LowRank, LowRank)
def multiply(a, b):
    assert_compatible(a, b)

    if structured(a.left, a.right, b.left, b.right):
        warn_upmodule(
            f"Multiplying {a} and {b}: converting factors to dense.",
            category=ToDenseWarning,
        )
    al, am, ar = B.dense(a.left), B.dense(a.middle), B.dense(a.right)
    bl, bm, br = B.dense(b.left), B.dense(b.middle), B.dense(b.right)

    # Pick apart the matrices.
    al, ar = B.unstack(al, axis=1), B.unstack(ar, axis=1)
    bl, br = B.unstack(bl, axis=1), B.unstack(br, axis=1)
    am = [B.unstack(x, axis=0) for x in B.unstack(am, axis=0)]
    bm = [B.unstack(x, axis=0) for x in B.unstack(bm, axis=0)]

    # Construct the factors.
    left = B.stack(*[B.multiply(ali, blk) for ali in al for blk in bl], axis=1)
    right = B.stack(*[B.multiply(arj, brl) for arj in ar for brl in br], axis=1)
    middle = B.stack(
        *[
            B.stack(*[amij * bmkl for amij in ami for bmkl in bmk], axis=0)
            for ami in am
            for bmk in bm
        ],
        axis=0,
    )

    return LowRank(left, right, middle)


@B.dispatch(Constant, LowRank)
def multiply(a, b):
    assert_compatible(a, b)
    return LowRank(b.left, b.right, B.multiply(a.const, b.middle))


@B.dispatch(LowRank, Constant)
def multiply(a, b):
    return multiply(b, a)


# Woodbury


@B.dispatch(Woodbury, AbstractMatrix)
def multiply(a, b):
    # Expand out Woodbury matrices.
    return B.add(B.multiply(a.diag, b), B.multiply(a.lr, b))


@B.dispatch(AbstractMatrix, Woodbury)
def multiply(a, b):
    return multiply(b, a)


redirect(B.multiply, (Woodbury, Woodbury), (Woodbury, AbstractMatrix), reverse=False)
redirect(B.multiply, (Woodbury, Diagonal), (AbstractMatrix, Diagonal))
redirect(B.multiply, (Woodbury, Constant), (Woodbury, AbstractMatrix))
redirect(B.multiply, (Woodbury, LowerTriangular), (Woodbury, AbstractMatrix))
redirect(B.multiply, (Woodbury, UpperTriangular), (Woodbury, AbstractMatrix))


# Kronecker


@B.dispatch(Kronecker, Kronecker)
def multiply(a, b):
    left_compatible = B.shape(a.left) == B.shape(b.left)
    right_compatible = B.shape(a.right) == B.shape(b.right)
    assert (
        left_compatible and right_compatible
    ), f"Kronecker products {a} and {b} must be compatible, but they are not."
    assert_compatible(a.left, b.left)
    assert_compatible(a.right, b.right)
    return Kronecker(B.multiply(a.left, b.left), B.multiply(a.right, b.right))


@B.dispatch(Constant, Kronecker)
def multiply(a, b):
    assert_compatible(a, b)
    return Kronecker(a.const * b.left, b.right)


@B.dispatch(Kronecker, Constant)
def multiply(a, b):
    return multiply(b, a)
