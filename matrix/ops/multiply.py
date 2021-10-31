import lab as B
from algebra import proven
from wbml.warning import warn_upmodule

from .util import align_batch
from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import assert_compatible
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning, redirect
from ..woodbury import Woodbury

__all__ = []


def _reverse_call(t0, t1):
    @B.multiply.dispatch
    def multiply(a: t1, b: t0):
        return multiply(b, a)


# Zero


@B.dispatch(precedence=proven())
def multiply(a: AbstractMatrix, b: Zero):
    assert_compatible(B.shape(a), B.shape(b))
    return B.broadcast_to(b, *B.shape_broadcast(a, b))


@B.dispatch(precedence=proven())
def multiply(a: Zero, b: AbstractMatrix):
    assert_compatible(B.shape(a), B.shape(b))
    return B.broadcast_to(a, *B.shape_broadcast(a, b))


# Dense


@B.dispatch
def multiply(a: AbstractMatrix, b: AbstractMatrix):
    if structured(a, b):
        warn_upmodule(
            f"Multiplying {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(B.multiply(B.dense(a), B.dense(b)))


@B.dispatch
def multiply(a: Dense, b: Dense):
    return Dense(B.multiply(a.mat, b.mat))


# Diagonal


@B.dispatch
def multiply(a: Diagonal, b: Diagonal):
    return Diagonal(B.multiply(a.diag, b.diag))


@B.dispatch
def multiply(a: Diagonal, b: AbstractMatrix):
    assert_compatible(B.shape(a), B.shape(b))
    # In the case of broadcasting, `B.diag(b)` will get the diagonal of `b`, which may
    # not be equalt to the diagonal broadcasted version of `b`.
    rows, cols = B.shape_matrix(b)
    if rows == 1:
        # Due to broadcasting, the diagonal will be the one column.
        b_diag = B.squeeze(B.dense(b), axis=-2)
    elif cols == 1:
        # Due to broadcasting, the diagonal will be the one row.
        b_diag = B.squeeze(B.dense(b), axis=-1)
    else:
        # No broadcasting happening. Just get the diagonal.
        b_diag = B.diag(b)
    return Diagonal(B.multiply(a.diag, b_diag))


_reverse_call(Diagonal, AbstractMatrix)


# Constant


@B.dispatch
def multiply(a: Constant, b: Constant):
    assert_compatible(B.shape(a), B.shape(b))
    return Constant(B.multiply(a.const, b.const), *B.shape_broadcast(a, b))


@B.dispatch
def multiply(a: Constant, b: AbstractMatrix):
    assert_compatible(B.shape(a), B.shape(b))
    if structured(b):
        warn_upmodule(
            f"Multiplying {a} and {b}: converting to dense.", category=ToDenseWarning
        )
    return Dense(
        B.broadcast_to(
            B.multiply(
                B.expand_dims(a.const, axis=-1, times=2, ignore_scalar=True),
                B.dense(b),
            ),
            *B.shape_broadcast(a, b),
        )
    )


@B.dispatch
def multiply(a: Constant, b: Diagonal):
    assert_compatible(B.shape(a), B.shape(b))
    return Diagonal(
        B.multiply(B.expand_dims(a.const, axis=-1, ignore_scalar=True), b.diag)
    )


_reverse_call(Constant, AbstractMatrix)
_reverse_call(Constant, Diagonal)


# LowerTriangular


@B.dispatch
def multiply(a: LowerTriangular, b: LowerTriangular):
    return LowerTriangular(B.multiply(a.mat, b.mat))


@B.dispatch
def multiply(a: LowerTriangular, b: AbstractMatrix):
    # TODO: Optimise away `B.dense` call.
    return LowerTriangular(B.multiply(a.mat, B.dense(b)))


@B.dispatch
def multiply(a: LowerTriangular, b: Constant):
    return LowerTriangular(
        B.multiply(a.mat, B.expand_dims(b.const, axis=-1, times=2, ignore_scalar=True))
    )


_reverse_call(LowerTriangular, AbstractMatrix)
_reverse_call(LowerTriangular, Constant)

redirect(B.multiply, (LowerTriangular, Diagonal), (AbstractMatrix, Diagonal))


# UpperTriangular


@B.dispatch
def multiply(a: UpperTriangular, b: UpperTriangular):
    return UpperTriangular(B.multiply(a.mat, b.mat))


@B.dispatch
def multiply(a: UpperTriangular, b: LowerTriangular):
    return Diagonal(B.multiply(B.diag(a), B.diag(b)))


@B.dispatch
def multiply(a: UpperTriangular, b: AbstractMatrix):
    # TODO: Optimise away `B.dense` call.
    return UpperTriangular(B.multiply(a.mat, B.dense(b)))


@B.dispatch
def multiply(a: UpperTriangular, b: Constant):
    return UpperTriangular(
        B.multiply(a.mat, B.expand_dims(b.const, axis=-1, times=2, ignore_scalar=True))
    )


_reverse_call(UpperTriangular, LowerTriangular)
_reverse_call(UpperTriangular, AbstractMatrix)
_reverse_call(UpperTriangular, Constant)

redirect(B.multiply, (UpperTriangular, Diagonal), (AbstractMatrix, Diagonal))


# LowRank


@B.dispatch
def multiply(a: LowRank, b: LowRank):
    assert_compatible(B.shape(a), B.shape(b))

    if structured(a.left, a.right, b.left, b.right):
        warn_upmodule(
            f"Multiplying {a} and {b}: converting factors to dense.",
            category=ToDenseWarning,
        )
    al, am, ar = B.dense(a.left), B.dense(a.middle), B.dense(a.right)
    bl, bm, br = B.dense(b.left), B.dense(b.middle), B.dense(b.right)

    # Perform broadcasting of batch dimensions.
    al, am, ar, bl, bm, br = align_batch(al, am, ar, bl, bm, br)
    offset = len(B.shape_batch(al))

    # Pick apart the matrices.
    al, ar = B.unstack(al, axis=offset + 1), B.unstack(ar, axis=offset + 1)
    bl, br = B.unstack(bl, axis=offset + 1), B.unstack(br, axis=offset + 1)
    am = [B.unstack(x, axis=offset) for x in B.unstack(am, axis=offset)]
    bm = [B.unstack(x, axis=offset) for x in B.unstack(bm, axis=offset)]

    # Construct the factors.
    left = B.stack(
        *[B.multiply(ali, blk) for ali in al for blk in bl],
        axis=offset + 1,
    )
    right = B.stack(
        *[B.multiply(arj, brl) for arj in ar for brl in br],
        axis=offset + 1,
    )
    middle = B.stack(
        *[
            B.stack(*[amij * bmkl for amij in ami for bmkl in bmk], axis=offset)
            for ami in am
            for bmk in bm
        ],
        axis=offset,
    )

    return LowRank(left, right, middle)


@B.dispatch
def multiply(a: Constant, b: LowRank):
    assert_compatible(B.shape(a), B.shape(b))
    return LowRank(
        b.left,
        b.right,
        B.multiply(
            B.expand_dims(a.const, axis=-1, times=2, ignore_scalar=True),
            b.middle,
        ),
    )


@B.dispatch
def multiply(a: LowRank, b: Constant):
    return multiply(b, a)


# Woodbury


@B.dispatch
def multiply(a: Woodbury, b: AbstractMatrix):
    # Expand out Woodbury matrices.
    return B.add(B.multiply(a.diag, b), B.multiply(a.lr, b))


@B.dispatch
def multiply(a: AbstractMatrix, b: Woodbury):
    return multiply(b, a)


redirect(B.multiply, (Woodbury, Woodbury), (Woodbury, AbstractMatrix), reverse=False)
redirect(B.multiply, (Woodbury, Diagonal), (AbstractMatrix, Diagonal))
redirect(B.multiply, (Woodbury, Constant), (Woodbury, AbstractMatrix))
redirect(B.multiply, (Woodbury, LowerTriangular), (Woodbury, AbstractMatrix))
redirect(B.multiply, (Woodbury, UpperTriangular), (Woodbury, AbstractMatrix))


# Kronecker


@B.dispatch
def multiply(a: Kronecker, b: Kronecker):
    left_compatible = B.shape_matrix(a.left) == B.shape_matrix(b.left)
    right_compatible = B.shape_matrix(a.right) == B.shape_matrix(b.right)
    if not (left_compatible and right_compatible):
        raise AssertionError(
            f"Kronecker products {a} and {b} must be compatible, but they are not."
        )
    return Kronecker(B.multiply(a.left, b.left), B.multiply(a.right, b.right))


@B.dispatch
def multiply(a: Constant, b: Kronecker):
    assert_compatible(B.shape(a), B.shape(b))
    return Kronecker(
        B.multiply(
            B.expand_dims(a.const, axis=-1, times=2, ignore_scalar=True), b.left
        ),
        b.right,
    )


@B.dispatch
def multiply(a: Kronecker, b: Constant):
    return multiply(b, a)
