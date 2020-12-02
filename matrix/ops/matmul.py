import lab as B
from algebra import proven
from wbml.warning import warn_upmodule
from plum import convert

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import AbstractMatrix, Dense, structured
from ..shape import get_shape
from ..triangular import LowerTriangular, UpperTriangular
from ..util import redirect, ToDenseWarning
from ..woodbury import Woodbury

__all__ = []


def _tr(a, do):
    return B.transpose(a) if do else a


def _shape_tr(a, tr_a):
    ar, ac = B.shape(a)
    if tr_a:
        return ac, ar
    else:
        return ar, ac


def _assert_composable(a, b, tr_a=False, tr_b=False):
    s1, s2 = get_shape(a, b)
    s1 = _tr(s1, tr_a)
    s2 = _tr(s2, tr_b)
    assert (
        s1[1] == s2[0]
    ), f"Objects {a} and {b} are asserted to be composable, but they are not."


# Zero


@B.dispatch(AbstractMatrix, Zero, precedence=proven())
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    ar, _ = _shape_tr(a, tr_a)
    _, bc = _shape_tr(b, tr_b)
    return Zero(b.dtype, ar, bc)


@B.dispatch(Zero, AbstractMatrix, precedence=proven())
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    ar, _ = _shape_tr(a, tr_a)
    _, bc = _shape_tr(b, tr_b)
    return Zero(b.dtype, ar, bc)


# Multiple multiplication:


@B.dispatch(object, object, object)
def matmul(a, b, c, tr_a=False, tr_b=False, tr_c=False):
    ar, ac = _shape_tr(a, tr_a)
    br, bc = _shape_tr(b, tr_b)
    cr, cc = _shape_tr(c, tr_c)

    cost_ab_first = ar * ac * bc + ar * bc * cc
    cost_bc_first = br * bc * cc + ar * br * cc

    if cost_ab_first <= cost_bc_first:
        return B.matmul(B.matmul(a, b, tr_a=tr_a, tr_b=tr_b), c, tr_b=tr_c)
    else:
        return B.matmul(a, B.matmul(b, c, tr_a=tr_b, tr_b=tr_c), tr_a=tr_a)


# Dense


@B.dispatch(Dense, Dense)
def matmul(a, b, tr_a=False, tr_b=False):
    return Dense(B.matmul(a.mat, b.mat, tr_a=tr_a, tr_b=tr_b))


# Diagonal


@B.dispatch(Diagonal, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return Diagonal(B.multiply(a.diag, b.diag))


@B.dispatch(Diagonal, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return Dense(B.expand_dims(a.diag, axis=1)) * _tr(b, tr_b)


@B.dispatch(AbstractMatrix, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    return _tr(a, tr_a) * Dense(B.expand_dims(b.diag, axis=0))


# Constant


@B.dispatch(Constant, Constant)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    return Constant(a.const * b.const * a.cols, a.rows, b.cols)


@B.dispatch(Constant, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    return LowRank(
        B.ones(B.dtype(a), a.rows, 1),
        B.expand_dims(B.sum(b, axis=0), axis=1),
        B.fill_diag(a.const, 1),
    )


@B.dispatch(AbstractMatrix, Constant)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    return LowRank(
        B.expand_dims(B.sum(a, axis=1), axis=1),
        B.ones(B.dtype(b), b.cols, 1),
        B.fill_diag(b.const, 1),
    )


redirect(B.matmul, (Constant, Diagonal), (Constant, AbstractMatrix))


# LowerTriangular


@B.dispatch(LowerTriangular, LowerTriangular)
def matmul(a, b, tr_a=False, tr_b=False):
    if not tr_a and not tr_b:
        return LowerTriangular(B.matmul(a.mat, b.mat))
    else:
        return matmul(_tr(a, tr_a), _tr(b, tr_b))


@B.dispatch(LowerTriangular, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    if structured(b):
        warn_upmodule(
            f"Matrix-multiplying {a} and {b}: converting to dense.",
            category=ToDenseWarning,
        )
    return B.matmul(B.dense(a), b, tr_a=tr_a, tr_b=tr_b)


@B.dispatch(AbstractMatrix, LowerTriangular)
def matmul(a, b, tr_a=False, tr_b=False):
    if structured(a):
        warn_upmodule(
            f"Matrix-multiplying {a} and {b}: converting to dense.",
            category=ToDenseWarning,
        )
    return B.matmul(a, B.dense(b), tr_a=tr_a, tr_b=tr_b)


@B.dispatch(LowerTriangular, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    if not tr_a:
        return LowerTriangular(B.matmul(a.mat, b))
    else:
        return matmul(_tr(a, tr_a), b)


@B.dispatch(Diagonal, LowerTriangular)
def matmul(a, b, tr_a=False, tr_b=False):
    if not tr_b:
        return LowerTriangular(B.matmul(a, b.mat))
    else:
        return matmul(a, _tr(b, tr_b))


redirect(B.matmul, (LowerTriangular, Constant), (AbstractMatrix, Constant))


# UpperTriangular


@B.dispatch(UpperTriangular, UpperTriangular)
def matmul(a, b, tr_a=False, tr_b=False):
    if not tr_a and not tr_b:
        return UpperTriangular(B.matmul(a.mat, b.mat))
    else:
        return matmul(_tr(a, tr_a), _tr(b, tr_b))


@B.dispatch(UpperTriangular, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    if not tr_a:
        return UpperTriangular(B.matmul(a.mat, b))
    else:
        return matmul(_tr(a, tr_a), b)


@B.dispatch(Diagonal, UpperTriangular)
def matmul(a, b, tr_a=False, tr_b=False):
    if not tr_b:
        return UpperTriangular(B.matmul(a, b.mat))
    else:
        return matmul(a, _tr(b, tr_b))


redirect(B.matmul, (UpperTriangular, AbstractMatrix), (LowerTriangular, AbstractMatrix))
redirect(B.matmul, (UpperTriangular, Constant), (AbstractMatrix, Constant))
redirect(
    B.matmul, (UpperTriangular, LowerTriangular), (AbstractMatrix, LowerTriangular)
)


# LowRank


@B.dispatch(LowRank, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    middle = B.matmul(a.right, b.left, tr_a=True)
    middle = B.matmul(a.middle, middle, b.middle)
    # Let `middle` be of type `Diagonal` if possible.
    if B.shape(middle) == (1, 1):
        return LowRank(a.left, b.right, Diagonal(middle[0]))
    else:
        return LowRank(a.left, b.right, middle)


@B.dispatch(LowRank, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    return LowRank(a.left, B.matmul(b, a.right, tr_a=True), a.middle)


@B.dispatch(AbstractMatrix, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a=tr_a, tr_b=tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)
    return LowRank(B.matmul(a, b.left), b.right, b.middle)


@B.dispatch(LowRank, Constant)
def matmul(a, b, tr_a=False, tr_b=False):
    return B.matmul(a, convert(b, LowRank), tr_a=tr_a, tr_b=tr_b)


@B.dispatch(Constant, LowRank)
def matmul(a, b, tr_a=False, tr_b=False):
    return B.matmul(convert(a, LowRank), b, tr_a=tr_a, tr_b=tr_b)


redirect(B.matmul, (LowRank, Diagonal), (LowRank, AbstractMatrix))
redirect(B.matmul, (LowRank, LowerTriangular), (LowRank, AbstractMatrix))
redirect(B.matmul, (LowRank, UpperTriangular), (LowRank, AbstractMatrix))


# Woodbury


@B.dispatch(Woodbury, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    # Expand out Woodbury matrices.
    return B.add(
        B.matmul(a.diag, b, tr_a=tr_a, tr_b=tr_b),
        B.matmul(a.lr, b, tr_a=tr_a, tr_b=tr_b),
    )


@B.dispatch(AbstractMatrix, Woodbury)
def matmul(a, b, tr_a=False, tr_b=False):
    # Expand out Woodbury matrices.
    return B.add(
        B.matmul(a, b.diag, tr_a=tr_a, tr_b=tr_b),
        B.matmul(a, b.lr, tr_a=tr_a, tr_b=tr_b),
    )


redirect(B.matmul, (Woodbury, Woodbury), (Woodbury, AbstractMatrix), reverse=False)
redirect(B.matmul, (Woodbury, Diagonal), (Woodbury, AbstractMatrix))
redirect(B.matmul, (Woodbury, Constant), (Woodbury, AbstractMatrix))
redirect(B.matmul, (Woodbury, LowRank), (Woodbury, AbstractMatrix))
redirect(B.matmul, (Woodbury, LowerTriangular), (Woodbury, AbstractMatrix))
redirect(B.matmul, (Woodbury, UpperTriangular), (Woodbury, AbstractMatrix))


# Kronecker


@B.dispatch(Kronecker, Kronecker)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a.left, b.left, tr_a=tr_a, tr_b=tr_b)
    _assert_composable(a.right, b.right, tr_a=tr_a, tr_b=tr_b)
    return Kronecker(
        B.matmul(a.left, b.left, tr_a=tr_a, tr_b=tr_b),
        B.matmul(a.right, b.right, tr_a=tr_a, tr_b=tr_b),
    )


def _reshape_cols(a, *indices):
    return B.transpose(B.reshape(B.transpose(a), *reversed(indices)))


_reshape_rows = B.reshape


def _kron_id_a_b(a, b):
    reshaped = _reshape_cols(b, B.shape(a)[1], -1)
    return _reshape_cols(B.matmul(a, reshaped), -1, B.shape(b)[1])


def _kron_a_id_b(a, b):
    reshaped = _reshape_rows(b, B.shape(a)[1], -1)
    return _reshape_rows(B.matmul(a, reshaped), -1, B.shape(b)[1])


@B.dispatch(Kronecker, AbstractMatrix)
def matmul(a, b, tr_a=False, tr_b=False):
    _assert_composable(a, b, tr_a, tr_b)
    a = _tr(a, tr_a)
    b = _tr(b, tr_b)

    # Extract the factors of the product.
    left = a.left
    right = a.right
    l_rows, l_cols = B.shape(left)
    r_rows, r_cols = B.shape(right)

    cost_first_left = l_rows * l_cols * r_cols + r_rows * r_cols * l_rows
    cost_first_right = r_rows * r_cols * l_cols + l_rows * l_cols * r_rows

    if cost_first_left <= cost_first_right:
        return _kron_id_a_b(right, _kron_a_id_b(left, b))
    else:
        return _kron_a_id_b(left, _kron_id_a_b(right, b))


@B.dispatch(AbstractMatrix, Kronecker)
def matmul(a, b, tr_a=False, tr_b=False):
    return B.transpose(B.matmul(b, a, tr_a=not tr_b, tr_b=not tr_a))


@B.dispatch(Kronecker, Diagonal)
def matmul(a, b, tr_a=False, tr_b=False):
    warn_upmodule(
        f"Cannot efficiently matrix-multiply {a} by {b}: "
        f"converting the Kronecker product to dense.",
        category=ToDenseWarning,
    )
    return B.matmul(B.dense(a), b, tr_a=tr_a, tr_b=tr_b)


@B.dispatch(Diagonal, Kronecker)
def matmul(a, b, tr_a=False, tr_b=False):
    warn_upmodule(
        f"Cannot efficiently matrix-multiply {a} by {b}: "
        f"converting the Kronecker product to dense.",
        category=ToDenseWarning,
    )
    return B.matmul(a, B.dense(b), tr_a=tr_a, tr_b=tr_b)


redirect(B.matmul, (Kronecker, Constant), (AbstractMatrix, Constant))
redirect(B.matmul, (Kronecker, LowRank), (AbstractMatrix, LowRank))
redirect(B.matmul, (Kronecker, Woodbury), (AbstractMatrix, Woodbury))
redirect(B.matmul, (Kronecker, LowerTriangular), (AbstractMatrix, LowerTriangular))
redirect(B.matmul, (Kronecker, UpperTriangular), (AbstractMatrix, UpperTriangular))
