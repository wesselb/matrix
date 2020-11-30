import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..lowrank import LowRank
from ..matrix import Dense
from ..woodbury import Woodbury
from ..kronecker import Kronecker
from ..triangular import LowerTriangular, UpperTriangular

__all__ = []


def _raise(axis):
    raise ValueError(f"Cannot sum over axis {axis}.")


@B.dispatch(Zero)
def sum(a, axis=None):
    if axis not in [None, 0, 1]:
        _raise(axis)
    return B.cast(a.dtype, 0)


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def sum(a, axis=None):
    return B.sum(a.mat, axis=axis)


@B.dispatch(Diagonal)
def sum(a, axis=None):
    if axis is None:
        return B.sum(a.diag)
    elif axis == 0 or axis == 1:
        return a.diag
    else:
        _raise(axis)


@B.dispatch(Constant)
def sum(a, axis=None):
    if axis is None:
        return a.const * a.rows * a.cols
    elif axis == 0:
        return a.const * a.rows * B.ones(B.dtype(a.const), a.cols)
    elif axis == 1:
        return a.const * a.cols * B.ones(B.dtype(a.const), a.rows)
    else:
        _raise(axis)


@B.dispatch(LowRank)
def sum(a, axis=None):
    if axis is None:
        return B.sum(B.sum(B.matmul(a.left, a.middle), axis=0) * B.sum(a.right, axis=0))
    elif axis == 0:
        return B.sum(
            B.multiply(
                B.expand_dims(B.sum(a.left, axis=0), axis=0),
                B.matmul(a.right, a.middle, tr_b=True),
            ),
            axis=1,
        )
    elif axis == 1:
        return B.sum(
            B.multiply(
                B.matmul(a.left, a.middle),
                B.expand_dims(B.sum(a.right, axis=0), axis=0),
            ),
            axis=1,
        )
    else:
        _raise(axis)


@B.dispatch(Woodbury)
def sum(a, axis=None):
    return B.sum(a.diag, axis=axis) + B.sum(a.lr, axis=axis)


@B.dispatch(Kronecker)
def sum(a, axis=None):
    if axis is None:
        return B.sum(a.left) * B.sum(a.right)
    elif axis == 0:
        return B.kron(B.sum(a.left, axis=0), B.sum(a.right, axis=0))
    elif axis == 1:
        return B.kron(B.sum(a.left, axis=1), B.sum(a.right, axis=1))
    else:
        _raise(axis)
