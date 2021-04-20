from typing import Union

import lab as B

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


def _raise(axis):
    raise ValueError(f"Cannot sum over axis {axis}.")


@B.dispatch
def sum(a: Zero, axis=None):
    if axis is None:
        return B.cast(a.dtype, 0)
    elif axis == 0:
        return B.zeros(a.dtype, a.cols)
    elif axis == 1:
        return B.zeros(a.dtype, a.rows)
    else:
        _raise(axis)


@B.dispatch
def sum(a: Union[Dense, LowerTriangular, UpperTriangular], axis=None):
    return B.sum(a.mat, axis=axis)


@B.dispatch
def sum(a: Diagonal, axis=None):
    if axis is None:
        return B.sum(a.diag)
    elif axis == 0 or axis == 1:
        return a.diag
    else:
        _raise(axis)


@B.dispatch
def sum(a: Constant, axis=None):
    if axis is None:
        return a.const * a.rows * a.cols
    elif axis == 0:
        return a.const * a.rows * B.ones(B.dtype(a.const), a.cols)
    elif axis == 1:
        return a.const * a.cols * B.ones(B.dtype(a.const), a.rows)
    else:
        _raise(axis)


@B.dispatch
def sum(a: LowRank, axis=None):
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


@B.dispatch
def sum(a: Woodbury, axis=None):
    return B.sum(a.diag, axis=axis) + B.sum(a.lr, axis=axis)


@B.dispatch
def sum(a: Kronecker, axis=None):
    if axis is None:
        return B.sum(a.left) * B.sum(a.right)
    elif axis == 0:
        return B.kron(B.sum(a.left, axis=0), B.sum(a.right, axis=0))
    elif axis == 1:
        return B.kron(B.sum(a.left, axis=1), B.sum(a.right, axis=1))
    else:
        _raise(axis)
