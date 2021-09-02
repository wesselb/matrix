from typing import Union

import lab as B
from lab.shape import Shape
from plum import convert

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, AbstractMatrix
from ..shape import expand_and_broadcast
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def shape_batch(a: B.Numeric):
    return B.shape_batch(convert(a, AbstractMatrix))


@B.dispatch
def shape_batch(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.shape(a.mat)[:-2]


@B.dispatch
def shape_batch(a: Diagonal):
    return B.shape(a.diag)[:-1]


@B.dispatch
def shape_batch(a: Zero):
    return a.batch


@B.dispatch
def shape_batch(a: Constant):
    return B.shape(a.const)


@B.dispatch
def shape_batch(a: LowRank):
    return expand_and_broadcast(
        Shape(*B.shape_batch(a.left)),
        Shape(*B.shape_batch(a.middle)),
        Shape(*B.shape_batch(a.right)),
    )


@B.dispatch
def shape_batch(a: Woodbury):
    return expand_and_broadcast(
        Shape(*B.shape_batch(a.lr)), Shape(*B.shape_batch(a.diag))
    )


@B.dispatch
def shape_batch(a: Kronecker):
    return expand_and_broadcast(
        Shape(*B.shape_batch(a.left)), Shape(*B.shape_batch(a.right))
    )


B.shape_batch = shape_batch
