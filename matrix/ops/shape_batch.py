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


@B.dispatch(B.Numeric)
def shape_batch(a):
    return B.shape_batch(convert(a, AbstractMatrix))


@B.dispatch({Dense, LowerTriangular, UpperTriangular})
def shape_batch(a):
    return B.shape(a.mat)[:-2]


@B.dispatch(Diagonal)
def shape_batch(a):
    return B.shape(a.diag)[:-1]


@B.dispatch(Zero)
def shape_batch(a):
    return a.batch


@B.dispatch(Constant)
def shape_batch(a):
    return B.shape(a.const)


@B.dispatch(LowRank)
def shape_batch(a):
    return expand_and_broadcast(
        Shape(*B.shape_batch(a.left)),
        Shape(*B.shape_batch(a.middle)),
        Shape(*B.shape_batch(a.right))
    )


@B.dispatch(Woodbury)
def shape_batch(a):
    return expand_and_broadcast(
        Shape(*B.shape_batch(a.lr)), Shape(*B.shape_batch(a.diag))
    )


@B.dispatch(Kronecker)
def shape_batch(a):
    return expand_and_broadcast(
        Shape(*B.shape_batch(a.left)), Shape(*B.shape_batch(a.right))
    )


B.shape_batch = shape_batch
