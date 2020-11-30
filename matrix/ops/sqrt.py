import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(Zero)
def sqrt(a):
    return a


@B.dispatch(AbstractMatrix)
def sqrt(a):
    if structured(a):
        warn_upmodule(
            f"Taking an element-wise square root of {a}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.sqrt(B.dense(a)))


@B.dispatch(Diagonal)
def sqrt(a):
    return Diagonal(B.sqrt(a.diag))


@B.dispatch(Constant)
def sqrt(a):
    return Constant(B.sqrt(a.const), a.rows, a.cols)


@B.dispatch(LowerTriangular)
def sqrt(a):
    return LowerTriangular(B.sqrt(a.mat))


@B.dispatch(UpperTriangular)
def sqrt(a):
    return UpperTriangular(B.sqrt(a.mat))


@B.dispatch(Kronecker)
def sqrt(a):
    return Kronecker(B.sqrt(a.left), B.sqrt(a.right))
