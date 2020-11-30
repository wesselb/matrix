import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning

__all__ = []


@B.dispatch(Zero, B.Numeric)
def power(a, b):
    return a


@B.dispatch(AbstractMatrix, B.Numeric)
def power(a, b):
    if structured(a):
        warn_upmodule(
            f"Taking an element-wise power of {a}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.power(B.dense(a), b))


@B.dispatch(Diagonal, B.Numeric)
def power(a, b):
    return Diagonal(B.power(a.diag, b))


@B.dispatch(Constant, B.Numeric)
def power(a, b):
    return Constant(B.power(a.const, b), a.rows, a.cols)


@B.dispatch(LowerTriangular, B.Numeric)
def power(a, b):
    return LowerTriangular(B.power(a.mat, b))


@B.dispatch(UpperTriangular, B.Numeric)
def power(a, b):
    return UpperTriangular(B.power(a.mat, b))


@B.dispatch(Kronecker, B.Numeric)
def power(a, b):
    return Kronecker(B.power(a.left, b), B.power(a.right, b))
