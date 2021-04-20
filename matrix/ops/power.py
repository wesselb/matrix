import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def power(a: Zero, b: B.Numeric):
    return a


@B.dispatch
def power(a: AbstractMatrix, b: B.Numeric):
    if structured(a):
        warn_upmodule(
            f"Taking an element-wise power of {a}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.power(B.dense(a), b))


@B.dispatch
def power(a: Diagonal, b: B.Numeric):
    return Diagonal(B.power(a.diag, b))


@B.dispatch
def power(a: Constant, b: B.Numeric):
    return Constant(B.power(a.const, b), a.rows, a.cols)


@B.dispatch
def power(a: LowerTriangular, b: B.Numeric):
    return LowerTriangular(B.power(a.mat, b))


@B.dispatch
def power(a: UpperTriangular, b: B.Numeric):
    return UpperTriangular(B.power(a.mat, b))


@B.dispatch
def power(a: Kronecker, b: B.Numeric):
    return Kronecker(B.power(a.left, b), B.power(a.right, b))