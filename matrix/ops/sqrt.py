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
def sqrt(a: Zero):
    return a


@B.dispatch
def sqrt(a: AbstractMatrix):
    if structured(a):
        warn_upmodule(
            f"Taking an element-wise square root of {a}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.sqrt(B.dense(a)))


@B.dispatch
def sqrt(a: Diagonal):
    return Diagonal(B.sqrt(a.diag))


@B.dispatch
def sqrt(a: Constant):
    return Constant(B.sqrt(a.const), a.rows, a.cols)


@B.dispatch
def sqrt(a: LowerTriangular):
    return LowerTriangular(B.sqrt(a.mat))


@B.dispatch
def sqrt(a: UpperTriangular):
    return UpperTriangular(B.sqrt(a.mat))


@B.dispatch
def sqrt(a: Kronecker):
    return Kronecker(B.sqrt(a.left), B.sqrt(a.right))