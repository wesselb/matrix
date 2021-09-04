import lab as B
from wbml.warning import warn_upmodule

from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..matrix import AbstractMatrix, Dense, structured
from ..triangular import LowerTriangular, UpperTriangular
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def abs(a: Zero):
    return a


@B.dispatch
def abs(a: AbstractMatrix):
    if structured(a):
        warn_upmodule(
            f"Taking an element-wise absolute value of {a}: converting to dense.",
            category=ToDenseWarning,
        )
    return Dense(B.abs(B.dense(a)))


@B.dispatch
def abs(a: Diagonal):
    return Diagonal(B.abs(a.diag))


@B.dispatch
def abs(a: Constant):
    return Constant(B.abs(a.const), a.rows, a.cols)


@B.dispatch
def abs(a: LowerTriangular):
    return LowerTriangular(B.abs(a.mat))


@B.dispatch
def abs(a: UpperTriangular):
    return UpperTriangular(B.abs(a.mat))


@B.dispatch
def abs(a: Kronecker):
    return Kronecker(B.abs(a.left), B.abs(a.right))
