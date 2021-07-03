import lab as B

from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def negative(a: Zero):
    return a


@B.dispatch
def negative(a: Dense):
    return Dense(B.negative(a.mat))


@B.dispatch
def negative(a: Diagonal):
    return Diagonal(B.negative(a.diag))


@B.dispatch
def negative(a: Constant):
    return Constant(-a.const, a.rows, a.cols)


@B.dispatch
def negative(a: LowerTriangular):
    return LowerTriangular(B.negative(a.mat))


@B.dispatch
def negative(a: UpperTriangular):
    return UpperTriangular(B.negative(a.mat))


@B.dispatch
def negative(a: LowRank):
    return LowRank(a.left, a.right, B.negative(a.middle))


@B.dispatch
def negative(a: Woodbury):
    return Woodbury(B.negative(a.diag), B.negative(a.lr))


@B.dispatch
def negative(a: Kronecker):
    return Kronecker(B.negative(a.left), a.right)
