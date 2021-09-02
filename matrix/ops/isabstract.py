from typing import Union

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
def isabstract(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.isabstract(a.mat)


@B.dispatch
def isabstract(a: Diagonal):
    return B.isabstract(a.diag)


@B.dispatch
def isabstract(a: Zero):
    return False


@B.dispatch
def isabstract(a: Constant):
    return B.isabstract(a.const)


@B.dispatch
def isabstract(a: LowRank):
    return B.isabstract(a.left) or B.isabstract(a.middle) or B.isabstract(a.right)


@B.dispatch
def isabstract(a: Woodbury):
    return B.isabstract(a.diag) or B.isabstract(a.lr)


@B.dispatch
def isabstract(a: Kronecker):
    return B.isabstract(a.left) or B.isabstract(a.right)
