from typing import Union

import lab as B

from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..tiledblocks import TiledBlocks
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def dtype(a: Zero):
    return a.dtype


@B.dispatch
def dtype(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.dtype(a.mat)


@B.dispatch
def dtype(a: Diagonal):
    return B.dtype(a.diag)


@B.dispatch
def dtype(a: Constant):
    return B.dtype(a.const)


@B.dispatch
def dtype(a: LowRank):
    return B.dtype(a.left, a.right, a.middle)


@B.dispatch
def dtype(a: Woodbury):
    return B.dtype(a.diag, a.lr)


@B.dispatch
def dtype(a: Kronecker):
    return B.dtype(a.left, a.right)


@B.dispatch
def dtype(a: TiledBlocks):
    return B.dtype(*a.blocks)
