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
def transpose(a: Zero):
    return Zero(a.dtype, a.cols, a.rows)


@B.dispatch
def transpose(a: Dense):
    return Dense(B.transpose(a.mat))


@B.dispatch
def transpose(a: Diagonal):
    return a


@B.dispatch
def transpose(a: Constant):
    return Constant(a.const, a.cols, a.rows)


@B.dispatch
def transpose(a: LowerTriangular):
    return UpperTriangular(B.transpose(a.mat))


@B.dispatch
def transpose(a: UpperTriangular):
    return LowerTriangular(B.transpose(a.mat))


@B.dispatch
def transpose(a: LowRank):
    return LowRank(a.right, a.left, B.transpose(a.middle))


@B.dispatch
def transpose(a: Woodbury):
    return Woodbury(B.transpose(a.diag), B.transpose(a.lr))


@B.dispatch
def transpose(a: Kronecker):
    return Kronecker(B.transpose(a.left), B.transpose(a.right))


@B.dispatch
def transpose(a: TiledBlocks):
    if a.axis not in {0, 1}:
        raise RuntimeError(f"Invalid axis {a.axis}.")
    return TiledBlocks(
        *((B.transpose(block), rep) for block, rep in zip(a.blocks, a.reps)),
        axis=1 - a.axis,
    )
