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
def cast(dtype: B.DType, a: Zero):
    return Zero(dtype, *B.shape(a))


@B.dispatch
def cast(dtype: B.DType, a: Dense):
    return Dense(B.cast(dtype, a.mat))


@B.dispatch
def cast(dtype: B.DType, a: LowerTriangular):
    return LowerTriangular(B.cast(dtype, a.mat))


@B.dispatch
def cast(dtype: B.DType, a: UpperTriangular):
    return UpperTriangular(B.cast(dtype, a.mat))


@B.dispatch
def cast(dtype: B.DType, a: Diagonal):
    return Diagonal(B.cast(dtype, a.diag))


@B.dispatch
def cast(dtype: B.DType, a: Constant):
    return Constant(B.cast(dtype, a.const), *B.shape(a))


@B.dispatch
def cast(dtype: B.DType, a: LowRank):
    return LowRank(
        B.cast(dtype, a.left),
        B.cast(dtype, a.right),
        B.cast(dtype, a.middle),
    )


@B.dispatch
def cast(dtype: B.DType, a: Woodbury):
    return Woodbury(B.cast(dtype, a.diag), B.cast(dtype, a.lr))


@B.dispatch
def cast(dtype: B.DType, a: Kronecker):
    return Kronecker(B.cast(dtype, a.left), B.cast(dtype, a.right))


@B.dispatch
def cast(dtype: B.DType, a: TiledBlocks):
    return TiledBlocks(
        *((B.cast(dtype, block), rep) for block, rep in zip(a.blocks, a.reps)),
        axis=a.axis,
    )
