import lab as B
from plum import convert, Union

from ..constant import Zero, Constant
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense, AbstractMatrix
from ..tiledblocks import TiledBlocks
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


@B.dispatch
def shape_batch(a, *dims: B.Int):
    """Get the batch shape of a tensor.

    Args:
        a (tensor): Tensor.
        *dims (int, optional): Dimensions to get.

    Returns:
        object: Batch shape of `a`.
    """
    a_shape_batch = B.shape_batch(a)
    return B.squeeze(tuple(a_shape_batch[i] for i in dims))


@B.dispatch
def shape_batch(a: B.Numeric):
    return B.shape_batch(convert(a, AbstractMatrix))


@B.dispatch
def shape_batch(a: Union[Dense, LowerTriangular, UpperTriangular]):
    return B.shape(a.mat)[:-2]


@B.dispatch
def shape_batch(a: Diagonal):
    return B.shape(a.diag)[:-1]


@B.dispatch
def shape_batch(a: Zero):
    return a.batch


@B.dispatch
def shape_batch(a: Constant):
    return B.shape(a.const)


@B.dispatch
def shape_batch(a: LowRank):
    return B.shape_batch_broadcast(a.left, a.right, a.middle)


@B.dispatch
def shape_batch(a: Woodbury):
    return B.shape_batch_broadcast(a.diag, a.lr)


@B.dispatch
def shape_batch(a: Kronecker):
    return B.shape_batch_broadcast(a.left, a.right)


@B.dispatch
def shape_batch(a: TiledBlocks):
    # `shape` is directly implemented for `TiledBlocks`, and the batch shape can be
    # inferred from it.
    return B.shape(a)[:-2]


B.shape_batch = shape_batch
