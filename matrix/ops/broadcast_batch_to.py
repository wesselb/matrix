import lab as B

from ..constant import Constant, Zero
from ..diagonal import Diagonal
from ..kronecker import Kronecker
from ..lowrank import LowRank
from ..matrix import Dense
from ..triangular import LowerTriangular, UpperTriangular
from ..woodbury import Woodbury

__all__ = []


# We don't implement a fallback method for `broadcast_batch_to`, because the function
# should always preserve the matrix type, which the fallback method will not do.


@B.dispatch
def broadcast_batch_to(a: B.Numeric, *batch: B.Int):
    """Broadcast the batch dimensions of a batched matrix.

    Args:
        a (matrix): Batched matrix.
        *batch (int): Desired batch dimensions.

    Returns:
        matrix: `a` with broadcasted batch dimensions.
    """
    return B.broadcast_to(a, *batch, *B.shape(a, -2, -1))


@B.dispatch
def broadcast_batch_to(a: Zero, *batch: B.Int):
    return Zero(a.dtype, *batch, a.rows, a.cols)


@B.dispatch
def broadcast_batch_to(a: Dense, *batch: B.Int):
    return Dense(broadcast_batch_to(B.dense(a), *batch))


@B.dispatch
def broadcast_batch_to(a: Constant, *batch: B.Int):
    return Constant(B.broadcast_to(a.const, *batch), a.rows, a.cols)


@B.dispatch
def broadcast_batch_to(a: Diagonal, *batch: B.Int):
    return Diagonal(B.broadcast_to(a.diag, *batch, B.shape(a.diag, -1)))


@B.dispatch
def broadcast_batch_to(a: LowerTriangular, *batch: B.Int):
    return LowerTriangular(B.broadcast_batch_to(a.mat, *batch))


@B.dispatch
def broadcast_batch_to(a: UpperTriangular, *batch: B.Int):
    return UpperTriangular(B.broadcast_batch_to(a.mat, *batch))


@B.dispatch
def broadcast_batch_to(a: LowRank, *batch: B.Int):
    return LowRank(
        B.broadcast_batch_to(a.left, *batch),
        B.broadcast_batch_to(a.right, *batch),
        B.broadcast_batch_to(a.middle, *batch),
    )


@B.dispatch
def broadcast_batch_to(a: Woodbury, *batch: B.Int):
    return Woodbury(
        B.broadcast_batch_to(a.diag, *batch),
        B.broadcast_batch_to(a.lr, *batch),
    )


@B.dispatch
def broadcast_batch_to(a: Kronecker, *batch: B.Int):
    return Kronecker(
        B.broadcast_batch_to(a.left, *batch),
        B.broadcast_batch_to(a.right, *batch),
    )


B.broadcast_batch_to = broadcast_batch_to
