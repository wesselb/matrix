import lab as B
from plum import Union
from wbml.warning import warn_upmodule

from ..constant import Zero
from ..diagonal import Diagonal
from ..matrix import AbstractMatrix, Dense
from ..tiledblocks import TiledBlocks
from ..util import ToDenseWarning

__all__ = []


@B.dispatch
def block_diag(
    block: Union[B.Numeric, AbstractMatrix], *blocks: Union[B.Numeric, AbstractMatrix]
):
    """Construct a block-diagonal matrix, preserving structure when possible.

    Args:
        *blocks (list): Blocks of the diagonal.

    Returns:
        matrix: Assembled matrix with as much structured as possible.
    """
    blocks = (block,) + blocks

    if len(blocks) == 1:
        # There is just one block. Return it.
        return blocks[0]

    res = _attempt_zero(blocks)
    if res is not None:
        return res

    res = _attempt_diagonal(blocks)
    if res is not None:
        return res

    # Could not preserve any structure. Simply concatenate them all densely.
    warn_upmodule(
        "Could not preserve structure in block-diagonal matrix: converting to dense.",
        category=ToDenseWarning,
    )
    return Dense(B.block_diag(*(B.dense(block) for block in blocks)))


def _attempt_zero(blocks):
    if all([isinstance(block, Zero) for block in blocks]):
        dtype = B.dtype(blocks[0])
        rows = sum([B.shape(block, 0) for block in blocks])
        cols = sum([B.shape(block, 1) for block in blocks])
        return Zero(dtype, rows, cols)
    else:
        return None


def _attempt_diagonal(blocks):
    if all([isinstance(block, Diagonal) for block in blocks]):
        return Diagonal(B.concat(*(B.diag(block) for block in blocks), axis=0))
    else:
        return None


@B.dispatch
def block_diag(a: TiledBlocks):
    return B.block_diag(
        *[
            B.kron(B.eye(B.dtype(block), rep), block)
            for block, rep in zip(a.blocks, a.reps)
        ]
    )
