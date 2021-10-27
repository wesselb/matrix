import lab as B
from wbml.warning import warn_upmodule

from .util import align_batch
from ..constant import Zero
from ..diagonal import Diagonal
from ..matrix import Dense
from ..util import ToDenseWarning

__all__ = []


def block(row, *rows):
    """Construct a matrix from its blocks, preserving structure when possible.

    Assumes that every row has an equal number of blocks and that the sizes
    of the blocks align to form a grid.

    Args:
        *rows (list): Rows of the block matrix.

    Returns:
        matrix: Assembled matrix with as much structured as possible.
    """
    rows = (row,) + rows

    if len(rows) == 1 and len(rows[0]) == 1:
        # There is just one block. Return it.
        return rows[0][0]

    res = _attempt_zero(rows)
    if res is not None:
        return res

    res = _attempt_diagonal(rows)
    if res is not None:
        return res

    # Could not preserve any structure. Simply concatenate them all densely.
    warn_upmodule(
        "Could not preserve structure in block matrix: converting to dense.",
        category=ToDenseWarning,
    )
    # Align batch dimensions to allow concatenation.
    rows = align_batch(*rows)
    return Dense(B.concat2d(*[[B.dense(x) for x in row] for row in rows]))


B.block = block


def _attempt_zero(rows):
    # Check whether the result is just zeros.
    if all([all([isinstance(x, Zero) for x in row]) for row in rows]):
        # Determine the resulting data type and shape.
        dtype = B.dtype(rows[0][0])
        batch = B.shape_batch_broadcast(*(x for row in rows for x in row))
        grid_rows = sum([B.shape_matrix(row[0], 0) for row in rows])
        grid_cols = sum([B.shape_matrix(x, 1) for x in rows[0]])
        return Zero(dtype, *batch, grid_rows, grid_cols)
    else:
        return None


def _attempt_diagonal(rows):
    # Check whether the result is diagonal.
    rows = align_batch(*rows)

    # Check that the blocks form a square.
    if not all([len(row) == len(rows) for row in rows]):
        return None

    # Collect the diagonal blocks.
    diagonal_blocks = []
    for r in range(len(rows)):
        for c in range(len(rows[0])):
            block_shape = B.shape_matrix(rows[r][c])
            if r == c:
                # Keep track of all the diagonal blocks.
                diagonal_blocks.append(rows[r][c])

                # All blocks on the diagonal must be diagonal or zero.
                if not isinstance(rows[r][c], (Diagonal, Zero)):
                    return None

                # All blocks on the diagonal must be square.
                if not block_shape[0] == block_shape[1]:
                    return None
            else:
                # All blocks not on the diagonal must be zero.
                if not isinstance(rows[r][c], Zero):
                    return None

    # Align the batch dimensions before concatenating.
    batch = B.shape_batch_broadcast(*(x for row in rows for x in row))
    diagonal_blocks = [B.broadcast_batch_to(x, *batch) for x in diagonal_blocks]
    return Diagonal(B.concat(*(B.diag(x) for x in diagonal_blocks), axis=-1))
