import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero
from ..diagonal import Diagonal
from ..matrix import Dense
from ..util import ToDenseWarning

__all__ = []


def block(*rows):
    """Construct a matrix from its blocks, preserving structure when possible.

    Assumes that every row has an equal number of blocks and that the sizes
    of the blocks align to form a grid.

    Args:
        *rows (list): Rows of the block matrix.

    Returns:
        matrix: Assembled matrix with as much structured as possible.
    """
    # Check whether the result is just zeros.
    if all([all([isinstance(x, Zero) for x in row]) for row in rows]):
        # Determine the resulting data type and shape.
        dtype = B.dtype(rows[0][0])
        grid_rows = sum([B.shape(row[0])[0] for row in rows])
        grid_cols = sum([B.shape(x)[1] for x in rows[0]])

        return Zero(dtype, grid_rows, grid_cols)

    # Check whether the result is diagonal.
    diagonal = True
    diagonal_blocks = []

    for r in range(len(rows)):
        for c in range(len(rows[0])):
            block_shape = B.shape(rows[r][c])
            if r == c:
                # Keep track of all the diagonal blocks.
                diagonal_blocks.append(rows[r][c])

                # All blocks on the diagonal must be diagonal or zero.
                if not isinstance(rows[r][c], (Diagonal, Zero)):
                    diagonal = False

                # All blocks on the diagonal must be square.
                if not block_shape[0] == block_shape[1]:
                    diagonal = False
            else:
                # All blocks not on the diagonal must be zero.
                if not isinstance(rows[r][c], Zero):
                    diagonal = False

    if diagonal:
        return Diagonal(B.concat(*[B.diag(x) for x in diagonal_blocks]))

    # Could not preserve any structure. Simply concatenate them all densely.
    warn_upmodule(
        "Could not preserve structure in block matrix: converting to dense.",
        category=ToDenseWarning,
    )
    return Dense(B.concat2d(*[[B.dense(x) for x in row] for row in rows]))


B.block = block
