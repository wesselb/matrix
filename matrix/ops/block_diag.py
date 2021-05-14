import lab as B
from wbml.warning import warn_upmodule

from ..constant import Zero
from ..diagonal import Diagonal
from ..matrix import Dense
from ..util import ToDenseWarning

__all__ = []


def block_diag(*blocks):
    """Construct a block-diagonal matrix, preserving structure when possible.

    Args:
        *blocks (list): Blocks of the diagonal.

    Returns:
        matrix: Assembled matrix with as much structured as possible.
    """
    rows = []
    for i in range(len(blocks)):
        row = []
        for j in range(len(blocks)):
            if i == j:
                row.append(blocks[i])
            else:
                row.append(
                    Zero(
                        B.dtype(blocks[i]),
                        B.shape(blocks[i], 0),
                        B.shape(blocks[j], 1),
                    )
                )
        rows.append(row)
    return B.block(*rows)


B.block_diag = block_diag
