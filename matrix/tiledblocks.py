import lab as B
from plum import Dispatcher, Tuple, Union
from lab.shape import Shape

from .matrix import AbstractMatrix, repr_format
from .shape import assert_matrix
from .util import dtype_str, indent

__all__ = ["TiledBlocks"]

_dispatch = Dispatcher()


class TiledBlocks(AbstractMatrix):
    """A concatenation of tiled blocks.

    Attributes:
        blocks (tuple[matrix]): Blocks.
        reps (tuple[int]): Repetition of every block.
        axis (int): Axis along which to concatenate the repeated blocks.
        dense (matrix or None): Dense version of the matrix, once it has been computed.

    Args:
        *blocks (tuple[matrix, int]): Tuples of matrices and integers representing
            how many times to repeat that matrix.
        axis (int, optional): Axis along which to concatenate the repeated blocks.
            Defaults to `-2`.
    """

    @_dispatch
    def __init__(
        self,
        block: Tuple[Union[B.Numeric, AbstractMatrix], B.Int],
        *blocks: Tuple[Union[B.Numeric, AbstractMatrix], B.Int],
        axis=-2,
    ):
        blocks = (block,) + blocks
        for block, rep in blocks:
            assert_matrix(
                block,
                f"Block {block} is not a rank-2 tensor. Can only construct a tile of "
                "blocks matrices from rank-2 tensors.",
            )
        # We could also check if the blocks are properly aligned, but this is pretty
        # complicated, so we leave this to a runtime error.
        self.blocks, self.reps = zip(*blocks)
        self.axis = axis
        self.dense = None

    @_dispatch
    def __init__(self, *blocks: Union[B.Numeric, AbstractMatrix], axis: B.Int = -2):
        TiledBlocks.__init__(self, *((block, 1) for block in blocks), axis=axis)

    @_dispatch
    def __init__(
        self, block: Union[B.Numeric, AbstractMatrix], reps: B.Int, axis: B.Int = -2
    ):
        TiledBlocks.__init__(self, (block, reps), axis=axis)

    def __str__(self):
        return (
            f"<tile of blocks:"
            f" batch={Shape(*B.shape_batch(self))},"
            f" shape={Shape(*B.shape_matrix(self))},"
            f" axis={self.axis},"
            f" dtype={dtype_str(self)}>"
        )

    def __repr__(self):
        out = str(self)[:-1] + "\n"
        for block, rep in zip(self.blocks, self.reps):
            out += f" <block: reps={rep}\n"
            out += (
                f"  matrix=" + indent(repr_format(repr(block)), " " * 10).strip() + "\n"
            )
        return out[:-1] + ">"
