import lab as B

from ..matrix import AbstractMatrix
from ..shape import broadcast
from ..tiledblocks import TiledBlocks

__all__ = []


@B.dispatch
def shape(a: AbstractMatrix):
    return B.shape_batch(a) + B.shape_matrix(a)


def _drop_axis(x, i):
    if i == 0:
        return x[1:]
    if i > 0:
        return x[:i] + x[i + 1 :]
    else:
        i = -i
        x = tuple(reversed(x))
        x = x[: i - 1] + x[i:]
        return tuple(reversed(x))


@B.dispatch
def shape(a: TiledBlocks):
    # Expand the shapes of the blocks.
    block_shapes = [B.shape(block) for block in a.blocks]
    rank = max([len(shape) for shape in block_shapes])
    block_shapes = [(1,) * (rank - len(shape)) + tuple(shape) for shape in block_shapes]

    # Compute the shape except for dimension `a.axis`. Work with the negative version of
    # `a.axis` for convenience.
    if a.axis >= 0:
        axis = a.axis - rank
    else:
        axis = a.axis
    shape = broadcast(*[_drop_axis(shape, axis) for shape in block_shapes])

    # Compute the dimension `a.axis` and add it to `shape`.
    repeated_dim = sum([rep * shape[axis] for rep, shape in zip(a.reps, block_shapes)])
    if axis == -1:
        shape = shape + (repeated_dim,)
    else:
        shape = shape[: axis + 1] + (repeated_dim,) + shape[axis + 1 :]

    return shape
