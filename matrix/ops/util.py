import lab as B
from plum import Union, List, Dispatcher

from ..matrix import AbstractMatrix

__all__ = ["align_batch", "batch_ones"]

_dispatch = Dispatcher()


@_dispatch
def align_batch(*xs: Union[B.Numeric, AbstractMatrix]):
    """Align the batch dimension of matrices.

    Args:
        *xs (matrix): Matrices.

    Returns:
        matrix or tuple[matrix]: `xs` with broadcasted batch dimensions.
    """
    batch = B.shape_batch_broadcast(*xs)
    return B.squeeze([B.broadcast_batch_to(x, *batch) for x in xs])


@_dispatch
def align_batch(*collections: List[Union[B.Numeric, AbstractMatrix]]):
    """Broadcast the batch dimension of dense matrices.

    Args:
        *collections (list[matrix]): Lists of matrices.

    Returns:
        list[list[matrix]]: Inputs with broadcasted batch dimensions.
    """
    batch = B.shape_batch_broadcast(*(x for xs in collections for x in xs))
    return [[B.broadcast_batch_to(x, *batch) for x in xs] for xs in collections]


def batch_ones(a):
    """Return a tuple of ones with length equal to the number of batch dimensions.

    Args:
        a (tensor): Tensor to use batch dimensions from.

    Returns:
        tuple[int]: Tuple of ones with length equal to the number of batch dimensions
            of `a`.
    """
    return (1,) * len(B.shape_batch(a))
