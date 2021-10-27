import lab as B

from ..shape import expand_and_broadcast

__all__ = []


@B.dispatch
def shape_batch_broadcast(*elements):
    """Get the broadcasted batch shape of tensors.

    Args:
        *elements (tensor): Tensors to broadcasts shapes of.

    Returns:
        object: Broadcasted batch shape.
    """
    return expand_and_broadcast(*(B.shape_batch(element) for element in elements))


B.shape_batch_broadcast = shape_batch_broadcast
