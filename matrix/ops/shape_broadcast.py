import lab as B

from ..shape import expand_and_broadcast

__all__ = []


@B.dispatch
def shape_broadcast(*elements):
    """Get the broadcasted shape of tensors.

    Args:
        *elements (tensor): Tensors to broadcasts shapes of.

    Returns:
        object: Broadcasted shape.
    """
    return expand_and_broadcast(*(B.shape(element) for element in elements))


B.shape_broadcast = shape_broadcast
