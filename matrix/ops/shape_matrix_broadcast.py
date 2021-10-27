import lab as B

from ..shape import expand_and_broadcast

__all__ = []


@B.dispatch
def shape_matrix_broadcast(*elements):
    """Get the broadcasted matrix shape of tensors.

    Args:
        *elements (tensor): Tensors to broadcasts shapes of.

    Returns:
        object: Broadcasted matrix shape.
    """
    return expand_and_broadcast(*(B.shape_matrix(element) for element in elements))


B.shape_matrix_broadcast = shape_matrix_broadcast
