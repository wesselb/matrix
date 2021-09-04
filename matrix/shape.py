import lab as B
from lab.shape import Shape, Dimension
from plum import Dispatcher

__all__ = [
    "assert_vector",
    "assert_matrix",
    "assert_square",
    "compatible",
    "assert_compatible",
    "broadcast",
    "expand_and_broadcast",
]

_dispatch = Dispatcher()


def assert_vector(x, message):
    """Assert that a tensor is a vector or a batch of vectors.

    Args
        x (tensor): Tensor that must be a vector.
        message (str): Error message to raise in the case that `x` is not a
            vector.
    """
    if B.rank(x) < 1:
        raise AssertionError(message)


def assert_matrix(x, message):
    """Assert that a tensor is a matrix or batch of matrices.

    Args
        x (tensor): Tensor that must be a matrix.
        message (str): Error message to raise in the case that `x` is not a
            matrix.
    """
    if B.rank(x) < 2:
        raise AssertionError(message)


def assert_square(x, message):
    """Assert that a tensor is a square matrix or a batch of square matrices.

    Args
        x (tensor): Tensor that must be a matrix.
        message (str): Error message to raise in the case that `x` is not a
            square matrix.
    """
    assert_matrix(x, message)
    shape = B.shape_matrix(x)
    if shape[0] != shape[1]:
        raise AssertionError(message)


@_dispatch
def compatible(s1: Shape, s2: Shape):
    """Assert that two shapes are compatible shapes.

    Args:
        s1 (:class:`lab.shape.Shape`): First shape.
        s2 (:class:`lab.shape.Shape`): Second shape.

    Returns:
        bool: Boolean indicating whether the two shapes are compatible.
    """
    try:
        expand_and_broadcast(s1, s2)
        return True
    except RuntimeError:
        return False


@_dispatch
def compatible(*xs):
    return compatible(*(Shape(*x) for x in xs))


def assert_compatible(s1, s2):
    """Assert that two shapes are compatible.

    Args:
        s1 (:class:`lab.shape.Shape`): First shape.
        s2 (:class:`lab.shape.Shape`): Second shape.
    """
    if not compatible(s1, s2):
        raise AssertionError(
            f"Objects {s1} and {s2} are asserted to be compatible, but they are not."
        )


@_dispatch
def broadcast(s1: Shape, s2: Shape):
    """Broadcast the shapes of two objects.

    Args:
        s1 (:class:`lab.shape.Shape`): First shape.
        s2 (:class:`lab.shape.Shape`): Second shape.

    Returns:
        :class:`lab.shape.Shape`: Broadcasted shape.
    """
    if len(s1) != len(s2):
        raise RuntimeError(f"Cannot broadcast shapes {s1} and {s2}.")
    return Shape(*(broadcast(d1, d2) for d1, d2 in zip(s1, s2)))


@_dispatch
def broadcast(shape1: Shape, shape2: Shape, *further_shapes: Shape):
    running_shape = broadcast(shape1, shape2)
    for s in further_shapes:
        running_shape = broadcast(running_shape, s)
    return running_shape


@_dispatch
def broadcast(d1: Dimension, d2: Dimension):
    if d1 != 1 and d2 != 1 and d1 != d2:
        raise RuntimeError(f"Cannot broadcast dimensions with values {d1} and {d2}.")
    return d2 if d1 == 1 else d1


@_dispatch
def broadcast(*xs):
    if len(xs) == 0:
        raise ValueError("No shapes to broadcast.")
    elif len(xs) == 1:
        return xs[0]
    else:
        return broadcast(*(Shape(*x) for x in xs))


@_dispatch
def expand_and_broadcast(s1: Shape, s2: Shape):
    """Expand two shapes to make them of equal rank and then broadcast.

    Args:
        s1 (:class:`lab.shape.Shape`): First shape.
        s2 (:class:`lab.shape.Shape`): Second shape.

    Returns:
        :class:`lab.shape.Shape`: Expanded and broadcasted shape.
    """
    len_diff = len(s1) - len(s2)
    if len_diff < 0:
        return broadcast((1,) * -len_diff + s1, s2)
    elif len_diff > 0:
        return broadcast(s1, (1,) * len_diff + s2)
    else:
        return broadcast(s1, s2)


@_dispatch
def expand_and_broadcast(shape1: Shape, shape2: Shape, *further_shapes: Shape):
    running_shape = expand_and_broadcast(shape1, shape2)
    for s in further_shapes:
        running_shape = expand_and_broadcast(running_shape, s)
    return running_shape


@_dispatch
def expand_and_broadcast(*xs):
    if len(xs) == 0:
        raise ValueError("No shapes to broadcast.")
    elif len(xs) == 1:
        return xs[0]
    else:
        return expand_and_broadcast(*(Shape(*x) for x in xs))
