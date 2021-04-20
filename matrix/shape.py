import lab as B
from plum import Dispatcher

__all__ = [
    "assert_scalar",
    "assert_vector",
    "assert_matrix",
    "assert_square",
    "get_shape",
    "compatible",
    "assert_compatible",
    "broadcast",
]

dispatch = Dispatcher()


def assert_scalar(x, message):
    """Assert that a tensor is a scalar.

    Args
        x (tensor): Tensor that must be a scalar.
        message (str): Error message to raise in the case that `x` is not a
            scalar.
    """
    if B.rank(x) != 0:
        raise AssertionError(message)


def assert_vector(x, message):
    """Assert that a tensor is a vector.

    Args
        x (tensor): Tensor that must be a vector.
        message (str): Error message to raise in the case that `x` is not a
            vector.
    """
    if B.rank(x) != 1:
        raise AssertionError(message)


def assert_matrix(x, message):
    """Assert that a tensor is a matrix.

    Args
        x (tensor): Tensor that must be a matrix.
        message (str): Error message to raise in the case that `x` is not a
            matrix.
    """
    if B.rank(x) != 2:
        raise AssertionError(message)


def assert_square(x, message):
    """Assert that a tensor is a square matrix.

    Args
        x (tensor): Tensor that must be a matrix.
        message (str): Error message to raise in the case that `x` is not a
            square matrix.
    """
    shape = B.shape(x)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise AssertionError(message)


class Dimension:
    """Dimension of a shape.

    Args:
        size (int): Size of the dimension.
    """

    def __init__(self, size):
        self.size = size

    def __eq__(self, other):
        return self.size == other.size

    def __str__(self):
        return str(self.size)

    def __repr__(self):
        return f"Dimension({self.size})"


class Shape:
    """A shape, which comprises dimensions.

    Args:
        *dims (:class:`.shape.Dimension`): Dimensions of the shape.
    """

    def __init__(self, *dims):
        self.dims = dims
        self.rank = len(dims)

    def __eq__(self, other):
        return self.dims == other.dims

    def __str__(self):
        return "({})".format(", ".join(str(d) for d in self.dims))

    def __repr__(self):
        return "Shape({})".format(", ".join(repr(d) for d in self.dims))

    def __getitem__(self, item):
        return self.dims[item]

    def as_tuple(self):
        return tuple(d.size for d in self.dims)


@B.dispatch
def transpose(a: Shape):
    return Shape(*reversed(a.dims))


@dispatch
def get_shape(*xs):
    """Get the shapes of objects as :class:`.shape.Shape` objects.

    Args:
        *xs (object): Objects to get the shape of.

    Returns:
        tuple: Tuple containing the shape objects.
    """
    return tuple(get_shape(x) for x in xs)


@dispatch
def get_shape(x):
    return Shape(*[Dimension(d) for d in B.shape(x)])


@dispatch
def compatible(x1, x2):
    """Assert that two objects have compatible shapes.

    Args:
        x1 (object): First object.
        x2 (object): Second object.

    Returns:
        bool: Boolean indicating whether the two objects have compatible shapes.
    """
    return compatible(*get_shape(x1, x2))


@dispatch
def compatible(s1: Shape, s2: Shape):
    return s1.rank == s2.rank and all(
        compatible(d1, d2) for d1, d2 in zip(s1.dims, s2.dims)
    )


@dispatch
def compatible(d1: Dimension, d2: Dimension):
    return d1.size == 1 or d2.size == 1 or d1.size == d2.size


def assert_compatible(x1, x2):
    """Assert that two objects, shapes, or dimensions are compatible.

    Args:
        x1 (object): First object.
        x2 (object): Second object.
    """
    assert compatible(x1, x2), (
        f"Objects {x1} and {x2} are asserted to be compatible, but they are not."
    )


@dispatch
def broadcast(x1, x2):
    """Broadcast the shapes of two objects.

    Args:
        x1 (object): First object.
        x2 (object): Second object.

    Returns:
        :class:`.shape.Shape`: Broadcasted shape.
    """
    return broadcast(*get_shape(x1, x2))


@dispatch
def broadcast(s1: Shape, s2: Shape):
    assert_compatible(s1, s2)
    return Shape(*(broadcast(d1, d2) for d1, d2 in zip(s1.dims, s2.dims)))


@dispatch
def broadcast(d1: Dimension, d2: Dimension):
    if not compatible(d1, d2):
        raise RuntimeError(f"Cannot broadcast dimensions {d1} and {d2}.")
    return d2 if d1.size == 1 else d1