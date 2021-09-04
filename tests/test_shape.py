import lab as B
import pytest

from matrix.shape import (
    compatible,
    assert_compatible,
    assert_vector,
    assert_matrix,
    assert_square,
    broadcast,
    expand_and_broadcast,
)

# noinspection PyUnresolvedReferences
from .util import dense1


def _shapes(*xs):
    return tuple(B.shape(x) for x in xs)


def test_assert_vector():
    assert_vector(B.ones(3), "test")
    assert_vector(B.ones(3, 4), "test")

    with pytest.raises(AssertionError):
        assert_vector(1, "test")


def test_assert_matrix():
    assert_matrix(B.ones(4, 5), "test")
    assert_matrix(B.ones(3, 4, 5), "test")

    with pytest.raises(AssertionError):
        assert_matrix(B.ones(3), "test")


def test_assert_square():
    assert_square(B.ones(3, 3), "test")
    assert_square(B.ones(4, 3, 3), "test")

    with pytest.raises(AssertionError):
        assert_square(B.ones(3), "test")
    with pytest.raises(AssertionError):
        assert_square(B.ones(3, 4), "test")
    with pytest.raises(AssertionError):
        assert_square(B.ones(3, 4, 5), "test")


def test_compatible():
    assert compatible(*_shapes(B.ones(5, 5), B.ones(5, 5)))
    assert compatible(*_shapes(B.ones(5, 5), B.ones(5, 1)))
    assert compatible(*_shapes(B.ones(4), B.ones(5, 4)))
    assert not compatible(*_shapes(B.ones(5, 3), B.ones(5)))
    assert not compatible(*_shapes(B.ones(5, 5), B.ones(5, 4)))
    assert not compatible(*_shapes(B.ones(5), B.ones(5, 4)))

    with pytest.raises(AssertionError):
        assert_compatible(*_shapes(B.ones(5), B.ones(4)))


def test_broadcast():
    # Assertions are tested in `compatible`.

    # Must give at least one shape.
    with pytest.raises(ValueError):
        broadcast()
    # If one thing is given, this is directly given back.
    assert broadcast(1) is 1

    # Shapes must have the same lengths, because this function does not expand.
    with pytest.raises(RuntimeError):
        broadcast(*_shapes(B.ones(5, 4), B.ones(5)))

    assert broadcast(*_shapes(B.ones(5, 4), B.ones(5, 4))) == (5, 4)
    assert broadcast(*_shapes(B.ones(5, 4), B.ones(5, 1))) == (5, 4)

    # Test more than two shapes.
    assert broadcast(*_shapes(B.ones(5, 4), B.ones(1, 1), B.ones(1, 4))) == (5, 4)


def test_expand_and_broadcast():
    # Must give at least one shape.
    with pytest.raises(ValueError):
        expand_and_broadcast()
    # If one thing is given, this is directly given back.
    assert expand_and_broadcast(1) is 1

    assert expand_and_broadcast(*_shapes(B.ones(4), B.ones(5, 4))) == (5, 4)
    assert expand_and_broadcast(*_shapes(B.ones(5, 4), B.ones(4))) == (5, 4)
    assert expand_and_broadcast(*_shapes(B.ones(5, 4), B.ones(5, 4))) == (5, 4)

    # Test more than two shapes.
    shape = expand_and_broadcast(*_shapes(B.ones(4), B.ones(5, 1), B.ones(5, 4)))
    assert shape == (5, 4)
