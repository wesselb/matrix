import lab as B
import pytest

from matrix.shape import (
<<<<<<< HEAD
    assert_vector,
    assert_matrix,
    compatible,
=======
    Dimension,
    Shape,
>>>>>>> master
    assert_compatible,
    assert_matrix,
    assert_scalar,
    assert_vector,
    broadcast,
<<<<<<< HEAD
    expand_and_broadcast,
=======
    compatible,
>>>>>>> master
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
    assert_matrix(B.ones(3, 3), "test")
    assert_matrix(B.ones(4, 3, 3), "test")

    with pytest.raises(AssertionError):
        assert_matrix(B.ones(3), "test")


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
    assert broadcast(*_shapes(B.ones(5, 4), B.ones(5, 4))) == (5, 4)
    assert broadcast(*_shapes(B.ones(5, 4), B.ones(5, 1))) == (5, 4)


def test_expand_and_broadcast():
    assert expand_and_broadcast(*_shapes(B.ones(4), B.ones(5, 4))) == (5, 4)
    assert expand_and_broadcast(*_shapes(B.ones(5, 4), B.ones(4))) == (5, 4)
    assert expand_and_broadcast(*_shapes(B.ones(5, 4), B.ones(5, 4))) == (5, 4)
