import lab as B
import pytest

from matrix.shape import (
    assert_scalar,
    assert_vector,
    assert_matrix,
    Dimension,
    Shape,
    compatible,
    assert_compatible,
    broadcast,
)

# noinspection PyUnresolvedReferences
from .util import dense1


def test_assert_scalar():
    assert_scalar(1, "test")

    with pytest.raises(AssertionError):
        assert_scalar(B.ones(3), "test")


def test_assert_vector():
    assert_vector(B.ones(3), "test")

    with pytest.raises(AssertionError):
        assert_vector(1, "test")


def test_assert_matrix():
    assert_matrix(B.ones(3, 3), "test")

    with pytest.raises(AssertionError):
        assert_matrix(B.ones(3), "test")


def test_dimension():
    d = Dimension(5)

    assert d.size == 5
    assert d == d
    assert d != Dimension(1)
    assert d != Dimension(3)
    assert str(d) == "5"
    assert repr(d) == "Dimension(5)"


def test_shape():
    s = Shape(Dimension(5), Dimension(4))

    assert s.rank == 2
    assert s == s
    assert s != Shape(Dimension(5), Dimension(1))
    assert s != Shape(Dimension(5), Dimension(3))
    assert str(s) == "(5, 4)"
    assert repr(s) == "Shape(Dimension(5), Dimension(4))"
    assert s[0] == Dimension(5)
    assert s[1] == Dimension(4)
    assert s.as_tuple() == (5, 4)


def test_compatible():
    assert compatible(B.ones(5, 5), B.ones(5, 5))
    assert compatible(B.ones(5, 5), B.ones(5, 1))
    assert not compatible(B.ones(5, 5), B.ones(5))
    assert not compatible(B.ones(5, 5), B.ones(5, 4))

    with pytest.raises(AssertionError):
        assert_compatible(B.ones(5), B.ones(4))


def test_broadcast():
    s = Shape(Dimension(5), Dimension(4))
    assert broadcast(B.ones(5, 4), B.ones(5, 4)) == s
    assert broadcast(B.ones(5, 4), B.ones(5, 1)) == s

    with pytest.raises(AssertionError):
        broadcast(B.ones(5, 5), B.ones(5, 4))
    with pytest.raises(RuntimeError):
        broadcast(Dimension(5), Dimension(4))
