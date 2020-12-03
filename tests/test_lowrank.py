import lab as B
import pytest

from matrix import LowRank
from .util import approx


def test_lowrank_formatting():
    assert (
        str(LowRank(B.ones(3, 1), 2 * B.ones(3, 1)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=1>"
    )
    assert (
        repr(LowRank(B.ones(3, 2), 2 * B.ones(3, 2)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=2\n"
        " left=[[1. 1.]\n"
        "       [1. 1.]\n"
        "       [1. 1.]]\n"
        " right=[[2. 2.]\n"
        "        [2. 2.]\n"
        "        [2. 2.]]>"
    )
    assert (
        repr(LowRank(B.ones(3, 2)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=2\n"
        " left=[[1. 1.]\n"
        "       [1. 1.]\n"
        "       [1. 1.]]>"
    )
    assert (
        repr(LowRank(B.ones(3, 2), middle=B.ones(2, 2)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=2\n"
        " left=[[1. 1.]\n"
        "       [1. 1.]\n"
        "       [1. 1.]]\n"
        " middle=[[1. 1.]\n"
        "         [1. 1.]]>"
    )


def test_lowrank_attributes():
    # Check default right and middle factor.
    left = B.ones(3, 2)
    lr = LowRank(left)
    assert lr.left is left
    assert lr.right is left
    approx(lr.middle, B.eye(2))
    assert lr.rank == 2

    # Check given identical right.
    lr = LowRank(left, left)
    assert lr.left is left
    assert lr.right is left
    approx(lr.middle, B.eye(2))
    assert lr.rank == 2

    # Check given identical right and middle.
    middle = B.ones(2, 2)
    lr = LowRank(left, left, middle=middle)
    assert lr.left is left
    assert lr.right is left
    approx(lr.middle, B.ones(2, 2))
    assert lr.rank == 2

    # Check given other right and middle factor.
    right = 2 * B.ones(3, 2)
    lr = LowRank(left, right, middle=middle)
    assert lr.left is left
    assert lr.right is right
    approx(lr.middle, B.ones(2, 2))
    assert lr.rank == 2

    # Check rank in non-square case.
    assert LowRank(B.ones(3, 2), B.ones(3, 2), B.ones(2, 2)).rank == 2
    assert LowRank(B.ones(3, 2), B.ones(3, 1), B.ones(2, 1)).rank == 1


def test_lowrank_shape_checks():
    # Check that matrices must be given.
    with pytest.raises(AssertionError):
        LowRank(B.ones(3))
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), B.ones(3))
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), B.ones(3, 1), B.ones(1))

    # Check that needs need to be compatible
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), B.ones(3, 2))
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), B.ones(3, 2), B.ones(1, 1))
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), middle=B.ones(2, 1))
