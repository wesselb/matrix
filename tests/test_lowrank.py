import lab as B
import pytest

from matrix import LowRank
from .util import allclose


def test_lowrank_formatting():
    assert (
        str(LowRank(B.ones(3, 1), 2 * B.ones(3, 1)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=1, sign=0>"
    )
    assert (
        repr(LowRank(B.ones(3, 2), 2 * B.ones(3, 2)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=2, sign=0\n"
        " left=[[1. 1.]\n"
        "       [1. 1.]\n"
        "       [1. 1.]]\n"
        " right=[[2. 2.]\n"
        "        [2. 2.]\n"
        "        [2. 2.]]>"
    )
    assert (
        repr(LowRank(B.ones(3, 2)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=2, sign=1\n"
        " left=[[1. 1.]\n"
        "       [1. 1.]\n"
        "       [1. 1.]]>"
    )
    assert (
        repr(LowRank(B.ones(3, 2), middle=B.ones(2, 2)))
        == "<low-rank matrix: shape=3x3, dtype=float64, rank=2, sign=1\n"
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
    allclose(lr.middle, B.eye(2))
    assert lr.rank == 2
    assert lr.sign == 1

    # Check given right and middle factor.
    right = 2 * B.ones(3, 2)
    middle = B.ones(2, 2)
    lr = LowRank(left, right, middle=middle)
    assert lr.left is left
    assert lr.right is right
    assert lr.middle is middle
    assert lr.sign == 0

    # Check given right factor and sign.
    lr = LowRank(left, left, sign=1)
    assert lr.sign == 1


def test_lowrank_shape_checks():
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), B.ones(3, 2))
    with pytest.raises(AssertionError):
        LowRank(B.ones(3, 1), B.ones(3, 1), B.eye(2))
