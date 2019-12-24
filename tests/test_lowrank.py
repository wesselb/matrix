import lab as B
import pytest
from matrix import LowRank

from .util import allclose


def test_lowrank_formatting():
    assert str(LowRank(B.ones(3, 1), 2 * B.ones(3, 1))) == \
           '<low-rank matrix: shape=3x3, dtype=float64, rank=1,' \
           ' symmetric=False>'
    assert repr(LowRank(B.ones(3, 2), 2 * B.ones(3, 2))) == \
           '<low-rank matrix: shape=3x3, dtype=float64, rank=2,' \
           ' symmetric=False\n' \
           ' left=[[1. 1.]\n' \
           '       [1. 1.]\n' \
           '       [1. 1.]]\n' \
           ' middle=<diagonal matrix: shape=2x2, dtype=float64\n' \
           '         diag=[1. 1.]>\n' \
           ' right=[[2. 2.]\n' \
           '        [2. 2.]\n' \
           '        [2. 2.]]>'


def test_lowrank_attributes():
    # Check the case where the left and right factor are given.
    left = B.ones(3, 1)
    right = 2 * B.ones(3, 1)
    lr = LowRank(left, right)
    assert lr.left is left
    allclose(B.dense(lr.middle), B.ones(1, 1))
    assert lr.right is right
    assert lr.rank == 1
    assert not lr.symmetric

    # Check the case where the left and middle factor are given.
    middle = B.ones(2, 2)
    lr = LowRank(B.ones(3, 2), middle=middle)
    assert lr.symmetric
    assert lr.middle is middle


def test_lowrank_shape_checks():
    with pytest.raises(ValueError):
        LowRank(B.ones(3, 1), B.ones(3, 2))
    with pytest.raises(ValueError):
        LowRank(B.ones(3, 1), B.ones(3, 1), B.eye(2))
