import lab as B
import pytest
from matrix import LowRank


def test_lowrank_str():
    assert str(LowRank(B.ones(3, 1), 2 * B.ones(3, 1))) == \
           'Rank-1 3x3 matrix of data type float64 with left factor\n' \
           '  [[1.]\n' \
           '   [1.]\n' \
           '   [1.]]\n' \
           'and right factor\n' \
           '  [[2.]\n' \
           '   [2.]\n' \
           '   [2.]]'


def test_lowrank_col_check():
    with pytest.raises(ValueError):
        LowRank(B.ones(3, 1), B.ones(3, 2))
