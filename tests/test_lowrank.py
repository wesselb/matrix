import lab as B
import pytest
from matrix import LowRank


def test_lowrank_str():
    assert str(LowRank(B.ones(3, 1), 2 * B.ones(3, 1))) == \
           '<low-rank matrix: shape=3x3, dtype=float64, rank=1,\n' \
           ' left=[[1.]\n' \
           '       [1.]\n' \
           '       [1.]],\n' \
           ' right=[[2.]\n' \
           '        [2.]\n' \
           '        [2.]]>'


def test_lowrank_col_check():
    with pytest.raises(ValueError):
        LowRank(B.ones(3, 1), B.ones(3, 2))
