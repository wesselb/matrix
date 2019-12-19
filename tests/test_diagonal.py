from matrix import Diagonal
import lab as B


def test_diagonal_str():
    assert str(Diagonal(B.ones(3))) == \
           '<diagonal matrix: shape=3x3, data type=float64,\n' \
           ' diagonal=[1. 1. 1.]>'
