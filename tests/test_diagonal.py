from matrix import Diagonal
import lab as B


def test_diagonal_str():
    assert str(Diagonal(B.ones(3))) == \
           '<diagonal matrix: shape=3x3, dtype=float64,\n' \
           ' diag=[1. 1. 1.]>'
