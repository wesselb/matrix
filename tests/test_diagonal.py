from matrix import Diagonal
import lab as B


def test_diagonal_str():
    assert str(Diagonal(B.ones(3))) == \
           'Diagonal matrix with diagonal\n  [1. 1. 1.]'
