import lab as B
from matrix import Diagonal, Kronecker


def test_kronecker_formatting():
    d1 = Diagonal(B.ones(2))
    d2 = Diagonal(B.ones(3))
    assert str(Kronecker(d1, d2)) == \
           '<Kronecker product: shape=6x6, dtype=float64>'
    assert repr(Kronecker(d1, d2)) == \
           '<Kronecker product: shape=6x6, dtype=float64\n' \
           ' left=<diagonal matrix: shape=2x2, dtype=float64\n' \
           '       diag=[1. 1.]>\n' \
           ' right=<diagonal matrix: shape=3x3, dtype=float64\n' \
           '        diag=[1. 1. 1.]>>'
