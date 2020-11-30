import lab as B
from matrix import Diagonal, Kronecker


def test_kronecker_formatting():
    left = Diagonal(B.ones(2))
    right = Diagonal(B.ones(3))
    assert (
        str(Kronecker(left, right)) == "<Kronecker product: shape=6x6, dtype=float64>"
    )
    assert (
        repr(Kronecker(left, right)) == "<Kronecker product: shape=6x6, dtype=float64\n"
        " left=<diagonal matrix: shape=2x2, dtype=float64\n"
        "       diag=[1. 1.]>\n"
        " right=<diagonal matrix: shape=3x3, dtype=float64\n"
        "        diag=[1. 1. 1.]>>"
    )


def test_kronecker_attributes():
    left = Diagonal(B.ones(2))
    right = Diagonal(B.ones(3))
    kron = Kronecker(left, right)
    assert kron.left is left
    assert kron.right is right
