import lab as B
from matrix import Diagonal

# noinspection PyUnresolvedReferences
from .util import approx, dense1


def test_diagonal_formatting():
    assert str(Diagonal(B.ones(3))) == "<diagonal matrix: shape=3x3, dtype=float64>"
    assert (
        repr(Diagonal(B.ones(3))) == "<diagonal matrix: shape=3x3, dtype=float64\n"
        " diag=[1. 1. 1.]>"
    )


def test_diagonal_attributes():
    diag = Diagonal(B.ones(3))
    approx(diag.diag, B.ones(3))


def test_conversion_to_diagonal(dense1):
    approx(Diagonal(dense1), B.diag(B.diag(dense1)))
