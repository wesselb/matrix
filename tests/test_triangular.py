import lab as B

from matrix import LowerTriangular, UpperTriangular

# noinspection PyUnresolvedReferences
from .util import approx, dense1, dense2, diag1


def test_lowertriangular_formatting():
    assert (
        str(LowerTriangular(B.ones(3, 3)))
        == "<lower-triangular matrix: shape=3x3, dtype=float64>"
    )
    assert (
        repr(LowerTriangular(B.ones(3, 3)))
        == "<lower-triangular matrix: shape=3x3, dtype=float64\n"
        " mat=[[1. 1. 1.]\n"
        "      [1. 1. 1.]\n"
        "      [1. 1. 1.]]>"
    )


def test_lowertriangular_attributes():
    mat = B.ones(3, 3)
    lt = LowerTriangular(mat)
    assert lt.mat is mat


def test_conversion_to_lowertriangular(diag1):
    approx(LowerTriangular(diag1), diag1)
    assert isinstance(LowerTriangular(diag1).mat, B.Numeric)


def test_uppertriangular_formatting():
    assert (
        str(UpperTriangular(B.ones(3, 3)))
        == "<upper-triangular matrix: shape=3x3, dtype=float64>"
    )
    assert (
        repr(UpperTriangular(B.ones(3, 3)))
        == "<upper-triangular matrix: shape=3x3, dtype=float64\n"
        " mat=[[1. 1. 1.]\n"
        "      [1. 1. 1.]\n"
        "      [1. 1. 1.]]>"
    )


def test_uppertriangular_attributes():
    mat = B.ones(3, 3)
    ut = UpperTriangular(mat)
    assert ut.mat is mat


def test_conversion_to_uppertriangular(diag1):
    approx(UpperTriangular(diag1), diag1)
    assert isinstance(UpperTriangular(diag1).mat, B.Numeric)
