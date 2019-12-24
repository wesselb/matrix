# noinspection PyUnresolvedReferences
import lab as B
from matrix import LowerTriangular, UpperTriangular


def test_lowertriangular_formatting():
    assert str(LowerTriangular(B.ones(3, 3))) == \
           '<lower-triangular matrix: shape=3x3, dtype=float64>'
    assert repr(LowerTriangular(B.ones(3, 3))) == \
           '<lower-triangular matrix: shape=3x3, dtype=float64\n' \
           ' mat=[[1. 1. 1.]\n' \
           '      [1. 1. 1.]\n' \
           '      [1. 1. 1.]]>'


def test_lowertriangular_attributes():
    mat = B.ones(3, 3)
    lt = LowerTriangular(mat)
    assert lt.mat is mat


def test_uppertriangular_formatting():
    assert str(UpperTriangular(B.ones(3, 3))) == \
           '<upper-triangular matrix: shape=3x3, dtype=float64>'
    assert repr(UpperTriangular(B.ones(3, 3))) == \
           '<upper-triangular matrix: shape=3x3, dtype=float64\n' \
           ' mat=[[1. 1. 1.]\n' \
           '      [1. 1. 1.]\n' \
           '      [1. 1. 1.]]>'


def test_uppertriangular_attributes():
    mat = B.ones(3, 3)
    ut = UpperTriangular(mat)
    assert ut.mat is mat
